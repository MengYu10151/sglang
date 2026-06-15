from __future__ import annotations

import logging
from enum import Enum, auto
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    EpV2OutputDtype,
    get_epv2_runner_capability,
)

logger = logging.getLogger(__name__)

_SCALE_BLOCK_SIZE = 128
_epv2_import_error: Optional[BaseException] = None
_fp8_quant_import_error: Optional[BaseException] = None
sglang_per_token_group_quant_fp8 = None

try:
    from deep_ep import ElasticBuffer

    use_epv2 = True
except (ImportError, OSError) as exc:
    use_epv2 = False
    _epv2_import_error = exc

if use_epv2:
    try:
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )
    except (ImportError, OSError) as exc:
        _fp8_quant_import_error = exc


class EpV2DispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.EPV2


class EpV2CombineInput(NamedTuple):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.EPV2


assert isinstance(EpV2DispatchOutput, DispatchOutput)
assert isinstance(EpV2CombineInput, CombineInput)


def _raise_epv2_import_error() -> None:
    detail = (
        f" Original import error: {_epv2_import_error}"
        if _epv2_import_error is not None
        else ""
    )
    raise ImportError(
        "DeepEP v2 (ElasticBuffer) is not available. Install DeepEP v2 from "
        "https://github.com/deepseek-ai/DeepEP."
        + detail
    )


def _ensure_epv2_available() -> None:
    if not use_epv2:
        _raise_epv2_import_error()


def _ensure_fp8_quant_available() -> None:
    _ensure_epv2_available()
    if sglang_per_token_group_quant_fp8 is None:
        detail = (
            f" Original import error: {_fp8_quant_import_error}"
            if _fp8_quant_import_error is not None
            else ""
        )
        raise ImportError(
            "DeepEP v2 FP8 dispatch requires the SGLang FP8 quantization kernel."
            + detail
        )


def _quantize_for_epv2_dispatch(hidden_states: torch.Tensor):
    _ensure_fp8_quant_available()
    return sglang_per_token_group_quant_fp8(
        hidden_states,
        _SCALE_BLOCK_SIZE,
        column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )


class EpV2Buffer:
    _buffer: Optional["ElasticBuffer"] = None
    _buffer_key: Optional[Tuple] = None

    @classmethod
    def get_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        router_topk: int,
        num_max_dispatch_tokens_per_rank: int,
        use_fp8_dispatch: bool,
    ) -> "ElasticBuffer":
        _ensure_epv2_available()

        allow_hybrid_mode = envs.SGLANG_EPV2_ALLOW_HYBRID_MODE.get()
        key = (
            id(group),
            hidden_size,
            router_topk,
            num_max_dispatch_tokens_per_rank,
            use_fp8_dispatch,
            allow_hybrid_mode,
            dist.get_world_size(group),
        )
        if cls._buffer is not None and cls._buffer_key == key:
            return cls._buffer

        if cls._buffer is not None:
            cls.destroy()

        cls._buffer = ElasticBuffer(
            group,
            num_max_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            hidden=hidden_size,
            num_topk=router_topk,
            use_fp8_dispatch=use_fp8_dispatch,
            allow_hybrid_mode=allow_hybrid_mode,
        )
        cls._buffer_key = key
        logger.info(
            "Initialized DeepEP v2 ElasticBuffer: world_size=%s hidden_size=%s "
            "num_topk=%s max_dispatch_tokens_per_rank=%s use_fp8_dispatch=%s "
            "allow_hybrid_mode=%s num_bytes=%s",
            dist.get_world_size(group),
            hidden_size,
            router_topk,
            num_max_dispatch_tokens_per_rank,
            use_fp8_dispatch,
            allow_hybrid_mode,
            cls._buffer.num_bytes,
        )
        return cls._buffer

    @classmethod
    def destroy(cls) -> None:
        cls._buffer = None
        cls._buffer_key = None


class _EpV2Impl:
    def __init__(
        self,
        group: dist.ProcessGroup,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        output_dtype: EpV2OutputDtype,
        expert_alignment: int,
        num_max_dispatch_tokens_per_rank: int,
    ):
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.output_dtype = output_dtype
        self.expert_alignment = expert_alignment
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.rank = dist.get_rank(group)
        self._handle = None
        self._num_input_tokens = 0

    def set_runner_capability(
        self, output_dtype: EpV2OutputDtype, expert_alignment: int
    ) -> None:
        if (
            self.output_dtype != output_dtype
            or self.expert_alignment != expert_alignment
        ):
            self._destroy_handle()
            self.output_dtype = output_dtype
            self.expert_alignment = expert_alignment

    def _uses_fp8_dispatch_output(self) -> bool:
        return self.output_dtype == EpV2OutputDtype.FP8

    def _destroy_handle(self) -> None:
        self._handle = None

    def _get_buffer(self) -> "ElasticBuffer":
        return EpV2Buffer.get_buffer(
            self.group,
            self.hidden_size,
            self.router_topk,
            self.num_max_dispatch_tokens_per_rank,
            self._uses_fp8_dispatch_output(),
        )

    def _validate_common(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor
    ) -> None:
        if hidden_states.shape[0] > self.num_max_dispatch_tokens_per_rank:
            raise ValueError(
                f"DeepEP v2 dispatch input exceeds the per-rank buffer capacity "
                f"{self.num_max_dispatch_tokens_per_rank}, got {hidden_states.shape[0]}. "
                "Increase SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK."
            )
        if hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"DeepEP v2 hidden size mismatch: expected {self.hidden_size}, "
                f"got {hidden_states.shape[1]}"
            )
        if self._uses_fp8_dispatch_output() and self.hidden_size % _SCALE_BLOCK_SIZE != 0:
            raise ValueError(
                "DeepEP v2 FP8 dispatch requires hidden_size multiple of "
                f"{_SCALE_BLOCK_SIZE}, got {self.hidden_size}"
            )
        if topk_ids.shape[1] != self.router_topk:
            raise ValueError(
                f"DeepEP v2 topk mismatch: expected {self.router_topk}, "
                f"got {topk_ids.shape[1]}"
            )

    @staticmethod
    def _to_local_topk_ids(
        topk_ids: torch.Tensor,
        rank: int,
        num_local_experts: int,
    ) -> torch.Tensor:
        valid_topk = topk_ids[topk_ids >= 0]
        if valid_topk.numel() == 0:
            return topk_ids
        if int(valid_topk.max().item()) < num_local_experts:
            local_topk_ids = topk_ids
        else:
            local_expert_start = rank * num_local_experts
            local_topk_ids = topk_ids - local_expert_start

        local_mask = (local_topk_ids >= 0) & (local_topk_ids < num_local_experts)
        return torch.where(
            local_mask, local_topk_ids, torch.full_like(local_topk_ids, -1)
        )

    def dispatch(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        _ensure_epv2_available()
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids.to(torch.int64)
        self._validate_common(hidden_states, topk_ids)
        self._num_input_tokens = hidden_states.shape[0]

        if self._uses_fp8_dispatch_output():
            dispatch_x = _quantize_for_epv2_dispatch(hidden_states)
            use_tma_aligned_col_major_sf = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        else:
            dispatch_x = hidden_states
            use_tma_aligned_col_major_sf = False

        buffer = self._get_buffer()
        self._destroy_handle()
        recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
            dispatch_x,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            num_experts=self.num_experts,
            num_max_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            expert_alignment=self.expert_alignment,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            # None keeps ElasticBuffer's documented default: CPU sync is used
            # for a fresh handle so exact receive counts are available.
            do_cpu_sync=None,
        )
        if event.event is not None:
            event.current_stream_wait()
        self._handle = handle

        if isinstance(recv_x, tuple):
            recv_hidden_states, recv_hidden_states_scale = recv_x
        else:
            recv_hidden_states = recv_x
            recv_hidden_states_scale = None

        num_recv_tokens = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
        recv_topk_idx = recv_topk_idx[:num_recv_tokens]
        recv_topk_weights = recv_topk_weights[:num_recv_tokens]
        recv_hidden_states = recv_hidden_states[:num_recv_tokens]
        if recv_hidden_states_scale is not None:
            recv_hidden_states_scale = recv_hidden_states_scale[:num_recv_tokens]

        local_topk_ids = self._to_local_topk_ids(
            recv_topk_idx, self.rank, self.num_local_experts
        )
        num_recv_tokens_per_expert = list(handle.num_recv_tokens_per_expert_list)

        return EpV2DispatchOutput(
            recv_hidden_states,
            recv_hidden_states_scale,
            local_topk_ids,
            recv_topk_weights,
            num_recv_tokens_per_expert,
        )

    def combine(self, combine_input: EpV2CombineInput) -> torch.Tensor:
        if self._handle is None:
            raise RuntimeError("DeepEP v2 combine called without a valid dispatch handle")

        buffer = self._get_buffer()
        try:
            combined_x, _, event = buffer.combine(
                combine_input.hidden_states,
                handle=self._handle,
                topk_weights=combine_input.topk_weights,
            )
            if event.event is not None:
                event.current_stream_wait()
            return combined_x
        finally:
            self._destroy_handle()


class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH = auto()


class EpV2Dispatcher(BaseDispatcher):
    def __init__(
        self,
        group: dist.ProcessGroup,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
    ):
        super().__init__()
        if params_dtype != torch.bfloat16:
            raise NotImplementedError(
                "DeepEP v2 dispatch adapter currently expects BF16 model activations, "
                f"got {params_dtype}"
            )
        self.quant_config = {}
        capability = get_epv2_runner_capability(self)
        self.output_dtype = capability.output_dtype
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        self._impl = _EpV2Impl(
            group=group,
            router_topk=router_topk,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            output_dtype=self.output_dtype,
            expert_alignment=capability.expert_alignment,
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
        )
        self._stage = _Stage.INITIAL

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config
        capability = get_epv2_runner_capability(self)
        self.output_dtype = capability.output_dtype
        self._impl.set_runner_capability(
            capability.output_dtype, capability.expert_alignment
        )

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        if self._stage != _Stage.INITIAL:
            raise RuntimeError(
                f"DeepEP v2 dispatch called in invalid stage: {self._stage}"
            )
        out = self._impl.dispatch(hidden_states, topk_output)
        self._stage = _Stage.AFTER_DISPATCH
        return out

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        if self._stage != _Stage.AFTER_DISPATCH:
            raise RuntimeError(
                f"DeepEP v2 combine called in invalid stage: {self._stage}"
            )
        if combine_input.format != CombineInputFormat.EPV2:
            raise TypeError(
                f"Expected DeepEP v2 combine input, got {combine_input.format}"
            )
        try:
            return self._impl.combine(combine_input)
        finally:
            self._stage = _Stage.INITIAL
