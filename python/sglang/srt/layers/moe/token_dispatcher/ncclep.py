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
from sglang.srt.layers.moe.utils import NcclEpMode
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

logger = logging.getLogger(__name__)

_SCALE_BLOCK_SIZE = 128

try:
    from nccl.ep import (
        Algorithm,
        CombineInputs,
        CombineOutputs,
        DispatchConfig,
        DispatchInputs,
        DispatchOutputs,
        Group,
        GroupConfig,
        Layout,
        LayoutInfo,
        PassDir,
        Tensor,
    )
    from nccl.ep.interop.torch import get_nccl_comm_from_group

    use_ncclep = True
except ImportError:
    use_ncclep = False


class NcclEpHighThroughputDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.NCCL_EP_HT


class NcclEpLowLatencyDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    masked_m: torch.Tensor
    expected_m: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.NCCL_EP_LL


assert isinstance(NcclEpHighThroughputDispatchOutput, DispatchOutput)
assert isinstance(NcclEpLowLatencyDispatchOutput, DispatchOutput)


class NcclEpHighThroughputCombineInput(NamedTuple):
    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.NCCL_EP_HT


class NcclEpLowLatencyCombineInput(NamedTuple):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.NCCL_EP_LL


assert isinstance(NcclEpHighThroughputCombineInput, CombineInput)
assert isinstance(NcclEpLowLatencyCombineInput, CombineInput)


def _algorithm(mode: NcclEpMode):
    if mode.is_high_throughput():
        return Algorithm.HIGH_THROUGHPUT
    if mode.is_low_latency():
        return Algorithm.LOW_LATENCY
    raise ValueError(f"Unsupported NCCL_EP mode: {mode}")


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def _quantize_for_ncclep_dispatch(hidden_states: torch.Tensor):
    return sglang_per_token_group_quant_fp8(
        hidden_states,
        _SCALE_BLOCK_SIZE,
        column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )


class NcclEpBuffer:
    _comm = None
    _ep_group: Optional["Group"] = None
    _group_key: Optional[Tuple] = None

    @classmethod
    def get_group(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        mode: NcclEpMode,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        num_local_experts: int,
    ):
        if not use_ncclep:
            raise ImportError(
                "NCCL_EP is not available. Build/install nccl4py from "
                "<nccl>/bindings/nccl4py and ensure libnccl_ep.so is loadable."
            )

        world_size = dist.get_world_size(group)
        # NCCL_EP HT FLAT receives at most one row from each source rank for a
        # dispatched token. Match the native Python UT / README group-config
        # budget: max_recv_tokens_per_rank = nRanks * max_dispatch_tokens_per_rank.
        max_recv_tokens_per_rank = num_max_dispatch_tokens_per_rank * world_size
        key = (
            id(group),
            hidden_size,
            mode.value,
            num_max_dispatch_tokens_per_rank,
            max_recv_tokens_per_rank,
            num_experts,
            num_local_experts,
            world_size,
        )
        if cls._ep_group is not None and cls._group_key == key:
            return cls._comm, cls._ep_group

        if cls._ep_group is not None:
            cls.destroy()

        cls._comm = get_nccl_comm_from_group(group)
        config = GroupConfig(
            algorithm=_algorithm(mode),
            num_experts=num_experts,
            max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            max_recv_tokens_per_rank=max_recv_tokens_per_rank,
            max_token_bytes=hidden_size * 2,
        )
        cls._ep_group = Group.create(cls._comm, config)
        cls._group_key = key
        logger.info(
            "Initialized NCCL_EP group: mode=%s world_size=%s num_experts=%s "
            "num_local_experts=%s max_dispatch_tokens_per_rank=%s "
            "max_recv_tokens_per_rank=%s hidden_size=%s",
            mode.value,
            world_size,
            num_experts,
            num_local_experts,
            num_max_dispatch_tokens_per_rank,
            max_recv_tokens_per_rank,
            hidden_size,
        )
        return cls._comm, cls._ep_group

    @classmethod
    def destroy(cls):
        if cls._ep_group is not None:
            cls._ep_group.destroy()
        cls._ep_group = None
        cls._comm = None
        cls._group_key = None


class _NcclEpImplBase:
    def __init__(
        self,
        group: dist.ProcessGroup,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        mode: NcclEpMode,
        num_max_dispatch_tokens_per_rank: int,
    ):
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.mode = mode
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self._handle = None

    @property
    def max_recv_tokens_per_rank(self) -> int:
        return self.num_max_dispatch_tokens_per_rank * self.world_size

    def _get_ep(self):
        return NcclEpBuffer.get_group(
            self.group,
            self.hidden_size,
            self.mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            self.num_local_experts,
        )

    def _destroy_handle(self) -> None:
        if self._handle is not None:
            self._handle.destroy()
            self._handle = None

    def _validate_common(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor
    ) -> None:
        if hidden_states.shape[0] > self.num_max_dispatch_tokens_per_rank:
            raise ValueError(
                f"NCCL_EP {self.mode.value} requires tokens per rank <= "
                f"{self.num_max_dispatch_tokens_per_rank}, got {hidden_states.shape[0]}. "
                "Increase SGLANG_NCCL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK."
            )
        if hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"NCCL_EP hidden size mismatch: expected {self.hidden_size}, got {hidden_states.shape[1]}"
            )
        if self.hidden_size % _SCALE_BLOCK_SIZE != 0:
            raise ValueError(
                f"NCCL_EP FP8 dispatch requires hidden_size multiple of {_SCALE_BLOCK_SIZE}, got {self.hidden_size}"
            )
        if topk_ids.shape[1] != self.router_topk:
            raise ValueError(
                f"NCCL_EP topk mismatch: expected {self.router_topk}, got {topk_ids.shape[1]}"
            )


class _NcclEpHighThroughputImpl(_NcclEpImplBase):
    def __init__(self, **kwargs):
        super().__init__(mode=NcclEpMode.HIGH_THROUGHPUT, **kwargs)
        self._recv_tokens: Optional[torch.Tensor] = None
        self._recv_scales: Optional[torch.Tensor] = None
        self._recv_topk_ids: Optional[torch.Tensor] = None
        self._recv_topk_weights: Optional[torch.Tensor] = None
        self._expert_counters: Optional[torch.Tensor] = None
        self._recv_total_counter: Optional[torch.Tensor] = None
        self._num_recv_tokens = 0
        self._num_input_tokens = 0

    def _ensure_buffers(self, max_recv_tokens: int) -> None:
        if (
            self._recv_tokens is not None
            and self._recv_tokens.shape[0] >= max_recv_tokens
        ):
            return
        self._recv_tokens = torch.empty(
            (max_recv_tokens, self.hidden_size),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        self._recv_scales = torch.empty(
            (max_recv_tokens, self.hidden_size // _SCALE_BLOCK_SIZE),
            dtype=torch.float32,
            device="cuda",
        )
        self._recv_topk_ids = torch.empty(
            (max_recv_tokens, self.router_topk), dtype=torch.int64, device="cuda"
        )
        self._recv_topk_weights = torch.empty(
            (max_recv_tokens, self.router_topk), dtype=torch.float32, device="cuda"
        )
        self._expert_counters = torch.zeros(
            self.num_local_experts, dtype=torch.int32, device="cuda"
        )
        self._recv_total_counter = torch.zeros(1, dtype=torch.int32, device="cuda")

    def dispatch(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids.to(torch.int64)
        self._validate_common(hidden_states, topk_ids)
        q_hidden_states, hidden_states_scale = _quantize_for_ncclep_dispatch(
            hidden_states
        )

        _, ep_group = self._get_ep()
        num_tokens = hidden_states.shape[0]
        self._num_input_tokens = num_tokens
        max_recv_tokens = self.max_recv_tokens_per_rank
        self._ensure_buffers(max_recv_tokens)
        self._expert_counters.zero_()
        self._recv_total_counter.zero_()
        self._recv_tokens.zero_()
        self._recv_scales.zero_()
        self._recv_topk_ids.fill_(-1)
        self._recv_topk_weights.zero_()

        self._destroy_handle()
        layout_info = LayoutInfo(expert_counters=Tensor(self._expert_counters))
        self._handle = ep_group.create_handle(
            Layout.FLAT,
            Tensor(topk_ids),
            stream=_stream(),
        )

        self._handle.dispatch(
            DispatchInputs(
                tokens=Tensor(q_hidden_states),
                topk_weights=Tensor(topk_weights),
                scales=Tensor(hidden_states_scale),
            ),
            DispatchOutputs(
                tokens=Tensor(self._recv_tokens),
                topk_weights=Tensor(self._recv_topk_weights),
                topk_idx=Tensor(self._recv_topk_ids),
                scales=Tensor(self._recv_scales),
            ),
            layout_info=layout_info,
            config=DispatchConfig(round_scales=0, pass_direction=PassDir.FWD),
            stream=_stream(),
        )
        self._handle.complete(stream=_stream())
        torch.cuda.synchronize()

        reported_recv = int(self._expert_counters.sum().item())
        valid_rows = (self._recv_topk_ids >= 0).any(dim=1)
        inferred_recv = int(valid_rows.sum().item())
        # NCCL_EP HT FLAT uses the static max_recv_tokens_per_rank slot space
        # in the handle metadata. Keep the full slot space through local MoE
        # and combine; invalid slots remain masked by topk_idx == -1.
        self._num_recv_tokens = max_recv_tokens

        recv_topk_ids = self._recv_topk_ids[: self._num_recv_tokens]
        recv_topk_weights = self._recv_topk_weights[: self._num_recv_tokens]

        valid_recv_topk = recv_topk_ids[recv_topk_ids >= 0]
        if valid_recv_topk.numel() == 0:
            local_topk_ids = recv_topk_ids
        elif int(valid_recv_topk.max().item()) < self.num_local_experts:
            # The current nccl.ep HT FLAT Python path returns local expert ids.
            local_topk_ids = recv_topk_ids
        else:
            # Keep a fallback for native paths that return global expert ids.
            local_expert_start = self.rank * self.num_local_experts
            local_topk_ids = recv_topk_ids - local_expert_start

        local_mask = (local_topk_ids >= 0) & (local_topk_ids < self.num_local_experts)
        local_topk_ids = torch.where(
            local_mask, local_topk_ids, torch.full_like(local_topk_ids, -1)
        )

        counts = [0] * self.num_local_experts
        if local_mask.any():
            count_tensor = torch.bincount(
                local_topk_ids[local_mask], minlength=self.num_local_experts
            ).cpu()
            counts = [int(x) for x in count_tensor.tolist()[: self.num_local_experts]]
        elif self._expert_counters is not None:
            counts = [int(x) for x in self._expert_counters.cpu().tolist()]

        return NcclEpHighThroughputDispatchOutput(
            self._recv_tokens[: self._num_recv_tokens],
            self._recv_scales[: self._num_recv_tokens],
            local_topk_ids,
            recv_topk_weights,
            counts,
        )

    def combine(self, combine_input: NcclEpHighThroughputCombineInput) -> torch.Tensor:
        output = combine_input.hidden_states
        combined = torch.empty(
            (self._num_input_tokens, self.hidden_size),
            dtype=torch.bfloat16,
            device=output.device,
        )
        self._handle.combine(
            CombineInputs(tokens=Tensor(output)),
            CombineOutputs(tokens=Tensor(combined)),
            stream=_stream(),
        )
        self._handle.complete(stream=_stream())
        torch.cuda.synchronize()
        self._destroy_handle()
        return combined


class _NcclEpLowLatencyImpl(_NcclEpImplBase):
    def __init__(self, **kwargs):
        super().__init__(mode=NcclEpMode.LOW_LATENCY, **kwargs)
        self._output_tokens: Optional[torch.Tensor] = None
        self._output_scales: Optional[torch.Tensor] = None
        self._recv_count: Optional[torch.Tensor] = None

    def _ensure_buffers(self) -> None:
        max_slots = self.max_recv_tokens_per_rank
        if self._output_tokens is not None:
            return
        self._output_tokens = torch.empty(
            (self.num_local_experts, max_slots, self.hidden_size),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        self._output_scales = torch.empty(
            (self.num_local_experts, max_slots, self.hidden_size // _SCALE_BLOCK_SIZE),
            dtype=torch.float32,
            device="cuda",
        )
        self._recv_count = torch.zeros(
            self.num_local_experts, dtype=torch.int32, device="cuda"
        )

    def dispatch(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids.to(torch.int64)
        self._validate_common(hidden_states, topk_ids)
        _, ep_group = self._get_ep()
        num_tokens = hidden_states.shape[0]
        self._ensure_buffers()
        self._recv_count.zero_()

        self._destroy_handle()
        self._handle = ep_group.create_handle(
            Layout.EXPERT_MAJOR,
            Tensor(topk_ids),
            stream=_stream(),
        )
        layout_info = LayoutInfo(expert_counters=Tensor(self._recv_count))
        self._handle.dispatch(
            DispatchInputs(tokens=Tensor(hidden_states)),
            DispatchOutputs(
                tokens=Tensor(self._output_tokens),
                scales=Tensor(self._output_scales),
            ),
            layout_info=layout_info,
            config=DispatchConfig(round_scales=0),
            stream=_stream(),
        )
        torch.cuda.synchronize()

        expected_m = (
            num_tokens * self.world_size * self.router_topk + self.num_experts
        ) // self.num_experts
        return NcclEpLowLatencyDispatchOutput(
            self._output_tokens,
            self._output_scales,
            topk_ids,
            topk_weights,
            self._recv_count,
            expected_m,
        )

    def combine(self, combine_input: NcclEpLowLatencyCombineInput) -> torch.Tensor:
        hidden_states, topk_ids, topk_weights = combine_input
        num_tokens = topk_weights.shape[0]
        combined = torch.empty(
            (num_tokens, self.hidden_size),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        self._handle.combine(
            CombineInputs(tokens=Tensor(hidden_states)),
            CombineOutputs(tokens=Tensor(combined), topk_weights=Tensor(topk_weights)),
            stream=_stream(),
        )
        torch.cuda.synchronize()
        self._destroy_handle()
        return combined


class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH = auto()


class NcclEpDispatcher(BaseDispatcher):
    def __init__(
        self,
        group: dist.ProcessGroup,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        ncclep_mode: NcclEpMode,
    ):
        super().__init__()
        if params_dtype != torch.bfloat16:
            raise NotImplementedError(
                f"NCCL_EP FP8 dispatch adapter currently expects BF16 model activations, got {params_dtype}"
            )
        self.ncclep_mode = ncclep_mode
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_NCCL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        common = dict(
            group=group,
            router_topk=router_topk,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
        )
        if ncclep_mode.is_high_throughput():
            self._impl = _NcclEpHighThroughputImpl(**common)
        elif ncclep_mode.is_low_latency():
            raise NotImplementedError(
                "NCCL_EP low_latency FP8 dispatch is not supported by native NCCL_EP "
                "yet. The current SGLang NCCL_EP backend only supports "
                "high_throughput FP8 dispatch for DeepGEMM runner correctness."
            )
        else:
            raise ValueError(f"Unsupported NCCL_EP mode: {ncclep_mode}")
        self._stage = _Stage.INITIAL

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        if self._stage != _Stage.INITIAL:
            raise RuntimeError(
                f"NCCL_EP dispatch called in invalid stage: {self._stage}"
            )
        out = self._impl.dispatch(hidden_states, topk_output)
        self._stage = _Stage.AFTER_DISPATCH
        return out

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        if self._stage != _Stage.AFTER_DISPATCH:
            raise RuntimeError(
                f"NCCL_EP combine called in invalid stage: {self._stage}"
            )
        if self.ncclep_mode.is_high_throughput():
            if combine_input.format != CombineInputFormat.NCCL_EP_HT:
                raise TypeError(
                    f"Expected NCCL_EP HT combine input, got {combine_input.format}"
                )
        else:
            if combine_input.format != CombineInputFormat.NCCL_EP_LL:
                raise TypeError(
                    f"Expected NCCL_EP LL combine input, got {combine_input.format}"
                )
        out = self._impl.combine(combine_input)
        self._stage = _Stage.INITIAL
        return out
