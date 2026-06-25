from __future__ import annotations

import ctypes
import logging
import os
from enum import Enum, auto
from pathlib import Path
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
    NcclEpMode,
    NcclEpOutputDtype,
    get_ncclep_output_dtype,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.utils.common import load_json_config

logger = logging.getLogger(__name__)

_SCALE_BLOCK_SIZE = 128
_LL_FP8_SCALE_ALIGNMENT = 512
_LL_MAX_TOPK = 9
_ncclep_import_error: Optional[BaseException] = None


class NcclEpConfig:
    def __init__(self, config_str: str):
        self.config = load_json_config(config_str) if config_str else {}
        if self.config and not isinstance(self.config, dict):
            raise ValueError("NCCL_EP config must be a JSON object")

    def max_num_sms(self, mode: NcclEpMode) -> int:
        mode_config = self.config.get(mode.value, {}) if self.config else {}
        if mode_config is None:
            mode_config = {}
        if not isinstance(mode_config, dict):
            raise ValueError(
                f"NCCL_EP config for {mode.value} must be a JSON object"
            )

        has_num_sms = "num_sms" in mode_config
        has_max_num_sms = "max_num_sms" in mode_config
        if has_num_sms and has_max_num_sms:
            raise ValueError(
                f"NCCL_EP config for {mode.value} should set only one of "
                "num_sms or max_num_sms"
            )
        value = mode_config.get("num_sms", mode_config.get("max_num_sms", 0))
        if value is None:
            value = 0
        if not isinstance(value, int):
            raise ValueError(
                f"NCCL_EP {mode.value} num_sms/max_num_sms must be an int, got {value!r}"
            )
        if value < 0:
            raise ValueError(
                f"NCCL_EP {mode.value} num_sms/max_num_sms must be >= 0, got {value}"
            )
        return value


def _dlopen_global(path: Path, label: str) -> None:
    mode = getattr(os, "RTLD_NOW", ctypes.DEFAULT_MODE) | getattr(
        os, "RTLD_GLOBAL", ctypes.RTLD_GLOBAL
    )
    try:
        ctypes.CDLL(str(path), mode=mode)
    except OSError as exc:
        raise ImportError(f"Failed to load {label} from {path}: {exc}") from exc


def _configured_ep_so_path() -> Optional[Path]:
    so_path = envs.SGLANG_NCCL_EP_SO_PATH.get()
    if not so_path:
        return None
    path = Path(so_path).expanduser()
    if not path.is_file():
        raise ImportError(f"SGLANG_NCCL_EP_SO_PATH does not exist: {path}")
    return path.resolve()


def _find_sibling_nccl_so(ep_so_path: Path) -> Optional[Path]:
    for name in ("libnccl.so.2", "libnccl.so"):
        candidate = ep_so_path.parent / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def _configure_jit_build_include_dir(ep_so_path: Optional[Path]) -> None:
    include_candidates = []
    if ep_so_path is not None:
        include_candidates.append(ep_so_path.parent.parent / "include")
    nccl_home = os.environ.get("NCCL_HOME")
    if nccl_home:
        home = Path(nccl_home).expanduser()
        include_candidates.extend([home / "build" / "include", home / "include"])

    for include_dir in include_candidates:
        if (include_dir / "nccl_device" / "gin.h").is_file():
            resolved = str(include_dir.resolve())
            if ep_so_path is not None:
                os.environ["NCCL_EP_JIT_BUILD_INCLUDE_DIR"] = resolved
            else:
                os.environ.setdefault("NCCL_EP_JIT_BUILD_INCLUDE_DIR", resolved)
            return


def _preload_configured_ncclep_libraries() -> None:
    ep_so_path = _configured_ep_so_path()
    _configure_jit_build_include_dir(ep_so_path)
    if ep_so_path is None:
        return
    nccl_so_path = _find_sibling_nccl_so(ep_so_path)
    if nccl_so_path is not None:
        _dlopen_global(nccl_so_path, "libnccl.so")
    _dlopen_global(ep_so_path, "libnccl_ep.so")


def _loaded_library_path(soname: str) -> Optional[Path]:
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.rstrip().split(maxsplit=5)
                if len(parts) < 6:
                    continue
                path = Path(parts[5])
                name = path.name
                if name == soname or name.startswith(f"{soname}."):
                    return path.resolve()
    except OSError:
        return None
    return None


try:
    _preload_configured_ncclep_libraries()
    from nccl.ep import (
        Algorithm,
        CombineConfig,
        CombineInputs,
        CombineOutputs,
        DispatchConfig,
        DispatchInputs,
        DispatchOutputs,
        Group,
        GroupConfig,
        HandleConfig,
        Layout,
        LayoutInfo,
        PassDir,
        Tensor,
    )
    from nccl.ep.interop.torch import get_nccl_comm_from_group

    use_ncclep = True
except (ImportError, OSError) as exc:
    use_ncclep = False
    _ncclep_import_error = exc


class NcclEpHighThroughputDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.NCCL_EP_HT


class NcclEpHighThroughputExpertMajorDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]
    expert_offsets: torch.Tensor
    max_recv_tokens: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.NCCL_EP_HT


class NcclEpLowLatencyDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    masked_m: torch.Tensor
    expected_m: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.NCCL_EP_LL


assert isinstance(NcclEpHighThroughputDispatchOutput, DispatchOutput)
assert isinstance(NcclEpHighThroughputExpertMajorDispatchOutput, DispatchOutput)
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


def _sync_mode() -> str:
    mode = os.getenv("SGLANG_NCCL_EP_SYNC_MODE", "event").lower()
    if mode not in ("event", "device"):
        raise ValueError(
            "SGLANG_NCCL_EP_SYNC_MODE must be 'event' or 'device', "
            f"got {mode!r}"
        )
    return mode


def _maybe_synchronize() -> None:
    if _sync_mode() == "device":
        torch.cuda.synchronize()


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
    _ll_buffer_key: Optional[Tuple] = None
    _ll_output_tokens: Optional[torch.Tensor] = None
    _ll_output_scales: Optional[torch.Tensor] = None
    _ll_recv_count: Optional[torch.Tensor] = None
    _library_paths_logged = False

    @classmethod
    def get_group(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        mode: NcclEpMode,
        output_dtype: NcclEpOutputDtype,
        num_max_dispatch_tokens_per_rank: int,
        max_num_sms: int,
        num_experts: int,
        num_local_experts: int,
    ):
        if not use_ncclep:
            detail = (
                f" Original import error: {_ncclep_import_error}"
                if _ncclep_import_error is not None
                else ""
            )
            raise ImportError(
                "NCCL_EP is not available. Build/install nccl4py from "
                "<nccl>/bindings/nccl4py and ensure libnccl_ep.so is loadable."
                + detail
            )
        cls._log_and_validate_loaded_libraries()

        world_size = dist.get_world_size(group)
        if num_max_dispatch_tokens_per_rank <= 0:
            raise ValueError(
                "NCCL_EP requires positive max_dispatch_tokens_per_rank, got "
                f"{num_max_dispatch_tokens_per_rank}"
            )
        if mode.is_low_latency():
            if (
                output_dtype == NcclEpOutputDtype.FP8
                and hidden_size % _LL_FP8_SCALE_ALIGNMENT != 0
            ):
                raise ValueError(
                    "NCCL_EP low_latency FP8 dispatch with output scales requires "
                    f"hidden_size multiple of {_LL_FP8_SCALE_ALIGNMENT}, got {hidden_size}."
                )

        # NCCL_EP HT FLAT receives at most one row from each source rank for a
        # dispatched token. Match the native Python UT / README group-config
        # budget: max_recv_tokens_per_rank = nRanks * max_dispatch_tokens_per_rank.
        max_recv_tokens_per_rank = num_max_dispatch_tokens_per_rank * world_size
        key = (
            id(group),
            hidden_size,
            mode.value,
            output_dtype.value,
            num_max_dispatch_tokens_per_rank,
            max_num_sms,
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
            max_num_sms=max_num_sms,
            num_qp_per_rank=(
                num_local_experts if mode.is_low_latency() else 0
            ),
        )
        cls._ep_group = Group.create(cls._comm, config)
        cls._group_key = key
        logger.info(
            "Initialized NCCL_EP group: mode=%s output_dtype=%s world_size=%s num_experts=%s "
            "num_local_experts=%s max_dispatch_tokens_per_rank=%s "
            "max_recv_tokens_per_rank=%s hidden_size=%s num_qp_per_rank=%s "
            "max_num_sms=%s",
            mode.value,
            output_dtype.value,
            world_size,
            num_experts,
            num_local_experts,
            num_max_dispatch_tokens_per_rank,
            max_recv_tokens_per_rank,
            hidden_size,
            num_local_experts if mode.is_low_latency() else 0,
            max_num_sms,
        )
        return cls._comm, cls._ep_group

    @classmethod
    def _log_and_validate_loaded_libraries(cls) -> None:
        if cls._library_paths_logged:
            return
        cls._library_paths_logged = True

        loaded_nccl = _loaded_library_path("libnccl.so")
        loaded_ep = _loaded_library_path("libnccl_ep.so")
        configured_ep = _configured_ep_so_path()
        logger.info(
            "NCCL_EP loaded libraries: libnccl=%s libnccl_ep=%s "
            "configured_libnccl_ep=%s jit_build_include=%s",
            loaded_nccl,
            loaded_ep,
            configured_ep,
            os.environ.get("NCCL_EP_JIT_BUILD_INCLUDE_DIR"),
        )

        if configured_ep is None:
            return
        if loaded_ep is not None and loaded_ep != configured_ep:
            logger.warning(
                "SGLANG_NCCL_EP_SO_PATH points to %s, but libnccl_ep.so was loaded from %s. "
                "Ensure this is intentional before comparing NCCL_EP results.",
                configured_ep,
                loaded_ep,
            )

        configured_nccl = _find_sibling_nccl_so(configured_ep)
        if (
            configured_nccl is not None
            and loaded_nccl is not None
            and loaded_nccl != configured_nccl
        ):
            logger.warning(
                "NCCL_EP is configured from %s, but libnccl.so was loaded from %s; "
                "the sibling NCCL library is %s. Use LD_PRELOAD or LD_LIBRARY_PATH "
                "when a strict NCCL/NCCL_EP pair is required.",
                configured_ep,
                loaded_nccl,
                configured_nccl,
            )

    @classmethod
    def destroy_low_latency_buffers(cls):
        cls._ll_buffer_key = None
        cls._ll_output_tokens = None
        cls._ll_output_scales = None
        cls._ll_recv_count = None

    @classmethod
    def get_low_latency_buffers(
        cls,
        *,
        output_dtype: NcclEpOutputDtype,
        num_local_experts: int,
        max_recv_tokens_per_rank: int,
        hidden_size: int,
    ):
        device = torch.cuda.current_device()
        token_shape = (num_local_experts, max_recv_tokens_per_rank, hidden_size)
        scale_shape = (
            num_local_experts,
            max_recv_tokens_per_rank,
            hidden_size // _SCALE_BLOCK_SIZE,
        )
        scale_storage_shape = (
            num_local_experts,
            hidden_size // _SCALE_BLOCK_SIZE,
            max_recv_tokens_per_rank,
        )
        count_shape = (num_local_experts,)
        key = (
            device,
            output_dtype.value,
            token_shape,
            scale_shape if output_dtype == NcclEpOutputDtype.FP8 else None,
            count_shape,
        )
        needs_allocate = cls._ll_buffer_key != key
        if not needs_allocate:
            needs_allocate = (
                cls._ll_output_tokens is None
                or cls._ll_recv_count is None
                or tuple(cls._ll_output_tokens.shape) != token_shape
                or tuple(cls._ll_recv_count.shape) != count_shape
                or (
                    output_dtype == NcclEpOutputDtype.FP8
                    and (
                        cls._ll_output_scales is None
                        or tuple(cls._ll_output_scales.shape) != scale_shape
                    )
                )
                or (
                    output_dtype != NcclEpOutputDtype.FP8
                    and cls._ll_output_scales is not None
                )
            )
            if needs_allocate:
                logger.warning(
                    "NCCL_EP low_latency shared buffer cache is invalid; "
                    "reallocating tokens=%s scales=%s recv_count=%s expected_tokens=%s",
                    None
                    if cls._ll_output_tokens is None
                    else tuple(cls._ll_output_tokens.shape),
                    None
                    if cls._ll_output_scales is None
                    else tuple(cls._ll_output_scales.shape),
                    None
                    if cls._ll_recv_count is None
                    else tuple(cls._ll_recv_count.shape),
                    token_shape,
                )

        if needs_allocate:
            cls._ll_output_tokens = torch.empty(
                token_shape,
                dtype=(
                    torch.float8_e4m3fn
                    if output_dtype == NcclEpOutputDtype.FP8
                    else torch.bfloat16
                ),
                device="cuda",
            )
            if output_dtype == NcclEpOutputDtype.FP8:
                # NCCL_EP LL writes scales in scale-major memory order:
                # [local_expert, scale_block, slot]. Expose a logical
                # [local_expert, slot, scale_block] view to the MoE runner.
                cls._ll_output_scales = torch.empty(
                    scale_storage_shape,
                    dtype=torch.float32,
                    device="cuda",
                ).transpose(1, 2)
            else:
                cls._ll_output_scales = None
            cls._ll_recv_count = torch.zeros(
                count_shape, dtype=torch.int32, device="cuda"
            )
            cls._ll_buffer_key = key
        if cls._ll_output_tokens is None or cls._ll_recv_count is None:
            raise RuntimeError("NCCL_EP low_latency shared buffers are not initialized")
        if tuple(cls._ll_output_tokens.shape) != token_shape:
            raise RuntimeError(
                "NCCL_EP low_latency shared token buffer shape mismatch: "
                f"got {tuple(cls._ll_output_tokens.shape)}, expected {token_shape}"
            )
        if (
            cls._ll_output_scales is not None
            and tuple(cls._ll_output_scales.shape) != scale_shape
        ):
            raise RuntimeError(
                "NCCL_EP low_latency shared scale buffer shape mismatch: "
                f"got {tuple(cls._ll_output_scales.shape)}, expected {scale_shape}"
            )
        if tuple(cls._ll_recv_count.shape) != count_shape:
            raise RuntimeError(
                "NCCL_EP low_latency shared recv_count shape mismatch: "
                f"got {tuple(cls._ll_recv_count.shape)}, expected {count_shape}"
            )
        return cls._ll_output_tokens, cls._ll_output_scales, cls._ll_recv_count

    @classmethod
    def destroy(cls):
        cls.destroy_low_latency_buffers()
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
        output_dtype: NcclEpOutputDtype,
        num_max_dispatch_tokens_per_rank: int,
        max_num_sms: int,
        layer_id: Optional[int] = None,
    ):
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.mode = mode
        self.output_dtype = output_dtype
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.max_num_sms = max_num_sms
        self.layer_id = layer_id
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self._handle = None
        self._deferred_handle_destroys = []

    @property
    def max_recv_tokens_per_rank(self) -> int:
        return self.num_max_dispatch_tokens_per_rank * self.world_size

    def _get_ep(self):
        return NcclEpBuffer.get_group(
            self.group,
            self.hidden_size,
            self.mode,
            self.output_dtype,
            self.num_max_dispatch_tokens_per_rank,
            self.max_num_sms,
            self.num_experts,
            self.num_local_experts,
        )

    def set_output_dtype(self, output_dtype: NcclEpOutputDtype) -> None:
        if self.output_dtype != output_dtype:
            self._destroy_handle()
            self.output_dtype = output_dtype

    def _uses_fp8_dispatch_output(self) -> bool:
        return self.output_dtype == NcclEpOutputDtype.FP8

    def _process_deferred_handle_destroys(self, force: bool = False) -> None:
        remaining = []
        for handle, event in self._deferred_handle_destroys:
            if force or event.query():
                try:
                    handle.destroy()
                except Exception:
                    logger.debug("Failed to destroy NCCL_EP handle", exc_info=True)
            else:
                remaining.append((handle, event))
        self._deferred_handle_destroys = remaining

    def _defer_or_destroy_handle(self, handle) -> None:
        if _sync_mode() == "device":
            handle.destroy()
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self._deferred_handle_destroys.append((handle, event))
        self._process_deferred_handle_destroys()

    def _destroy_handle(self) -> None:
        self._process_deferred_handle_destroys()
        if self._handle is not None:
            self._defer_or_destroy_handle(self._handle)
            self._handle = None

    def __del__(self):
        try:
            if self._handle is not None:
                try:
                    self._handle.destroy()
                except Exception:
                    pass
                self._handle = None
            self._process_deferred_handle_destroys(force=True)
        except Exception:
            pass

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
                "NCCL_EP dispatch requires hidden_size multiple of "
                f"{_SCALE_BLOCK_SIZE}, got {self.hidden_size}"
            )
        if topk_ids.shape[1] != self.router_topk:
            raise ValueError(
                f"NCCL_EP topk mismatch: expected {self.router_topk}, got {topk_ids.shape[1]}"
            )
        if self.mode.is_low_latency() and topk_ids.shape[1] > _LL_MAX_TOPK:
            raise ValueError(
                f"NCCL_EP low_latency supports topk <= {_LL_MAX_TOPK}, got {topk_ids.shape[1]}"
            )


class _NcclEpHighThroughputImpl(_NcclEpImplBase):
    def __init__(self, **kwargs):
        super().__init__(mode=NcclEpMode.HIGH_THROUGHPUT, **kwargs)
        self._recv_tokens: Optional[torch.Tensor] = None
        self._recv_scales: Optional[torch.Tensor] = None
        self._recv_topk_ids: Optional[torch.Tensor] = None
        self._recv_topk_weights: Optional[torch.Tensor] = None
        self._expert_counters: Optional[torch.Tensor] = None
        self._expert_offsets: Optional[torch.Tensor] = None
        self._recv_total_counter: Optional[torch.Tensor] = None
        self._num_recv_tokens = 0
        self._num_input_tokens = 0
        self._recv_layout_expert_major: Optional[bool] = None

    def _is_expert_major(self) -> bool:
        return self._uses_fp8_dispatch_output()

    def set_output_dtype(self, output_dtype: NcclEpOutputDtype) -> None:
        if self.output_dtype != output_dtype:
            super().set_output_dtype(output_dtype)
            self._recv_tokens = None
            self._recv_scales = None
            self._recv_topk_ids = None
            self._recv_topk_weights = None
            self._recv_layout_expert_major = None

    def _ensure_buffers(self, max_recv_tokens: int) -> None:
        expert_major = self._is_expert_major()
        token_dtype = (
            torch.float8_e4m3fn if self._uses_fp8_dispatch_output() else torch.bfloat16
        )
        if (
            self._recv_tokens is not None
            and self._recv_tokens.shape[0] >= max_recv_tokens
            and self._recv_tokens.dtype == token_dtype
            and self._recv_layout_expert_major == expert_major
        ):
            return
        self._recv_tokens = torch.empty(
            (max_recv_tokens, self.hidden_size),
            dtype=token_dtype,
            device="cuda",
        )
        if self._uses_fp8_dispatch_output():
            self._recv_scales = torch.empty(
                (max_recv_tokens, self.hidden_size // _SCALE_BLOCK_SIZE),
                dtype=torch.float32,
                device="cuda",
            )
        else:
            self._recv_scales = None
        if expert_major:
            self._recv_topk_ids = None
            self._recv_topk_weights = torch.empty(
                max_recv_tokens, dtype=torch.float32, device="cuda"
            )
        else:
            self._recv_topk_ids = torch.empty(
                (max_recv_tokens, self.router_topk), dtype=torch.int64, device="cuda"
            )
            self._recv_topk_weights = torch.empty(
                (max_recv_tokens, self.router_topk),
                dtype=torch.float32,
                device="cuda",
            )
        self._expert_counters = torch.zeros(
            self.num_local_experts, dtype=torch.int32, device="cuda"
        )
        self._expert_offsets = torch.zeros(
            self.num_local_experts, dtype=torch.int32, device="cuda"
        )
        self._recv_total_counter = torch.zeros(1, dtype=torch.int32, device="cuda")
        self._recv_layout_expert_major = expert_major

    def _bind_handle(
        self,
        ep_group,
        topk_ids: torch.Tensor,
        layout_info: Optional["LayoutInfo"] = None,
        handle_config: Optional["HandleConfig"] = None,
    ) -> None:
        topk_tensor = Tensor(topk_ids)
        layout = Layout.EXPERT_MAJOR if self._is_expert_major() else Layout.FLAT
        if self._handle is None:
            self._handle = ep_group.create_handle(
                layout,
                topk_tensor,
                layout_info=layout_info,
                config=handle_config,
                stream=_stream(),
            )
        else:
            self._handle.update(topk_tensor, layout_info=layout_info, stream=_stream())

    def dispatch(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        hidden_states = hidden_states.contiguous()
        topk_weights = topk_output.topk_weights.contiguous()
        topk_ids = topk_output.topk_ids.to(torch.int64).contiguous()
        self._validate_common(hidden_states, topk_ids)
        if self._uses_fp8_dispatch_output():
            q_hidden_states, hidden_states_scale = _quantize_for_ncclep_dispatch(
                hidden_states
            )
        else:
            q_hidden_states = hidden_states
            hidden_states_scale = None

        _, ep_group = self._get_ep()
        num_tokens = hidden_states.shape[0]
        self._num_input_tokens = num_tokens
        max_recv_tokens = self.max_recv_tokens_per_rank
        self._ensure_buffers(max_recv_tokens)
        self._expert_counters.zero_()
        self._expert_offsets.zero_()
        self._recv_total_counter.zero_()
        self._recv_tokens.zero_()
        if self._recv_scales is not None:
            self._recv_scales.zero_()
        if self._recv_topk_ids is not None:
            self._recv_topk_ids.fill_(-1)
        self._recv_topk_weights.zero_()

        layout_info = LayoutInfo(
            expert_counters=Tensor(self._expert_counters),
            **(
                {
                    "expert_offsets": Tensor(self._expert_offsets),
                    "recv_total_counter": Tensor(self._recv_total_counter),
                }
                if self._is_expert_major()
                else {}
            ),
        )
        handle_config = (
            HandleConfig(dispatch_output_per_expert_alignment=128)
            if self._is_expert_major()
            else None
        )
        self._bind_handle(
            ep_group,
            topk_ids,
            layout_info=layout_info if self._is_expert_major() else None,
            handle_config=handle_config,
        )

        dispatch_inputs = DispatchInputs(
            tokens=Tensor(q_hidden_states),
            topk_weights=Tensor(topk_weights),
            **(
                {"scales": Tensor(hidden_states_scale)}
                if hidden_states_scale is not None
                else {}
            ),
        )
        dispatch_outputs = DispatchOutputs(
            tokens=Tensor(self._recv_tokens),
            topk_weights=Tensor(self._recv_topk_weights),
            **(
                {"topk_idx": Tensor(self._recv_topk_ids)}
                if self._recv_topk_ids is not None
                else {}
            ),
            **(
                {"scales": Tensor(self._recv_scales)}
                if self._recv_scales is not None
                else {}
            ),
        )
        self._handle.dispatch(
            dispatch_inputs,
            dispatch_outputs,
            layout_info=layout_info,
            config=DispatchConfig(round_scales=0, pass_direction=PassDir.FWD),
            stream=_stream(),
        )
        self._handle.complete(stream=_stream())
        _maybe_synchronize()

        if self._is_expert_major():
            counts = [int(x) for x in self._expert_counters.cpu().tolist()]
            self._num_recv_tokens = max_recv_tokens
            return NcclEpHighThroughputExpertMajorDispatchOutput(
                self._recv_tokens[: self._num_recv_tokens],
                (
                    self._recv_scales[: self._num_recv_tokens]
                    if self._recv_scales is not None
                    else None
                ),
                self._recv_topk_weights[: self._num_recv_tokens],
                counts,
                self._expert_offsets,
                self._num_recv_tokens,
            )

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
            (
                self._recv_scales[: self._num_recv_tokens]
                if self._recv_scales is not None
                else None
            ),
            local_topk_ids.to(torch.int32),
            recv_topk_weights,
            counts,
        )

    def combine(self, combine_input: NcclEpHighThroughputCombineInput) -> torch.Tensor:
        output = combine_input.hidden_states
        if self._handle is None:
            raise RuntimeError("NCCL_EP HT combine called before dispatch handle init")
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
        _maybe_synchronize()
        return combined


class _NcclEpLowLatencyImpl(_NcclEpImplBase):
    def __init__(self, **kwargs):
        super().__init__(mode=NcclEpMode.LOW_LATENCY, **kwargs)
        self._output_tokens: Optional[torch.Tensor] = None
        self._output_scales: Optional[torch.Tensor] = None
        self._recv_count: Optional[torch.Tensor] = None
        self._topk_ids: Optional[torch.Tensor] = None

    def set_output_dtype(self, output_dtype: NcclEpOutputDtype) -> None:
        if self.output_dtype != output_dtype:
            super().set_output_dtype(output_dtype)
            self._output_tokens = None
            self._output_scales = None
            self._recv_count = None
            NcclEpBuffer.destroy_low_latency_buffers()

    def _ensure_buffers(self) -> None:
        max_slots = self.max_recv_tokens_per_rank
        token_shape = (self.num_local_experts, max_slots, self.hidden_size)
        scale_shape = (
            self.num_local_experts,
            max_slots,
            self.hidden_size // _SCALE_BLOCK_SIZE,
        )
        (
            self._output_tokens,
            self._output_scales,
            self._recv_count,
        ) = NcclEpBuffer.get_low_latency_buffers(
            output_dtype=self.output_dtype,
            num_local_experts=self.num_local_experts,
            max_recv_tokens_per_rank=max_slots,
            hidden_size=self.hidden_size,
        )
        if tuple(self._output_tokens.shape) != token_shape:
            raise RuntimeError(
                "Failed to allocate NCCL_EP low_latency output token buffer: "
                f"got {tuple(self._output_tokens.shape)}, expected {token_shape}"
            )
        if (
            self._output_scales is not None
            and tuple(self._output_scales.shape) != scale_shape
        ):
            raise RuntimeError(
                "Failed to allocate NCCL_EP low_latency output scale buffer: "
                f"got {tuple(self._output_scales.shape)}, expected {scale_shape}"
            )

    def dispatch(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        hidden_states = hidden_states.contiguous()
        topk_weights = topk_output.topk_weights.contiguous()
        topk_ids = topk_output.topk_ids.to(torch.int64).contiguous()
        self._topk_ids = topk_ids
        self._validate_common(hidden_states, topk_ids)
        _, ep_group = self._get_ep()
        num_tokens = hidden_states.shape[0]
        self._ensure_buffers()
        self._recv_count.zero_()
        self._output_tokens.zero_()
        if self._output_scales is not None:
            self._output_scales.zero_()

        self._destroy_handle()
        self._handle = ep_group.create_handle(
            Layout.EXPERT_MAJOR,
            Tensor(self._topk_ids),
            stream=_stream(),
        )
        output_tokens = Tensor(self._output_tokens)
        output_scales = (
            Tensor(self._output_scales) if self._output_scales is not None else None
        )
        if output_tokens.ndim != 3 or (
            output_scales is not None and output_scales.ndim != 3
        ):
            raise RuntimeError(
                "NCCL_EP low_latency dispatch expects 3D output tensors, got "
                "tokens=%s, scales=%s"
                % (output_tokens.sizes, getattr(output_scales, "sizes", None))
            )
        layout_info = LayoutInfo(expert_counters=Tensor(self._recv_count))
        dispatch_inputs = DispatchInputs(tokens=Tensor(hidden_states))
        dispatch_outputs = DispatchOutputs(
            tokens=output_tokens,
            **({"scales": output_scales} if output_scales is not None else {}),
        )
        dispatch_config = DispatchConfig(round_scales=0)
        self._handle.dispatch(
            dispatch_inputs,
            dispatch_outputs,
            layout_info=layout_info,
            config=dispatch_config,
            stream=_stream(),
        )
        self._handle.complete(stream=_stream())
        _maybe_synchronize()

        expected_m = (
            num_tokens * self.world_size * self.router_topk + self.num_experts
        ) // self.num_experts
        return NcclEpLowLatencyDispatchOutput(
            self._output_tokens,
            self._output_scales,
            topk_ids.to(torch.int32),
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
        combine_inputs = CombineInputs(tokens=Tensor(hidden_states))
        combine_outputs = CombineOutputs(
            tokens=Tensor(combined),
            topk_weights=Tensor(topk_weights),
        )
        combine_config = CombineConfig(send_only=0)
        self._handle.combine(
            combine_inputs,
            combine_outputs,
            config=combine_config,
            stream=_stream(),
        )
        self._handle.complete(stream=_stream())
        _maybe_synchronize()
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
        ncclep_config: str,
        layer_id: Optional[int] = None,
    ):
        super().__init__()
        if params_dtype != torch.bfloat16:
            raise NotImplementedError(
                "NCCL_EP dispatch adapter currently expects BF16 model activations, "
                f"got {params_dtype}"
            )
        self.ncclep_mode = ncclep_mode
        self.quant_config = {}
        self.output_dtype = get_ncclep_output_dtype(self)
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_NCCL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        self.max_num_sms = NcclEpConfig(ncclep_config).max_num_sms(ncclep_mode)
        common = dict(
            group=group,
            router_topk=router_topk,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            output_dtype=self.output_dtype,
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            max_num_sms=self.max_num_sms,
            layer_id=layer_id,
        )
        if ncclep_mode.is_high_throughput():
            self._impl = _NcclEpHighThroughputImpl(**common)
        elif ncclep_mode.is_low_latency():
            self._impl = _NcclEpLowLatencyImpl(**common)
        else:
            raise ValueError(f"Unsupported NCCL_EP mode: {ncclep_mode}")
        self._stage = _Stage.INITIAL

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config
        self.output_dtype = get_ncclep_output_dtype(self)
        self._impl.set_output_dtype(self.output_dtype)

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
