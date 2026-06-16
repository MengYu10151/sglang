from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

MARLIN_MOE_WORKSPACE: Optional[torch.Tensor] = None


@dataclass
class MarlinRunnerInput(RunnerInput):
    """Input bundle passed to the Marlin runner core."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@dataclass
class MarlinRunnerOutput(RunnerOutput):
    """Output bundle returned from the Marlin runner core."""

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@dataclass
class MarlinMoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by the Marlin backend."""

    w13_qweight: torch.Tensor
    w2_qweight: torch.Tensor
    w13_scales: torch.Tensor
    w2_scales: torch.Tensor
    w13_g_idx_sort_indices: Optional[torch.Tensor]
    w2_g_idx_sort_indices: Optional[torch.Tensor]
    weight_bits: int

    # GPTQ specific (Optional)
    w13_g_idx: Optional[torch.Tensor] = None
    w2_g_idx: Optional[torch.Tensor] = None
    is_k_full: bool = True

    # AWQ specific (Optional)
    w13_qzeros: Optional[torch.Tensor] = None
    w2_qzeros: Optional[torch.Tensor] = None

    # Optional
    expert_map: Optional[torch.Tensor] = None
    global_num_experts: int = -1
    w13_global_scale: Optional[torch.Tensor] = None
    w2_global_scale: Optional[torch.Tensor] = None


def _run_marlin_fused(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    router_logits: torch.Tensor,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> torch.Tensor:
    global MARLIN_MOE_WORKSPACE
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    if runner_config.is_gated:
        assert runner_config.activation == "silu", "Only gated SiLU is supported."
    elif runner_config.activation not in {"silu", "relu2"}:
        raise ValueError(
            f"Unsupported Marlin MoE activation: {runner_config.activation}"
        )

    if (
        MARLIN_MOE_WORKSPACE is None
        or MARLIN_MOE_WORKSPACE.device != hidden_states.device
    ):
        MARLIN_MOE_WORKSPACE = marlin_make_workspace(
            hidden_states.device, max_blocks_per_sm=4
        )

    marlin_hidden_states = hidden_states
    # Avoid aliasing the MoE input buffer until Marlin output semantics are
    # fully validated across shared-expert and overlap paths.
    marlin_inplace = False
    if (
        quant_info.weight_bits == 4
        and quant_info.w13_qzeros is None
        and quant_info.w2_qzeros is None
        and quant_info.w13_scales.dtype == torch.float8_e8m0fnu
        and quant_info.w2_scales.dtype == torch.float8_e8m0fnu
        and hidden_states.dtype == torch.float16
    ):
        # MXFP4(E8M0) Marlin kernels are only numerically valid on the bf16
        # activation path. The fp16 + E8M0 path is intentionally not generated
        # in sgl-kernel, so upcast activations here and cast the result back.
        marlin_hidden_states = hidden_states.to(torch.bfloat16)
        marlin_inplace = False

    return fused_marlin_moe(
        hidden_states=marlin_hidden_states,
        w1=quant_info.w13_qweight,
        w2=quant_info.w2_qweight,
        w1_scale=quant_info.w13_scales,
        w2_scale=quant_info.w2_scales,
        gating_output=router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=quant_info.global_num_experts,
        expert_map=quant_info.expert_map,
        g_idx1=quant_info.w13_g_idx,
        g_idx2=quant_info.w2_g_idx,
        sort_indices1=quant_info.w13_g_idx_sort_indices,
        sort_indices2=quant_info.w2_g_idx_sort_indices,
        w1_zeros=quant_info.w13_qzeros,
        w2_zeros=quant_info.w2_qzeros,
        w1_global_scale=quant_info.w13_global_scale,
        w2_global_scale=quant_info.w2_global_scale,
        workspace=MARLIN_MOE_WORKSPACE,
        num_bits=quant_info.weight_bits,
        is_k_full=quant_info.is_k_full,
        inplace=marlin_inplace,
        routed_scaling_factor=runner_config.routed_scaling_factor,
        clamp_limit=runner_config.swiglu_limit,
        activation=runner_config.activation,
        is_gated=runner_config.is_gated,
    ).to(hidden_states.dtype)


@register_fused_func("none", "marlin")
def fused_experts_none_to_marlin(
    dispatch_output: StandardDispatchOutput,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    output = _run_marlin_fused(
        hidden_states=hidden_states,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        router_logits=topk_output.router_logits,
        quant_info=quant_info,
        runner_config=runner_config,
    )

    return StandardCombineInput(
        hidden_states=output,
    )


def _empty_router_logits(num_tokens: int, device: torch.device) -> torch.Tensor:
    return torch.empty((num_tokens, 0), dtype=torch.float32, device=device)


def _sanitize_topk_for_marlin(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_topk = topk_ids >= 0
    topk_weights = torch.where(
        valid_topk,
        topk_weights,
        torch.zeros_like(topk_weights),
    )
    topk_ids = topk_ids.clamp_min(0).contiguous()
    return topk_ids, topk_weights.contiguous()


@register_fused_func("ncclep", "marlin")
def fused_experts_ncclep_to_marlin(
    dispatch_output,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
):
    from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutputChecker
    from sglang.srt.layers.moe.token_dispatcher.ncclep import (
        NcclEpHighThroughputCombineInput,
        NcclEpHighThroughputExpertMajorDispatchOutput,
        NcclEpLowLatencyCombineInput,
    )

    if DispatchOutputChecker.format_is_ncclep_low_latency(dispatch_output):
        hidden_states, hidden_states_scale, topk_ids, topk_weights, masked_m, _ = (
            dispatch_output
        )
        if hidden_states_scale is not None or hidden_states.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            raise RuntimeError(
                "NCCL_EP low_latency -> Marlin expects BF16/FP16 dispatch output "
                "without activation scales. Use --ncclep-dispatcher-output-dtype bf16."
            )

        num_local_experts, max_slots, _ = hidden_states.shape
        slot_ids = torch.arange(max_slots, device=hidden_states.device)
        valid_mask = slot_ids.unsqueeze(0) < masked_m.to(torch.long).unsqueeze(1)
        compact_hidden = hidden_states[valid_mask].contiguous()
        compact_tokens = compact_hidden.shape[0]
        output = hidden_states

        if compact_tokens > 0:
            compact_topk_ids = (
                torch.arange(
                    num_local_experts,
                    device=hidden_states.device,
                    dtype=torch.int32,
                )
                .unsqueeze(1)
                .expand(num_local_experts, max_slots)[valid_mask]
                .reshape(-1, 1)
                .contiguous()
            )
            compact_topk_weights = torch.ones(
                (compact_tokens, 1), dtype=torch.float32, device=hidden_states.device
            )
            output[valid_mask] = _run_marlin_fused(
                hidden_states=compact_hidden,
                topk_weights=compact_topk_weights,
                topk_ids=compact_topk_ids,
                router_logits=_empty_router_logits(compact_tokens, hidden_states.device),
                quant_info=quant_info,
                runner_config=runner_config,
            )

        return NcclEpLowLatencyCombineInput(
            hidden_states=output,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    if DispatchOutputChecker.format_is_ncclep_high_throughput(dispatch_output):
        if isinstance(dispatch_output, NcclEpHighThroughputExpertMajorDispatchOutput):
            (
                hidden_states,
                hidden_states_scale,
                topk_weights,
                num_recv_tokens_per_expert,
                _,
                _,
            ) = dispatch_output
            if hidden_states_scale is not None or hidden_states.dtype not in (
                torch.bfloat16,
                torch.float16,
            ):
                raise RuntimeError(
                    "NCCL_EP high_throughput expert-major -> Marlin expects "
                    "BF16/FP16 dispatch output without activation scales. "
                    "Use --ncclep-dispatcher-output-dtype bf16."
                )

            all_tokens = sum(num_recv_tokens_per_expert)
            output = torch.zeros_like(hidden_states)
            if all_tokens > 0:
                expert_ids = [
                    torch.full(
                        (count,),
                        expert_id,
                        dtype=torch.int32,
                        device=hidden_states.device,
                    )
                    for expert_id, count in enumerate(num_recv_tokens_per_expert)
                    if count > 0
                ]
                compact_topk_ids = torch.cat(expert_ids).reshape(-1, 1).contiguous()
                compact_topk_weights = topk_weights[:all_tokens]
                if compact_topk_weights.dim() == 1:
                    compact_topk_weights = compact_topk_weights.reshape(-1, 1)
                else:
                    compact_topk_weights = compact_topk_weights[:, :1]
                output[:all_tokens] = _run_marlin_fused(
                    hidden_states=hidden_states[:all_tokens].contiguous(),
                    topk_weights=compact_topk_weights.to(torch.float32).contiguous(),
                    topk_ids=compact_topk_ids,
                    router_logits=_empty_router_logits(all_tokens, hidden_states.device),
                    quant_info=quant_info,
                    runner_config=runner_config,
                )
            return NcclEpHighThroughputCombineInput(hidden_states=output)

        hidden_states, hidden_states_scale, topk_ids, topk_weights, _ = dispatch_output
        if hidden_states_scale is not None or hidden_states.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            raise RuntimeError(
                "NCCL_EP high_throughput -> Marlin expects BF16/FP16 dispatch "
                "output without activation scales. Use --ncclep-dispatcher-output-dtype bf16."
            )

        valid_rows = (topk_ids >= 0).any(dim=1)
        output = torch.zeros_like(hidden_states)
        if valid_rows.any():
            compact_hidden = hidden_states[valid_rows].contiguous()
            compact_topk_ids = topk_ids[valid_rows].contiguous()
            compact_topk_weights = topk_weights[valid_rows].contiguous()
            compact_topk_ids, compact_topk_weights = _sanitize_topk_for_marlin(
                compact_topk_ids, compact_topk_weights
            )
            output[valid_rows] = _run_marlin_fused(
                hidden_states=compact_hidden,
                topk_weights=compact_topk_weights,
                topk_ids=compact_topk_ids,
                router_logits=_empty_router_logits(
                    compact_hidden.shape[0], hidden_states.device
                ),
                quant_info=quant_info,
                runner_config=runner_config,
            )
        return NcclEpHighThroughputCombineInput(hidden_states=output)

    raise RuntimeError(
        f"Unsupported NCCL_EP dispatch output for Marlin: {dispatch_output.format}"
    )
