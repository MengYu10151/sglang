# DeepEP v2 / EPv2 SGLang Integration

This branch adds an independent DeepEP v2 MoE all-to-all backend to SGLang.
The backend is exposed as `epv2` and is intentionally separate from the legacy
`deepep` backend. It should not reuse DeepEP v1 dispatcher objects or mode
semantics.

## Scope

- Add `--moe-a2a-backend epv2` for MoE expert-parallel dispatch and combine.
- Wrap DeepEP v2 `ElasticBuffer` through a new SGLang dispatcher.
- Keep EPv2 mode semantics explicit with `--epv2-mode {direct,hybrid}`.
- Keep dispatcher output dtype explicit or auto-selected with
  `--epv2-dispatcher-output-dtype {auto,bf16,fp8}`.
- Support the runner paths that currently have matching adapters:
  DeepGEMM with FP8 activation dispatch and Triton with BF16 activation dispatch.

This branch does not contain NCCL_EP integration code.

## Runtime Interface

### Required backend option

```bash
--moe-a2a-backend epv2
```

### EPv2 mode

```bash
--epv2-mode direct
--epv2-mode hybrid
```

`direct` and `hybrid` map to DeepEP v2 `ElasticBuffer(allow_hybrid_mode=...)`.
They are independent from legacy DeepEP `--deepep-mode normal/low_latency`.

### Dispatcher output dtype

```bash
--epv2-dispatcher-output-dtype auto
--epv2-dispatcher-output-dtype fp8
--epv2-dispatcher-output-dtype bf16
```

`auto` currently resolves by MoE runner:

- `deep_gemm` -> `fp8`
- `triton` -> `bf16`

Explicit dtype is allowed only when a matching runner adapter exists.
Unsupported combinations fail fast during server argument validation or runner
capability selection.

### Environment variables

```bash
SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
SGLANG_EPV2_ALLOW_HYBRID_MODE=0
SGLANG_DEEPEP_ALLOW_MNNVL=1
```

`SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK` is a per-rank EPv2
communication buffer capacity. It is not a model semantic token limit. Large
prefill, chunked-prefill, or high-concurrency decode workloads may need a larger
value such as `1024`.

`SGLANG_EPV2_ALLOW_HYBRID_MODE` is only a compatibility fallback for synthetic
or unit tests that instantiate the dispatcher without `ServerArgs`. Server runs
should use `--epv2-mode`.

`SGLANG_DEEPEP_ALLOW_MNNVL` belongs to the legacy DeepEP baseline path. It is
kept in this branch because the fair DeepEP baseline on H20-like environments
may need to disable fabric memory handles, but it is not an EPv2 option.

## Supported Matrix

| MoE runner | EPv2 output dtype | Status | Notes |
| --- | --- | --- | --- |
| `deep_gemm` | `fp8` | Supported | Dispatcher returns FP8 activation plus scale. Adapter uses 128-token expert alignment and DeepGEMM scale layout. |
| `triton` | `bf16` | Supported | Dispatcher returns BF16 activation without scale. Adapter compacts valid rows before Triton and expands them before EPv2 combine. |
| `deep_gemm` | `bf16` | Rejected | Current DeepGEMM adapter expects FP8 activation and scale. |
| `triton` | `fp8` | Rejected | Current Triton adapter expects BF16 activation and no dispatcher scale. |
| Other runners | Any | Rejected | Add an explicit runner adapter and capability contract before enabling. |

## Implementation Map

- `python/sglang/srt/layers/moe/token_dispatcher/epv2.py`
  - Defines EPv2 dispatch/combine input and output containers.
  - Owns EPv2 stage checks, capacity checks, hidden-size checks, top-k checks,
    dispatch output quantization, and `ElasticBuffer` calls.
  - Uses a dedicated singleton buffer key that includes process group,
    hidden size, top-k, capacity, FP8/BF16 output mode, hybrid/direct mode, and
    world size.
- `python/sglang/srt/layers/moe/utils.py`
  - Adds `MoeA2ABackend.EPV2`.
  - Defines `EpV2OutputDtype` and `EpV2RunnerCapability`.
  - Resolves EPv2 runner capability from server args and MoE runner backend.
- `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`
  - Adds EPv2 -> DeepGEMM pre-permute and DeepGEMM -> EPv2 post-permute.
  - Consumes EPv2 FP8 activation and scale directly.
- `python/sglang/srt/layers/moe/moe_runner/triton.py`
  - Adds EPv2 -> Triton pre-permute and Triton -> EPv2 post-permute.
  - Handles BF16 valid-row compaction and expansion.
- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
  - Selects `EpV2Dispatcher` when `--moe-a2a-backend epv2` is used.
- `python/sglang/srt/models/deepseek_v2.py`
  - Routes MoE forward through the generic all-to-all MoE helper for EPv2.
- `python/sglang/srt/server_args.py`
  - Adds CLI arguments and fail-fast checks for unsupported EPv2 runtime
    features.
- `python/sglang/srt/environ.py`
  - Adds EPv2 buffer capacity and hybrid-mode fallback environment variables.

## Example Server Commands

### DeepGEMM FP8, prefill-like EPv2 hybrid

```bash
SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
python3 -m sglang.launch_server \
  --model-path /models/DeepSeek-V4-Flash-FP8 \
  --trust-remote-code \
  --tp-size 8 --dp-size 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend epv2 \
  --epv2-mode hybrid \
  --epv2-dispatcher-output-dtype fp8 \
  --moe-runner-backend deep_gemm \
  --kv-cache-dtype fp8_e4m3 \
  --disable-cuda-graph --disable-piecewise-cuda-graph
```

### DeepGEMM FP8, decode-like EPv2 direct

```bash
SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
python3 -m sglang.launch_server \
  --model-path /models/DeepSeek-V4-Flash-FP8 \
  --trust-remote-code \
  --tp-size 8 --dp-size 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend epv2 \
  --epv2-mode direct \
  --epv2-dispatcher-output-dtype fp8 \
  --moe-runner-backend deep_gemm \
  --kv-cache-dtype fp8_e4m3 \
  --disable-cuda-graph --disable-piecewise-cuda-graph
```

### Triton BF16 smoke path

```bash
SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
python3 -m sglang.launch_server \
  --model-path /models/DeepSeek-V4-Flash-FP8 \
  --trust-remote-code \
  --tp-size 8 --dp-size 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend epv2 \
  --epv2-mode direct \
  --epv2-dispatcher-output-dtype bf16 \
  --moe-runner-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --disable-cuda-graph --disable-piecewise-cuda-graph
```
## Validation Summary

The latest validation was run on the H20 8-GPU node under the privileged
`sglang_deepep_epv2_menyu` container with DSv4 Flash FP8.

Basic checks completed:

- `git diff --check`
- Python compile checks for touched SGLang files.
- EPv2 import and capability resolution inside the runtime container.
- E2E prompt checks for DeepGEMM FP8 and Triton BF16 paths, including factual,
  arithmetic, and translation prompts.
- Fail-fast checks for unsupported EPv2 runner/dtype combinations and overlap
  features.
- Synthetic dispatcher capacity guard checks.

Representative performance results with fair capacity alignment are below.
Both DeepEP and EPv2 used the same model, runner, DP-attention setting, disabled
CUDA graph, disabled TBO/SBO, `max_prefill_tokens=8192`, and
`SGLANG_*_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024` unless noted.

### Prefill-like: ISL=1024, OSL=1

| Backend | Mode | CC | Input tok/s | Output tok/s | Mean TTFT |
| --- | --- | ---: | ---: | ---: | ---: |
| DeepEP | normal | 64 | 21713.72 | 21.20 | 1944.17 ms |
| EPv2 | hybrid | 64 | 21042.05 | 20.55 | 1993.08 ms |
| DeepEP | normal | 128 | 25839.65 | 25.23 | 3039.61 ms |
| EPv2 | hybrid | 128 | 24958.80 | 24.37 | 3168.68 ms |

EPv2 hybrid was functional and close to DeepEP normal, about 3-4% slower in this
prefill-like test.

### Decode-like: ISL=1, OSL=1024

| Backend | Mode | Capacity | CC | Output tok/s | Mean TPOT | Mean TTFT | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| DeepEP | low_latency | 128 | 64 | 486.89 | 130.54 ms | 1025.37 ms | PASS |
| EPv2 | direct | 128 | 64 | 386.18 | 164.61 ms | 1244.02 ms | PASS |
| DeepEP | low_latency | 128 | 128 | - | - | - | FAIL, capacity assert |
| EPv2 | direct | 128 | 128 | - | - | - | FAIL, capacity guard |
| DeepEP | low_latency | 1024 | 64 | 231.47 | 275.11 ms | 1517.50 ms | PASS |
| EPv2 | direct | 1024 | 64 | 378.46 | 168.10 ms | 1163.30 ms | PASS |
| DeepEP | low_latency | 1024 | 128 | 441.15 | 283.87 ms | 1810.45 ms | PASS |
| EPv2 | direct | 1024 | 128 | 730.17 | 170.77 ms | 1319.88 ms | PASS |

Capacity materially changes both correctness and performance. At CC=128 the
runtime can issue larger MoE dispatch batches than the nominal single decode
step, so capacity 128 is not sufficient. With capacity aligned to 1024, EPv2
direct was faster than DeepEP low_latency in this decode-like test.

Detailed logs and iteration notes are kept outside the source tree under:

```text
/root/menyu/comm_docs/epv2/progress.md
/root/menyu/logs/deepep_epv2_menyu_bench_*
```

## Known Limitations

- EPv2 currently supports only the DeepGEMM FP8 and Triton BF16 runner adapters.
- TBO and SBO overlap hooks are not implemented for EPv2 and are rejected at
  launch time.
- CUDA graph is disabled for EPv2 until graph-capture safety is validated.
- Shared expert fusion is disabled for EPv2 until the fused path is validated.
- E2E capacity boundary testing is not deterministic enough yet because SGLang
  scheduling, tokenization, and chunked-prefill logic can handle requests before
  they reach dispatcher capacity. Synthetic dispatcher capacity tests are used
  for now, but a stronger in-server instrumentation test is still needed.
- The current EPv2 buffer is a singleton with a detailed key. Multi-model or
  dynamic multi-config serving should move toward explicit per-key buffer
  lifecycle management.
- `_to_local_topk_ids` currently relies on the existing global/local expert-id
  range convention. This should be formalized against the DeepEP v2 API contract.
- DeepEP v2 direct/hybrid mode is selected for the server lifetime. There is no
  DeepEP v1-style automatic normal/low_latency switching in this branch.

## Next Optimization Items

- Compare DeepEP v2 and DeepEP v1 timelines at the dispatcher, communication,
  and MoE runner boundaries.
- Audit `ElasticBuffer`, workspace, and handle lifecycle overhead.
- Test whether `do_cpu_sync=False` is valid for the supported EPv2 paths.
- Replace singleton buffer handling with explicit per-key lifetime management.
- Extend the runner capability contract before enabling FlashInfer, Cutlass, or
  other MoE runners.
- Add deterministic E2E capacity instrumentation so capacity behavior can be
  validated inside real server execution rather than only by synthetic tests.
