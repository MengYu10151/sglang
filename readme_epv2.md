# DeepEP v2 / EPv2 SGLang 集成说明

这个分支在 SGLang 中新增了独立的 DeepEP v2 MoE all-to-all 后端。
后端名是 `epv2`，语义上与已有的 legacy `deepep` 后端分离，不复用
DeepEP v1 的 dispatcher 对象、mode 语义或 dispatch/combine 数据结构。

## 当前进展总览（2026-06-18）

这一节是当前分支的状态页，用于快速判断 EPv2 集成已经完成什么、现在的性能问题在哪里、以及后续优化应该优先看什么。下面的详细启动命令、支持矩阵和实验表格保留在后续章节。

### 当前完成状态

- EPv2 已作为独立 MoE A2A backend 接入 SGLang，后端名为 `epv2`。它不复用 legacy `deepep` 的 dispatcher、mode 语义或 dispatch/combine 数据结构。
- Runtime 接口已经显式化：`--moe-a2a-backend epv2`、`--epv2-mode {direct,hybrid}`、`--epv2-dispatcher-output-dtype {auto,fp8,bf16}`。
- 当前 `auto` 只做 runner capability 映射：DeepGEMM -> FP8 activation dispatch，Triton -> BF16 activation dispatch。未补 adapter 的 runner/dtype 组合会 fail-fast。
- DSv4 Flash FP8、H20 8 卡、DP attention、DeepGEMM runner、CUDA graph 关闭条件下，EPv2 direct 和 hybrid 均已完成 server E2E correctness smoke 与性能测试。
- DeepEP v2 `ElasticBuffer` 初始化依赖 NCCL process group 的 `device_id`。当前分支已在 EPv2 路径补稳定性修复，避免 `ElasticBuffer.calculate_buffer_size -> ncclTeamWorld` 早期 segfault。
- 当前分支只整理 EPv2，不包含 NCCL_EP/NicoEP 内容；NCCL_EP 的 README 和代码在独立分支维护。

### 数据流和关键 contract

- Attention 后的 hidden states 是 BF16；router 产生 `topk_ids/topk_weights`；EPv2 dispatcher 负责按 expert ownership 做 dispatch/combine；MoE runner 负责本地 expert MLP。
- DeepGEMM FP8 路径下，adapter 在 dispatch 前把 BF16 activation 量化成 FP8 activation + activation scale，再交给 `ElasticBuffer.dispatch`。DeepGEMM 后续直接消费 dispatcher 返回的 FP8 activation 和 scale。
- Triton BF16 路径下，dispatcher 返回 BF16 activation，不返回 scale；adapter 在 Triton 前 compact valid rows，在 combine 前 expand 回 EPv2 layout。
- EPv2 direct + DeepGEMM FP8 默认使用 native expanded layout。DeepEP v2 `dispatch_copy_epilogue_impl<kDoExpand>` 会把收到的 token 写成 one-row-per-local-expert-slot 的 flat expert-packed layout，SGLang 侧因此可以跳过非 expanded 路径中的 `ep_scatter/ep_gather`。
- EPv2 hybrid + DeepGEMM FP8 保持 native non-expanded layout。实测 hybrid/prefill-like 场景打开 expanded 会严重退化，因此 direct 和 hybrid 不能共享同一个 layout 策略。
- EPv2 expanded layout 不是 DeepGEMM masked path 需要的 `[num_local_experts, expected_m, hidden]` 固定 slab，而是 expert-packed variable padded segments。SGLang adapter 侧强行转 masked 会引入额外 copy/pad，并且 combine 前还要转回 EPv2 flat slot order。
- EPv2 native expanded combine 当前不消费 `topk_weights`。因此 SGLang adapter 必须在 combine 前对每个 expanded expert slot 先乘 top-k 权重，这也是当前 direct 路径的一个额外开销来源。

### Correctness 口径

- DSv4 Flash FP8 本地 tokenizer 缺少 `chat_template`，raw `/generate` plain prompt 可能输出模板碎片。这个现象在 DeepEP 和 EPv2 都出现过，因此 raw `/generate` 只作为接口 smoke，不作为 strict correctness。
- Strict correctness 固定使用 `/v1/chat/completions`，覆盖三类 prompt：事实问答（中日首都）、算术（`17*23+19`）、翻译（fox 句子）。
- 当前恢复 DeepEP 原始安装包后，EPv2 direct 默认路径三问 correctness PASS。日志：`/root/menyu/logs/epv2_post_restore_default_correctness_20260618_114356`。
- 历史上 `SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1` + swizzle=True 的 EPv2 contiguous path 会产生错误输出；当前 no-swizzle guard 是 correctness 必需条件。

### 性能结论

- Prefill-like `ISL=1024/OSL=1`：EPv2 hybrid 已稳定可跑，和 DeepEP normal 同量级，目前约慢 3% 到 4%。
- Decode-like `ISL=1/OSL=1024`：EPv2 direct expanded path 从 corrected non-expanded baseline `808.29 output tok/s` 提升到 `862.24 output tok/s`，约 +6.7%；但 corrected DeepEP LL baseline 为 `953.93 output tok/s`，EPv2 仍慢约 9.6%。
- 最近短 decode `ISL=1/OSL=256/CC=128` adapter 实验显示：quant output buffer reuse 无收益（`864.00 -> 860.14 output tok/s`）；`m_indices` buffer reuse 只有小幅提升（`864.00 -> 867.72 output tok/s`），接近噪声。
- 因此当前性能问题主要不是单个 `torch.empty` 或 quant v1/v2，而是 EPv2 direct expanded layout 接 DeepGEMM contiguous path 时的整体 adapter/layout/pre-post-processing 成本。

### Profiling 判断

- Decode/direct corrected profile 显示差距集中在 MoE A2A adapter 和 runner 衔接，而不是 attention 或单纯 expert GEMM：
  - EPv2 dispatch wrapper median `0.462 ms/layer`，DeepEP LL dispatch median `0.171 ms/layer`。
  - EPv2 pre-permute median `0.444 ms/layer`，DeepEP LL pre-permute median `0.0049 ms/layer`。
  - EPv2 post-permute median `0.128 ms/layer`，DeepEP LL post-permute median `0.0036 ms/layer`。
  - EPv2 contiguous GEMM median `0.488 ms/layer`，DeepEP LL masked GEMM median `0.345 ms/layer`。
- `do_cpu_sync=False` 不能直接打开。最小实验触发 segfault/请求失败，后续需要重新设计异步 recv-count、handle lifecycle 或 fixed-capacity adapter。
- BF16 dispatch + DeepGEMM 前 post-quant 可以作为诊断路径，但不是产品主路径：EPv2 原生支持 FP8 dispatch + scale，BF16 搬运后再本地量化会重复通信和量化成本。
- 临时修改 native combine 支持 weighted expanded combine 的实验未通过 correctness：第一版触发 multiple-reduction device assert，第二版补 weighted reduce 后首个请求卡死。这个方向需要 EPv2 native API/kernel 级设计，不是 SGLang glue 里补几行即可安全解决。

### 当前主要风险

- 当前工作区仍包含实验开关和文档更新，PR 前需要区分稳定实现、诊断开关和失败实验，避免把临时变量误暴露成正式接口。
- EPv2 direct/hybrid mode 在 server 生命周期内固定；当前没有 legacy DeepEP v1 `auto` 那种按 prefill/decode 自动切换的语义。
- E2E capacity 边界还需要 server 内 instrumentation。普通请求可能先被 SGLang scheduler/tokenization/chunking 拦截或切分，不能直接证明 dispatcher capacity guard 的真实覆盖效果。
- 当前 adapter 只覆盖 DeepGEMM FP8 和 Triton BF16。FlashInfer、Cutlass 或其他 MoE runner 不能默认共享，需要显式 capability contract 和 adapter。

### 下一步优先级

1. PR 前 clean-up：保留稳定修复，移除或隔离实验开关，整理 commit 信息和 README。
2. 固化 correctness/unit tests：DeepGEMM FP8 direct/hybrid、Triton BF16、empty-token rank、capacity guard、expanded local expert id contract。
3. 增加 E2E dispatcher instrumentation，记录每次进入 EPv2 dispatch 的真实 token/count/capacity，避免只依赖 synthetic test。
4. 继续 profile direct/hybrid 的 adapter 和 handle 生命周期，确认 `m_indices` buffer reuse 是否值得作为正式优化。
5. 和 DeepEP v2 native 侧对齐两个真正可能缩小 gap 的接口：masked-compatible dispatch output，或正确支持 top-k weighted expanded combine。

## 集成范围

- 新增 `--moe-a2a-backend epv2`，用于 MoE expert-parallel dispatch/combine。
- 通过新的 SGLang dispatcher 封装 DeepEP v2 `ElasticBuffer`。
- 通过 `--epv2-mode {direct,hybrid}` 显式选择 DeepEP v2 direct/hybrid 模式。
- 通过 `--epv2-dispatcher-output-dtype {auto,bf16,fp8}` 显式或自动选择 dispatcher 输出 dtype。
- 当前只启用已经补齐 adapter 的 MoE runner：
  - DeepGEMM + FP8 activation dispatch
  - Triton + BF16 activation dispatch

这个分支不包含 NCCL_EP 集成内容。

## Runtime 接口

### 必选后端参数

```bash
--moe-a2a-backend epv2
```

### EPv2 mode

```bash
--epv2-mode direct
--epv2-mode hybrid
```

`direct` / `hybrid` 对应 DeepEP v2 `ElasticBuffer(allow_hybrid_mode=...)`。
这套语义独立于 legacy DeepEP 的 `--deepep-mode normal/low_latency/auto`。

### Dispatcher 输出 dtype

```bash
--epv2-dispatcher-output-dtype auto
--epv2-dispatcher-output-dtype fp8
--epv2-dispatcher-output-dtype bf16
```

当前 `auto` 按 MoE runner capability 选择：

- `deep_gemm` -> `fp8`
- `triton` -> `bf16`

只有已经实现 adapter 的 runner/dtype 组合可以使用。其他组合会在启动或
capability 解析阶段 fail-fast，避免静默进入错误 layout 或 scale 语义。

### 环境变量

```bash
SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
SGLANG_EPV2_ALLOW_HYBRID_MODE=0
SGLANG_DEEPEP_ALLOW_MNNVL=1
NVSHMEM_DISABLE_CUDA_VMM=0
```

`SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 是每个 rank 的 EPv2 通信
buffer capacity，不是模型语义上的 token limit。大 prefill、chunked-prefill
或高并发 decode 场景可能需要设置为 `1024` 或更高。

`SGLANG_EPV2_ALLOW_HYBRID_MODE` 只用于兼容不带 `ServerArgs` 直接构造 dispatcher
的 synthetic/unit test。真实 server 启动应使用 `--epv2-mode`。

`SGLANG_DEEPEP_ALLOW_MNNVL` 属于 legacy DeepEP baseline 路径，不是 EPv2 参数。
它保留在这个分支里，是因为 H20 类环境在做公平 DeepEP baseline 对比时，可能
需要关闭不可用的 fabric memory handle。

`NVSHMEM_DISABLE_CUDA_VMM=0` 也是 legacy DeepEP baseline 的环境要求。H20 上
DeepEP low_latency 曾经能稳定通过；后来 bench/profile 脚本把该变量默认成 `1`，
会在 legacy DeepEP LL buffer 初始化阶段触发 `cudaErrorInvalidValue`。复测 DeepEP
LL decode baseline 时需要显式设为 `0`。EPv2 direct 本身不依赖这个 baseline 规避项。

## 支持矩阵

| MoE runner | EPv2 output dtype | 状态 | 说明 |
| --- | --- | --- | --- |
| `deep_gemm` | `fp8` | 支持 | Dispatcher 返回 FP8 activation 和 scale；direct 模式使用 EPv2 native expanded layout；hybrid 模式保持 EPv2 原生默认 non-expanded layout。 |
| `triton` | `bf16` | 支持 | Dispatcher 返回 BF16 activation，不返回 scale；adapter 在 Triton 前 compact valid rows，在 combine 前 expand 回 EPv2 layout。 |
| `deep_gemm` | `bf16` | 拒绝 | 当前 DeepGEMM adapter 要求 FP8 activation + scale。 |
| `triton` | `fp8` | 拒绝 | 当前 Triton adapter 要求 BF16 activation，且不消费 dispatcher scale。 |
| 其他 runner | 任意 | 拒绝 | 需要先补显式 runner adapter 和 capability contract。 |

## 代码结构

- `python/sglang/srt/layers/moe/token_dispatcher/epv2.py`
  - 定义 EPv2 专属 dispatch/combine input/output 类型。
  - 负责 stage 检查、capacity 检查、hidden size 检查、top-k 检查、dispatch 输出量化，以及 `ElasticBuffer` 调用。
  - 使用专属 singleton buffer key，key 包含 process group、hidden size、top-k、capacity、FP8/BF16 输出模式、direct/hybrid mode 和 world size。
- `python/sglang/srt/layers/moe/utils.py`
  - 新增 `MoeA2ABackend.EPV2`。
  - 定义 `EpV2OutputDtype` 和 `EpV2RunnerCapability`。
  - 根据 server args 和 MoE runner backend 解析 EPv2 runner capability。
- `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`
  - 新增 EPv2 -> DeepGEMM pre-permute 与 DeepGEMM -> EPv2 post-permute。
  - 直接消费 EPv2 FP8 activation 和 scale。
  - DeepGEMM FP8 direct/decode 路径使用 EPv2 native expanded layout：dispatch copy epilogue 直接写 one-row-per-local-expert-slot 的 `recv_x`，从而跳过 SGLang 侧额外的 `ep_scatter/ep_gather` layout round-trip。hybrid/prefill 路径保持 native 默认 non-expanded layout，避免大 token 场景 expanded rows 放大带来的性能退化。
- `python/sglang/srt/layers/moe/moe_runner/triton.py`
  - 新增 EPv2 -> Triton pre-permute 与 Triton -> EPv2 post-permute。
  - 处理 BF16 valid-row compaction/expansion。
- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
  - 当 `--moe-a2a-backend epv2` 时创建 `EpV2Dispatcher`。
- `python/sglang/srt/models/deepseek_v2.py`
  - 让 EPv2 走通用 A2A MoE forward helper。
- `python/sglang/srt/server_args.py`
  - 新增 CLI 参数和不支持功能的 fail-fast 检查。
- `python/sglang/srt/environ.py`
  - 新增 EPv2 capacity 与 hybrid fallback 环境变量。

## 示例启动命令

### DeepGEMM FP8，prefill-like EPv2 hybrid

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

### DeepGEMM FP8，decode-like EPv2 direct

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

### Triton BF16 smoke 路径

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

## 验证情况

最近一次验证在 H20 8 卡节点的 privileged container `sglang_deepep_epv2_menyu`
中完成，模型为 DSv4 Flash FP8。

已完成的基础检查：

- `git diff --check`
- 修改文件的 Python compile 检查。
- runtime container 内 EPv2 import 和 capability resolution 检查。
- DeepGEMM FP8 与 Triton BF16 路径的 E2E prompt 检查，包括事实、算术和翻译类 prompt。
- 不支持 runner/dtype 组合和 overlap 功能的 fail-fast 检查。
- synthetic dispatcher capacity guard 检查。

## 性能结果摘要

以下结果使用公平 capacity 对齐方式复测。DeepEP 与 EPv2 使用相同模型、runner、
DP-attention 设置，均关闭 CUDA graph 与 TBO/SBO，`max_prefill_tokens=8192`，
且除特别说明外都设置 `SGLANG_*_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024`。

### Prefill-like：ISL=1024，OSL=1

| Backend | Mode | CC | Input tok/s | Output tok/s | Mean TTFT |
| --- | --- | ---: | ---: | ---: | ---: |
| DeepEP | normal | 64 | 21713.72 | 21.20 | 1944.17 ms |
| EPv2 | hybrid | 64 | 21042.05 | 20.55 | 1993.08 ms |
| DeepEP | normal | 128 | 25839.65 | 25.23 | 3039.61 ms |
| EPv2 | hybrid | 128 | 24958.80 | 24.37 | 3168.68 ms |

结论：EPv2 hybrid 可以稳定运行，prefill-like 场景与 DeepEP normal 同量级，
当前约慢 3% 到 4%。

### Decode-like：ISL=1，OSL=1024

| Backend | Mode | Capacity | CC | Output tok/s | Mean TPOT | Mean TTFT | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| DeepEP | low_latency | 128 | 64 | 486.89 | 130.54 ms | 1025.37 ms | PASS |
| EPv2 | direct | 128 | 64 | 386.18 | 164.61 ms | 1244.02 ms | PASS |
| DeepEP | low_latency | 128 | 128 | - | - | - | FAIL，capacity assert |
| EPv2 | direct | 128 | 128 | - | - | - | FAIL，capacity guard |
| DeepEP | low_latency | 1024 | 64 | 231.47 | 275.11 ms | 1517.50 ms | PASS |
| EPv2 | direct | 1024 | 64 | 378.46 | 168.10 ms | 1163.30 ms | PASS |
| DeepEP | low_latency | 1024 | 128 | 441.15 | 283.87 ms | 1810.45 ms | PASS，旧 wrapper：`SGLANG_OPT_SWIGLU_CLAMP_FUSION=false` |
| DeepEP | low_latency | 1024 | 128 | 953.93 | 129.27 ms | 1796.47 ms | PASS，corrected baseline：`SGLANG_OPT_SWIGLU_CLAMP_FUSION=true` |
| EPv2 | direct | 1024 | 128 | 808.29 | 154.02 ms | 1237.50 ms | PASS，corrected no-swizzle baseline，非 expanded layout |
| EPv2 | direct | 1024 | 128 | 862.24 | 144.18 ms | 1193.45 ms | PASS，native expanded layout 默认路径；相对 corrected baseline output tok/s 提升约 6.7% |
| EPv2 | direct | 1024 | 128 | 774.37 | 160.88 ms | 1255.55 ms | EXPERIMENT，BF16 dispatch + masked DeepGEMM preprocess，本地重新量化；仅作隔离对照，不是 DeepGEMM 主路径 |
| EPv2 | direct | 1024 | 128 | 760.72 | 163.76 ms | 1278.87 ms | INVALID，旧实验方案 1：FP8 dispatch + fused contiguous activation + swizzle=True，correctness 失败 |
| EPv2 | direct | 1024 | 128 | 804.54 | 154.68 ms | 1234.64 ms | INVALID，旧实验方案 1 + topk sync fix + swizzle=True，correctness 失败，不作为性能结论 |

结论：capacity 对正确性和性能都有明显影响。CC=128 时 runtime 可能发出比单步
decode 更大的 MoE dispatch batch，因此 capacity=128 不足。capacity 对齐为 1024 后，
如果 DeepEP LL 使用旧 wrapper 里的 `SGLANG_OPT_SWIGLU_CLAMP_FUSION=false`，EPv2 direct 更快；
但 corrected DeepEP LL baseline 使用 `SGLANG_OPT_SWIGLU_CLAMP_FUSION=true` 后，DeepEP LL
当前快于 EPv2 direct。后续 decode 对比必须同时标注 capacity、VMM 和 swiglu clamp fusion。

补充实验结论：EPv2 原生支持 FP8 dispatch，DeepGEMM 主路径必须使用 FP8 dispatch output + activation scale。方案 2 的 BF16 dispatch 只是隔离 masked DeepGEMM preprocess 的实验对照：它会在本地重新生成 FP8 activation 和 scale，语义重复且不应产品化。方案 1 保持 FP8 dispatch，方向更干净；严格 correctness 证明旧 `SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1` + swizzle=True 会产生错误输出。当前已在 EPv2 contiguous adapter 下禁用 swizzled activation reader，strict chat correctness 通过；性能/profile 需要重新复测。

2026-06-17 更新：参考 DeepEP v2 `dispatch_copy_epilogue_impl` 源码确认，EPv2 dispatch epilogue 已经把 global expert id 转成 local expert id，并把非本 rank 选择写为 `-1`。因此 SGLang 侧删除了 `_to_local_topk_ids()` 的 `max().item()` 分支，避免 decode 热路径上的 GPU->CPU sync。注意：此前基于 `SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1` + swizzle=True 的 `804.54 output tok/s / 154.68 ms TPOT` 已被 correctness 判定为无效。后续补充 no-swizzle guard 后，`FIX_MEGA=1` strict chat correctness 通过，但性能/profile 尚未重测。

2026-06-17 更新：EPv2 native `ElasticBuffer.dispatch` 的 `do_expand` 源码默认值是 `False`。SGLang 侧只在 `epv2 direct + DeepGEMM FP8` 路径启用 native expanded layout；`epv2 hybrid + DeepGEMM FP8` 保持 EPv2 原生默认 non-expanded layout。源码依据是 `ElasticBuffer.dispatch(do_expand=True)` 与 `dispatch_copy_epilogue_impl<kDoExpand>`：跨 rank dispatch 仍按目标 rank 接收 token，copy epilogue 在目标 rank 本地把 received token 写成 one-row-per-local-expert-slot 的 `recv_x`。这与 direct/decode-like DeepGEMM expert-major grouped activation 更匹配，可以跳过 SGLang adapter 中的 `ep_scatter/ep_gather`。H20 8 卡 decode-like `ISL=1/OSL=1024/CC=128/capacity=1024` strict correctness PASS；direct expanded path output tok/s 从 corrected baseline `808.29` 提升到 `862.24`，约 +6.7%；相比 corrected DeepEP LL `953.93` 仍慢约 9.6%。日志：`/root/menyu/logs/epv2_correctness_expand_default_20260617_060832` 和 `/root/menyu/logs/deepep_epv2_expand_default_bench_epv2_isl1_osl1024_cc128_20260617_061244`。

2026-06-17 更新：补测 `epv2 hybrid + DeepGEMM FP8 + do_expand=True` 后确认 hybrid/prefill-like 场景是负收益。`ISL=1024/OSL=1/CC=64` 从旧 non-expanded `20.55 tok/s / 1993.08 ms TTFT` 退化到 expanded `7.39 tok/s / 3670.68 ms TTFT`；`CC=128` 从旧 non-expanded `24.37 tok/s / 3168.68 ms TTFT` 退化到 expanded `11.12 tok/s / 10115.74 ms TTFT`。正确性仍 PASS，但性能不适合作为 hybrid 默认。日志：`/root/menyu/logs/epv2_hybrid_expand_default_20260617_062956`。

详细过程和日志保存在源码树外：

```text
/root/menyu/comm_docs/epv2/progress.md
/root/menyu/logs/deepep_epv2_menyu_bench_*
```

## Timeline profiling 状态与当前瓶颈判断

已补齐 H20 8 卡、DSv4 Flash FP8、DeepGEMM runner、DP attention、关闭 CUDA graph
和 TBO/SBO 条件下的 torch profiler 对比。trace 与 bench 日志保存在：

```text
/root/menyu/logs/deepep_epv2_torch_profile_warm_20260616_175003/
/root/menyu/logs/deepep_epv2_torch_profile_warm_decode_epv2_180505/
/root/menyu/logs/deepep_epv2_torch_profile_warm_decode_deepep_vmm0_214145/
/root/menyu/logs/deepep_epv2_torch_profile_warm_decode_deepep_fusion_true_vmm0_20260616_222235/
/root/menyu/logs/deepep_epv2_torch_profile_fixmega_decode_epv2_20260617_003023/
/root/menyu/logs/deepep_epv2_torch_profile_topkfix_decode_epv2_20260617_010354/
/root/menyu/logs/deepep_epv2_corrected_bench_epv2_fixmega1_isl1_osl1024_cc128_20260617_024425/
/root/menyu/logs/deepep_epv2_corrected_profile_epv2_fixmega1_decode_isl1_osl1024_cc128_20260617_025141/
/root/menyu/logs/deepep_epv2_menyu_bench_*
```

### Prefill-like profiling：DeepEP normal vs EPv2 hybrid

测试条件：`ISL=1024, OSL=1, CC=128, capacity=1024`。
非 profiler 性能：DeepEP normal 约 `25.23 tok/s`，EPv2 hybrid 约 `24.37 tok/s`，
EPv2 慢约 3% 到 4%。

profile 结论：

- DeepGEMM 和 attention 基本同量级，不是主要差距来源。
- raw trace 里两边都有 long-running/waiting 型 EP dispatch kernel；不能把该 kernel 的
  trace duration 直接理解成纯计算时间，但同一统计口径下 DeepEP normal 与 EPv2 hybrid
  的 EP kernel 总量非常接近。
- EPv2 hybrid 的小额额外开销主要体现在 dispatcher/adapter 包装层、combine/copy 以及
  runner 衔接路径上，而不是 native ElasticBuffer kernel 明显慢。
- 当前 EPv2 只有单段 `dispatch/combine`，没有 legacy DeepEP 的
  `dispatch_a/dispatch_b/combine_a/combine_b` overlap hook 结构；这是后续优化和对齐项。

### Decode-like profiling：DeepEP low_latency vs EPv2 direct

测试条件：`ISL=1, OSL=1024, CC=128, capacity=1024`。
DeepEP low_latency 复测必须设置 `NVSHMEM_DISABLE_CUDA_VMM=0`。

这轮 profiling 发现旧 wrapper 还强制设置了 `SGLANG_OPT_SWIGLU_CLAMP_FUSION=false`。
该设置会让 DeepEP LL + DeepGEMM masked path 在 padded layout 上单独执行 swiglu clamp，
从而严重拖慢 DeepEP baseline。corrected baseline 应使用 `SGLANG_OPT_SWIGLU_CLAMP_FUSION=true`。

非 profiler 性能：

| Backend | Mode | Extra env | Output tok/s | Mean TPOT | 结论 |
| --- | --- | --- | ---: | ---: | --- |
| DeepEP | low_latency | `SGLANG_OPT_SWIGLU_CLAMP_FUSION=false` | 441.49 | 283.87 ms | 旧 wrapper，非公平 baseline |
| DeepEP | low_latency | `SGLANG_OPT_SWIGLU_CLAMP_FUSION=true` | 953.93 | 129.27 ms | corrected baseline，当前最快 |
| EPv2 | direct | capacity=1024, `FIX_MEGA=0` | 730.17 | 170.77 ms | correctness PASS 的基础路径，慢于 corrected DeepEP LL |
| EPv2 | direct | `FIX_MEGA=1` + no-swizzle guard | 808.29 | 154.02 ms | strict chat correctness PASS 后的 corrected fused contiguous path，仍慢于 corrected DeepEP LL |
| EPv2 | direct | BF16 masked experiment | 774.37 | 160.88 ms | 仅作隔离对照；DeepGEMM 主路径不应使用 BF16 dispatch 后二次量化 |
| EPv2 | direct | fused contiguous + empty guard, swizzle=True | 760.72 | 163.76 ms | INVALID，严格 correctness 失败 |
| EPv2 | direct | topk sync fix + fused contiguous, swizzle=True | 804.54 | 154.68 ms | INVALID，严格 correctness 失败 |

profile 结论：

- DeepEP LL fusion=false 的主要异常是 `_apply_swiglu_limit` 对 `[262144, 2048]` padded tensor 做单独 `torch.clamp`，全 rank median 约 `1023.68 ms/rank`。
- DeepEP LL fusion=true 后，大 clamp 消失，swiglu clamp median 降到约 `0.02 ms/rank`，activation/quant 约 `14.70 ms/rank`。
- EPv2 direct 走 contiguous adapter，swiglu clamp shape 是 actual-token 规模，通常 `[1536~2944, 2048]`，median 约 `8.70 ms/rank`；但 DeepGEMM/adapter 路径整体仍慢于 corrected DeepEP LL。
- topk sync fix 后，`_to_local_topk_ids()` 从 trace 消失；但对应 profile/bench 使用了后续证明 correctness 失败的 `SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1` path，因此不能作为有效性能结论。
- 当前有效 correctness 路径包括 EPv2 FP8 dispatch + DeepGEMM + `SGLANG_OPT_FIX_MEGA_MOE_MEMORY=0`，以及 `FIX_MEGA=1` + EPv2 no-swizzle guard。后者已完成 corrected bench/profile：非 profiler `808.29 output tok/s / 154.02 ms TPOT`，profiler 日志见 `deepep_epv2_corrected_profile_epv2_fixmega1_decode_isl1_osl1024_cc128_20260617_025141`。
- DeepEP LL 和 EPv2 direct 的 EP dispatch kernel 在 torch profiler 中都是 long-running/waiting 型通信 kernel，不能把 trace duration 直接等价为单步计算耗时。
- 因此 decode/direct 场景里，EPv2 相比旧 DeepEP wrapper 有收益，但相比 corrected DeepEP LL baseline 仍需要优化。

- Corrected no-swizzle profile 的函数级统计显示，EPv2 direct 当前主要慢在 SGLang adapter/同步路径，而不是 attention：
  - `epv2.py dispatch` median `0.462 ms/layer`，DeepEP LL `deepep.py dispatch` median `0.171 ms/layer`。
  - `pre_permute_epv2_to_deep_gemm` median `0.444 ms/layer`，DeepEP LL pre-permute median `0.0049 ms/layer`。
  - `post_permute_deep_gemm_to_epv2` median `0.128 ms/layer`，DeepEP LL post-permute median `0.0036 ms/layer`。
  - `_run_contiguous_gemm` median `0.488 ms/layer`，DeepEP LL `_run_masked_gemm` median `0.345 ms/layer`。
- `ElasticBuffer.dispatch(do_cpu_sync=None)` 在 fresh handle 下按 DeepEP v2 API 默认等价于 CPU sync。最小实验把它改成 `False` 后，首个 chat 请求阶段 segfault，日志 `/root/menyu/logs/epv2_correctness_docusync_false_20260617_030808`。因此不能简单关闭；需要先设计异步 recv-count / handle 生命周期，或者让 adapter 不依赖同步后的精确 CPU count。
- 实验方案 2 只说明 masked DeepGEMM 路径本身可作为参考；但 EPv2 BF16 dispatch 后重新 preprocess/quant 会重复搬运和量化 activation。由于 EPv2 原生支持 FP8 dispatch + scales，DeepGEMM 主路径应直接消费 FP8 dispatch output。
- 实验方案 1 的 fused contiguous activation 失败根因收敛到 swizzled activation reader：MegaMoE 的 swizzle=True 假设 gran=8 interleaved gate/up，EPv2 contiguous adapter 需要禁用该 reader。no-swizzle guard 已通过 strict chat correctness。

### 当前性能判断

- EPv2 direct decode 相比旧 DeepEP LL wrapper 有收益；当前默认 direct expanded path 为 `862.24 tok/s / 144.18 ms TPOT`，相比 corrected DeepEP LL `953.93 tok/s / 129.27 ms TPOT` 仍落后约 9.6%。
- EPv2 hybrid prefill 已经稳定可跑，但还略慢于 DeepEP normal；优化重点应放在
  SGLang adapter/handle 生命周期/overlap hook/runner 衔接，而不是先假设 native
  EPv2 kernel 本身慢。


### Correctness 验证口径

- 对 DSv4 Flash FP8，raw `/generate` 输入普通中文 prompt 时，DeepEP 与 EPv2 都会出现文件名/模板碎片；该模型本地 `tokenizer_config.json` 没有 `chat_template` 字段，不能把 plain prompt raw completion 当作 strict chat correctness。
- Strict correctness 使用 `/v1/chat/completions`。同一三组 prompt 下，DeepEP LL 与 EPv2 direct (`SGLANG_OPT_FIX_MEGA_MOE_MEMORY=0`) 均输出正确答案。raw `/generate` 不再标记为 correctness PASS，只能作为底层接口 smoke。
- EPv2 direct + `SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1` 原 swizzle=True 路径在同一 chat correctness 下输出乱码；no-swizzle guard 后三组 chat correctness 通过，日志 `/root/menyu/logs/epv2_correctness_epv2_fixmega_noswizzle_20260617_023626`。模板化 `/generate` 复核见 `/root/menyu/logs/epv2_template_generate_20260617_022510`，HF tokenizer 因缺少 `chat_template` 无法生成标准模板。


### 2026-06-18 复核：`do_cpu_sync=False` 与 expanded no-count

本轮只在 H20 22 节点、`sglang_deepep_epv2_menyu` 容器、`/workspace/sglang_epv2`
工作区内验证；未 push。

#### Correctness 口径修正

- raw `/generate` plain prompt 继续出现文件名/模板碎片，且历史复核中 DeepEP 与 EPv2
  都会出现同类现象。因此 raw `/generate` 不作为 DSv4 Flash FP8 的 strict correctness。
- strict correctness 固定使用 `/v1/chat/completions`，三组 prompt 为首都、`17*23+19`、
  fox 翻译。
- 当前 EPv2 direct + FP8 dispatch + DeepGEMM 在 chat endpoint 下通过：
  `/root/menyu/logs/epv2_chat_correctness_current_20260617_170355`。

#### `do_cpu_sync=False` 临时 patch

- 远端工作区中发现 `do_cpu_sync=False if use_expand_layout else None` 临时 patch 后，先检查已有
  correctness 日志：`/root/menyu/logs/epv2_expand_docusync_false_correctness_20260617_163604`。
- 结果为 FAIL：server 首个请求阶段触发 scheduler exception，错误包括
  `CUBLAS_STATUS_EXECUTION_FAILED` 和后续 DeepEP handle assert。
- 该 patch 已回退为 `do_cpu_sync=None`，保留 ElasticBuffer fresh handle 的默认同步语义。
- 结论：不能直接关闭 `do_cpu_sync`。后续若要优化，需要设计异步 count/handle 生命周期，
  或让 adapter 完全不依赖同步后的 CPU metadata。

#### Expanded no-count 优化

- 优化内容：在 EPv2 expanded layout 下，SGLang adapter 不再读取
  `handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()`，也不再返回
  `handle.num_recv_tokens_per_expert_list`；这些 exact-count metadata 只供非 expanded slicing/scatter 使用。
- 通信 payload、native ElasticBuffer dispatch/combine、DeepGEMM 输入 dtype/layout 不变。
- Chat correctness PASS：
  `/root/menyu/logs/epv2_no_count_chat_correctness_20260617_170819`。
- Torch profile：
  `/root/menyu/logs/epv2_no_count_decode_profile_20260618_011213`。
- 本轮 profile 参数：`ISL=1, OSL=128, CC=128, capacity=1024, epv2 direct, fp8 dispatch`。
  profiler run 的 serving 性能为 `281.30 output tok/s / 405.96 ms TPOT`，只用于热点定位，
  不作为非 profiler 性能结论。
- Trace 解析已落盘：
  `focus_decode.txt`、`focus_extend.txt`、`key_stats.txt`。
- Decode trace 仍显示主要热点在 EP dispatch wrapper/native dispatch 和 prequant/scale 路径：
  `epv2.py dispatch`/`deep_ep.buffs.elastic.dispatch`/`dispatch_impl`，以及
  `per_token_group_quant_fp8`、`tma_align_input_scale`、contiguous DeepGEMM。

#### 当前保留/不保留

- 保留：expanded no-count；direct 使用 expanded layout，hybrid 保持非 expanded layout。
- 保留：top-k 权重在 expanded combine 前 out-of-place 应用；不采用 in-place `mul_`，避免未来 buffer
  复用语义不清导致隐性风险。
- 不保留：`do_cpu_sync=False`。
- 不 push：当前只是远端工作区验证状态。

## 已知限制

- EPv2 当前只支持 DeepGEMM FP8 与 Triton BF16 两条 runner adapter。
- TBO/SBO overlap hooks 尚未实现，server 启动阶段会直接拒绝。
- CUDA graph 当前对 EPv2 关闭，后续需要验证 graph capture 安全性。
- Shared expert fusion 当前对 EPv2 关闭，后续需要单独验证 fused path。
- Decode 场景存在 empty-token rank。fused contiguous activation 已补 `all_tokens == 0` guard；EPv2 下还必须禁用 MegaMoE swizzled activation reader，否则会混淆 gate/up layout 并导致错误输出。
- E2E capacity 边界测试还不够确定。SGLang scheduler、tokenization、chunked-prefill
  可能在请求进入 EPv2 dispatcher 前就先拦截或切分请求；当前只能用 synthetic
  dispatcher test 覆盖 capacity guard，后续需要 server 内 instrumentation。
- 当前 `EpV2Buffer` 是 singleton + 详细 key。单模型单配置可用，但多模型、多 group、
  混合 dtype/capacity 切换时应改成显式 per-key buffer 生命周期管理。
- EPv2 `recv_topk_idx` local expert id 语义已按 DeepEP v2 native epilogue 源码对齐，后续应增加 unit/synthetic test 固化该 API contract，避免未来 DeepEP v2 变更时静默破坏。
- EPv2 direct/hybrid mode 在 server 生命周期内固定；当前没有 DeepEP v1 风格的
  normal/low_latency 自动切换。

## 后续优化项

- 基于已完成 timeline 继续细化 EPv2 hybrid prefill 的 dispatcher/adapter 包装层开销。
- 检查 `ElasticBuffer`、workspace 和 handle 生命周期开销。
- `do_cpu_sync=False` 不能直接启用：最小实验已触发 segfault。后续需要设计异步 recv-count、cached handle 或固定容量 adapter 后再重试。
- 从 singleton buffer 演进到显式 per-key buffer 生命周期管理。
- 扩展 runner capability contract，再考虑支持 FlashInfer、Cutlass 或其他 MoE runner。
- 增加确定性的 E2E capacity instrumentation，记录每次进入 EPv2 dispatch 的实际 token 数，避免只能依赖 synthetic test。

## 近期 Profiling / 优化记录

### 2026-06-18 Adapter 优化实验：BF16 dispatch、quant、topk 权重

测试基准：DeepSeek-V4-Flash-FP8，8 卡，`--tp-size 8 --dp-size 8 --enable-dp-attention --ep-size 8`，EPv2 direct，DeepGEMM，`SGLANG_EPV2_NUM_SMS=20`，`SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8192`，CUDA graph 关闭。正确性使用 `/v1/chat/completions` 三条明确问题校验，性能使用 `bench_serving` random，ISL=1、OSL=256、CC=128。日志位于 `/root/menyu/logs/epv2_deviceid_fp8_correctness_20260618_083530`、`/root/menyu/logs/epv2_adapter_opt_deviceid_screen_20260618_084203`、`/root/menyu/logs/epv2_bf16_postquant_v1_deviceid_20260618_085945`、`/root/menyu/logs/epv2_inplace_topk_retry_deviceid_20260618_085319`、`/root/menyu/logs/epv2_quant_reuse_20260618_110825`、`/root/menyu/logs/epv2_m_indices_reuse_20260618_111522`。

| 方案 | Correctness | Output tok/s | Mean TTFT | Mean TPOT | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| FP8 prequant + quant v2 | PASS | 864.00 | 1202.29 ms | 143.77 ms | 当前主路径基准。 |
| FP8 prequant + quant v1 | PASS | 868.10 | 1211.29 ms | 143.06 ms | 与 v2 基本持平，quant v1/v2 不是主瓶颈。 |
| BF16 dispatch + post-quant v2 | PASS | 861.82 | 1272.24 ms | 143.56 ms | 正确但没有收益；BF16 搬运后再量化不能替代当前 FP8 dispatch 主路径。 |
| BF16 dispatch + post-quant v1 | PASS | 864.05 | 1261.21 ms | 143.28 ms | 与 v2/baseline 持平，post-quant 的 v1/v2 也不是主瓶颈。 |
| FP8 prequant + in-place topk weight | PASS | 860.57 | 1203.60 ms | 144.42 ms | 正确但无收益，默认仍保留 out-of-place，避免引入 buffer aliasing 风险。 |
| FP8 prequant + quant output buffer reuse | PASS | 860.14 | 1271.72 ms | 143.90 ms | 正确但无收益；复用 FP8 activation/scale 输出 buffer 没有降低 decode E2E 热路径开销，暂不作为主优化方向。 |
| FP8 prequant + `m_indices` buffer reuse | PASS | 867.72 | 1216.06 ms | 143.09 ms | 正确，小幅提升约 +0.4%；收益接近噪声，需要重复 profile 后再决定是否保留为正式优化。 |

阶段结论：

1. `BF16 dispatch -> DeepGEMM 前 post-quant` 可以作为诊断/兼容路径，但 decode E2E 没有比 `FP8 prequant -> FP8 dispatch` 更快，同时通信 payload 更大，不适合作为性能主路径。
2. `quant v1/v2` 只表示 SGLang `sglang_per_token_group_quant_fp8` 的两个 kernel 实现，不是 EPv1/EPv2，也不是 DeepEP 版本。切换 v1/v2 没有稳定收益，但这不代表 quant 开销不重要；当前每层 MoE decode 前的 BF16 activation -> FP8 activation + scale 仍是主要热路径之一。
3. expanded combine 下 EPv2 native combine 不消费 `topk_weights`，SGLang adapter 必须在 combine 前补权重乘法。in-place 乘法正确性通过，但性能无收益，暂不保留为默认优化。
4. 本轮还定位到 EPv2 `ElasticBuffer` 初始化稳定性问题：SGLang world process group 不带 `device_id` 时，DeepEP v2 可能在 `ElasticBuffer.calculate_buffer_size -> ncclTeamWorld` segfault；按 DeepEP v2 官方测试方式给 NCCL `init_process_group` 传入 `device_id` 后，E2E baseline 恢复稳定。该改动涉及 SGLang 分布式初始化，后续需要整理成最小、可解释的正式实现。
5. 剩余主要 gap 仍在 adapter/layout 级别：EPv2 当前 direct expanded 输出接 DeepGEMM contiguous path，需要 `m_indices`、topk 权重前乘和 combine adapter；DeepEP LL 更贴近 masked DeepGEMM path。源码复核显示，EPv2 expanded flat layout 是按 expert packed 的变长 padded segment，不能零拷贝 view 成 DeepGEMM masked 需要的 `[num_local_experts, expected_m, hidden]` slab；若在 SGLang 侧强行转 masked，需要额外 pad/copy 并且 combine 前还要转回 EPv2 flat slot order，短期不适合作为 adapter-only 优化。真正干净的方向是让 EPv2 native dispatch 直接产出 masked-compatible layout，或让 EPv2 native combine 支持 weighted expanded combine。
6. 已尝试 quant adapter 侧 buffer reuse：预分配并复用 FP8 activation/scale 输出 buffer，correctness PASS，但 decode E2E 无收益（`864.00 -> 860.14 tok/s`），说明当前 gap 主要不是这类 quant output allocation；该实验保留为诊断开关，不进入主路径。
7. 已尝试 expanded path 的 `m_indices` buffer reuse：correctness PASS，decode E2E 从 `864.00` 到 `867.72 tok/s`，提升很小且可能受波动影响。它只能减少 `torch.empty`，不能去掉 `ep_expand_init_m_indices_from_psum` kernel，也不能改变 contiguous GEMM contract，因此不是解决 DeepEP LL 差距的主方向。
8. 已尝试临时修改 DeepEP/EPv2 native combine 支持 expanded slot 级 1D topk weight。第一版只覆盖 non-multiple-reduction expanded send，真实运行触发 `kUseExpandedLayout && kAllowMultipleReduction` 的 device assert；第二版补 weighted local-reduce 后不再 assert，但首个 correctness 请求卡死，未通过验证。结论：weighted expanded combine 需要按 EPv2 combine 的 reduce/同步协议重新设计 native API 和 kernel，不是 SGLang adapter 侧能安全补出来的优化。失败日志：`/root/menyu/logs/epv2_native_weighted_combine_20260618_112533`、`/root/menyu/logs/epv2_native_weighted_combine_v2_20260618_113056`。DeepEP 源码和安装包已恢复到原始 `2.0.0+d4f41e4`，恢复日志：`/root/menyu/logs/epv2_restore_deepep_build_20260618_114123`；恢复后 EPv2 direct 默认路径三问 correctness PASS，日志：`/root/menyu/logs/epv2_post_restore_default_correctness_20260618_114356`。
