# DeepEP v2 / EPv2 SGLang 集成说明

这个分支在 SGLang 中新增了独立的 DeepEP v2 MoE all-to-all 后端。
后端名是 `epv2`，语义上与已有的 legacy `deepep` 后端分离，不复用
DeepEP v1 的 dispatcher 对象、mode 语义或 dispatch/combine 数据结构。

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
| `deep_gemm` | `fp8` | 支持 | Dispatcher 返回 FP8 activation 和 scale；adapter 使用 128-token expert alignment 与 DeepGEMM scale layout。 |
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
| EPv2 | direct | 1024 | 128 | 730.17 | 170.77 ms | 1319.88 ms | PASS，当前默认路径 |
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
| EPv2 | direct | capacity=1024 | 730.17 | 170.77 ms | 慢于 corrected DeepEP LL，当前默认路径 |
| EPv2 | direct | BF16 masked experiment | 774.37 | 160.88 ms | 仅作隔离对照；DeepGEMM 主路径不应使用 BF16 dispatch 后二次量化 |
| EPv2 | direct | fused contiguous + empty guard, swizzle=True | 760.72 | 163.76 ms | INVALID，严格 correctness 失败 |
| EPv2 | direct | topk sync fix + fused contiguous, swizzle=True | 804.54 | 154.68 ms | INVALID，严格 correctness 失败 |

profile 结论：

- DeepEP LL fusion=false 的主要异常是 `_apply_swiglu_limit` 对 `[262144, 2048]` padded tensor 做单独 `torch.clamp`，全 rank median 约 `1023.68 ms/rank`。
- DeepEP LL fusion=true 后，大 clamp 消失，swiglu clamp median 降到约 `0.02 ms/rank`，activation/quant 约 `14.70 ms/rank`。
- EPv2 direct 走 contiguous adapter，swiglu clamp shape 是 actual-token 规模，通常 `[1536~2944, 2048]`，median 约 `8.70 ms/rank`；但 DeepGEMM/adapter 路径整体仍慢于 corrected DeepEP LL。
- topk sync fix 后，`_to_local_topk_ids()` 从 trace 消失；但对应 profile/bench 使用了后续证明 correctness 失败的 `SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1` path，因此不能作为有效性能结论。
- 当前有效 correctness 路径包括 EPv2 FP8 dispatch + DeepGEMM + `SGLANG_OPT_FIX_MEGA_MOE_MEMORY=0`，以及 `FIX_MEGA=1` + EPv2 no-swizzle guard。后者刚通过 strict chat correctness，性能/profile 仍需重测。
- DeepEP LL 和 EPv2 direct 的 EP dispatch kernel 在 torch profiler 中都是 long-running/waiting 型通信 kernel，不能把 trace duration 直接等价为单步计算耗时。
- 因此 decode/direct 场景里，EPv2 相比旧 DeepEP wrapper 有收益，但相比 corrected DeepEP LL baseline 仍需要优化。
- 实验方案 2 只说明 masked DeepGEMM 路径本身可作为参考；但 EPv2 BF16 dispatch 后重新 preprocess/quant 会重复搬运和量化 activation。由于 EPv2 原生支持 FP8 dispatch + scales，DeepGEMM 主路径应直接消费 FP8 dispatch output。
- 实验方案 1 的 fused contiguous activation 失败根因收敛到 swizzled activation reader：MegaMoE 的 swizzle=True 假设 gran=8 interleaved gate/up，EPv2 contiguous adapter 需要禁用该 reader。no-swizzle guard 已通过 strict chat correctness。

### 当前性能判断

- EPv2 direct decode 相比旧 DeepEP LL wrapper 有收益，但相比 corrected DeepEP LL baseline 仍落后。
- EPv2 hybrid prefill 已经稳定可跑，但还略慢于 DeepEP normal；优化重点应放在
  SGLang adapter/handle 生命周期/overlap hook/runner 衔接，而不是先假设 native
  EPv2 kernel 本身慢。


### Correctness 验证口径

- 对 DSv4 Flash FP8，raw `/generate` 输入普通中文 prompt 时，DeepEP 与 EPv2 都会出现文件名/模板碎片；该模型本地 `tokenizer_config.json` 没有 `chat_template` 字段，不能把 plain prompt raw completion 当作 strict chat correctness。
- Strict correctness 使用 `/v1/chat/completions`。同一三组 prompt 下，DeepEP LL 与 EPv2 direct (`SGLANG_OPT_FIX_MEGA_MOE_MEMORY=0`) 均输出正确答案。raw `/generate` 不再标记为 correctness PASS，只能作为底层接口 smoke。
- EPv2 direct + `SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1` 原 swizzle=True 路径在同一 chat correctness 下输出乱码；no-swizzle guard 后三组 chat correctness 通过，日志 `/root/menyu/logs/epv2_correctness_epv2_fixmega_noswizzle_20260617_023626`。模板化 `/generate` 复核见 `/root/menyu/logs/epv2_template_generate_20260617_022510`，HF tokenizer 因缺少 `chat_template` 无法生成标准模板。

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
- 实验 `do_cpu_sync=False` 是否正确且有性能收益；只有确认不会破坏 recv token 计数后再考虑启用。
- 从 singleton buffer 演进到显式 per-key buffer 生命周期管理。
- 扩展 runner capability contract，再考虑支持 FlashInfer、Cutlass 或其他 MoE runner。
- 增加确定性的 E2E capacity instrumentation，记录每次进入 EPv2 dispatch 的实际 token 数，避免只能依赖 synthetic test。
