# DeepEP v2 / EPv2 SGLang 集成说明

这个分支在 SGLang 中新增了独立的 DeepEP v2 MoE all-to-all 后端。
后端名是 `epv2`，语义上与已有的 legacy `deepep` 后端分离，不复用
DeepEP v1 的 dispatcher 对象、mode 语义或 dispatch/combine 数据结构。

## 最终设计总览（2026-06-23）

这一节是当前分支的状态页。相比早期版本，本轮把 **direct decode 改成了 masked-GEMM path**，
并在 direct 模式下**打开了 CUDA graph**，decode 性能从慢于 DeepEP LL 约 9.6% 翻转为
与 DeepEP LL 持平/略快。下面详细描述最终设计、性能数据、根因与优化历程；更早期的
expanded→contiguous / 关 CUDA graph 调查保留在文末「历史 profiling 与设计演进」一节，仅作记录。

### 当前完成状态

- EPv2 已作为独立 MoE A2A backend 接入 SGLang，后端名为 `epv2`。它不复用 legacy `deepep` 的 dispatcher、mode 语义或 dispatch/combine 数据结构。
- Runtime 接口：`--moe-a2a-backend epv2`、`--epv2-mode {direct,hybrid}`、`--epv2-dispatcher-output-dtype {auto,fp8,bf16}`。`auto` 按 runner capability 映射：DeepGEMM -> FP8，Triton -> BF16；未补 adapter 的组合 fail-fast。
- **direct decode = masked-GEMM path（纯 SGLang 侧）**：dispatch 在 decode 批次走 `do_cpu_sync=False` + 静态 `expected_m`，adapter 把 expanded buffer 重打包成固定 `[E_local, max_m, hidden]` slab 喂 `grouped_gemm_nt_f8f8bf16_masked`，combine 前把 top-k 权重融进回写。由此 direct decode 路径**形状静态、无 host readback，可被 CUDA graph capture**。
- **direct 模式打开 CUDA graph**；hybrid 模式保持关闭（non-expanded cpu_sync 路径不可 capture，会 `cudaErrorStreamCaptureUnjoined`）。门控在 `server_args.py`，按 `epv2_mode` 决定，不依赖环境变量。
- direct 的 extend/prefill 批次（`get_is_extend_in_batch()` 为真）仍走 expanded contiguous path；masked 只在 decode 批次启用。hybrid + DeepGEMM 保持 native non-expanded layout。
- DeepEP v2 `ElasticBuffer` 初始化依赖 NCCL process group 的 `device_id`。分支已在 EPv2 路径补稳定性修复，避免 `ElasticBuffer.calculate_buffer_size -> ncclTeamWorld` 早期 segfault；该修复 gate 在 `moe_a2a_backend=="epv2"`，不影响其它 backend。
- 当前分支只整理 EPv2，不包含 NCCL_EP/NicoEP 内容。

### 数据流和关键 contract

- Attention 后的 hidden states 是 BF16；router 产生 `topk_ids/topk_weights`；EPv2 dispatcher 负责按 expert ownership 做 dispatch/combine；MoE runner 负责本地 expert MLP。
- DeepGEMM FP8 路径下，adapter 在 dispatch 前把 BF16 activation 量化成 FP8 activation + scale，再交给 `ElasticBuffer.dispatch`；DeepGEMM 直接消费 dispatcher 返回的 FP8 activation 和 scale。
- Triton BF16 路径下，dispatcher 返回 BF16 activation、不返回 scale；adapter 在 Triton 前 compact valid rows，在 combine 前 expand 回 EPv2 layout。
- **EPv2 direct + DeepGEMM FP8 + decode 批次** 使用 masked-GEMM path：
  - dispatch 仍走 native expanded layout（`do_expand=True`），但设 `do_cpu_sync=False`、按 buffer cap 做固定分配、用静态 `expected_m = max(1, (num_input_tokens*router_topk + num_experts)//num_experts)`、量化用 plain row-major fp8 scale（不在 dispatch 前做 e8m0/TMA 对齐，交给 masked GEMM 自己对齐）。
  - pre-permute `expand_to_masked_slab(...)` 把 expert-packed expanded buffer 重打包成固定 `[E_local, max_m, hidden]` slab + `masked_m`（按真实 per-expert 计数、clamp 到 `max_m`）。
  - GEMM 用 `grouped_gemm_nt_f8f8bf16_masked`，`masked_m` 把计算量收敛到真实 per-expert 行数（不跑 worst-case buffer 全量）。
  - post-permute `masked_slab_to_expand(...)` 把结果写回 EPv2 expanded layout，并**只在真实行上融入 top-k 权重**（不再单独跑全 buffer 的 weight-mul kernel），输出 buffer 用 `torch.empty`（不 zero-fill）。
  - combine 走 native expanded combine（`topk_weights=None`）。
- **EPv2 direct + DeepGEMM FP8 + extend/prefill 批次** 与 **hybrid** 仍是早期路径：direct extend 用 expanded contiguous（exact-count、`do_cpu_sync` 默认）；hybrid 用 native non-expanded layout（native combine 自带权重）。实测 hybrid/prefill-like 打开 expanded 会严重退化，因此 direct/hybrid 不共享同一 layout 策略。

### Correctness 口径

- DSv4 Flash FP8 本地 tokenizer 缺 `chat_template`，raw `/generate` plain prompt 会输出模板碎片（DeepEP 与 EPv2 都有），只作接口 smoke，不作 strict correctness。
- Strict correctness 固定用 `/v1/chat/completions`，三类 prompt：事实问答（中日首都）、算术（`17*23+19`=410）、翻译（fox 句子）。
- 任何性能数据都必须先过这三问 correctness 才认。masked path + CUDA graph 的 direct decode 已多轮 `CORRECTNESS_PASS=True`。

### 性能结论（DSv4 Flash FP8 / H20 8 卡 / DP attn / DeepGEMM / TBO+SBO 关 / cap=1024 / VMM=0 / SWIGLU clamp fusion=true）

- **Decode-like `ISL=1/OSL=1024/CC=128`，CUDA graph 开**：
  - DeepEP low_latency = **~3487 output tok/s**。
  - EPv2 direct（masked path）= **~3618 output tok/s / 31.47 ms TPOT**，与 DeepEP LL 持平/略快。
  - 对照早期关 CUDA graph 的同口径：DeepEP LL ~954、EPv2 direct ~862（expanded contiguous）。CUDA graph 对两者都带来约 3.7x 提升，是 decode 单项最大杠杆。
- **Prefill-like `ISL=1024/OSL=1/CC=128`**：
  - DeepEP normal（CUDA graph 开）= **~21945 input tok/s**。
  - EPv2 hybrid（hybrid 不可 capture，跑 eager）= **~22223 input tok/s**，与 DeepEP normal 持平。
- 结论：EPv2 集成的两条主线（decode=direct、prefill=hybrid）在 H20 + DSv4 Flash FP8 上都已达到与 DeepEP 基线持平的水平。

### direct decode 为什么慢、又怎么追平（根因与优化历程）

1. **关 CUDA graph 时**（早期默认）：每层几十个 kernel 的 launch 开销主导，掩盖了 EPv2 的额外 elementwise 成本；此时 EPv2 direct ~862 vs LL ~954，落后约 9.6%，gap 被判断为 adapter/layout glue。
2. **打开 CUDA graph 后**：launch 开销被消除，DeepEP LL 直接跳到 ~3487；但早期 EPv2 expanded 路径只到 ~1654（落后约 2.1x）。原因暴露出来——EPv2 在 **worst-case expanded buffer（cap=1024 时约 49152 行，真实仅约 96 行）** 上做的两件 elementwise 工作成了串行主成本：
   - combine 前 out-of-place 的 top-k 权重乘（全 buffer）。
   - 回写 buffer 的 zero-fill。
   - 以及 contiguous GEMM 在 worst-case 行数上 tile（而非真实 per-expert 行数）。
3. **修复（全部纯 SGLang 侧，不动 DeepEP native）**：
   - masked-GEMM repack：`expand_to_masked_slab` + `grouped_gemm_nt_f8f8bf16_masked`，用 `masked_m` 把计算收敛到真实 per-expert 行数。
   - 权重融合：`masked_slab_to_expand` 只在真实行上乘 top-k 权重，去掉全 buffer 的独立 weight-mul kernel。
   - 回写 buffer 改 `torch.empty`，去掉 zero-fill。
   - 同时 `do_cpu_sync=False` + 静态 `expected_m` 让 direct decode 形状静态、无 host readback，从而可被 CUDA graph capture。
   - 结果：EPv2 direct 1654 → ~3618，追平并略超 DeepEP LL 3487。
4. **被否决的方向**（避免重蹈）：直接翻 `do_cpu_sync=False` 而不收敛 GEMM 尺寸 → worst-case GEMM 暴涨（830→544）；降 buffer cap → expanded dispatch native dedup assert（cap 不能降）；改 DeepEP native 让 expanded combine 吃权重 → device assert/卡死。L1 单测亦证实 native ElasticBuffer 通信 kernel 本身不比 v1 LL 慢（dispatch 20µs vs 24µs、combine 23µs vs 37µs），瓶颈不在 native kernel。

### 当前主要风险 / 限制

- direct/hybrid mode 在 server 生命周期内固定，没有 DeepEP v1 `auto` 那种 prefill/decode 自动切换。
- masked path 只覆盖 `direct + deep_gemm + decode`；direct extend、hybrid、Triton 仍走各自路径。CUDA graph 门控限定在 `direct + deep_gemm + fp8` masked 路径，其它组合（hybrid、`direct + triton/bf16`）由 server_args 自动关闭 graph。
- masked slab 单 expert 容量 = `max_m`（buffer cap）。若某个 expert 收到的 token 数超过 `max_m`，`expand_to_masked_slab` 会 fail-fast（写 overflow flag、host 在非 graph capture 时检查），不静默截断；graph capture 期间跳过该检查以保持可 capture，由 eager warmup 用代表性 shape 兜底。
- adapter 只覆盖 DeepGEMM FP8 与 Triton BF16，其它 runner fail-fast。
- `EpV2Buffer` 是 singleton + 详细 key；多模型/多 group/混合 dtype 切换需改为显式 per-key 生命周期管理。
- E2E capacity 边界仍需 server 内 instrumentation。

### 下一步优先级

1. 固化 correctness/unit test：masked slab repack（`expand_to_masked_slab`/`masked_slab_to_expand`）的 round-trip 与 top-k 权重融合、empty-token rank、expanded local expert id contract、capacity guard。
2. CUDA graph 覆盖面：`direct + triton/bf16` 与 hybrid 目前自动关 graph；若要扩面需先在 graph 下复测对应路径，并评估 hybrid 是否可改造成 capturable。
3. E2E dispatcher instrumentation：记录真实进入 dispatch 的 token/count/capacity。
4. TBO/SBO overlap hooks（较大工作量，目前 server 启动阶段直接拒绝）。
5. 与 DeepEP v2 native 对齐更省的接口（masked-compatible dispatch output、weighted expanded combine、减少 handle/count 生命周期开销）——可进一步缩小残余开销，但当前 decode 已达 parity，优先级降低。

## 集成范围

- 新增 `--moe-a2a-backend epv2`，用于 MoE expert-parallel dispatch/combine。
- 通过新的 SGLang dispatcher 封装 DeepEP v2 `ElasticBuffer`。
- 通过 `--epv2-mode {direct,hybrid}` 显式选择 DeepEP v2 direct/hybrid 模式。
- 通过 `--epv2-dispatcher-output-dtype {auto,bf16,fp8}` 显式或自动选择 dispatcher 输出 dtype。
- 当前只启用已补齐 adapter 的 MoE runner：DeepGEMM + FP8、Triton + BF16。
- 这个分支不包含 NCCL_EP 集成内容。

## Runtime 接口

### 必选后端参数

```bash
--moe-a2a-backend epv2
```

### EPv2 mode

```bash
--epv2-mode direct     # decode 主线；masked-GEMM path + CUDA graph
--epv2-mode hybrid     # prefill 主线；non-expanded layout，CUDA graph 自动关闭
```

`direct` / `hybrid` 对应 DeepEP v2 `ElasticBuffer(allow_hybrid_mode=...)`，独立于 legacy
DeepEP 的 `--deepep-mode normal/low_latency/auto`。mode 在 server init 固定。

### Dispatcher 输出 dtype

```bash
--epv2-dispatcher-output-dtype auto    # deep_gemm->fp8, triton->bf16
--epv2-dispatcher-output-dtype fp8
--epv2-dispatcher-output-dtype bf16
```

只有已实现 adapter 的 runner/dtype 组合可用，其它组合在 capability 解析阶段 fail-fast。

### 环境变量

```bash
SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024   # 每 rank 通信 buffer 容量（不是模型 token limit）
SGLANG_EPV2_NUM_SMS=0                                # EPv2 通信 kernel SM 数；0=默认
SGLANG_EPV2_ALLOW_HYBRID_MODE=0                      # 仅给不带 ServerArgs 的 synthetic/unit test
SGLANG_DEEPEP_ALLOW_MNNVL=1                          # legacy DeepEP baseline 路径用
NVSHMEM_DISABLE_CUDA_VMM=0                           # legacy DeepEP LL baseline 复测必需
```

- `SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 是每 rank EPv2 通信 buffer 容量，不是模型语义 token limit。大 prefill、chunked-prefill、高并发 decode 需设为 `1024` 或更高。**expanded dispatch 有 native dedup assert，cap 不能降到 1024 以下**；masked path 用 `do_cpu_sync=False` + masked_m 收敛计算，不靠降 cap。
- `SGLANG_EPV2_NUM_SMS` 控制 EPv2 通信 kernel 占用的 SM 数，`0` 为 native 默认。
- `SGLANG_EPV2_ALLOW_HYBRID_MODE` 只用于不带 `ServerArgs` 直接构造 dispatcher 的 synthetic/unit test，真实 server 用 `--epv2-mode`。
- `SGLANG_DEEPEP_ALLOW_MNNVL` / `NVSHMEM_DISABLE_CUDA_VMM=0` 属于 legacy DeepEP baseline 路径：H20 上 DeepEP LL 复测需 `NVSHMEM_DISABLE_CUDA_VMM=0`，否则 buffer 初始化阶段 `cudaErrorInvalidValue`。EPv2 本身不依赖这两项。

## 支持矩阵

| MoE runner | EPv2 output dtype | 状态 | 说明 |
| --- | --- | --- | --- |
| `deep_gemm` | `fp8` | 支持 | direct decode 走 masked-GEMM path（`grouped_gemm_nt_f8f8bf16_masked`）；direct extend 走 expanded contiguous；hybrid 保持 non-expanded layout。Dispatcher 返回 FP8 activation + scale。 |
| `triton` | `bf16` | 支持 | Dispatcher 返回 BF16 activation、不返回 scale；adapter 在 Triton 前 compact valid rows、combine 前 expand 回 EPv2 layout。功能路径。 |
| `deep_gemm` | `bf16` | 拒绝 | DeepGEMM adapter 要求 FP8 activation + scale。 |
| `triton` | `fp8` | 拒绝 | Triton adapter 要求 BF16 activation、不消费 scale。 |
| 其他 runner | 任意 | 拒绝 | 需先补显式 runner adapter 和 capability contract。 |

CUDA graph 支持：

| 路径 | CUDA graph | 说明 |
| --- | --- | --- |
| EPv2 direct（decode，deep_gemm + fp8） | 支持 | masked path 形状静态、无 host readback，可 capture。server_args 只对这一组合默认开启。 |
| EPv2 hybrid | 不支持 | non-expanded cpu_sync 路径 capture 时 `cudaErrorStreamCaptureUnjoined`；server_args 自动关闭。 |
| EPv2 direct + triton/bf16 | 不支持 | 非 masked 路径，server_args 自动关闭 graph。 |
| DeepEP normal / low_latency | 支持 | 基线对照，均可 capture。 |

## 代码结构

- `python/sglang/srt/layers/moe/token_dispatcher/epv2.py`
  - EPv2 专属 dispatch/combine input/output 类型；stage/capacity/hidden/top-k 检查；dispatch 输出量化；`ElasticBuffer` 调用。
  - masked path：`use_masked = use_expand_layout and not get_is_extend_in_batch()`；masked 时 `do_cpu_sync=False`、静态 `expected_m`、`masked_max_m`、`total_expanded` 随 `EpV2DispatchOutput` 下传 adapter。
  - 使用专属 singleton buffer key（process group、hidden、top-k、capacity、输出 dtype、direct/hybrid、world size）。
- `python/sglang/srt/layers/moe/ep_moe/kernels.py`
  - `expand_to_masked_slab(recv_x, recv_x_scale, psum, num_local_experts, max_m, expert_alignment)`：expanded buffer -> 固定 `[E_local, max_m, hidden]` slab + `masked_m`（count clamp 到 max_m）。
  - `masked_slab_to_expand(slab, psum, total_expanded_tokens, expert_alignment, topk_weights=None)`：slab -> EPv2 expanded layout，`topk_weights` 非空时只在真实行融权重，输出 buffer 用 `torch.empty`。
- `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`
  - EPv2 <-> DeepGEMM pre/post-permute；FP8 + scale 契约。
  - masked 分支（`epv2_use_masked`）：pre-permute 调 `expand_to_masked_slab` 返回 `use_masked_gemm=True`/`masked_m`/`expected_m`；GEMM 用 `grouped_gemm_nt_f8f8bf16_masked`；post-permute 调 `masked_slab_to_expand(..., topk_weights=...)` 后早返回。
  - 非 masked 分支（direct extend / 早期路径）：expanded contiguous（`grouped_gemm_nt_f8f8bf16_contig` + `m_indices`）；no-swizzle correctness guard。
- `python/sglang/srt/layers/moe/moe_runner/triton.py`：EPv2 <-> Triton pre/post-permute，BF16 valid-row compaction/expansion。
- `python/sglang/srt/layers/moe/utils.py`：`MoeA2ABackend.EPV2`、`EpV2OutputDtype`、`EpV2RunnerCapability`，不支持组合 fail-fast。
- `python/sglang/srt/distributed/parallel_state.py`：EPv2 process group `device_id` 稳定性修复（gate 在 epv2 路径）。
- `python/sglang/srt/server_args.py`：CLI 参数、fail-fast 检查、按 `epv2_mode` 门控 CUDA graph。
- `python/sglang/srt/environ.py`：`SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK`、`SGLANG_EPV2_NUM_SMS`、`SGLANG_EPV2_ALLOW_HYBRID_MODE`。

## 示例启动命令

### DeepGEMM FP8，decode-like EPv2 direct（masked path + CUDA graph）

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
  --kv-cache-dtype fp8_e4m3
```

注意：direct 模式**不要**再传 `--disable-cuda-graph`——masked path 已使其可 capture，CUDA graph 是 decode 性能的主要来源。

### DeepGEMM FP8，prefill-like EPv2 hybrid（CUDA graph 自动关闭）

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
  --kv-cache-dtype fp8_e4m3
```

hybrid 模式 server_args 会自动关闭 CUDA graph（不可 capture），无需手动加 `--disable-cuda-graph`。

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

Triton 是功能路径，未在 CUDA graph 下复测，smoke 时显式关 graph。

## 性能结果摘要

口径：DSv4 Flash FP8、H20 8 卡、`--tp-size 8 --dp-size 8 --ep-size 8 --enable-dp-attention`、
DeepGEMM runner、TBO/SBO 关、`cap=1024`、`VMM=0`、`SGLANG_OPT_SWIGLU_CLAMP_FUSION=true`。
correctness 先过 `/v1/chat/completions` 三问。

### Decode-like：ISL=1，OSL=1024，CC=128，**CUDA graph 开**（最终主线）

| Backend | Mode | Output tok/s | Mean TPOT | 状态 |
| --- | --- | ---: | ---: | --- |
| DeepEP | low_latency | ~3487 | — | PASS，corrected baseline |
| EPv2 | direct（masked path） | **~3618** | 31.47 ms | PASS，与 LL 持平/略快 |

### Prefill-like：ISL=1024，OSL=1，CC=128

| Backend | Mode | CUDA graph | Input tok/s | 状态 |
| --- | --- | --- | ---: | --- |
| DeepEP | normal | 开 | ~21945 | PASS |
| EPv2 | hybrid | 关（不可 capture） | ~22223 | PASS，与 normal 持平 |

### 历史对照：Decode-like，**CUDA graph 关**

| Backend | Mode | Output tok/s | Mean TPOT | 备注 |
| --- | --- | ---: | ---: | --- |
| DeepEP | low_latency | 953.93 | 129.27 ms | corrected baseline（`SWIGLU_CLAMP_FUSION=true`） |
| EPv2 | direct | 862.24 | 144.18 ms | 早期 expanded contiguous path，慢约 9.6% |
| EPv2 | direct | 808.29 | 154.02 ms | 早期 corrected no-swizzle，非 expanded |

CUDA graph 对两个 backend 都带来约 3.7x decode 提升；它是 decode 单项最大杠杆，也是早期
关 graph 口径下「EPv2 慢 9.6%」结论失效的原因——真实瓶颈直到打开 graph、消除 launch
开销后才暴露（见上文「根因与优化历程」）。

---

# 历史 profiling 与设计演进（pre-masked / no-cuda-graph，仅作记录）

> 以下内容记录 masked-GEMM path 之前、关 CUDA graph 口径下的调查与失败实验。结论已被上文最终设计取代，保留用于追溯设计演进，不要据此理解当前路径。

## Timeline profiling 状态与当时瓶颈判断（关 CUDA graph）

已补齐 H20 8 卡、DSv4 Flash FP8、DeepGEMM runner、DP attention、关闭 CUDA graph
和 TBO/SBO 条件下的 torch profiler 对比。trace 与 bench 日志保存在 `/root/menyu/logs/` 下
（`deepep_epv2_torch_profile_*`、`deepep_epv2_corrected_*` 等）。

### Prefill-like profiling：DeepEP normal vs EPv2 hybrid

测试条件 `ISL=1024/OSL=1/CC=128/capacity=1024`，关 CUDA graph。非 profiler 性能 DeepEP
normal ~25.23 tok/s、EPv2 hybrid ~24.37 tok/s（慢约 3%~4%）。profile 结论：DeepGEMM 与
attention 同量级；两边 EP dispatch kernel 总量接近；EPv2 小额额外开销在 dispatcher/adapter
包装层、combine/copy 与 runner 衔接，而非 native ElasticBuffer kernel 慢。

### Decode-like profiling：DeepEP low_latency vs EPv2 direct

测试条件 `ISL=1/OSL=1024/CC=128/capacity=1024`，关 CUDA graph。DeepEP LL 复测必须
`NVSHMEM_DISABLE_CUDA_VMM=0`。这轮发现旧 wrapper 强制 `SGLANG_OPT_SWIGLU_CLAMP_FUSION=false`，
会让 DeepEP LL 在 padded layout 上单独跑 swiglu clamp 严重拖慢；corrected baseline 应用
`SGLANG_OPT_SWIGLU_CLAMP_FUSION=true`。

| Backend | Mode | Extra env | Output tok/s | Mean TPOT | 结论 |
| --- | --- | --- | ---: | ---: | --- |
| DeepEP | low_latency | `SWIGLU_CLAMP_FUSION=false` | 441.49 | 283.87 ms | 旧 wrapper，非公平 baseline |
| DeepEP | low_latency | `SWIGLU_CLAMP_FUSION=true` | 953.93 | 129.27 ms | corrected baseline |
| EPv2 | direct | `FIX_MEGA=0` | 730.17 | 170.77 ms | correctness PASS 基础路径 |
| EPv2 | direct | `FIX_MEGA=1` + no-swizzle guard | 808.29 | 154.02 ms | corrected fused contiguous |
| EPv2 | direct | expanded layout 默认路径 | 862.24 | 144.18 ms | 慢 corrected LL 约 9.6% |
| EPv2 | direct | BF16 masked experiment | 774.37 | 160.88 ms | 仅隔离对照，非主路径 |

关 CUDA graph 口径下的函数级 profile（已被最终设计取代）显示 EPv2 direct 慢在 adapter/同步路径：
`epv2.py dispatch` median 0.462 vs LL 0.171 ms/layer、`pre_permute` 0.444 vs 0.0049、
`post_permute` 0.128 vs 0.0036、contiguous GEMM 0.488 vs masked GEMM 0.345 ms/layer。当时
判断 gap 在 per-layer host 同步 + adapter glue；打开 CUDA graph 后才确认真实瓶颈是
worst-case expanded buffer 上的 elementwise（weight-mul + zero-fill）与 contiguous GEMM 行数，
最终用 masked path 解决。

### 关 CUDA graph 时期被否决/失败的实验

- `do_cpu_sync=False` 单独翻：worst-case 分配下 `CUBLAS_STATUS_EXECUTION_FAILED` / GEMM 暴涨（830→544）。承重墙是 GEMM 定尺，不能单独关——最终由 masked path 配套解决（masked_m 收敛 + 静态 expected_m）。
- 降 buffer cap（128/256/512）：correctness 过但 CC=128 并发 bench 容量不足崩；expanded dispatch 还有 native dedup assert。cap 必须 1024。
- BF16 dispatch + post-quant：correctness 过但无收益，通信 payload 更大，仅诊断。
- quant v1/v2、quant output buffer reuse、in-place topk、`m_indices` reuse：均无稳定收益（接近噪声）。
- 改 DeepEP native 让 expanded combine 吃权重：第一版 device assert，第二版卡死。weighted expanded combine 需 native API/kernel 级重设计，非 SGLang glue 能安全补出——最终改为在 `masked_slab_to_expand` 里于真实行融权重，绕开 native combine 限制。
- L1 纯通信 kernel 单测（kineto busy time，hidden=4096/experts=256/topk=6/tokens=128/8×H20）：DeepEP v2 elastic dispatch 20.2µs / combine 22.9µs，均快于 v1 LL 的 24.5µs / 37.4µs。证实 native 通信库不是瓶颈。

详细过程和日志保存在源码树外：`/root/menyu/comm_docs/epv2/progress.md`、`/root/menyu/logs/deepep_epv2_*`。
