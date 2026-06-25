# DeepEP v2 / EPv2 SGLang 集成说明

这个分支在 SGLang 中新增了独立的 DeepEP v2 MoE all-to-all 后端。
后端名是 `epv2`，语义上与已有的 legacy `deepep` 后端分离，不复用
DeepEP v1 的 dispatcher 对象、mode 语义或 dispatch/combine 数据结构。

## 收口状态（2026-06-25，以本节为准）

> 下面「最终设计总览（2026-06-23）」及 exp1 等小节是更早的优化历程，**perf 数字与 exp1「反超 LL +5.4%」结论已被本节取代**，设计与坑仍可参考。

- **生产口径真实 gap：decode（masked + 全 CUDA graph）比 DeepEP LL 慢 ~1.75%。** 测量口径：普通 server（**不是** disagg decode-only fake，后者会放大到 ~12%）、DSv4-Flash-FP8 / H20×8 / DP attn / deep_gemm / cap=128 / ISL=1 OSL=1024 / CC=1024 满批。EPv2 14670 vs LL 14932 tok/s。
- **exp1（动态 masked slab 尺寸）已回退**（commit `5314799`）：masked slab `max_m` 与 dispatch `num_max_tokens` 改回固定 `cap × ep_group_size` / `cap`。原因 ① ragged DP（SUM_LEN/skewed）下按本地 batch 定尺会溢出本地 slab，不安全；② 打满时 batch=cap，max_m 与 LL 恒等，exp1 优势本就归零。
- **expected_m 回归实际 batch（对齐 LL，语义修复，非性能）**：上面的 exp1 回退把 `expected_m`（DeepGEMM masked GEMM 的**调度提示**，不是硬界——真正每-expert 上界是 GPU 上的 `masked_m`）也一并钉成了 cap，但 DeepEP LL（`deepep.py` dispatch_a）一直用**实际 batch** `hidden_states.shape[0]`。当时注释写「应该用 local tokens」却与代码矛盾。改回 `expected_m = (local_tokens × group × topk + E) // E`（容量/slab 仍固定 cap，仅此 hint 用实际 batch；它是 per-rank-local、不需跨 rank 一致，ragged DP 安全）。**A/B 实测（cap=1024 欠载 CC=128，bs≈16/rank）fixed 3600 vs oldcap 3578 tok/s，+0.6% 在噪声内** —— masked GEMM 是 weight-bandwidth-bound，expected_m 给 3 还是 192 不改变要读的权重量，故无性能收益；价值在正确性/对齐。满批下 local_tokens==cap，与旧版逐位相等（no-op）。
- **repack 向量化**（commit `91592c5`，本轮主要优化）：`expand_to_masked_slab` / `masked_slab_to_expand` 从「grid=32、串行逐行拷」改成 2D block-row grid（`(E, cdiv(MAX_M,8))`，每 program 拷 8 行，cuda-graph-safe）。repack 56.7→8.6ms；**combine 自愈 57.2→35.5ms（未改 combine，repack 均匀后消除 rank 失同步，spin-wait 尾部收缩，反超 LL 41ms）**；总 GPU kernel 576.7→505.1ms（LL 501.2，≈parity）；吞吐 14118→14670（−5.6%→−1.75%）。
- **combine gap 根因 = repack 引起的 spin-wait，不是 combine kernel**：EPv2 combine 中位 23µs 比 LL 31µs 快（单测与生产 trace 一致），慢的是被 repack 拖出的尾部；repack 向量化后该尾部自动收敛。
- **contiguous（免 repack）= 死路**（A/B 实测否决）：关 graph 下 contig≈masked（do_cpu_sync host 气泡抵消省下的 repack）；且 contiguous 的 do_cpu_sync=True 不能 cuda-graph capture → 拿不到 cuda graph 的 **2.7×**（masked+graph 14077 vs nograph 5170）。**生产必须 masked + 全 CUDA graph。**
- 残余 −1.75% 主要是 quant +2.7ms（不可消，native dispatch 要预量化 fp8）+ per-step 墙钟/噪声；GPU-kernel 已与 LL parity，repack 这条线挖到底。

## 最终设计总览（2026-06-23）

这一节是当前分支的状态页。相比早期版本，本轮把 **direct decode 改成了 masked-GEMM path**，
并在 direct 模式下**打开了 CUDA graph**，decode 性能从慢于 DeepEP LL 约 9.6% 翻转为
与 DeepEP LL 持平/略快（**注：该「持平/略快」基于已回退的 exp1；当前真实为慢 ~1.75%，见顶部收口状态**）。下面详细描述最终设计、性能数据、根因与优化历程；更早期的
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
  - dispatch 仍走 native expanded layout（`do_expand=True`），但设 `do_cpu_sync=False`、按 buffer **cap** 做固定分配（masked slab `max_m = cap × ep_group_size`）、用 `expected_m = max(1, (local_tokens × ep_group_size × router_topk + num_experts)//num_experts)`（`local_tokens = hidden_states.shape[0]` 实际 batch，对齐 LL，见顶部收口节）、量化用 plain row-major fp8 scale（不在 dispatch 前做 e8m0/TMA 对齐，交给 masked GEMM 自己对齐）。
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
  - EPv2 direct（masked path）= **~3674 output tok/s，反超 DeepEP LL +5.4%**。关键：masked slab 尺寸 `max_m = 实际batch × num_ranks`（不是 buffer cap × num_ranks），decode bs≈16/rank 时 max_m≈128，远小于 LL 固定的 `cap × num_ranks = 8192`。LL 的 native masked buffer 绑死在 cap，silu_mul_quant 要跑 8192 行 padding（~59µs/层）；EPv2 收到 max_m≈128 只跑 128 行（~2.5µs），省 ~56µs/层（GEMM 两者都 weight-bandwidth-bound、不可压、持平）。详见「根因与优化历程」exp1。
  - 对照早期关 CUDA graph 的同口径：DeepEP LL ~954、EPv2 direct ~862（expanded contiguous）。CUDA graph 对两者都带来约 3.7x 提升，是 decode 单项最大杠杆。
- **Prefill-like `ISL=1024/OSL=1/CC=128`**：
  - DeepEP normal（CUDA graph 开）= **~21945 input tok/s**。
  - EPv2 hybrid（hybrid 不可 capture，跑 eager）= **~22223 input tok/s**，与 DeepEP normal 持平。
- 结论：EPv2 集成的两条主线在 H20 + DSv4 Flash FP8 上达到/超过 DeepEP 基线——decode（direct）**反超 DeepEP LL +5.4%**，prefill（hybrid）与 DeepEP normal 持平。

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
   - 结果：EPv2 direct 1654 → ~3252（cap=1024）/ ~3448（cap=512），≈ DeepEP LL 3487。残余差距是 EPv2 比 LL 多的 repack + 每步 `[E, max_m, N]` slab 分配（实测 repack kernel 与 max_m 无关、是 count-bounded；只有 masked-GEMM grid 和这些分配随 `max_m = cap × num_ranks` 线性增长，所以差距随 cap 变大）。LL 是 native 持久 masked buffer、无 repack、无每步分配。
4. **torch-profiler timeline 归因 + mn-major scale 优化（cap=1024：3252 → 3376，−6.7% → −3.2%）**：抓稳态单层 decode（cuda graph，bs=16/rank）的 GPU timeline 发现——计算 stream busy 628/640µs、idle 仅 11µs（kernel 背靠背，**不是** spin-wait 主导；通信只占 ~5%）。逐 kernel diff EPv2 vs LL，EPv2 每层多出三项 LL 没有的开销：`expand_to_masked_slab`（~9.5µs）、**GEMM0 前的 scale 转置 `transpose_fp32`（~28.9µs，最大单项）**、`masked_slab_to_expand`（~11.7µs），合计 ~50µs/层 ≈ step 的 7.8%，正好对上 gap。其中 scale 转置可消除：原先 masked path 用 row-major scale，DeepGEMM 侧每层调 `get_mn_major_tma_aligned_tensor` 转成 mn-major TMA-aligned；改为让 `expand_to_masked_slab` 直接产出 mn-major 布局（物理 `[E, sh, max_m]` contiguous、view 成 `[E, max_m, sh]`），该转置变零拷贝 no-op。实测 cap=1024 下 decode **3252 → 3376（+3.8%）**，gap 收窄到 −3.2%。该函数仍兜底转置，correctness 不依赖此优化。
5. **exp1：masked slab 收到实际 batch（cap=1024：3376 → 3674，反超 LL +5.4%）**。前面 max_m = buffer cap × num_ranks = 8192（与 LL 对齐）。但 EPv2 的 ElasticBuffer 接受 per-call `num_max_tokens_per_rank`，所以 masked decode 改成传**实际 batch `_num_input_tokens`**，max_m = batch × num_ranks（decode bs≈16 → max_m≈128），buffer cap 仍 1024（prefill 安全）。timeline 实证：GEMM（181+99µs）是 weight-bandwidth-bound、不随 max_m 变（读 537+268MB expert 权重是底线，EPv2/LL 同）；真正随 max_m 缩的是 **silu_mul_quant**——LL 卡死 max_m=8192 要跑 8192 行 padding（~59µs），EPv2 max_m≈128 只跑 128 行（~2.5µs），省 ~56µs/层。这是 ElasticBuffer per-call cap 灵活性带来的、LL 拿不到的优势。无回归：大 batch 时 max_m 自动涨回 8192。一个 local expert 最多从每个 rank 收 batch 个 token，所以 batch × num_ranks 仍是该步真实最坏界，overflow guard 兜底 misconfig。
6. **exp2（quant）/ exp3（hybrid decode）：评估后不做**。exp2：pre-dispatch quant（`sglang_per_token_group_quant_fp8`）实测仅 ~1.6µs/层（~0.3%），且 EPv2 dispatch 的 native API 要求传入已量化 fp8（LL 是融在 dispatch 内核），没法消，ROI 太低。exp3：UT（num_tokens=128）显示 hybrid（non-expanded）通信无优势——dispatch 与 expanded 持平（~23µs），带权 reduced combine（38.8µs）反比 expanded combine（31.9µs）贵 ~7µs；且 non-expanded decode 路径有 host readback（`all_tokens=int(psum[-1].item())`）→ 不可 cuda graph capture，要可 capture 得仿照 masked 大改写；contiguous GEMM 历史上也比 masked 慢（0.488 vs 0.345 ms/层）。结论：留在 direct，不切 hybrid（负结果，省一次大改写）。
7. **被否决的方向**（避免重蹈）：直接翻 `do_cpu_sync=False` 而不收敛 GEMM 尺寸 → worst-case GEMM 暴涨（830→544）；降 buffer cap → expanded dispatch native dedup assert（cap 不能降）；改 DeepEP native 让 expanded combine 吃权重 → device assert/卡死；纯 SGLang 用 `expert_alignment=max_m` 让 recv_x 直接当 slab（消 repack）→ `do_cpu_sync=False` 下 native async psum 非理想 e*max_m 布局，masked_m 越界 silu illegal instruction（消 repack 需 native 配合）。L1 单测亦证实 native ElasticBuffer 通信 kernel 本身不比 v1 LL 慢（dispatch 20µs vs 24µs、combine 23µs vs 37µs），瓶颈不在 native kernel。

### 当前主要风险 / 限制

- direct/hybrid mode 在 server 生命周期内固定，没有 DeepEP v1 `auto` 那种 prefill/decode 自动切换。
- masked path 只覆盖 `direct + deep_gemm + decode`；direct extend、hybrid、Triton 仍走各自路径。CUDA graph 门控限定在 `direct + deep_gemm + fp8` masked 路径，其它组合（hybrid、`direct + triton/bf16`）由 server_args 自动关闭 graph。
- masked slab 每 expert 容量 `max_m = 实际batch × num_ranks`（exp1）：masked decode dispatch 传 per-call `num_max_tokens_per_rank = _num_input_tokens`（当前 forward 的 per-rank batch），而非 buffer cap。一个 local expert 最多从每个 rank 收 batch 个 token，所以 batch × num_ranks 是该步真实最坏界（decode bs≈16 → max_m≈128）。buffer cap（`num_max_dispatch_tokens_per_rank`）仍保持 1024（prefill 安全）。对比：DeepEP LL 的 native masked buffer 绑死 `cap × num_ranks = 8192`，没法收紧——这是 EPv2 反超 LL 的来源（silu 不跑 padding）。`overflow guard` 只在 misconfig（实际 per-expert > max_m）时触发。
- `expand_to_masked_slab` 仍保留 overflow guard（写 flag、host 在非 graph capture 时 fail-fast、不静默截断），但只在 `cap` 配置异常（小于真实 decode batch）时才可能触发；`cap` 设置合理时它是纯防御断言。graph capture 期间跳过该检查以保持可 capture。
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

- `SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK`（cap）是每 rank EPv2 通信 buffer 容量，不是模型语义 token limit。**代码默认值是 128**（`environ.py`），但真实混合 serving 必须 ≥ `chunked_prefill_size`。
- **cap 的真实下限由 prefill 决定，不是 decode**（cap=128 实测结论）：DP attention 下 `chunked_prefill_size` 被强制为 1024，prefill 一个 chunk = 1024 token/rank，dispatch 入口 assert 要求 `cap ≥ 该 chunk`。实测 cap=128 时 decode 阶段（cuda graph capture + correctness）全过，但一进 prefill 就 `dispatch input exceeds per-rank buffer capacity 128, got 1024` 崩溃 → 所以**有 prefill 的服务 cap 必须 ≥ 1024**。
- **decode 不被这个大 cap 连累**：exp1 让 masked decode 用 per-call 实际 batch（不是 cap）定 max_m，所以 cap=128 vs 1024 对 decode 性能无差别（masked slab 只由 batch 决定）。这正是 exp1 解耦 cap/decode 的价值——cap 被 prefill 绑死在 1024，decode 仍按真实 batch 走。纯 decode 节点（如 PD 分离的 D 节点、无 prefill chunk）才可用小 cap。
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

### Decode-like：ISL=1，OSL=1024，CC=128，**CUDA graph 开**（主线单点）

| Backend | Mode | Output tok/s | Mean TPOT | 状态 |
| --- | --- | ---: | ---: | --- |
| DeepEP | low_latency | ~3487 | — | corrected baseline |
| EPv2 | direct（masked path，exp1：max_m=实际batch×ranks） | **~3674** | 30.9 ms | PASS，**反超 LL +5.4%** |
| EPv2 | direct（masked path，exp0：max_m=cap×ranks=8192，mn-major scale） | ~3376 | 34.0 ms | 历史，exp1 前 |
| EPv2 | direct（masked path，row-major scale） | ~3252 | 35.4 ms | 历史，mn-major 前 |

优化演进（cap=1024，CC=128）：1654（关 graph 前的 expanded）→ 开 graph 后 masked path 3252 → mn-major scale 3376 → **exp1 收紧 max_m 到实际 batch：3674（反超 LL）**。

### Decode-like CC 扫描：ISL=1，OSL=256（exp1 vs DeepEP LL，per_rank_bs=CC/8，exp1 max_m≈CC）

| CC | per_rank_bs | EPv2 exp1 | DeepEP LL | 反超 |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 489 | 424 | **+15%** |
| 32 | 4 | 1457 | 1351 | +8% |
| 64 | 8 | 2488 | 2331 | +7% |
| 128 | 16 | 4438 | 4115 | +8% |
| 256 | 32 | 5451 | 5290 | +3% |
| 512 | 64 | 6472 | 6425 | +1% |
| 1024 | 128 | 7870 | 7777 | +1% |

- **exp1 在全 CC 段反超/持平 LL**；趋势小 CC 大赢、大 CC 收窄到持平。原因：小 CC 时 LL 的 silu 跑 max_m=8192 padding（~59µs/层）占比大，EPv2 max_m≈CC 只跑真实行（~5µs）→ 大赢；CC 增大后 GEMM 本身（weight-bandwidth-bound、两者相同）摊薄 silu 优势，EPv2 多出的 repack（~30µs/层、临界 compute stream 不可重叠）逐渐变净拖累 → 收窄到持平。
- **口径**：表内两边均取**首次测量**（公平对比，不混入多跑取优）。CC=512 单点 E2E 抖动较大（同 backend 复测 6472/8258/8326，首次受 warmup 影响偏低）；大 CC 判性能须两边同等多跑取中位，或看不受客户端抖动影响的 GPU 侧 timeline。
- **GPU 侧 timeline（CC=512，bs=50，profiler 口径）**：EPv2 step 757µs vs LL 738µs。EPv2 silu 5.1 vs LL 59.1（省 54）、combine 14.1 vs LL 101.9（省 88），但多 repack ~30µs（expand 18.9 + slab-back 11.0，不可重叠）+ GEMM 两者相同（319µs，weight-bound）。E2E（无 profiler 负载）EPv2 反超。

每层 MoE transient 分配峰值（`max_memory_allocated`，与 `max_m` 线性）：exp1 后 `max_m≈CC`（decode CC=128 → max_m≈128 → ~50 MiB/层），不再是固定 8192（3.1GB/层）。`max_m`=1024/4096/8192 对照值 388/1552/3104 MiB。被 `mem-fraction-static` 预留池吸收，无 OOM。

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
