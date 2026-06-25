# DeepEP v2 / EPv2 SGLang 集成

在 SGLang 中新增独立的 DeepEP v2 MoE all-to-all 后端，后端名 `epv2`，与 legacy `deepep` 后端分离（不复用 DeepEP v1 的 dispatcher、mode 语义或 dispatch/combine 数据结构）。底层封装 DeepEP v2 的 `ElasticBuffer`。

本文三部分：**① 部署配置 ② 运行方式与测试数据 ③ 优化总结**。

---

## 一、部署配置（Deploy Guide）

已在干净的 sglang 官方镜像里从零验证可一键装通（EPv2 direct 三问 correctness PASS）。除 DeepEP v2 本身需源码 build 外，其余全部 pip 化，**不需要任何定制 NCCL**。

### 前置

- 硬件：本文档以 H20 × 8（Hopper, sm90）为例；其它卡改 `TORCH_CUDA_ARCH_LIST`。
- Base 镜像：sglang 官方镜像（`lmsysorg/sglang:dev` 或按 `docker/Dockerfile` 以 `FLASHINFER_VERSION=0.6.12` 构建的镜像）。容器需 `--privileged --network host --ipc host --shm-size 64g --gpus all`，并 mount 模型目录。

### 步骤 1 — 安装 SGLang（epv2 分支）

```bash
git clone -b epv2-integration https://github.com/MengYu10151/sglang.git
cd sglang && pip install -e python
```

`pip install -e python` 会按 `pyproject.toml` **自动拉取 `sglang-kernel==0.4.4` 与 `flashinfer 0.6.12`**，无需手动指定。

### 步骤 2 — NCCL ≥ 2.30.7（含 GIN，官方 pip 版即可）

DeepEP v2 的通信 kernel 用了 NCCL 的 GIN（Generalized Internode）device API（`ncclGetLsaDevicePointer`/`ginExclusiveContexts` 等），这是 NVIDIA NCCL 2.28+ 的官方特性。官方 pip wheel 自带完整 GIN 开发 header，**不需要定制 NCCL**：

```bash
pip install nvidia-nccl-cu13==2.30.7 --no-deps --force-reinstall   # cu12 环境用 nvidia-nccl-cu12==2.30.7
```

镜像默认可能是旧版（如 2.28.9，GIN 符号不全 → DeepEP v2 编译报 undefined）；务必升级到 2.30.7。

### 步骤 3 — 从源码 build DeepEP v2（pip 上只有 v1，必须源码装）

```bash
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP && git checkout d4f41e4
TORCH_CUDA_ARCH_LIST=9.0a python setup.py bdist_wheel    # H20=9.0a；其它卡相应调整
pip install dist/*.whl --force-reinstall --no-deps
```

DeepEP `setup.py` 的 `find_nccl_root` / `find_nvshmem_root` 会**自动找到 pip 的 `nvidia-nccl` 与 `nvidia-nvshmem`**，无需设 `EP_NCCL_ROOT_DIR`、无需 symlink、无需本地 NCCL build。

### 步骤 4 — 对齐 flashinfer-jit-cache

`pip install` 会把 flashinfer 升到 0.6.12，但镜像若自带旧的 `flashinfer-jit-cache`（如 0.6.11）会在 boot 时报版本不匹配。用按 `FLASHINFER_VERSION=0.6.12` 构建的镜像即可；否则装匹配版本或临时绕过：

```bash
pip install flashinfer-jit-cache==0.6.12   # 或临时: export FLASHINFER_DISABLE_VERSION_CHECK=1
```

### 步骤 5 — 运行期环境变量

```bash
export NVSHMEM_BOOTSTRAP=UID NVSHMEM_DISABLE_CUDA_VMM=0 NVSHMEM_QP_DEPTH=4096 NVSHMEM_IBGDA_NIC_HANDLER=cpu
export NCCL_CUMEM_ENABLE=1 NCCL_WIN_ENABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export SGLANG_DEEPEP_ALLOW_MNNVL=0          # H20 单节点 NVL，避免 legacy fabric 路径
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=true SGLANG_OPT_USE_JIT_EP_ACTIVATION=1
export SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024   # 见运行部分对 cap 的说明
```

### 验证安装

```bash
python3 -c "import sglang; from deep_ep import ElasticBuffer; print('OK')"
```

---

## 二、运行方式与测试数据

### 启动命令

**Decode 主线（EPv2 direct，masked-GEMM path + CUDA graph）**：

```bash
SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
python3 -m sglang.launch_server \
  --model-path /models/DeepSeek-V4-Flash-FP8 --trust-remote-code \
  --tp-size 8 --dp-size 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend epv2 --epv2-mode direct --epv2-dispatcher-output-dtype fp8 \
  --moe-runner-backend deep_gemm --kv-cache-dtype fp8_e4m3
```

direct 模式 **不要**加 `--disable-cuda-graph`——CUDA graph 是 decode 性能的主要来源。

**Prefill 主线（EPv2 hybrid，CUDA graph 自动关）**：把 `--epv2-mode direct` 换成 `--epv2-mode hybrid`，其余相同。hybrid 不可 capture，server_args 自动关 graph，无需手动加 `--disable-cuda-graph`。

**Triton BF16（功能 smoke）**：`--epv2-dispatcher-output-dtype bf16 --moe-runner-backend triton --disable-cuda-graph --disable-piecewise-cuda-graph`。

### Runtime 接口

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--moe-a2a-backend` | `epv2` | 启用 EPv2 后端 |
| `--epv2-mode` | `direct` / `hybrid` | direct=decode 主线（masked+graph）；hybrid=prefill 主线（non-expanded，自动关 graph）。server init 固定 |
| `--epv2-dispatcher-output-dtype` | `auto` / `fp8` / `bf16` | auto: deep_gemm→fp8、triton→bf16；不匹配组合 fail-fast |

`SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK`（cap）是每 rank 通信 buffer 容量，**不是模型 token limit**。**有 prefill 的服务 cap 必须 ≥ chunked_prefill_size（DP attention 下强制 1024）**——否则 prefill 一进来就 `dispatch input exceeds per-rank buffer capacity`。decode 性能不被大 cap 连累（masked slab 固定 `cap × ep_group_size`，GEMM 由 GPU 上的真实 `masked_m` 收敛计算量）。纯 decode 节点（PD 分离的 D 节点）可用小 cap。

### 支持矩阵

| MoE runner | output dtype | CUDA graph | 状态 |
| --- | --- | --- | --- |
| `deep_gemm` | `fp8` | direct decode 支持 / hybrid 关 | 支持（主线）|
| `triton` | `bf16` | 关 | 支持（功能路径）|
| `deep_gemm` | `bf16` / `triton` | `fp8` / 其它 runner | — | fail-fast |

### Correctness 口径

DSv4 Flash FP8 本地 tokenizer 缺 `chat_template`，raw `/generate` plain prompt 会输出模板碎片，**不作 strict correctness**。Strict 固定用 `/v1/chat/completions` 三问：事实问答（中日首都）、算术（`17*23+19`=410）、翻译（fox 句）。任何性能数据都先过这三问才认。

### 最新测试数据

口径：DSv4 Flash FP8 / H20×8 / `--tp 8 --dp 8 --ep 8 --enable-dp-attention` / deep_gemm / **普通 server（非 disagg）+ CUDA graph 开** / TBO+SBO 关。对照基线 DeepEP `low_latency`（LL）。

**满批 throughput（ISL=1, OSL=512, batch/rank = cap，CC = cap × 8）**：

| cap | CC | EPv2 direct (tok/s) | DeepEP LL (tok/s) | gap |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 17766 | 18134 | **−2.0%** |
| 1024 | 8192 | 13287 | 13313 | **−0.2%（持平）** |

cap 越大、batch 越大，repack 的固定开销被 GEMM 摊薄，gap 越小（大 cap 下与 LL 持平）。

**repack 向量化前后（ISL=1, OSL=1024, CC=1024 满批, cap=128）**：

| | EPv2 (tok/s) | vs LL 14932 | 总 GPU kernel | repack | combine |
| --- | ---: | ---: | ---: | ---: | ---: |
| 优化前（串行 repack） | 14118 | −5.6% | 576.7 ms | 56.7 ms | 57.2 ms |
| **优化后（向量化 repack）** | **14670** | **−1.75%** | **505.1 ms** | **8.6 ms** | **35.5 ms** |
| DeepEP LL | 14932 | — | 501.2 ms | 0 | 41.0 ms |

向量化 repack 后 EPv2 的 GPU kernel 总时间与 LL parity（505 vs 501ms）。

**Prefill（ISL=1024, OSL=1, CC=128）**：EPv2 hybrid ~22223 input tok/s vs DeepEP normal ~21945，**持平**。

**CUDA graph 杠杆**：masked + graph 14077 vs masked + 关 graph 5170 tok/s = **2.7×**，是 decode 单项最大因素。

---

## 三、优化总结

### 设计：direct decode = masked-GEMM path + CUDA graph（纯 SGLang 侧）

DeepEP v2 dispatch 原生出 **expanded** 布局，而 DeepGEMM masked GEMM 要 `[E_local, max_m, hidden]` slab。decode 批次走 `do_cpu_sync=False`（静态 recv shape，可 capture）+ adapter 把 expanded buffer 重打包成 masked slab 喂 `grouped_gemm_nt_f8f8bf16_masked`，combine 前在真实行上融 top-k 权重。由此 decode 路径形状静态、无 host readback，**可被 CUDA graph capture**（hybrid/extend 走各自 non-expanded/contiguous 路径，自动关 graph）。这一切不动 DeepEP native。

### 优化 1 — repack 向量化（本轮主要优化，commit `91592c5`）

`expand_to_masked_slab` / `masked_slab_to_expand` 两个 repack kernel 原是 grid=`(num_local_experts,)`=32、每个 program 串行 `for j in range(count)` 拷行 → 只用 ~32/132 SM、~96 GB/s、方差大。改成 **2D block-row grid `(E, cdiv(MAX_M, 8))`，每 program 拷 8 行**，打满 SM、每 call 时间恒定，静态 grid 仍 cuda-graph-safe。效果：repack 56.7→8.6ms；吞吐 14118→14670（gap −5.6%→−1.75%）。

### 优化 2 — masked slab 固定 cap×ranks，DP-ragged 安全（commit `5314799`）

masked slab `max_m` 与 dispatch `num_max_tokens` 固定为 `cap × ep_group_size` / `cap`（不按本地实际 batch 动态定尺）。原因：① ragged DP（SUM_LEN/skewed decode）下其它 rank 的更大 batch 会溢出本地 slab，按本地 batch 定尺不安全；② 打满（压测口径）时 batch=cap，动态尺寸与固定值恒等、本就无收益。这回退了早先「按实际 batch 收紧 max_m 反超 LL」的实验（该实验只在大 cap + 欠载这种非压测场景才有收益，且不安全）。

### 优化 3 — expected_m 回归实际 batch，对齐 LL（commit `81dcfc5`，语义修复）

`expected_m` 是 DeepGEMM masked GEMM 的**调度提示**（非硬界，真正每-expert 上界是 GPU 上的 `masked_m`），DeepEP LL 一直用实际 batch。优化 2 曾把它误钉成 cap，此 commit 改回 `(local_tokens × group × topk + E) // E`（per-rank-local，ragged DP 安全）。A/B 实测无性能差异（masked GEMM 是 weight-bandwidth-bound），价值在与 LL 语义一致。

### 根因：combine 的 gap 是 spin-wait，不是 kernel

同参数单测 + 生产 trace 都显示 **EPv2 combine kernel 比 LL 快**（中位 23µs vs 31µs）。优化前生产里 combine 显得慢（57.2ms）是 **spin-wait 尾部**——repack 是 combine 正前方一道变长、串行的工序，把 8 个 rank 拖失同步，集合通信 combine 要等最慢的 rank。repack 向量化后 combine 自动收敛到 35.5ms（甚至快过 LL 41ms），**没有改 combine 一行代码**——直接坐实了 spin-wait 假说。

### 关键结论与边界

- **CUDA graph 必需**：decode 2.7× 来自它；生产必须 masked + 全 CUDA graph。
- **contiguous（免 repack）是死路**（A/B 否决）：contiguous 的 `do_cpu_sync=True` 不能 cuda-graph capture → 拿不到 2.7×；且关 graph 下 contig ≈ masked（省 repack 的收益被 host 同步气泡抵消，净≈0）。
- **测量口径**：测 decode 性能用普通 server + ISL=1，**不要**用 disagg decode-only fake（后者 rank 失同步会把 gap 放大到 ~12%）。
- **残余 ~2%**：主要是 EPv2 独有的 standalone 预量化 +2.7ms（不可消——DeepEP v2 dispatch API 要求传入已量化 fp8，LL 把量化融在 dispatch kernel 内）+ per-step 墙钟/噪声。GPU kernel 已与 LL parity，repack 这条线已挖到底。

### 限制

- direct/hybrid mode 在 server 生命周期内固定，无 DeepEP v1 `auto` 那种 prefill/decode 自动切换。
- masked path 只覆盖 `direct + deep_gemm + decode`；direct extend、hybrid、Triton 走各自路径并自动关 graph。
- adapter 只覆盖 DeepGEMM FP8 与 Triton BF16，其它 runner fail-fast。
- `EpV2Buffer` 是 singleton（按 group/hidden/topk/cap/dtype/mode/world 做 key），多模型/多 group 混合切换需改显式 per-key 生命周期。
