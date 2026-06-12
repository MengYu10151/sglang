# DeepSeek-V4-Flash-FP8 + SGLang epv2 (DeepEP v2) 集成与测试报告

**日期:** 2026-06-12  
**硬件:** 2× NVIDIA H20-3e 节点（各 8 GPU，NVLink + RoCE）  
**节点:** `10.6.131.22` (H20-GPU-22)、`10.6.131.23` (H20-GPU-23)  
**模型:** `/models/DeepSeek-V4-Flash-FP8`  
**SGLang 代码:** `/root/jianxiong/sglang`  
**DeepEP v2:** `/root/jianxiong/DeepEP` (2.0.0)  
**自定义 NCCL:** `/root/jianxiong/nccl/build` (libnccl.so.2.30.7, sm_90)

---

## 1. 项目目标

在 SGLang 中新增 **epv2** MoE A2A 后端，对接 DeepEP v2 的 `ElasticBuffer`，使 DeepSeek-V4-Flash-FP8 能在 H20 上以 EP 方式推理；并完成单机/双机功能验证与性能压测。

参考实现：menyu@nvidia.com 在 SGLang 中集成的 **ncclep** 后端模式。

---

## 2. epv2 Backend Patch 制作与说明

### 2.1 Patch 文件

| 项 | 路径 |
|----|------|
| Patch 文件 | `/root/jianxiong/sglang-epv2-backend.patch` |
| 行数 | 864 行 |
| 涉及文件 | 9 个（+643 行） |

### 2.2 生成方式

在 `/root/jianxiong/sglang` 仓库内完成修改后：

```bash
cd /root/jianxiong/sglang
git diff > /root/jianxiong/sglang-epv2-backend.patch
```

应用 patch：

```bash
cd /path/to/sglang
git apply /root/jianxiong/sglang-epv2-backend.patch
```

### 2.3 修改文件清单

| 文件 | 变更说明 |
|------|----------|
| `python/sglang/srt/layers/moe/token_dispatcher/epv2.py` | **新增** — `EpV2Dispatcher`，封装 DeepEP v2 `ElasticBuffer` 的 dispatch/combine |
| `python/sglang/srt/layers/moe/utils.py` | 新增 `MoeA2ABackend.EPV2`、`EpV2OutputDtype`、`get_epv2_output_dtype()` |
| `python/sglang/srt/layers/moe/token_dispatcher/base.py` | 新增 `DispatchOutputFormat.EPV2`、`CombineInputFormat.EPV2` |
| `python/sglang/srt/layers/moe/token_dispatcher/__init__.py` | 导出 epv2 相关类型 |
| `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` | 在 `create_moe_dispatcher()` 中 wire `EpV2Dispatcher` |
| `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py` | epv2 pre/post permute 适配 DeepGEMM runner |
| `python/sglang/srt/layers/moe/moe_runner/triton.py` | epv2 pre/post permute 适配 Triton runner |
| `python/sglang/srt/server_args.py` | 新增 `--moe-a2a-backend epv2`、`--epv2-dispatcher-output-dtype` |
| `python/sglang/srt/environ.py` | 新增 `SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK`、`SGLANG_EPV2_ALLOW_HYBRID_MODE` |

### 2.4 架构设计（对标 ncclep）

```
Token → EpV2Dispatcher.dispatch()
          └─ ElasticBuffer.dispatch()   # DeepEP v2 A2A
       → MoeRunner (deep_gemm / triton)  # 本地 expert 计算
       → EpV2Dispatcher.combine()
          └─ ElasticBuffer.combine()
       → 输出 hidden states
```

**关键设计点：**

- **ElasticBuffer 单例缓存** (`EpV2Buffer`)：按 `(group, hidden, topk, max_tokens, fp8, world_size)` 复用 buffer。
- **FP8 dispatch**：`--epv2-dispatcher-output-dtype fp8` 时，dispatch 前做 per-token-group FP8 量化，与 DeepGEMM 路径对齐。
- **ep_size 自动对齐 tp_size**：与 deepep/ncclep 一致，启用 epv2 时 `ep_size = tp_size`。
- **CUDA Graph 自动禁用**：DeepEP v2 高吞吐 dispatch 与 cudagraph 不兼容。

### 2.5 新增 CLI 参数

```bash
--moe-a2a-backend epv2
--epv2-dispatcher-output-dtype {auto,bf16,fp8}   # 推荐 fp8 + deep_gemm
--moe-runner-backend deep_gemm
```

### 2.6 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | 128 | 每 rank 最大 dispatch token 数（压测用 1024） |
| `SGLANG_EPV2_ALLOW_HYBRID_MODE` | False | True=Hybrid(RDMA+NVLink)，False=Direct(NVLink) |
| `EP_REUSE_NCCL_COMM` | 0 | 0=DeepEP 自建 NCCL comm，避免与 PyTorch comm 混用 segfault |
| `EP_DISABLE_GIN` | 1（单机） | 无 IB GIN 的单机 NVLink 需关闭 GIN |
| `EP_NCCL_ROOT_DIR` | — | 指向自定义 NCCL build 目录 |

### 2.7 运行时 Bug 修复（epv2.py，未单独进 patch 摘要）

测试过程中在 `epv2.py` 额外修复：

1. **移除** `ElasticBuffer.get_buffer_size_hint()` 预调用 — 与 PyTorch NCCL comm 混用时 segfault。
2. **`allow_hybrid_mode=False`** 默认（单机 NVLink）；Hybrid 测试时显式设 `SGLANG_EPV2_ALLOW_HYBRID_MODE=1`。
3. **`event.current_stream_wait()` 守卫** — sync dispatch (`do_cpu_sync=True`) 时 `event.event` 可能为 None。

---

## 3. 环境与依赖构建

### 3.1 最新 NCCL 编译

脚本：`/root/jianxiong/build_nccl.sh`

```bash
git clone --depth 1 https://github.com/NVIDIA/nccl.git /root/jianxiong/nccl
make -C /root/jianxiong/nccl -j$(nproc) src.build \
  CUDA_HOME=/usr/local/cuda \
  NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
# 产物: /root/jianxiong/nccl/build/lib/libnccl.so.2.30.7
```

NCCL 需支持 **NCCL GIN / Device API**，供 DeepEP v2 Hybrid 模式使用。

### 3.2 DeepEP v2 安装

在 `lmsysorg/sglang:dev` 容器内：

```bash
cd /DeepEP
EP_NCCL_ROOT_DIR=/nccl-src/build pip install --no-build-isolation -e .
python3 -c "import deep_ep; print(deep_ep.__version__)"  # 2.0.0
```

### 3.3 运行时环境脚本

`/root/jianxiong/env_epv2.sh` — 容器内 `source /env_epv2.sh`：

- `LD_PRELOAD` 指向自定义 `libnccl.so.2`
- symlink pip 自带 NCCL 到自定义 build
- `NCCL_CUMEM_ENABLE=1`、`NCCL_WIN_ENABLE=1`
- 单机默认 `EP_DISABLE_GIN=1`、`SGLANG_EPV2_ALLOW_HYBRID_MODE=0`

### 3.4 Docker 容器

**Node 22:**

```bash
docker run -d --name sglang_epv2_run --gpus all --ipc=host --network=host \
  -v /root/jianxiong/sglang:/workspace/sglang_epv2 \
  -v /root/jianxiong/DeepEP:/DeepEP \
  -v /root/jianxiong/nccl:/nccl-src \
  -v /root/jianxiong/env_epv2.sh:/env_epv2.sh:ro \
  -v /root/menyu/models:/models:ro \
  lmsysorg/sglang:dev sleep infinity
```

**Node 23:** 同镜像、同挂载（代码/rsync 同步）。

### 3.5 网络配置（双机）

| 变量 | 值 |
|------|-----|
| `NCCL_SOCKET_IFNAME` | `bond0` |
| `GLOO_SOCKET_IFNAME` | `bond0` |
| `NCCL_IB_HCA` | `mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7` |
| IB 设备（PD KV 传输） | `mlx5_bond_0` |

---

## 4. 测试 Case 总览

| # | Case | 拓扑 | epv2 模式 | 传输 | 状态 | 日志 |
|---|------|------|-----------|------|------|------|
| 1 | 单机服务启动 | 1×8 GPU TP8 EP8 | Direct | — | ✅ | `logs/dsv4_epv2_tp8_20260612_*` |
| 2 | 功能 — 长江 | Case 1 | Direct | — | ✅ | 对话测试 |
| 3 | 压测 isl=1024 osl=128 | Case 1 | Direct | — | ✅ | `logs/dsv4_epv2_bench_20260612_162612/` |
| 4 | 压测 isl=1024 osl=1024 | Case 1 | Direct | — | ✅ | `logs/dsv4_epv2_bench_1024_1024_20260612_163342/` |
| 5 | Hybrid 模式切换 | Case 1 | Hybrid | — | ✅ | `logs/dsv4_epv2_hybrid_20260612_170001/` |
| 6 | 功能 — 黄河（Hybrid） | Case 5 | Hybrid | — | ✅ | ~15s |
| 7 | 功能 — 珠江（Hybrid） | Case 5 | Hybrid | — | ✅ | ~15s |
| 8 | 双机 Colocated TP16 | 2×8 GPU | Hybrid | NCCL IB | ❌ | DSV4 `o_groups=8` 与 TP16 不兼容 |
| 9 | PD 1P1D（Mooncake） | P@.22 D@.23 | Hybrid | Mooncake | ❌ | Mooncake TransferEngine 不可用 |
| 10 | PD 1P1D（NIXL） | P@.22 D@.23 | Hybrid | NIXL/UCX | ✅ | `logs/pd_1p1d_nixl_20260612_180704/` |
| 11 | PD 功能 — 珠江 | Case 10 | Hybrid | NIXL | ✅ | Router :8000 |
| 12 | PD 压测 isl=1024 osl=128 | Case 10 | Hybrid | NIXL | ✅ | `logs/pd_1p1d_bench_20260612_181951/` |

---

## 5. 单机测试详情

### 5.1 启动命令

脚本：`/root/jianxiong/run_dsv4_epv2_tp8.sh`

```bash
python3 -m sglang.launch_server \
  --model-path /models/DeepSeek-V4-Flash-FP8 \
  --trust-remote-code --host 0.0.0.0 --port 30000 \
  --tp-size 8 --ep-size 8 \
  --moe-a2a-backend epv2 \
  --moe-runner-backend deep_gemm \
  --epv2-dispatcher-output-dtype fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.70 \
  --context-length 4096 \
  --disable-cuda-graph --skip-server-warmup
```

环境：`SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024`

### 5.2 Direct vs Hybrid

| 模式 | 环境变量 | 日志特征 |
|------|----------|----------|
| **Direct** | `SGLANG_EPV2_ALLOW_HYBRID_MODE=0`, `EP_DISABLE_GIN=1` | `allow_hybrid_mode=False` |
| **Hybrid** | `SGLANG_EPV2_ALLOW_HYBRID_MODE=1`, `EP_DISABLE_GIN=1` | `allow_hybrid_mode=True` |

单机 8×H20 NVLink 无 IB GIN，两者均设 `EP_DISABLE_GIN=1`；Hybrid 主要影响 EP kernel 路径选择。

### 5.3 功能测试结果

| Prompt | 模式 | 回答 | 耗时 |
|--------|------|------|------|
| 中国最长的河流是什么 | Direct | 长江，~6300 km | ~17s |
| 中国第二长的河流是哪个 | Direct | 黄河，~5464 km | ~17s |
| 中国第三长的河流是哪个 | Hybrid | 珠江，~2320 km | ~15s |

---

## 6. 压测结果汇总

**公共参数：** random dataset, `random-range-ratio=1.0`, `num-prompts=32`, `max-concurrency=8`, `warmup-requests=4`, `request-rate=inf`

### 6.1 单机 Colocated（Direct, TP8 EP8, port 30000）

**isl=1024 / osl=128** — `logs/dsv4_epv2_bench_20260612_162612/bench.jsonl`

| 指标 | 值 |
|------|-----|
| 总耗时 | 139.9 s |
| Mean TTFT | **3668 ms** |
| Median TTFT | 1346 ms |
| P99 TTFT | 10642 ms |
| Mean TPOT | **246 ms** |
| Median TPOT | 247 ms |
| Output throughput | **29.3 tok/s** |
| Total throughput | 263.6 tok/s |

**isl=1024 / osl=1024** — `logs/dsv4_epv2_bench_1024_1024_20260612_163342/bench.log`

| 指标 | 值 |
|------|-----|
| 总耗时 | 918.6 s (~15 min) |
| Mean TTFT | **7204 ms** |
| Median TTFT | 808 ms |
| P99 TTFT | 45603 ms |
| Mean TPOT | **183 ms** |
| Median TPOT | 176 ms |
| Output throughput | **35.7 tok/s** |
| Total throughput | 71.3 tok/s |

> osl 从 128 增到 1024 后，decode batch 更满，TPOT 下降、输出吞吐上升；Mean TTFT 受少数长 prefill 拉高。

### 6.2 双机 PD 1P1D（Hybrid, NIXL, Router :8000）

**isl=1024 / osl=128** — `logs/pd_1p1d_bench_20260612_181951/bench.log`

| 指标 | 值 |
|------|-----|
| 总耗时 | **94.4 s** |
| Mean TTFT | **1070 ms** |
| Median TTFT | 1093 ms |
| P99 TTFT | 1389 ms |
| Mean TPOT | **177 ms** |
| Median TPOT | 177 ms |
| Output throughput | **43.4 tok/s** |
| Input throughput | 347.2 tok/s |
| Total throughput | 390.5 tok/s |
| Mean E2E | 23.5 s |

### 6.3 单机 vs PD 对比（isl=1024, osl=128）

| 指标 | 单机 TP8 Direct | PD 1P1D Hybrid |
|------|---------------|----------------|
| Mean TTFT | 3668 ms | **1070 ms** ↓71% |
| Median TTFT | 1346 ms | 1093 ms |
| Mean TPOT | 246 ms | **177 ms** ↓28% |
| Output tok/s | 29.3 | **43.4** ↑48% |
| 总耗时 (32 req) | 140 s | **94 s** ↓33% |

PD 分离后 Prefill 不再打断 Decode，TTFT 更稳定、decode 吞吐更高。

---

## 7. 双机 PD 1P1D 部署详情

### 7.1 拓扑

```
10.6.131.22  ── Prefill  (TP8 EP8, :30000, bootstrap :8998)
      │ NIXL/UCX KV transfer (mlx5_bond_0)
10.6.131.23  ── Decode   (TP8 EP8, :30001)
10.6.131.22  ── Router   (:8000)
```

### 7.2 启动脚本

| 脚本 | 运行节点 |
|------|----------|
| `run_dsv4_epv2_2node_pd_prefill.sh` | .22 |
| `run_dsv4_epv2_2node_pd_decode.sh` | .23 |
| `run_dsv4_epv2_2node_pd_router.sh` | .22 |
| `run_pd_1p1d_test.sh` | 编排脚本 |

关键参数：

```bash
--disaggregation-mode prefill|decode
--disaggregation-transfer-backend nixl   # 非 mooncake
--disaggregation-ib-device mlx5_bond_0
--tp-size 8 --ep-size 8 --moe-a2a-backend epv2
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
export UCX_NET_DEVICES=bond0
export EP_DISABLE_GIN=1   # 各节点单机 8 GPU
```

### 7.3 压测命令（经 Router）

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url http://10.6.131.22:8000 \
  --model /models/DeepSeek-V4-Flash-FP8 \
  --tokenizer /models/DeepSeek-V4-Flash-FP8 \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 128 \
  --num-prompts 32 --max-concurrency 8 --warmup-requests 4
```

---

## 8. 问题与解决记录

| 问题 | 现象 | 根因 | 解决 |
|------|------|------|------|
| ImportError: ElasticBuffer | 服务无法启动 | DeepEP v2 未安装 | 容器内 `pip install -e /DeepEP` |
| NCCL GIN unavailable | Hybrid 跨节点失败 | 单机无 IB GIN | 单机设 `EP_DISABLE_GIN=1` |
| Segfault in calculate_buffer_size | 启动崩溃 | PyTorch NCCL comm 与 DeepEP 混用 | `EP_REUSE_NCCL_COMM=0`，移除 buffer size hint 预调用 |
| AssertionError: event is not None | dispatch 崩溃 | sync 模式 event 为空 | 守卫 `event.event is not None` 再 wait |
| TP16 Colocated 崩溃 | `n_local_groups=0` | DSV4 `o_groups=8` 无法被 TP16 整除 | 需 `--enable-dp-attention --dp-size 2`（未完整验证） |
| PD Mooncake 启动失败 | P/D 秒退 | `mooncake.engine.TransferEngine` 不可用 | 改用 `--disaggregation-transfer-backend nixl` |
| PD 首请求超时 | health 失败、Router 挂住 | Decode 首请求触发 DeepGEMM JIT 编译 | 等待 JIT 完成或预跑 `sglang.compile_deep_gemm` |
| Node 23 SSH | 默认 22 端口 key 失败 | 集群 SSH 端口/认证 | 使用 `sshpass -p root2930 ssh -p 22` |

---

## 9. 产物清单

| 类型 | 路径 |
|------|------|
| **Patch** | `/root/jianxiong/sglang-epv2-backend.patch` |
| **环境脚本** | `/root/jianxiong/env_epv2.sh` |
| **NCCL 构建** | `/root/jianxiong/build_nccl.sh` |
| **单机启动** | `/root/jianxiong/run_dsv4_epv2_tp8.sh` |
| **双机 Colocated** | `/root/jianxiong/run_dsv4_epv2_2node_colocated.sh`, `launch_colocated_rank.sh` |
| **双机 PD** | `/root/jianxiong/run_dsv4_epv2_2node_pd_{prefill,decode,router}.sh` |
| **PD 编排** | `/root/jianxiong/run_pd_1p1d_test.sh` |
| **压测日志** | `/root/jianxiong/logs/dsv4_epv2_bench_*`, `pd_1p1d_bench_*` |
| **服务日志** | `/root/jianxiong/logs/dsv4_epv2_tp8_*`, `dsv4_epv2_hybrid_*`, `pd_1p1d_*` |

---

## 10. 结论与建议

1. **epv2 backend 集成成功**：DeepSeek-V4-Flash-FP8 在 8×H20 上以 TP8/EP8 + epv2 + DeepGEMM 稳定推理，Direct/Hybrid 均可工作。
2. **PD 1P1D 显著优于单机 Colocated**（同 isl/osl）：TTFT 降低 71%，输出吞吐提升 48%；推荐使用 NIXL 而非 Mooncake（当前容器环境）。
3. **双机 Colocated TP16** 需 DPA（`dp-size=2`）适配 DSV4 的 `o_groups=8`，尚未完成端到端验证。
4. **生产建议**：
   - 预编译 DeepGEMM：`python3 -m sglang.compile_deep_gemm --model-path ... --tp 8`
   - PD 部署使用 NIXL + UCX，`NCCL_IB_HCA=mlx5_0,...,mlx5_7`
   - 多机 Hybrid 场景设 `EP_DISABLE_GIN=0` 并确保 NCCL GIN 可用

---

## 附录 A：推荐单机启动（Direct）

```bash
source /env_epv2.sh
export SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
bash /root/jianxiong/run_dsv4_epv2_tp8.sh
```

## 附录 B：推荐双机 PD 1P1D 启动

```bash
# Node 23
export EP_DISABLE_GIN=1
bash /root/jianxiong/run_dsv4_epv2_2node_pd_decode.sh

# Node 22
export EP_DISABLE_GIN=1
bash /root/jianxiong/run_dsv4_epv2_2node_pd_prefill.sh
bash /root/jianxiong/run_dsv4_epv2_2node_pd_router.sh

# 测试
curl http://10.6.131.22:8000/v1/chat/completions ...
```

## 附录 C：Patch 应用与 DeepEP 安装一键流程

```bash
# 1. 应用 patch
cd /root/jianxiong/sglang && git apply /root/jianxiong/sglang-epv2-backend.patch

# 2. 构建 NCCL
bash /root/jianxiong/build_nccl.sh

# 3. 容器内安装 DeepEP v2
docker exec -it sglang_epv2_run bash
source /env_epv2.sh
cd /DeepEP && EP_NCCL_ROOT_DIR=/nccl-src/build pip install --no-build-isolation -e .

# 4. 启动服务
bash /root/jianxiong/run_dsv4_epv2_tp8.sh
```
