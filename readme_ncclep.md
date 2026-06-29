# SGLang NCCL_EP Integration Guide

本文档记录当前 NCCL_EP/Nico EP 在 SGLang 中的可运行部署方式、验证方法和已经完成的主要优化。

当前目标不是覆盖所有模型和所有 backend，而是明确一条已经验证过的 SM120 路径：

- SGLang fork + NCCL_EP 独立 MoE A2A backend。
- 自编译 NCCL/NCCL_EP。
- SM120 上需要应用 NCCL_EP low-latency patch。
- DeepSeek-V4-Flash + DeepGEMM + NCCL_EP low_latency 可运行并通过 correctness。

## 1. Deploy Guide

### 1.1 Validated Environment

当前验证环境：

| Item | Value |
|---|---|
| Host | 5k11, `10.6.142.11`, no-NVL 8 GPU SM120 machine |
| Workdir | `/root/menyu` |
| Docker | `5k11_ncclep_qwen` |
| SGLang repo | `/root/menyu/sglang` |
| SGLang branch | `ncclep-dsv4-sm120-deepgemm-opt` |
| Last validated commit | `d57063e17 Clean up NCCL EP DSV4 integration` |
| NCCL lib | `/root/menyu/nccl/build/lib/libnccl.so.2.30.7` |
| NCCL_EP lib | `/root/menyu/nccl/build/lib/libnccl_ep.so.0.0.1` |
| Python | `3.12.3` |
| PyTorch | `2.11.0+cu130` |
| DeepGEMM | `2.5.0` |
| DSV4 model | `/root/menyu/models/DeepSeek-V4-Flash` |

The important point is that SGLang does not use a system NCCL_EP package. Runtime must load the locally built NCCL/NCCL_EP pair and the matching `nccl4py` binding.

### 1.2 SGLang Fork

Use our SGLang fork and the NCCL_EP integration branch:

```bash
cd /root/menyu
git clone https://github.com/MengYu10151/sglang.git
cd /root/menyu/sglang
git checkout ncclep-dsv4-sm120-deepgemm-opt
```

Install SGLang in the dev container:

```bash
cd /root/menyu/sglang
pip install -e python
```

For DSV4 Flash + DeepGEMM on SM120, the current branch contains the needed SGLang-side changes:

- `--moe-a2a-backend ncclep`
- `--ncclep-mode high_throughput | low_latency`
- `--ncclep-dispatcher-output-dtype bf16 | fp8`
- `--ncclep-config <json>`
- SM120 DeepGEMM wrapper changes for DSV4 FP8/FP4 expert path
- NCCL_EP event-based sync and handle lifetime management

### 1.3 NCCL/NCCL_EP for SM120

SM120 needs an NCCL_EP patch before building low_latency. Use the public patch and guide here:

```text
https://github.com/qijiaxing/nccl/tree/v2.30u1-sm120/contrib/nccl_ep/sm120
```

That guide provides:

- `contrib/nccl_ep/sm120/nccl_ep_sm120_ll_low_latency.patch`
- build instructions for `sm_120`
- an `ep_bench` validation command

The patch reduces NCCL_EP low_latency kernel shared-memory pressure for RTX 5k/6kD style SM120 GPUs. Do not use an unpatched NCCL_EP low_latency build on SM120 for this integration test.

Public-source flow:

```bash
cd /root/menyu
git clone https://github.com/qijiaxing/nccl.git
cd /root/menyu/nccl
git checkout v2.30u1-sm120

git apply contrib/nccl_ep/sm120/nccl_ep_sm120_ll_low_latency.patch

export NVCC_GENCODE="-gencode=arch=compute_120,code=sm_120"
export CUDA_HOME=/usr/local/cuda
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1

make -C . src.build BUILDDIR=$PWD/build -j
make -C contrib/nccl_ep lib ep_bench MPI=1 BUILDDIR=$PWD/build -j
pip install -e bindings/nccl4py
```

Current internal environment uses an equivalent locally patched tree at:

```text
/root/menyu/nccl
```

### 1.4 Runtime Environment Variables

Load the matching NCCL/NCCL_EP pair before launching SGLang:

```bash
source /root/menyu/ncclep_build/ncclep_env.sh
```

Current `ncclep_env.sh`:

```bash
export NCCL_HOME="/root/menyu/nccl/build"
export LD_LIBRARY_PATH="/root/menyu/nccl/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD="/root/menyu/nccl/build/lib/libnccl.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"
export PYTHONPATH="/root/menyu/nccl/bindings/nccl4py${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_NCCL_EP_SO_PATH="/root/menyu/nccl/build/lib/libnccl_ep.so"
export NCCL_EP_JIT_BUILD_INCLUDE_DIR="/root/menyu/nccl/build/include"
export NCCL_EP_JIT_SOURCE_DIR="/root/menyu/nccl/bindings/nccl4py/nccl/ep/include"
export NCCL_CUMEM_ENABLE=1
export NCCL_WIN_ENABLE=1
```

For 5k11 no-NVL runs, also use:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_DSV4_FP4_EXPERTS=1
export SGLANG_OPT_USE_JIT_EP_ACTIVATION=true
export SGLANG_NCCL_EP_SYNC_MODE=event
export SGLANG_NCCL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export NCCL_LSA_TEAM_SIZE=1
export NCCL_NET_MERGE_LEVEL=LOC
export NCCL_CUMEM_ENABLE=1
export NCCL_WIN_ENABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_EP_TIMEOUT_MS=10000
```

### 1.5 Import Check

Run this before starting SGLang:

```bash
source /root/menyu/ncclep_build/ncclep_env.sh
cd /root/menyu/sglang/python

python3 - <<'PY'
import os
import torch
import nccl
import nccl.ep

print("torch =", torch.__version__)
print("nccl =", nccl.__file__)
print("nccl.ep =", nccl.ep.__file__)
print("SGLANG_NCCL_EP_SO_PATH =", os.environ.get("SGLANG_NCCL_EP_SO_PATH"))
PY
```

### 1.6 DSV4 + NCCL_EP LL Launch Template

This is the validated SM120 DSV4 server shape:

```bash
source /root/menyu/ncclep_build/ncclep_env.sh

export PYTHONPATH="/root/menyu/sglang/python:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_DSV4_FP4_EXPERTS=1
export SGLANG_OPT_USE_JIT_EP_ACTIVATION=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=true
export SGLANG_NCCL_EP_SYNC_MODE=event
export SGLANG_NCCL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export NCCL_LSA_TEAM_SIZE=1
export NCCL_NET_MERGE_LEVEL=LOC
export NCCL_NVLS_ENABLE=0

cat >/tmp/ncclep_ll_sm32.json <<'JSON'
{"low_latency":{"num_sms":32,"layout":"expert_major"}}
JSON

python3 -m sglang.launch_server \
  --model-path /root/menyu/models/DeepSeek-V4-Flash \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 32123 \
  --context-length 16384 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --tp-size 8 \
  --dp-size 8 \
  --ep-size 8 \
  --enable-dp-attention \
  --attention-backend dsv4 \
  --sampling-backend flashinfer \
  --mamba-backend triton \
  --moe-runner-backend deep_gemm \
  --moe-a2a-backend ncclep \
  --ncclep-mode low_latency \
  --ncclep-dispatcher-output-dtype fp8 \
  --ncclep-config /tmp/ncclep_ll_sm32.json \
  --disable-overlap-schedule \
  --disable-cuda-graph \
  --disable-radix-cache \
  --disable-chunked-prefix-cache \
  --disable-shared-experts-fusion \
  --disable-flashinfer-autotune \
  --chunked-prefill-size 2048 \
  --page-size 256 \
  --max-running-requests 256 \
  --sm-group-num 8 \
  --random-seed 42 \
  --skip-server-warmup
```

Notes:

- `--ncclep-mode low_latency` is NCCL_EP mode. Do not use `--deepep-mode`.
- For DSV4 Flash + DeepGEMM, use `--ncclep-dispatcher-output-dtype fp8`.
- The LL config uses `num_sms=32` and `expert_major`.
- CUDA graph is still disabled in this validated path.

## 2. Test Guide and Sample Data

### 2.1 Correctness Smoke

Validated script:

```bash
RUN_ID=clean_commit_correctness_$(date +%Y%m%d_%H%M%S) \
SGLANG_NCCL_EP_SYNC_MODE=event \
SGLANG_OPT_SWIGLU_CLAMP_FUSION=true \
NCCLEP_MAX_DISPATCH=256 \
/root/menyu/run_dsv4_ncclep_deepgemm_correctness.sh
```

Latest validated result:

```text
/root/menyu/ncclep_vs_tp_sglang_dsv4_flash/deepgemm_correctness_clean_commit_correctness_20260625_231230
```

Expected checks:

- The capital prompt should include Beijing and Tokyo.
- `17*23+19` should output `410`.
- `NCCL_EP low_latency shared buffer cache is invalid` count should be `0`.
- Server log should not contain traceback/runtime error/shape mismatch.

### 2.2 Serving Benchmark

Validated script:

```bash
RUN_ID=sync_event_latency_$(date +%Y%m%d_%H%M%S) \
CONFIGS="tp8 tp8_dp dp8_tp dp8_ep_ncclep_ll" \
SHAPES="8192:1024 1024:1024 1024:8192" \
NUM_PROMPTS=3 \
CONCURRENCY=1 \
NCCLEP_MAX_DISPATCH=256 \
/root/menyu/run_dsv4_sync_event_latency_matrix.sh
```

Important benchmark settings:

- Model: `/root/menyu/models/DeepSeek-V4-Flash`
- Dataset: `random-ids`
- `--tokenize-prompt`
- `--random-range-ratio 1.0`
- `--max-concurrency 1`
- `--num-prompts 3`
- Shapes: `8192/1024`, `1024/1024`, `1024/8192`
- DP+EP: `--tp-size 8 --dp-size 8 --ep-size 8 --enable-dp-attention`
- NCCL_EP: `low_latency`, FP8 dispatcher output, cap=256
- DeepGEMM runner, DSV4 attention backend

Latest validated result:

```text
/root/menyu/ncclep_vs_tp_sglang_dsv4_flash/sync_event_latency_matrix_20260625_115116
```

Result table:

| Config | ISL | OSL | Mean TTFT ms | Mean TPOT ms | Output tok/s | Total tok/s | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| TP | 8192 | 1024 | 24919.38 | 85.84 | 9.08 | 81.75 | PASS |
| TP | 1024 | 1024 | 2975.47 | 84.47 | 11.46 | 22.91 | PASS |
| TP | 1024 | 8192 | 2977.47 | 84.58 | 11.77 | 13.25 | PASS |
| TP+DP | N/A | N/A | N/A | N/A | N/A | N/A | SERVER_FAIL |
| DP+TP | 8192 | 1024 | 26729.78 | 85.41 | 8.97 | 80.77 | PASS |
| DP+TP | 1024 | 1024 | 5992.29 | 85.50 | 10.96 | 21.91 | PASS |
| DP+TP | 1024 | 8192 | 4206.90 | 87.40 | 11.38 | 12.80 | PASS |
| DP+EP NCCL_EP LL | 8192 | 1024 | 26957.38 | 89.42 | 8.65 | 77.81 | PASS |
| DP+EP NCCL_EP LL | 1024 | 1024 | 6945.98 | 86.32 | 10.75 | 21.50 | PASS |
| DP+EP NCCL_EP LL | 1024 | 8192 | 4626.85 | 87.41 | 11.37 | 12.79 | PASS |

Supplemental prefill/decode benchmark:

```bash
RUN_ID=ds_supp_20260629_1008 \
CONFIGS="tp8 dp8_tp dp8_ep_ncclep_ll" \
SHAPES="1024:1 8192:1 1:1024" \
NUM_PROMPTS=3 \
CONCURRENCY=1 \
NCCLEP_MAX_DISPATCH=256 \
/root/menyu/run_dsv4_sync_event_latency_matrix.sh
```

Latest supplemental result:

```text
/root/menyu/ncclep_vs_tp_sglang_dsv4_flash/sync_event_latency_matrix_ds_supp_20260629_1008
```

| Config | ISL | OSL | Mean TTFT ms | Mean TPOT ms | Output tok/s | Total tok/s | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| TP | 1024 | 1 | 3082.14 | 0.00 | 0.32 | 332.34 | PASS |
| DP+TP | 1024 | 1 | 3851.31 | 0.00 | 0.17 | 176.70 | PASS |
| DP+EP NCCL_EP LL | 1024 | 1 | 4351.52 | 0.00 | 0.15 | 156.03 | PASS |
| TP | 8192 | 1 | 24816.15 | 0.00 | 0.04 | 330.12 | PASS |
| DP+TP | 8192 | 1 | 17643.91 | 0.00 | 0.04 | 309.18 | PASS |
| DP+EP NCCL_EP LL | 8192 | 1 | 17928.70 | 0.00 | 0.04 | 304.42 | PASS |
| TP | 1 | 1024 | 92.34 | 84.53 | 11.83 | 11.84 | PASS |
| DP+TP | 1 | 1024 | 915.00 | 85.00 | 11.65 | 11.66 | PASS |
| DP+EP NCCL_EP LL | 1 | 1024 | 1124.44 | 85.37 | 11.58 | 11.59 | PASS |

Observations:

- For long prefill (`8192/1`), DP+EP NCCL_EP LL is close to DP+TP: total throughput is 304.42 tok/s vs 309.18 tok/s, about 1.5% lower. It is about 7.8% lower than pure TP.
- For short prefill (`1024/1`), DP+EP NCCL_EP LL is weaker: total throughput is 156.03 tok/s, about 11.7% lower than DP+TP and about 53.0% lower than pure TP.
- For decode-heavy (`1/1024`), DP+EP NCCL_EP LL is close to pure TP and DP+TP: TPOT is 85.37 ms vs 84.53 ms for pure TP and 85.00 ms for DP+TP.

TP+DP note:

- `tp=8, dp=8, no-dp-attn` is not a valid single-node 8-GPU setting because it tries to allocate beyond 8 device ordinals.
- A reduced `tp=4, dp=2, no-dp-attn` variant failed in the DSV4 DeepGEMM `silu_and_mul_masked_post_quant` path.
- This failure is not NCCL_EP related, so use `TP`, `DP+TP`, and `DP+EP` for NCCL_EP comparison.

### 2.3 Native NCCL_EP SM120 Check

Use the SM120 guide's `ep_bench` before running SGLang:

```bash
export GPUS=4
export NCCL_HOME=/root/menyu/nccl/build
export CUDA_HOME=/usr/local/cuda
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_EP_JIT_CACHE_DIR=".jit-cache/nccl_ep_jit_sm120_lsa1"
export NCCL_LSA_TEAM_SIZE=1
export NCCL_CUMEM_ENABLE=1
export NCCL_WIN_ENABLE=1

mpirun --allow-run-as-root -np ${GPUS} \
  -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES \
  -x NCCL_LSA_TEAM_SIZE -x NCCL_CUMEM_ENABLE -x NCCL_WIN_ENABLE \
  -x NCCL_EP_JIT_CACHE_DIR \
  ${NCCL_HOME}/test/nccl_ep/ep_bench \
  --algorithm ll \
  --layout em \
  --tokens 4096 \
  --hidden 4096 \
  --max-num-sms 32 \
  --validate \
  --top-k 6 \
  --experts 256 \
  --warmup 10 \
  --iters 50
```

For SGLang + DeepGEMM, `expert_major`/`em` is the validated serving layout. `rank_major` can reduce receive-buffer shape in native testing, but it needs extra compaction before grouped GEMM and is not the current clean integration path.

## 3. Optimization Summary

### 3.1 Backend and Semantics

- Added NCCL_EP as an independent MoE A2A backend: `--moe-a2a-backend ncclep`.
- Added NCCL_EP-specific mode: `--ncclep-mode high_throughput | low_latency`.
- Avoided mixing NCCL_EP semantics with DeepEP mode/config.
- Added dispatcher output dtype control: `--ncclep-dispatcher-output-dtype bf16 | fp8 | auto`.

### 3.2 DSV4 + DeepGEMM Path

- Added FP8 activation + scale adapter for NCCL_EP low_latency into DeepGEMM.
- Added DSV4 FP8xFP4 DeepGEMM wrapper calls for SM120 path.
- Preserved NCCL_EP shared output buffers by preventing DeepGEMM from disposing them.
- Added SM120 DeepGEMM enablement in the SGLang wrapper path.

### 3.3 NCCL_EP LL Runtime

- Replaced per-dispatch/combine device-wide synchronize with event-based mode:
  - default: `SGLANG_NCCL_EP_SYNC_MODE=event`
  - fallback: `SGLANG_NCCL_EP_SYNC_MODE=device`
- Added deferred NCCL_EP handle destroy using CUDA events.
- Kept shared-buffer shape validation to catch accidental resize/dispose bugs.
- Removed temporary LL trace/dump/barrier/debug env paths before committing clean code.

### 3.4 Cap and Buffer Tuning

- DSV4 non-PD LL serving uses cap=256:
  - `SGLANG_NCCL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256`
- cap=128 can be useful for decode-only/PD-D style tests, but it fails prefill chunks that reach 256 tokens.
- cap=512 is stable but increases masked DeepGEMM buffer and temporary tensor cost.
- Current no-NVL SM120 path uses:
  - `NCCL_LSA_TEAM_SIZE=1`
  - `NCCL_NET_MERGE_LEVEL=LOC`
  - `NCCL_NVLS_ENABLE=0`

### 3.5 Current Known Limits

- CUDA graph remains disabled for this validated NCCL_EP LL path.
- DP+EP is close to DP+TP at `CC=1`, but does not clearly outperform it yet.
- Performance work should focus on:
  - expert-major buffer/layout overhead
  - DeepGEMM masked path capacity cost
  - real routing long-tail on no-NVL topology
  - CUDA graph capture safety
  - static/dynamic expert placement and load balance

## 4. Quick Recap

The clean SM120 reproduction path is:

1. Use our SGLang fork and NCCL_EP integration branch.
2. Build NCCL/NCCL_EP with the SM120 low_latency patch from `qijiaxing/nccl`.
3. Source `ncclep_env.sh` so SGLang loads the matching NCCL/NCCL_EP pair.
4. Run DSV4 Flash with DeepGEMM, DP attention, NCCL_EP low_latency, FP8 dispatcher output, `num_sms=32`, cap=256.
5. Validate with correctness prompts and then run the latency matrix.
