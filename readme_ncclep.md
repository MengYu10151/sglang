# SGLang NCCL_EP Integration README

本文档总结当前 SGLang 中 NCCL_EP 的模型适配、runner 适配、部署方式和已知限制。

## 当前结论

- NCCL_EP 作为独立 MoE A2A backend 使用：`--moe-a2a-backend ncclep`。
- NCCL_EP mode 使用独立参数：`--ncclep-mode high_throughput | low_latency`，不要复用 `--deepep-mode`。
- HT/high_throughput 面向 prefill / large-token 场景。
- LL/low_latency 面向 decode / small-token 场景。
- dispatcher output dtype 使用：`--ncclep-dispatcher-output-dtype auto | bf16 | fp8`。
- 部署目标是不依赖已有 `/root/menyu/nccl` 本地源码树。可以在干净容器中直接 clone 最新 NCCL 源码、编译 NCCL + NCCL_EP、安装 `nccl4py`，然后运行 SGLang。

## 模型适配

NCCL_EP 只替换 MoE expert-parallel 的 dispatch/combine 通信，不替换 attention、router、MLP kernel、scheduler 或 KV cache。一个模型要能接入 NCCL_EP，需要同时满足：

1. 模型 MoE forward 能进入 SGLang 的 EP MoE 路径。
2. MoE runner 有对应的 NCCL_EP pre/post adapter。
3. dispatcher 输出的 dtype/layout 与 runner 输入契约匹配。
4. 模型 forward 中如果有 post-expert TP all-reduce、shared expert、empty-token rank 等特殊逻辑，需要与 NCCL_EP 的 empty rank 行为对齐。

### 已验证模型矩阵

| Model | Hardware / topology | Runner | NCCL_EP mode | Dtype | 当前状态 | 备注 |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash-FP8 / DSv4 Flash | H20 8 GPU / NVL | DeepGEMM | HT | FP8 | 可运行 | 推荐用于 prefill-like / large-token 验证；HT 与 DeepEP normal 仍有性能差距。 |
| DeepSeek-V4-Flash-FP8 / DSv4 Flash | H20 8 GPU / NVL | DeepGEMM | LL | FP8 | 可运行 | 推荐只用于 decode-like / small-token 验证；不要用 LL 代表 prefill 性能。 |
| DeepSeek-V4-Flash-FP8 / DSv4 Flash | H20 8 GPU / NVL | Triton | HT/LL | BF16 | 可运行 | 适合作为 BF16 通信正确性路径；不是 DSv4 FP8 性能主线。 |
| Mixtral-8x7B-Instruct-v0.1-FP8 | 8 GPU | Triton / patch-based repro | HT/LL | BF16/FP8 smoke | 可运行 | 已复现多卡 `/generate`；EP8 下每 rank 只有 1 个 expert，必须处理 empty-token rank。 |
| Qwen3.5-35B-A3B / Qwen3.5-35B-A3B-FP8 | 5k11 no-NVL / SM120 | Triton / Marlin fallback | LL | BF16 | 功能可跑，性能不佳 | no-NVL 下推荐 `NCCL_LSA_TEAM_SIZE=1`；HT 在 no-NVL 上未作为通过路径。 |
| Qwen3.5-35B-A3B-FP8 | 5k11 no-NVL / SM120 | DeepGEMM | HT/LL | FP8 | 不适用 | 当前 latest SGLang 在 SM120 上默认禁用 DeepGEMM，不能作为合法验证路径。 |
| 其他 MoE 模型 | 未验证 | 未验证 | 未验证 | 未验证 | 不保证 | 需要先确认模型 forward、runner adapter、dtype/layout。 |

### 模型适配注意点

- DSv4 Flash FP8 的主线是 `DeepGEMM + FP8 activation + activation scale`。
- Triton runner 需要 BF16 activation，不消费 FP8 activation scale。
- Qwen2-MoE / Qwen3.5 路径需要显式让 `is_ncclep()` 进入 EP MoE forward 语义；不能简单落回普通 TP MoE 路径，否则 empty rank 和 post-expert all-reduce 调用顺序可能不一致。
- no-NVL 拓扑下，LL 可以通过 `NCCL_LSA_TEAM_SIZE=1` 规避部分 HT/LSA/NVL 假设，但性能和稳定性仍需单独验证。
- PD disaggregation 的语义应保持：P 节点用 HT，D 节点用 LL。PD 会混入 NIXL/router/KV transfer，不适合作为 NCCL_EP dispatcher 的第一验证入口。

## MoE Runner 适配矩阵

| MoE runner | HT BF16 | HT FP8 | LL BF16 | LL FP8 | 当前结论 |
|---|---|---|---|---|---|
| `triton` | 可以 | 不建议 | 可以 | 不建议 | BF16 activation 路径可运行。 |
| `deep_gemm` | 不建议 | 可以 | 不建议 | 可以 | FP8 activation + scale 路径可运行，是 DSv4 Flash FP8 主线。 |
| `marlin` | 未作为主线 | 不适用 | 可以 | 不适用 | 5k11/DSv4 fallback 可跑，但性能差，只能算功能性路径。 |
| `triton_kernel` | 不可以 | 不可以 | 不可以 | 不可以 | DSv4 Flash FP8 权重/layout 不匹配。 |
| `flashinfer_trtllm` | 不可以 | 不可以 | 不可以 | 不可以 | 缺少 `ncclep` fused func。 |
| `flashinfer_trtllm_routed` | 不可以 | 不可以 | 不可以 | 不可以 | 同上。 |
| `flashinfer_cutlass` | 不可以 | 不可以 | 不可以 | 不可以 | 当前量化路径未创建 runner，也没有 NCCL_EP adapter。 |
| `flashinfer_mxfp4` | 不可以 | 不可以 | 不可以 | 不可以 | 当前 DSv4 Flash 测试走 FP8，不是 MXFP4 runner 路径。 |
| `flashinfer_cutedsl` | 不可以 | 不可以 | 不可以 | 不可以 | 只支持特定 FP4/modelopt 路径，且缺少 NCCL_EP fused func。 |
| `cutlass` | 不可以 | 不可以 | 不可以 | 不可以 | runner 配置断言失败。 |
| `aiter` | 不可以 | 不可以 | 不可以 | 不可以 | `Fp8MoEMethod` 没有 AITER 分支。 |

## 部署方式

### 推荐方式：干净环境直接下载并编译 NCCL/NCCL_EP

当前已有脚本 `scripts/build_nccl_ep.sh`，设计目标就是在新容器内自包含地准备 NCCL_EP，不依赖 host 上已有的 `/root/menyu/nccl`。

典型流程：

```bash
# 在带 nvcc/git/make/python/pip 的 SGLang dev 容器中
cd /workspace/sglang

# 从 NVIDIA/nccl 拉取并编译最新 NCCL + NCCL_EP
WORKDIR=/opt/ncclep \
NCCL_REF=master \
CUDA_ARCH=90 \
BUILD_JOBS=24 \
bash scripts/build_nccl_ep.sh

# 运行 SGLang 前加载环境
source /opt/ncclep/ncclep_env.sh
```

如果要固定 release/tag/commit：

```bash
WORKDIR=/opt/ncclep \
NCCL_REF=v2.30.7-1 \
CUDA_ARCH=90 \
bash scripts/build_nccl_ep.sh
```

如果是 SM120 / 5k11：

```bash
WORKDIR=/opt/ncclep \
NCCL_REF=master \
CUDA_ARCH=120 \
bash scripts/build_nccl_ep.sh
```

脚本会完成：

- clone 或复用 NCCL source。
- 编译 NCCL base library。
- 编译 `contrib/nccl_ep`。
- 将 `libnccl_ep.so`、`libnccl.so.2`、headers staging 到 `bindings/nccl4py/nccl/ep`。
- `pip install -e <nccl>/bindings/nccl4py`。
- 生成 `${WORKDIR}/ncclep_env.sh`。

生成的环境变量包括：

```bash
export NCCL_HOME=<nccl>/build
export LD_LIBRARY_PATH=<nccl>/build/lib:${LD_LIBRARY_PATH}
export LD_PRELOAD=<nccl>/build/lib/libnccl.so.2:${LD_PRELOAD}
export PYTHONPATH=<nccl>/bindings/nccl4py:${PYTHONPATH}
export SGLANG_NCCL_EP_SO_PATH=<nccl>/build/lib/libnccl_ep.so
export NCCL_EP_JIT_BUILD_INCLUDE_DIR=<nccl>/build/include
export NCCL_EP_JIT_SOURCE_DIR=<nccl>/bindings/nccl4py/nccl/ep/include
export NCCL_CUMEM_ENABLE=1
export NCCL_WIN_ENABLE=1
```

结论：部署上可以不再依赖本地固定 repo。只要 NCCL upstream API 与 SGLang adapter 当前绑定兼容，就可以直接下载最新 NCCL 编译使用。若 upstream API 变化，需要同步更新 Python binding / adapter 并重新跑 correctness。

### Debug 方式：复用已有 NCCL 源码树

如果要复用已有 dirty NCCL 源码树，避免脚本 checkout 覆盖：

```bash
WORKDIR=/opt/ncclep \
NCCL_SRC=/root/menyu/nccl \
NCCL_REF=local \
CUDA_ARCH=90 \
bash scripts/build_nccl_ep.sh

source /opt/ncclep/ncclep_env.sh
```

这种方式只建议用于调试 NCCL_EP native patch，不建议作为可复现部署文档的默认路径。

### 一键干净容器 bootstrap

`scripts/bootstrap_clean_ncclep_container.sh` 进一步封装了：

1. 编译 NCCL/NCCL_EP。
2. clone `MengYu10151/sglang` 的 `ncclep-integration` 分支。
3. editable install SGLang。
4. import check。
5. 跑 NCCL_EP boundary cases。

示例：

```bash
WORKDIR=/opt/ncclep \
NCCL_REF=master \
CUDA_ARCH=90 \
bash scripts/bootstrap_clean_ncclep_container.sh
```

这个脚本适合验证“新机器从零准备是否可运行”。

## SGLang 启动模板

### DSv4 Flash + DeepGEMM + HT

```bash
source /opt/ncclep/ncclep_env.sh

export SGLANG_NCCL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8192

python -m sglang.launch_server \
  --model-path /models/DeepSeek-V4-Flash-FP8 \
  --trust-remote-code \
  --tp-size 8 \
  --ep-size 8 \
  --moe-a2a-backend ncclep \
  --ncclep-mode high_throughput \
  --ncclep-dispatcher-output-dtype fp8 \
  --moe-runner-backend deep_gemm \
  --kv-cache-dtype fp8_e4m3 \
  --disable-cuda-graph
```

### DSv4 Flash + DeepGEMM + LL

```bash
source /opt/ncclep/ncclep_env.sh

export SGLANG_NCCL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128

python -m sglang.launch_server \
  --model-path /models/DeepSeek-V4-Flash-FP8 \
  --trust-remote-code \
  --tp-size 8 \
  --ep-size 8 \
  --moe-a2a-backend ncclep \
  --ncclep-mode low_latency \
  --ncclep-dispatcher-output-dtype fp8 \
  --moe-runner-backend deep_gemm \
  --kv-cache-dtype fp8_e4m3 \
  --disable-cuda-graph
```

### Triton BF16 runner

```bash
source /opt/ncclep/ncclep_env.sh

python -m sglang.launch_server \
  --model-path <model> \
  --trust-remote-code \
  --tp-size 8 \
  --ep-size 8 \
  --moe-a2a-backend ncclep \
  --ncclep-mode high_throughput \
  --ncclep-dispatcher-output-dtype bf16 \
  --moe-runner-backend triton \
  --disable-cuda-graph
```

### no-NVL / 5k11 LL

```bash
source /opt/ncclep/ncclep_env.sh

export NCCL_LSA_TEAM_SIZE=1
export NCCL_NET_MERGE_LEVEL=LOC
export NCCL_NVLS_ENABLE=0
export SGLANG_NCCL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024

python -m sglang.launch_server \
  --model-path /root/menyu/models/Qwen3.5-35B-A3B-FP8 \
  --trust-remote-code \
  --dp-size 4 \
  --ep-size 4 \
  --enable-dp-attention \
  --moe-a2a-backend ncclep \
  --ncclep-mode low_latency \
  --ncclep-dispatcher-output-dtype bf16 \
  --moe-runner-backend triton \
  --disable-cuda-graph
```

## 验证顺序

推荐按下面顺序验证，避免一上来用真实 serving 混入模型、runner、scheduler、KV、router 等变量。

1. Import check：

```bash
source /opt/ncclep/ncclep_env.sh
python - <<'PY'
import os
import nccl
import nccl.ep
print("nccl =", nccl.__file__)
print("nccl.ep =", nccl.ep.__file__)
print("SGLANG_NCCL_EP_SO_PATH =", os.environ.get("SGLANG_NCCL_EP_SO_PATH"))
PY
```

2. Native NCCL_EP benchmark：

```bash
# 示例，具体 binary 路径以 NCCL build 输出为准
<nccl>/build/test/nccl_ep/ep_bench \
  --algorithm ht \
  --hidden 4096 \
  --top-k 6 \
  --experts 256 \
  --tokens 8192 \
  --validate

<nccl>/build/test/nccl_ep/ep_bench \
  --algorithm ll \
  --layout em \
  --hidden 4096 \
  --top-k 6 \
  --experts 256 \
  --tokens 128 \
  --validate
```

3. SGLang synthetic communication boundary：

```bash
source /opt/ncclep/ncclep_env.sh
bash scripts/run_ncclep_boundary_cases.sh
```

4. MoE runner smoke：

- Triton: `--ncclep-dispatcher-output-dtype bf16`
- DeepGEMM: `--ncclep-dispatcher-output-dtype fp8`

5. Server `/generate` correctness：

- 固定 deterministic prompt。
- 使用 chat template / OpenAI chat endpoint，避免 raw generate 造成模型续写异常。
- HT/LL 分开测，不混用 prefill/decode 语义。

6. Serving benchmark：

- HT: `ISL=1024/8192, OSL=1`，prefill-like。
- LL: `ISL=1, OSL=1024`，decode-like。
- 对比 DeepEP 时保持同模型、同 runner、同 dtype、同 CUDA graph/TBO/SBO 开关。

## 已知限制

- HT 与 DeepEP normal 在 DSv4 Flash + DeepGEMM + DP attention 下仍有性能差距，主要集中在 EP dispatch/combine 路径，不是 MLP 或 attention 本身。
- LL 不应作为 prefill 大 token 的性能结论。LL 在 prefill 大 token 下容易放大 expert-major buffer、per-layer routed hotspot 和 no-NVL combine slow path。
- LL + CUDA graph 尚未作为正式支持路径验证，建议启动期 fail-fast 或关闭 CUDA graph。
- no-NVL 拓扑下 HT 未作为可用路径确认；LL 需要 `NCCL_LSA_TEAM_SIZE=1` 等配置，并且性能需单独评估。
- 使用 NCCL upstream 最新 master 时，API/headers/binding 可能变化。每次更新 NCCL 后都必须重新编译 `nccl4py` 并跑 native + SGLang boundary correctness。
- 当前 NCCL_EP 安装产品化还不是 pip wheel 级别；脚本能从源码自举，但 PR 级部署仍需要进一步收敛版本锁定和 CI 验证。

## 推荐 PR 表述边界

可以表述：

- NCCL_EP backend 已独立于 DeepEP backend。
- 支持 `high_throughput` 和 `low_latency` 两种 NCCL_EP mode。
- 支持 Triton BF16 和 DeepGEMM FP8 两类 runner 适配。
- 支持从干净容器下载、编译、安装 NCCL/NCCL_EP，不强依赖本地 `/root/menyu/nccl`。

暂不建议表述：

- NCCL_EP 全面优于 DeepEP。
- LL 适合作为 prefill 性能路径。
- FlashInfer/Cutlass/AITER 等 runner 已支持。
- no-NVL 拓扑下 HT 已稳定可用。
