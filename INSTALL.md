# Installation

Tested on Linux x86_64 (RHEL 8, glibc 2.28), October 2026. Two conda envs: `webshop` (CPU side:
WebShop + our scripts) and `vllm` (GPU side: model server).

## 0. Requirements

| | |
|---|---|
| WebShop env | Linux x86_64, Java via conda |
| vLLM | NVIDIA GPU, driver for CUDA 12.4 (>= 550 recommended); 7B bf16 needs ~16 GB + KV cache (A100 40 GB is fine) |
| disk | ~4 GB webshop env, ~8 GB vllm env, ~18 GB models (Qwen2.5-1.5B/7B), ~15 GB GiGPO-7B |

## 1. Code

```bash
git clone https://github.com/langfengQ/verl-agent.git
cd verl-agent && git checkout 20bd331bdbc9026a5668e11362178e10ab7400c8
export CREDIT_VERL_AGENT_DIR=$PWD
WS=$CREDIT_VERL_AGENT_DIR/agent_system/environments/env_package/webshop/webshop
```

## 2. `webshop` env (Python 3.10)

```bash
conda create -y -p $CREDIT_ROOT/envs/webshop python=3.10
conda activate $CREDIT_ROOT/envs/webshop
cd $WS && pip install -r requirements.txt
conda install -y -c conda-forge mkl faiss-cpu openjdk=11
conda install -y -c conda-forge numpy=1.26.4   # faiss pulls numpy 2.x over pip's 1.26.4; spaCy breaks
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

Do not run WebShop's `setup.sh` as is: its Google Drive data links return 404 and `spacy download`
cannot resolve a version. Only `en_core_web_sm` is used by the env.

## 3. Data and search index

```bash
cd $WS/data
for f in items_shuffle_1000.json items_ins_v2_1000.json items_human_ins.json; do
  curl -L -o $f https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/$f
done
git hash-object items_shuffle_1000.json items_ins_v2_1000.json items_human_ins.json
# 08dfda9a1c896e0f69d50e3437dd3fe94419b7d7
# 22837ad99f3173712fc9cb1cce7de6b75fe90595
# b2ab7479b9798e0b62ea394e065f3d506875f090
# (identical files in HongbangYuan/webshop and Skyler215/VERL_WEBSHOP)
cd $WS/search_engine
mkdir -p resources resources_100 resources_1k resources_100k indexes
python convert_product_file_format.py && bash run_indexing.sh
```

The env then loads 1,000 products and 6,910 synthetic goals (`human_goals=False`), as verl-agent does.

## 4. `vllm` env (Python 3.10)

```bash
conda create -y -p $CREDIT_ROOT/envs/vllm python=3.10
conda activate $CREDIT_ROOT/envs/vllm
pip install uv
uv pip install --no-build vllm==0.8.5.post1      # torch 2.6.0+cu124, wheels only
uv pip install --no-build transformers==4.51.3   # vllm 0.8.5 has no upper bound; 5.x gets picked otherwise
```

- `vllm==0.8.2` (verl-agent's pin) is no longer installable: its `xgrammar==0.1.16` was removed from PyPI.
- Recent vLLM releases need a source build of `llguidance` (Rust) on glibc 2.28.

## 5. Models

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
huggingface-cli download Qwen/Qwen2.5-7B-Instruct  --revision a09a35458c702b33eeacc393d103063234e8bc28
huggingface-cli download langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop --revision 8e58a29cdbad307671427340b1fe06a15ce05eeb
```

## 6. Node-local copies of the envs (network file systems)

On a network file system with slow small-file access (BeeGFS on Zaratan) importing torch/vllm took
20-35 min on compute nodes. Pack the envs once and let `cluster/local_envs.sh` unpack them to the
node's `/tmp` inside each job (~1 min, then `import vllm` takes ~6 s):

```bash
tar -C $CREDIT_ROOT/envs -cf $CREDIT_ROOT/envs/vllm-env.tar vllm
tar -C $CREDIT_ROOT/envs -cf $CREDIT_ROOT/envs/webshop-env.tar webshop
```

## 7. Pitfalls we hit

- **vLLM sampling defaults**: vLLM fills unset sampling parameters from the model's
  `generation_config.json` (Qwen2.5: `repetition_penalty=1.05`, `top_k=20`). That made base 7B write
  `[action>` and 1.5B reason in Chinese. `cluster/vllm_server.sh` passes `--generation-config vllm`
  and `credit/rollout.py` sends `top_k=-1, repetition_penalty=1.0` (verl-agent's rollout settings).
- **`prompt_logprobs` memory**: full-vocabulary log-probs per prompt token OOM'd vLLM at 0.90 GPU
  memory utilization with 32 concurrent requests; score at ~0.60 with 2 requests.
- **Thread limits**: login nodes may cap threads per user (256 on Zaratan); the JVM and torch size
  their pools to all cores. `cluster/local_envs.sh` sets `JAVA_TOOL_OPTIONS` and `OMP_NUM_THREADS=1`.
- **gym**: build `WebAgentTextEnv` directly; gym 0.24's `gym.make` calls `reset`/`step` internally.
- **Goal order** depends on the env seed (SimServer shuffles goals); we fix seed 0 everywhere.
