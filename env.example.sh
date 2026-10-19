# Copy to env.sh (git-ignored), edit, and `source env.sh` before running anything.

# Checkout of https://github.com/langfengQ/verl-agent (we used commit 20bd331). Its vendored WebShop
# is the environment, and the prompt templates / action projection are loaded from it.
export CREDIT_VERL_AGENT_DIR=/path/to/verl-agent
# Optional: a different WebShop install (default: the copy inside verl-agent).
# export CREDIT_WEBSHOP_DIR=$CREDIT_VERL_AGENT_DIR/agent_system/environments/env_package/webshop/webshop

# Large-storage root: conda env tarballs (envs/vllm-env.tar, envs/webshop-env.tar), HF cache, Slurm logs.
export CREDIT_ROOT=/path/to/scratch/credit
# Where experiment outputs are written.
export CREDIT_RUNS=$CREDIT_ROOT/runs
export HF_HOME=$CREDIT_ROOT/.cache/huggingface

# Slurm (cluster/submit_hold.sh); leave CREDIT_NOTIFY_EMAIL empty to skip mails.
export CREDIT_SLURM_ACCOUNT=your-allocation
export CREDIT_SLURM_PARTITION=gpu
export CREDIT_NOTIFY_EMAIL=

# Python of the WebShop env for CPU-side scripts (checks/, credit/goal_ceiling.py). Activating the
# env also sets JAVA_HOME, which pyserini needs.
# conda activate $CREDIT_ROOT/envs/webshop
export PY=python

# On machines that cap threads per user (Zaratan login nodes: 256), keep the JVM / torch pools small.
export JAVA_TOOL_OPTIONS="-XX:ActiveProcessorCount=1 -XX:+UseSerialGC"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
