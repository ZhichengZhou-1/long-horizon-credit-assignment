# Source inside a GPU job to run from node-local copies of the two conda envs.
# Importing torch/vllm/pyserini straight from a network file system (BeeGFS on Zaratan) took
# 20-35 min on compute nodes; from local disk `import vllm` takes ~6 s. Unpacking the two
# tarballs takes ~20-60 s per env. Build the tarballs once on a login node:
#   tar -C $CREDIT_ROOT/envs -cf $CREDIT_ROOT/envs/vllm-env.tar vllm
#   tar -C $CREDIT_ROOT/envs -cf $CREDIT_ROOT/envs/webshop-env.tar webshop
# Defines VLLM_PY and WS_PY and sets the WebShop runtime variables.
: "${CREDIT_ROOT:?set CREDIT_ROOT (see env.example.sh)}"
LOCAL_ENVS=${CREDIT_LOCAL_ENVS:-/tmp/$USER/envs}
mkdir -p "$LOCAL_ENVS"
for name in vllm webshop; do
    if [ ! -x "$LOCAL_ENVS/$name/bin/python" ]; then
        echo "$(date +%T) unpacking $name env to $LOCAL_ENVS"
        tar -C "$LOCAL_ENVS" -xf "$CREDIT_ROOT/envs/$name-env.tar"
    fi
done
VLLM_PY=$LOCAL_ENVS/vllm/bin/python
WS_PY=$LOCAL_ENVS/webshop/bin/python

# WebShop runtime (conda's openjdk normally exports JAVA_HOME from its activate.d script).
export JAVA_HOME=$LOCAL_ENVS/webshop
# Keep each process small: login nodes cap a user at 256 threads and the JVM / torch otherwise
# size their thread pools to all cores.
export JAVA_TOOL_OPTIONS="-XX:ActiveProcessorCount=1 -XX:+UseSerialGC"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$CREDIT_ROOT/.cache/huggingface} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=/tmp/$USER/cache VLLM_CACHE_ROOT=/tmp/$USER/cache/vllm TRITON_CACHE_DIR=/tmp/$USER/cache/triton
