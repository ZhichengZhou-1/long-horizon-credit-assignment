# Source after local_envs.sh. start_server MODEL LOGFILE [GPU_MEM_UTIL] / stop_server.
# Sampling defaults: --generation-config vllm, so the model's generation_config.json (Qwen2.5:
# repetition_penalty 1.05, top_k 20) is NOT applied; that silently changed outputs in our first run.
# Use GPU_MEM_UTIL ~0.6 when requesting prompt_logprobs (hcapo_score.py): full-vocabulary log-probs
# for every prompt token ran out of memory at 0.90.
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_URL=http://127.0.0.1:$VLLM_PORT/v1

start_server() {
    local model=$1 log=$2 util=${3:-0.90}
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server --model "$model" --served-model-name "$model" \
        --host 127.0.0.1 --port "$VLLM_PORT" --max-model-len 8192 --gpu-memory-utilization "$util" \
        --seed 0 --generation-config vllm --disable-log-requests --dtype "${DTYPE:-auto}" > "$log" 2>&1 &
    VPID=$!
    for _ in $(seq 1 $(( ${WAIT_MIN:-15} * 12 ))); do
        curl -sf "http://127.0.0.1:$VLLM_PORT/health" > /dev/null && { echo "$(date '+%F %T') vLLM up: $model (util $util)"; return 0; }
        kill -0 $VPID 2>/dev/null || break
        sleep 5
    done
    echo "$(date '+%F %T') vLLM failed to start: $model"; tail -20 "$log"; return 1
}

stop_server() {
    kill $VPID 2>/dev/null; wait $VPID 2>/dev/null; sleep 5
}
