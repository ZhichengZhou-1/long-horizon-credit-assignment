#!/bin/bash
# Submit a GPU "hold" job: it only sleeps, so work is attached with srun --jobid=<ID> --overlap
# (by hand, or automatically with watch_and_run.sh). Account, mail and log dir come from env.sh.
# Usage: bash submit_hold.sh GPU_SPEC TIME [extra sbatch args]
#   e.g. bash submit_hold.sh a100:1 04:00:00 --begin=2026-10-20T14:00:00
set -eu
: "${CREDIT_SLURM_ACCOUNT:?}" "${CREDIT_ROOT:?}"
GPUS=$1; TIME=$2; shift 2
MAIL=()
[ -n "${CREDIT_NOTIFY_EMAIL:-}" ] && MAIL=(--mail-user="$CREDIT_NOTIFY_EMAIL" --mail-type=BEGIN,END,FAIL)
mkdir -p "$CREDIT_ROOT/slurm"
sbatch --parsable -A "$CREDIT_SLURM_ACCOUNT" -p "${CREDIT_SLURM_PARTITION:-gpu}" --gpus="$GPUS" \
    --cpus-per-task="${CREDIT_CPUS:-16}" --mem="${CREDIT_MEM:-96G}" -t "$TIME" -J credit-hold \
    -o "$CREDIT_ROOT/slurm/%x-%j.out" "${MAIL[@]}" "$@" \
    --wrap 'hostname; nvidia-smi --query-gpu=name,memory.total --format=csv; echo "allocation ready $(date)"; sleep infinity'
