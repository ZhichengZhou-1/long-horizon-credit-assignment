#!/bin/bash
# Run on a login node with nohup: wait until a pending GPU hold job starts, then run a session
# script inside it right away (srun --overlap), so a start at 3 am does not leave the GPU idle.
# When the session ends the job is cancelled to release the GPU, unless the 4th argument is
# "keep" (then the job stays allocated for more work until its time limit).
# Usage: nohup setsid bash watch_and_run.sh JOBID SESSION_SCRIPT LOG [keep] < /dev/null > LOG.watch 2>&1 &
J=$1; SCRIPT=$2; LOG=$3; KEEP=${4:-}
echo "$(date '+%F %T') watching job $J${KEEP:+ (keep the job after the session)}"
while true; do
    st=$(squeue -h -j "$J" -o %T 2>/dev/null)
    [ -z "$st" ] && { echo "$(date '+%F %T') job $J no longer in the queue, stopping"; exit 0; }
    [ "$st" = RUNNING ] && break
    sleep 60
done
echo "$(date '+%F %T') job $J is running, launching $SCRIPT (log: $LOG)"
srun --jobid="$J" --overlap bash "$SCRIPT" > "$LOG" 2>&1
rc=$?
if [ "$KEEP" = keep ]; then
    echo "$(date '+%F %T') session finished (exit $rc), keeping job $J"
else
    echo "$(date '+%F %T') session finished (exit $rc), cancelling $J to release the GPU"
    scancel "$J"
fi
