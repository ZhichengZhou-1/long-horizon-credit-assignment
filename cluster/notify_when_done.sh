#!/bin/bash
# Run on a login node with nohup: email when a watched session ends (its log prints "===== ... done",
# the watcher reports "session finished", or the job leaves the queue first). The mail carries the
# summary lines of the session log. Slurm's own BEGIN/END mails do not cover this when the job is
# kept allocated after the session.
# Usage: nohup setsid bash notify_when_done.sh JOBID SESSION_LOG WATCH_LOG TO SUBJECT < /dev/null > notify.log 2>&1 &
J=$1; LOG=$2; WATCH=$3; TO=$4; SUBJ=$5
echo "$(date '+%F %T') notifier for job $J -> $TO"
while true; do
    if grep -q "===== .* done" "$LOG" 2>/dev/null; then status="finished"; break; fi
    if grep -q "session finished" "$WATCH" 2>/dev/null; then status="session ended early: $(grep 'session finished' "$WATCH" | tail -1)"; break; fi
    if [ -z "$(squeue -h -j "$J" -o %T 2>/dev/null)" ]; then status="job $J left the queue before the session finished"; break; fi
    sleep 60
done
{
    echo "$SUBJ"
    echo "status: $status"
    echo "job: $J    session log: $LOG"
    echo
    grep -E "^=====|success_rate|mean_score|invalid_action_rate|wall_s|verify|FAILED|giving up|failed to start" "$LOG" 2>/dev/null | tail -40
} | mail -s "[credit] $SUBJ: $status" "$TO"
echo "$(date '+%F %T') mail sent: $status"
