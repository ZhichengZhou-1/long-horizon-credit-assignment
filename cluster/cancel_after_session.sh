#!/bin/bash
# Run with nohup on the login node that launched the session's srun: once that srun exits
# (session finished or crashed), wait a minute for files and mail, then scancel the job.
# Usage: nohup setsid bash cancel_after_session.sh JOBID SESSION_SCRIPT_NAME < /dev/null > cancel.log 2>&1 &
J=$1; NAME=$2
PATTERN="srun --jobid=$J --overlap bash .*$NAME"
echo "$(date '+%F %T') on $(hostname): will scancel $J after '$PATTERN' exits"
pgrep -u "$USER" -f "$PATTERN" > /dev/null || { echo "no matching srun on this node, not cancelling"; exit 1; }
while pgrep -u "$USER" -f "$PATTERN" > /dev/null; do sleep 60; done
echo "$(date '+%F %T') session srun exited; cancelling $J in 60 s"
sleep 60
scancel "$J" && echo "$(date '+%F %T') scancel $J done"
