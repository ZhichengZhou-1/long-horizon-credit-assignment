#!/bin/bash
# Run with nohup on the login node that launched the session's srun: when that srun exits,
# build the Phase 1 premise-check report (premise_check.py, also on partial data) and mail it.
# Usage: nohup setsid bash report_after_session.sh JOBID SESSION_SCRIPT_NAME RUN_DIR SOURCE_EPISODES TO < /dev/null > report.log 2>&1 &
J=$1; NAME=$2; RUN=$3; SOURCE=$4; TO=$5
HERE=$(cd "$(dirname "$0")" && pwd)
PATTERN="srun --jobid=$J --overlap bash .*$NAME"
echo "$(date '+%F %T') on $(hostname): report after '$PATTERN' exits"
pgrep -u "$USER" -f "$PATTERN" > /dev/null || { echo "no matching srun on this node"; exit 1; }
while pgrep -u "$USER" -f "$PATTERN" > /dev/null; do sleep 60; done
sleep 30
/usr/bin/python3 "$HERE/../analysis/premise_check.py" "$RUN" "$SOURCE" > "$RUN/premise_report.txt" 2>&1
echo "$(date '+%F %T') report written: $RUN/premise_report.txt"
mail -s "[credit] Phase 1 premise check report (job $J)" "$TO" < "$RUN/premise_report.txt" \
    && echo "$(date '+%F %T') mailed to $TO"
