#!/usr/bin/env bash
# Train every scenario one after another, headless, logging each to logs/<task>.out.
#
#   nohup bash scripts/train_all.sh > logs/train_all.out 2>&1 &
#   tail -f logs/train_all.out
#
# Pass task names to run only some of them, e.g.:
#   bash scripts/train_all.sh hopper ant_constant_speed
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-$HOME/workspaces/.venv/bin/python}"
if [[ $# -gt 0 ]]; then
    TASKS=("$@")
else
    TASKS=(inverted_pendulum inverted_double_pendulum hopper ant_constant_speed)
fi

mkdir -p logs
for task in "${TASKS[@]}"; do
    echo "=== [$(date '+%F %T')] training $task  (log: logs/$task.out)"
    start=$SECONDS
    "$PY" "scripts/train_$task.py" > "logs/$task.out" 2>&1
    echo "=== [$(date '+%F %T')] finished $task in $(( (SECONDS - start) / 60 )) min"
    grep -E "Episode 1:|Saved policy|Saved video" "logs/$task.out" || true
done
echo "=== all done"
