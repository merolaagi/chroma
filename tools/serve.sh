#!/usr/bin/env bash
# Run CHROMA Lab as a background service. Idempotent: safe to re-run.
#   tools/serve.sh start | stop | restart | status | log
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${CHROMA_PORT:-51847}"
PID=.lab.pid
mkdir -p logs

start() {
  if [ -f $PID ] && kill -0 "$(cat $PID)" 2>/dev/null; then
    echo "already running (pid $(cat $PID)) on :$PORT"; return 0
  fi
  [ -d .venv ] || python3 -m venv .venv
  source .venv/bin/activate
  python -c "import torch" 2>/dev/null || pip install -q torch numpy
  nohup python webapp/server.py "$PORT" >> logs/lab.log 2>&1 &
  echo $! > $PID
  sleep 2
  kill -0 "$(cat $PID)" 2>/dev/null \
    && echo "started pid $(cat $PID)  ->  http://localhost:$PORT" \
    || { echo "failed to start; see logs/lab.log"; tail -20 logs/lab.log; exit 1; }
}
stop() {
  [ -f $PID ] && kill "$(cat $PID)" 2>/dev/null && echo "stopped $(cat $PID)" || echo "not running"
  rm -f $PID
  pkill -f "webapp/server.py $PORT" 2>/dev/null || true
}
case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  restart) stop; sleep 1; start ;;
  status) [ -f $PID ] && kill -0 "$(cat $PID)" 2>/dev/null \
            && echo "running pid $(cat $PID) on :$PORT" || echo "not running" ;;
  log) tail -f logs/lab.log ;;
  *) echo "usage: tools/serve.sh {start|stop|restart|status|log}"; exit 1 ;;
esac
