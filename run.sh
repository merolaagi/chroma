#!/usr/bin/env bash
# CHROMA Lab: set up if needed, then serve.
#   ./run.sh              start on the default port
#   ./run.sh 51848        start on another port
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-51847}"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -c "import torch" 2>/dev/null || pip install -q torch numpy
pkill -f "webapp/server.py $PORT" 2>/dev/null || true
echo "CHROMA Lab  ->  http://localhost:$PORT"
exec python webapp/server.py "$PORT"
