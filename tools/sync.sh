#!/usr/bin/env bash
# Apply a downloaded CHROMA archive, commit, push, restart the lab.
#   tools/sync.sh ~/Downloads/chroma-YYYY-MM-DD-vNN.tar.gz
# Commit message is the first "## " heading in CHANGES.md.
set -euo pipefail
cd "$(dirname "$0")/.."
ARCHIVE="${1:?usage: tools/sync.sh <archive.tar.gz>}"
[ -f "$ARCHIVE" ] || { echo "no such archive: $ARCHIVE"; exit 1; }

TMP=$(mktemp -d)
tar xzf "$ARCHIVE" -C "$TMP"
SRC="$TMP/chroma"
[ -d "$SRC" ] || SRC=$(find "$TMP" -maxdepth 1 -mindepth 1 -type d | head -1)

# never let the archive's git history or local state clobber ours
rsync -a --exclude '.git' --exclude '.venv' --exclude 'logs' \
      --exclude 'results' "$SRC"/ .
rsync -a --ignore-existing "$SRC"/results/ results/ 2>/dev/null || true
rm -rf "$TMP"
chmod +x run.sh tools/*.sh scripts/*.sh 2>/dev/null || true

MSG=$(grep -m1 '^## ' CHANGES.md | sed 's/^## //')
git add -A
git diff --cached --quiet && { echo "nothing changed"; } || {
  git commit -q -m "$MSG"
  echo "committed: $MSG"
}
git push origin "$(git rev-parse --abbrev-ref HEAD)"
tools/serve.sh restart
