#!/usr/bin/env bash
# One command per iteration: verify, regenerate docs from live results, commit,
# push. Refuses to ship if the proposition tests fail.
#
#   scripts/ship.sh "message"          verify + docs + commit + push
#   scripts/ship.sh -n "message"       everything except the push
#   scripts/ship.sh -q "message"       skip the test gate (not recommended)
set -euo pipefail
cd "$(dirname "$0")/.."

PUSH=1; GATE=1
while getopts "nq" o; do case $o in n) PUSH=0;; q) GATE=0;; esac; done
shift $((OPTIND-1))
MSG="${1:-iterate}"

say() { printf '\033[1m==> %s\033[0m\n' "$1"; }

if [ "$GATE" = 1 ]; then
  say "proposition tests"
  if ! PYTHONPATH=. python tests/test_propositions.py | tee /tmp/chroma_tests.txt; then
    echo "test run errored - not shipping"; exit 1
  fi
  grep -q "ALL PASS" /tmp/chroma_tests.txt || {
    echo
    echo "A proposition test FAILED. That means a claim in the README or the"
    echo "explainer PDF is now false. Fix the claim or the code before shipping."
    exit 1; }
fi

say "aggregate experiment results"
python aggregate.py | tee results/SUMMARY.txt || true

say "regenerate figures and explainer PDF"
python docs/make_figures.py
python docs/build_pdf.py

if git diff --quiet && git diff --cached --quiet && [ -z "$(git status --porcelain)" ]; then
  say "nothing changed"; exit 0
fi

say "commit"
git add -A
git commit -q -m "$MSG" || { say "nothing to commit"; exit 0; }
git --no-pager log --oneline -1

if [ "$PUSH" = 1 ]; then
  say "push"
  git push origin "$(git rev-parse --abbrev-ref HEAD)"
else
  say "skipped push (-n)"
fi
