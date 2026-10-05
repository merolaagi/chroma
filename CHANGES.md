# Changes

## v19 — E10 bisection, and a prediction that failed
- `experiments/e10_bisect.py`: eight-way bisection of the gap between CHROMA and
  the Bayes-optimal observer of E9
- On an idealised recogniser the hypothesis grid dominates (+0.500 against
  +0.200 each for representation and likelihood)
- **It does not transfer.** End to end the grid change moves nothing: 0.250 at
  extent 2, 0.250 at extent 4. The harness built templates from ground truth,
  which removed memory — the very thing that limits the real pipeline
- `grid_extent` defaulted to 4 anyway: correct, just not the fix
- Server answers HEAD, so `curl -I` no longer returns 501

## v18 — public serving at chroma.fueldeskpro.com
- Cloudflare tunnel ingress documented; `tools/serve.sh` runs the lab as a
  background service with a pidfile so restarts are idempotent
- `tools/sync.sh`: one command to apply a downloaded archive, commit from the
  top CHANGES entry, push to GitHub, and restart the server
- Server honours `CHROMA_PORT` and logs to `logs/lab.log`

## v17 — CHROMA Lab
- Interactive web demonstration on port 51847; eight live panels including the
  gene/regulatory manipulation surface

## v16 — E9 ceiling
- Bayes-optimal observer reaches 1.000 in three touches; CHROMA sits at 0.287
  in twelve. Recognition is a model failure, not a task limit.
