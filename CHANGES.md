# Changes

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
