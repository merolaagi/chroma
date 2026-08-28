"""Aggregate E1 arm checkpoints into the three-arm comparison table."""
import glob, json, statistics as st
from collections import defaultdict

rows = defaultdict(list)
for f in sorted(glob.glob('results/e1_*.json')):
    if f.endswith('_smoke.json'):
        continue   # 2k-step smoke runs are not sweep results

    d = json.load(open(f))
    rows[d['arm']].append(d)

ARMS = ['chroma', 'ablate_hysteresis', 'ablate_grn']
LABEL = {'chroma': 'CHROMA (recurrence + basins)',
         'ablate_hysteresis': 'flat landscape (recurrence only)',
         'ablate_grn': 'MLP (neither)'}
def agg(rs, k):
    v = [r[k] for r in rs]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)

print(f"{'arm':34s} {'n':>2s} {'switch rate':>18s} {'mean err':>16s} {'basins':>7s}")
base = None
for a in ARMS:
    rs = rows.get(a)
    if not rs: continue
    sm, ss = agg(rs, 'switch_rate')
    errs = [st.mean([r[f'err_regime{i}'] for i in range(3)]) for r in rs]
    em, es = st.mean(errs), (st.stdev(errs) if len(errs) > 1 else 0.0)
    nb = st.mean([r.get('n_basins', 1) for r in rs])
    if a == 'chroma': base = sm
    print(f"{LABEL[a]:34s} {len(rs):2d} {sm:8.5f} +/-{ss:7.5f} "
          f"{em:8.5f} +/-{es:6.5f} {nb:7.1f}")

print()
for a in ARMS[1:]:
    if a in rows:
        r = agg(rows[a], 'switch_rate')[0] / max(base, 1e-9)
        print(f"switch-rate ratio  {a:20s} / chroma = {r:6.2f}x")
print()
print("Pre-registered threshold: ratio > 5x for the regulatory layer to earn "
      "its keep.\nIf ablate_hysteresis ~= chroma, the credit belongs to slow "
      "integration, not\nto multistability, and Section 4.1 overclaims.")
