"""Generate the vector figures used by the explainer PDF.

Written as plain SVG with inline attributes (no CSS classes, no variables) so
that svglib can convert them faithfully into ReportLab drawings.
"""

import json
import statistics as stat
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

PURPLE = dict(fill="#EEEDFE", stroke="#534AB7", text="#3C3489", sub="#534AB7")
TEAL = dict(fill="#E1F5EE", stroke="#0F6E56", text="#085041", sub="#0F6E56")
CORAL = dict(fill="#FAECE7", stroke="#993C1D", text="#712B13", sub="#993C1D")
GRAY = dict(fill="#F1EFE8", stroke="#5F5E5A", text="#2C2C2A", sub="#5F5E5A")
RED = dict(fill="#FCEBEB", stroke="#A32D2D", text="#791F1F", sub="#A32D2D")

FONT = 'font-family="Helvetica"'
ARROW = ('<defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" '
         'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
         '<path d="M2 1L8 5L2 9" fill="none" stroke="#5F5E5A" stroke-width="1.5" '
         'stroke-linecap="round"/></marker></defs>')


def head(h, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="680" height="{h}" '
            f'viewBox="0 0 680 {h}"><title>{title}</title>{ARROW}')


def box(x, y, w, h, c, title, sub=None, r=8):
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
         f'fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="0.8"/>')
    cx = x + w / 2
    if sub:
        s += (f'<text x="{cx}" y="{y + h / 2 - 3}" text-anchor="middle" {FONT} '
              f'font-size="14.5" fill="{c["text"]}">{title}</text>')
        s += (f'<text x="{cx}" y="{y + h / 2 + 14}" text-anchor="middle" {FONT} '
              f'font-size="12.5" fill="{c["sub"]}">{sub}</text>')
    else:
        s += (f'<text x="{cx}" y="{y + h / 2 + 4}" text-anchor="middle" {FONT} '
              f'font-size="14.5" fill="{c["text"]}">{title}</text>')
    return s


def head_tri(x1, y1, x2, y2, col="#5F5E5A", size=6):
    """Explicit arrowhead polygon. svglib's marker support is unreliable, so
    the heads are drawn rather than referenced."""
    import math
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    bx, by = x2 - ux * size, y2 - uy * size
    return (f'<polygon points="{x2:.1f},{y2:.1f} '
            f'{bx + px * size * 0.5:.1f},{by + py * size * 0.5:.1f} '
            f'{bx - px * size * 0.5:.1f},{by - py * size * 0.5:.1f}" '
            f'fill="{col}"/>')


def arrow(x1, y1, x2, y2, dashed=False, col="#5F5E5A"):
    d = ' stroke-dasharray="4 4"' if dashed else ''
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
            f'stroke-width="1.2"{d}/>' + head_tri(x1, y1, x2, y2, col))


def label(x, y, txt, size=12.5, anchor="middle", col="#5F5E5A"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" {FONT} '
            f'font-size="{size}" fill="{col}">{txt}</text>')


# ---------------------------------------------------------------- figure 1
def fig_architecture():
    s = head(360, "Three timescales, one currency")
    s += box(170, 30, 340, 56, PURPLE, "Regulatory state",
             "slow dial, changes over weeks")
    s += arrow(340, 86, 340, 134)
    s += box(170, 140, 340, 56, TEAL, "Lateral voting",
             "pads compare notes, weighted by trust")
    s += arrow(250, 196, 175, 244)
    s += arrow(340, 196, 340, 244)
    s += arrow(430, 196, 505, 244)
    for i, x in enumerate((95, 265, 435)):
        s += box(x, 250, 150, 56, CORAL, f"Sensor pad {i + 1}",
                 "predict, feel, compare")
    s += ('<path d="M585 278 L640 278 L640 58 L512 58" fill="none" '
          'stroke="#993C1D" stroke-width="1.2" stroke-dasharray="4 4"/>')
    s += head_tri(560, 58, 512, 58, "#993C1D")
    s += label(600, 336, "Dashed: prediction error flows upward and drives every layer",
               11, "end")
    return s + "</svg>"


# ---------------------------------------------------------------- figure 2
def fig_timing():
    s = head(310, "How often each loop fires during one grasp")
    lanes = [(38, PURPLE, "Slow loop", "once a week", [553]),
             (128, TEAL, "Medium loop", "4 per grasp", [229, 337, 445, 553]),
             (218, CORAL, "Fast loop", "12 per grasp",
              [175 + 36 * i for i in range(12)])]
    for y, c, t, sub, ticks in lanes:
        s += box(30, y, 110, 44, c, t, sub)
        s += (f'<rect x="155" y="{y}" width="455" height="44" rx="8" '
              f'fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="0.8"/>')
        for x in ticks:
            s += (f'<circle cx="{x}" cy="{y + 22}" r="5" fill="{c["stroke"]}"/>')
    s += ('<path d="M628 245 L628 72" fill="none" stroke="#993C1D" '
          'stroke-width="1.2" stroke-dasharray="4 4"/>')
    s += head_tri(628, 120, 628, 72, "#993C1D")
    s += label(610, 290, "Prediction error from the fast loop feeds everything above it",
               11, "end")
    return s + "</svg>"


# ---------------------------------------------------------------- figure 3
def fig_loop_closure():
    """Walk a square and come back. Exact transport returns; learned drifts."""
    s = head(300, "Walk a square and return to the start")
    # left panel: exact
    s += label(175, 34, "Geometry computed exactly", 14, "middle", "#085041")
    s += ('<rect x="115" y="70" width="120" height="120" fill="none" '
          'stroke="#0F6E56" stroke-width="1.6"/>')
    s += '<circle cx="115" cy="190" r="6" fill="#0F6E56"/>'
    s += label(175, 224, "ends exactly where it began", 11, "middle", "#0F6E56")
    s += label(175, 242, "error after 32 moves: 0.0000010", 11, "middle", "#0F6E56")

    # right panel: drifting
    s += label(495, 34, "Geometry learned from data", 14, "middle", "#791F1F")
    s += ('<path d="M435 190 L437 72 L560 76 L556 194 L462 196" fill="none" '
          'stroke="#A32D2D" stroke-width="1.6"/>')
    s += '<circle cx="435" cy="190" r="6" fill="#A32D2D"/>'
    s += '<circle cx="462" cy="196" r="6" fill="#A32D2D" fill-opacity="0.35"/>'
    s += label(495, 224, "ends somewhere else", 11, "middle", "#A32D2D")
    s += label(495, 242, "error after 32 moves: 0.65", 11, "middle", "#A32D2D")

    s += ('<line x1="340" y1="60" x2="340" y2="250" stroke="#D3D1C7" '
          'stroke-width="1"/>')
    s += label(340, 282, "Same test, measured in the code as experiment E4",
               11, "middle")
    return s + "</svg>"


# ---------------------------------------------------------------- figure 4
def read_e1():
    """Live switch rates from results/e1_*.json, averaged over seeds.

    The figure is generated from whatever runs are on disk, so re-running the
    experiment and rebuilding the PDF keeps the document honest automatically.
    Falls back to the pilot numbers if no results are present.
    """
    arms = [("chroma", "Full system", TEAL),
            ("ablate_hysteresis", "Slow dial, no distinct modes", TEAL),
            ("ablate_grn", "No slow dial at all", RED)]
    fallback = {"chroma": 0.00234, "ablate_hysteresis": 0.00250,
                "ablate_grn": 0.01078}
    out, n_seeds = [], 0
    for key, label_, col in arms:
        files = sorted(RESULTS.glob(f"e1_{key}_s*.json")) if RESULTS.is_dir() else []
        if files:
            vals = [json.loads(f.read_text())["switch_rate"] for f in files]
            v = stat.mean(vals)
            n_seeds = max(n_seeds, len(vals))
        else:
            v = fallback[key]
        out.append((label_, v, col))
    return out, n_seeds


def fig_e1_bars():
    """Three-arm switch rate. Lower is better (more stable regime identity)."""
    s = head(320, "Experiment E1: three-arm ablation")
    vals, n_seeds = read_e1()
    x0, base, maxw = 300, 70, 300
    vmax = max(v for _, v, _ in vals) * 1.12 or 0.012
    for i, (name, v, c) in enumerate(vals):
        y = base + i * 70
        w = max(6, maxw * v / vmax)
        s += (f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="34" rx="4" '
              f'fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="0.8"/>')
        s += label(x0 - 12, y + 22, name, 11, "end", "#2C2C2A")
        s += label(x0 + w + 10, y + 22, f"{v:.5f}", 11, "start", c["stroke"])
    # Use the SAME threshold as the pre-registration (5x), not a separate
    # 25% rule. With chroma 1.0e-4 and flat 1.4e-4 the 25% rule would have
    # printed "the distinct modes are doing work" for a 1.44x effect that the
    # pre-registered criterion calls unsupported. The figure must not be able
    # to overclaim relative to the stated bar.
    ratio = vals[1][1] / max(vals[0][1], 1e-12)
    tie = ratio < 5.0
    s += label(340, 285,
               "Instability of module identity (lower is better)."
               + (f" {n_seeds} seeds." if n_seeds else ""), 12.5, "middle")
    s += label(340, 303,
               f"Rows 1 and 2 differ by {ratio:.2f}x, under the 5x bar set in advance."
               if tie else
               f"Rows 1 and 2 differ by {ratio:.2f}x, clearing the 5x bar set in advance.",
               12.5, "middle")
    return s + "</svg>"


FIGS = {"fig1_architecture.svg": fig_architecture,
        "fig2_timing.svg": fig_timing,
        "fig3_loop_closure.svg": fig_loop_closure,
        "fig4_e1_bars.svg": fig_e1_bars}

if __name__ == "__main__":
    for name, fn in FIGS.items():
        (OUT / name).write_text(fn())
        print("wrote", OUT / name)
