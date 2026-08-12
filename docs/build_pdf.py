"""Build docs/CHROMA-explained.pdf -- the plain-language explainer.

Figures are embedded as vector drawings (svglib -> ReportLab), so the PDF stays
sharp at any zoom and small enough to sit comfortably in a git repo.

    python docs/build_pdf.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)
from svglib.svglib import svg2rlg

HERE = Path(__file__).parent
FIGS = HERE / "figures"
OUT = HERE / "CHROMA-explained.pdf"

INK = colors.HexColor("#2C2C2A")
MUTED = colors.HexColor("#5F5E5A")
RULE = colors.HexColor("#D3D1C7")
ACCENT = colors.HexColor("#993C1D")
TEAL = colors.HexColor("#0F6E56")
RED = colors.HexColor("#A32D2D")
TINT = colors.HexColor("#F1EFE8")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica",
                            fontSize=26, leading=31, textColor=INK,
                            alignment=0, spaceAfter=4),
    "sub": ParagraphStyle("s", fontName="Helvetica", fontSize=12.5,
                          leading=17, textColor=MUTED, spaceAfter=16),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=15,
                         leading=19, textColor=INK, spaceBefore=17,
                         spaceAfter=7),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12,
                         leading=16, textColor=ACCENT, spaceBefore=12,
                         spaceAfter=5),
    "p": ParagraphStyle("p", fontName="Helvetica", fontSize=10.2, leading=15.4,
                        textColor=INK, alignment=TA_JUSTIFY, spaceAfter=8),
    "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=9,
                          leading=12.5, textColor=MUTED, spaceBefore=5,
                          spaceAfter=13, alignment=1),
    "pull": ParagraphStyle("pull", fontName="Helvetica", fontSize=10.6,
                           leading=15.5, textColor=INK, leftIndent=11,
                           borderPadding=(8, 8, 8, 8), spaceBefore=6,
                           spaceAfter=11, backColor=TINT),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=9,
                           leading=12.4, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=9,
                            leading=12.4, textColor=INK),
    "foot": ParagraphStyle("foot", fontName="Helvetica", fontSize=8,
                           textColor=MUTED),
}


def P(t, k="p"):
    return Paragraph(t, S[k])


def fig(name, caption, width=158 * mm):
    d = svg2rlg(str(FIGS / name))
    sc = width / d.width
    d.width, d.height = d.width * sc, d.height * sc
    d.scale(sc, sc)
    d.hAlign = "CENTER"
    return [d, P(caption, "cap")]


def table(rows, widths, header=True):
    data = [[Paragraph(c, S["cellb"] if (header and i == 0) else S["cell"])
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT",
              repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
             ("LEFTPADDING", (0, 0), (-1, -1), 7),
             ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), TINT),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED)]
    t.setStyle(TableStyle(style))
    return t


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(20 * mm, 16 * mm, 190 * mm, 16 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 11 * mm, "CHROMA - plain-language explainer")
    canvas.drawRightString(190 * mm, 11 * mm, f"{doc.page}")
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(str(OUT), pagesize=A4,
                          leftMargin=20 * mm, rightMargin=20 * mm,
                          topMargin=18 * mm, bottomMargin=22 * mm,
                          title="CHROMA - plain-language explainer",
                          author="Manish Bhattarai")
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="body")
    doc.addPageTemplates(PageTemplate(id="main", frames=[frame],
                                      onPage=decorate))
    W = doc.width
    st = []

    # ---------------------------------------------------------------- cover
    st += [Spacer(1, 8 * mm), P("CHROMA", "title"),
           P("How a robot can learn to recognise things by touch &mdash; and why "
             "two thirds of this design currently works and one third does not.",
             "sub")]
    st += [table([["Status", "Research pilot. Sections 1&ndash;3 proved and "
                             "confirmed; Section 4 failing its own tests."],
                  ["Code", "github.com/merolaagi/chroma-arch &mdash; MIT licence"],
                  ["Audience", "No mathematics assumed."]],
                 [30 * mm, W - 30 * mm], header=False)]
    st += [Spacer(1, 6 * mm)]

    # ---------------------------------------------------------------- problem
    st += [P("The problem", "h1")]
    st += [P(
        "Picture a robot arm in a wholesale grocery warehouse. Its job is to "
        "reach into a mixed pallet, feel around, and work out what it is "
        "touching before it grips. Cameras are little help: the bin is dark, "
        "items are shrink-wrapped and reflective, and half of them are buried "
        "under other items. So the arm carries eight small tactile pads, like "
        "fingertips, and must identify the item by touch alone.")]
    st += [P("Three things make this hard. Each one corresponds to a piece of "
             "the design.")]
    st += [table([
        ["Difficulty", "Why it hurts"],
        ["The robot moves", "Every time a fingertip slides two centimetres, "
         "everything it feels changes. A ridge that was under the index pad is "
         "now under the middle pad."],
        ["Memory explodes", "A cereal box feels completely different depending "
         "on which corner you touch and which way it is turned. One memory per "
         "viewpoint turns 1,800 products into hundreds of thousands of stored "
         "patterns."],
        ["The warehouse changes", "In summer the pallets hold soft produce and "
         "thin bags; in winter, boxed dry goods and cans. The robot's sense of "
         "how things usually feel has to shift &mdash; slowly, and not because "
         "of one odd item."]], [38 * mm, W - 38 * mm])]

    # ---------------------------------------------------------------- ideas
    st += [P("Three ideas", "h1")]

    st += [P("1. Geometry is arithmetic, not something to learn", "h2")]
    st += [P(
        "Think about reading braille. If you know your finger moved exactly one "
        "character to the right, you know precisely how the dot pattern should "
        "shift. You do not guess. You do not need training examples. It is "
        "bookkeeping.")]
    st += [P(
        "CHROMA builds that bookkeeping in as a hard rule. When the gripper "
        "moves, the robot's internal picture of what it is touching is shifted "
        "by exactly the right amount &mdash; computed, never learned. The "
        "learned part covers only what geometry cannot explain: the item was "
        "squishier than expected, the wrap slipped, there was a seam.")]
    st += fig("fig3_loop_closure.svg",
              "Figure 1. The test: trace a square and return to the start. "
              "Exact geometry returns to within rounding error. Learned "
              "geometry ends up somewhere else.")

    st += [P("2. Store the object, not the photographs", "h2")]
    st += [P(
        "Rather than recording what a noodle carton feels like from five "
        "hundred angles, CHROMA stores one description in the carton's own "
        "coordinates: corner here, seam there, ridged panel over there. When a "
        "fingertip makes contact at some arbitrary angle, the geometry "
        "bookkeeping translates the sensation back into carton coordinates "
        "before comparing.")]
    st += [P("It is the difference between a floor plan of a building and five "
             "hundred photographs taken from different doorways. With a floor "
             "plan you only have to walk the building once.", "pull")]

    st += [P("3. A slow dial that gates everything", "h2")]
    st += [P(
        "The robot carries a small internal state that changes over weeks, not "
        "milliseconds. Think of a chef who has decided that tonight is Italian, "
        "versus one who re-evaluates their entire technique after every single "
        "ingredient.")]
    st += [P(
        "This state does not identify items. It decides which of the robot's "
        "skills are turned up, which are turned down, and &mdash; the "
        "interesting part &mdash; which skills are still allowed to change. A "
        "grip strategy that has worked reliably for months gets locked. One "
        "that has started failing gets unlocked and becomes trainable again. "
        "The idea is borrowed from how a stem cell commits to becoming a "
        "specific cell type, and can un-commit under injury.")]

    st += [PageBreak()]

    # ------------------------------------------------------------ how it fits
    st += [P("How the three fit together", "h1")]
    st += [P(
        "Three loops run at three speeds. One quantity connects them all: the "
        "<b>prediction error</b> &mdash; the gap between what the robot expected "
        "to feel and what it actually felt. That single number trains the fast "
        "loop, weights the middle loop, and steers the slow one.")]
    st += fig("fig1_architecture.svg",
              "Figure 2. The fast loop predicts and feels; the middle loop pools "
              "opinions across pads; the slow loop sets the operating mode.")
    st += fig("fig2_timing.svg",
              "Figure 3. The same three loops seen as timing. One grasp is "
              "twelve finger movements; the slow dial barely moves in a week.")

    # ---------------------------------------------------------------- walkthrough
    st += [P("One grasp, step by step", "h1")]
    st += [P("The gripper descends into a mixed pallet. Target: a twelve-pack of "
             "instant noodle cups, shrink-wrapped, partly under a bag of rice.")]
    st += [table([
        ["Step", "What happens"],
        ["0. Contact", "Eight pads touch the item at eight spots. Each pad knows "
         "exactly where it sits relative to the others &mdash; that is fixed by "
         "the hardware, so it is given, not learned."],
        ["1. The guess", "Pad 3 feels something ridged. Before moving, the robot "
         "commits to a prediction: slide two millimetres and this ridge should "
         "shift by exactly this much, with the texture unchanged. The geometric "
         "half is computed exactly; only the texture half is a learned guess."],
        ["2. Reality check", "The hand moves and the pad reports what it actually "
         "feels. The gap is the prediction error. Note that the robot predicts "
         "at the level of &ldquo;ridged cardboard edge&rdquo;, not raw pressure "
         "readings &mdash; the plot of a film, not every pixel."],
        ["3&ndash;6. Evidence", "Each pad translates its sensation back into "
         "object coordinates and asks the catalogue for a match. Pad 3 says "
         "&ldquo;noodle carton, rotated about thirty degrees, left face&rdquo;. "
         "Pad 7 is torn between two products."],
        ["7. Compare notes", "Because the hand geometry is known, a confident "
         "pad's guess can be translated into any other pad's frame. They pool "
         "evidence, weighted by how well each has been predicting lately. A pad "
         "with high recent error is discounted automatically. Recognition "
         "usually settles here."],
        ["8&ndash;12. Grip", "Remaining moves confirm the hypothesis and locate "
         "the seam for a safe grip."],
        ["Once a week", "The slow dial has been quietly integrating average error "
         "and disagreement across thousands of grasps. As the seasonal mix "
         "shifts, skills that keep working stay locked; skills that start "
         "failing unlock and retrain."]], [26 * mm, W - 26 * mm])]

    st += [Spacer(1, 4 * mm)]
    st += [P(
        "That last row addresses a real and expensive problem. Retrain a normal "
        "robot on new items and it often gets <i>worse</i> at the old ones "
        "&mdash; catastrophic forgetting. The locking mechanism here comes with "
        "a mathematical guarantee: as long as a skill keeps performing, how far "
        "it can drift is capped by a fixed number that does not grow with time, "
        "no matter how many months pass.")]

    st += [PageBreak()]

    # ------------------------------------------------------------- ablations
    st += [P("What breaks if you remove each piece", "h1")]
    st += [table([
        ["Remove", "What happens"],
        ["Exact geometry", "The hand loses track of where it is after about ten "
         "moves. Errors compound and long exploration becomes useless."],
        ["Object-coordinate memory", "Every item must be felt from every angle. "
         "1,800 products becomes an impossible amount of training."],
        ["Multiple voting pads", "One misleading contact &mdash; a fingertip on a "
         "label instead of cardboard &mdash; throws the whole identification."],
        ["The slow dial", "The robot re-tunes its entire strategy every time it "
         "touches something unusual. It never settles."],
        ["Plasticity locking", "Teaching it this season's items degrades last "
         "season's. Every changeover costs accuracy on things it already knew."],
    ], [45 * mm, W - 45 * mm])]

    # ---------------------------------------------------------------- results
    st += [P("Where it actually stands", "h1")]
    st += [P(
        "Ideas 1 and 2 are proved, both on paper and in the running code. They "
        "are not especially new; the contribution is welding them together. "
        "Idea 3 is the invented part, and the honest result so far is that it "
        "mostly does not work as claimed.")]
    st += [P(
        "The decisive test compares three versions: the full system; a version "
        "with the &ldquo;settles into distinct modes&rdquo; property removed but "
        "the slowness kept; and a version with neither. The middle version "
        "performs identically to the full one.", "pull")]
    st += fig("fig4_e1_bars.svg",
              "Figure 4. Instability of module identity across the three "
              "versions. Lower is better. The top two bars are the same height.")
    st += [P(
        "The benefit therefore comes from the dial simply being <b>slow</b> "
        "&mdash; not from it having distinct settings, which was the actual "
        "claim. A plain running average would deliver the same thing with none "
        "of the biology-inspired machinery. A second test, asking whether the "
        "eight pads specialise into distinct roles over time the way stem cells "
        "differentiate, also came back negative: they stayed nearly identical.")]
    st += [P(
        "Both results come from a run at one fiftieth of the intended size, so "
        "they may yet flip. But as things stand, the most elaborate part of the "
        "design is the part failing its own test, and the plain parts are the "
        "ones carrying the system.")]

    st += [P("Scoreboard", "h1")]
    st += [table([
        ["Claim", "Verdict", "Evidence"],
        ["Exact geometric transport", "Supported", "Error 0.0000010 vs 0.65 for "
         "the learned alternative"],
        ["Viewpoint-free memory", "Supported", "Proved, and confirmed by test"],
        ["Bounded forgetting", "Supported in isolation", "Measured drift 19.53 "
         "against a predicted ceiling of 20.0"],
        ["Distinct modes help", "Not supported", "Flat-landscape version ties "
         "the full system"],
        ["Slow integration helps", "Weakly supported", "4.6x better than no dial; "
         "the threshold set in advance was 5x"],
        ["Pads specialise over time", "Not supported", "0.214 against a "
         "pre-set threshold of 0.7"],
    ], [48 * mm, 34 * mm, W - 82 * mm])]

    st += [P("Reproducing this", "h1")]
    st += [P(
        "Everything above is generated by the repository, and every claim on "
        "paper is an executable test. The full comparison is fifteen runs "
        "&mdash; five random seeds across three versions &mdash; which is about "
        "one overnight session on a desktop machine.")]
    code = ParagraphStyle("code", fontName="Courier", fontSize=8.6,
                          leading=13, textColor=INK, backColor=TINT,
                          borderPadding=(8, 8, 8, 8), spaceBefore=4,
                          spaceAfter=6)
    st += [Paragraph(
        "git clone https://github.com/merolaagi/chroma-arch<br/>"
        "cd chroma-arch &amp;&amp; pip install torch<br/>"
        "PYTHONPATH=. python tests/test_propositions.py<br/>"
        "python experiments/e1_to_e4.py e4<br/>"
        "python run_e1_chunk.py chroma 20000 0<br/>"
        "python aggregate.py", code)]
    st += [Spacer(1, 3 * mm)]
    st += [P(
        "If the flat-landscape version still ties the full system at full "
        "scale, the right response is to delete the elaborate machinery and "
        "keep the timescale. That would be a smaller claim than the one the "
        "design started with, and a true one.")]

    doc.build(st)
    print("wrote", OUT, f"({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build()
