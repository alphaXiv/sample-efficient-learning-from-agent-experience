#!/usr/bin/env python3
"""Generate the report's dependency-free SVG evidence figures."""

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path("reports/experience-distillation/images")
OUT.mkdir(parents=True, exist_ok=True)

BG, FG, GRID = "#fbfaf7", "#202124", "#d8d6cf"
BLUE, TEAL, GOLD, RED, PURPLE, GRAY = (
    "#3b6fb6",
    "#2a9d8f",
    "#e9a23b",
    "#d65f5f",
    "#7768ae",
    "#7b8087",
)


def doc(width, height, body, title, desc):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>
<rect width="100%" height="100%" fill="{BG}"/>
<style>text{{font-family:Inter,ui-sans-serif,system-ui,sans-serif;fill:{FG}}}.small{{font-size:13px}}.label{{font-size:15px}}.title{{font-size:22px;font-weight:700}}.value{{font-size:14px;font-weight:700}}</style>
{body}</svg>"""


def bar_chart(path, title, subtitle, labels, values, errors, colors, ymax, ylabel, ymin=0):
    w, h = 920, 500
    left, top, right, bottom = 90, 85, 130, 110
    pw, ph = w - left - right, h - top - bottom
    parts = [
        f'<text x="{left}" y="36" class="title">{escape(title)}</text>',
        f'<text x="{left}" y="61" class="small">{escape(subtitle)}</text>',
        f'<text transform="translate(22 {top + ph/2}) rotate(-90)" class="label">{escape(ylabel)}</text>',
    ]
    step = 20 if ymax - ymin <= 100 else 50
    for tick in range(int(ymin), int(ymax) + 1, step):
        y = top + ph * (ymax - tick) / (ymax - ymin)
        parts += [
            f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>',
            f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" class="small">{tick}</text>',
        ]
    slot = pw / len(labels)
    bw = min(88, slot * 0.62)
    for i, (lab, val, err, color) in enumerate(zip(labels, values, errors, colors)):
        x = left + slot * (i + 0.5) - bw / 2
        zero_y = top + ph * ymax / (ymax - ymin)
        value_y = top + ph * (ymax - val) / (ymax - ymin)
        y = min(zero_y, value_y)
        bh = abs(zero_y - value_y)
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="5" fill="{color}"/>')
        if err:
            ey1 = top + ph * (ymax - min(ymax, val + err)) / (ymax - ymin)
            ey2 = top + ph * (ymax - max(ymin, val - err)) / (ymax - ymin)
            cx = x + bw / 2
            parts += [
                f'<line x1="{cx:.1f}" y1="{ey1:.1f}" x2="{cx:.1f}" y2="{ey2:.1f}" stroke="{FG}" stroke-width="2"/>',
                f'<line x1="{cx-7:.1f}" y1="{ey1:.1f}" x2="{cx+7:.1f}" y2="{ey1:.1f}" stroke="{FG}" stroke-width="2"/>',
                f'<line x1="{cx-7:.1f}" y1="{ey2:.1f}" x2="{cx+7:.1f}" y2="{ey2:.1f}" stroke="{FG}" stroke-width="2"/>',
            ]
        parts += [
            f'<text x="{x+bw/2:.1f}" y="{value_y-10 if val >= 0 else value_y+20:.1f}" text-anchor="middle" class="value">{val:.1f}</text>',
            f'<text x="{x+bw/2:.1f}" y="{top+ph+26:.1f}" text-anchor="middle" class="small">{escape(lab)}</text>',
        ]
    path.write_text(doc(w, h, "\n".join(parts), title, subtitle))


bar_chart(
    OUT / "headline_scores.svg",
    "Direct SFT led this substitute study",
    "Mean normalized TextWorldExpress score; error bars are sample SD across four adapter replications.",
    ["Zero-shot", "Experience", "Direct SFT", "Unpacked", "Packed", "No exp."],
    [12.5, 38.5417, 47.1354, 40.8854, 35.9375, 7.0313],
    [0, 0, 10.1897, 0.9974, 3.5586, 2.9929],
    [GRAY, BLUE, GOLD, TEAL, PURPLE, RED],
    80,
    "Normalized task score",
)

bar_chart(
    OUT / "retained_gain.svg",
    "Experience targets retained the context gain",
    "Zero-shot = 0%; experience in context = 100%. Error bars show sample SD (n=4).",
    ["Direct SFT", "Unpacked", "Packed", "No experience"],
    [133, 109, 90, -21],
    [39.1067, 3.8297, 13.6626, 11.4891],
    [GOLD, TEAL, PURPLE, RED],
    200,
    "Retained in-context gain (%)",
    -50,
)


def efficiency():
    w, h = 900, 480
    left, top, pw = 210, 90, 500
    rows = [
        ("Training instances", 196, 71, ""),
        ("Optimization steps", 125, 45, ""),
        ("Training time", 8.47, 6.39, " s"),
    ]
    parts = [
        f'<text x="70" y="36" class="title">Packing reduced optimization work</text>',
        f'<text x="70" y="61" class="small">Absolute measurements; unpacked and packed preserve the same 790 supervised target tokens.</text>',
    ]
    for j, (label, a, b, suffix) in enumerate(rows):
        y = top + j * 115
        mx = max(a, b)
        parts.append(f'<text x="20" y="{y-10}" class="label">{escape(label)}</text>')
        for k, (name, val, color) in enumerate([("Unpacked", a, TEAL), ("Packed", b, PURPLE)]):
            yy = y + k * 38
            width = pw * val / mx
            value_x = left + width - 10 if val == mx else left + width + 9
            value_anchor = "end" if val == mx else "start"
            value_fill = ' style="fill:#ffffff"' if val == mx else ""
            parts += [
                f'<rect x="{left}" y="{yy}" width="{width:.1f}" height="25" rx="4" fill="{color}"/>',
                f'<text x="{value_x:.1f}" y="{yy+18}" text-anchor="{value_anchor}" class="value"{value_fill}>{val:g}{suffix}</text>',
                f'<text x="{left-10}" y="{yy+18}" text-anchor="end" class="small">{name}</text>',
            ]
    (OUT / "packing_efficiency.svg").write_text(
        doc(w, h, "\n".join(parts), "Packing reduced optimization work", "Instances, steps, and training seconds for unpacked and lossless packed distillation.")
    )


efficiency()


def heatmap():
    w, h = 890, 470
    left, top, cw, ch = 260, 105, 170, 65
    cols = ["Coin Collector", "Map Reader", "Commonsense"]
    rows = [
        ("Zero-shot", [25.0, 0.0, 12.5]),
        ("Experience in context", [87.5, 18.75, 9.375]),
        ("Direct SFT", [87.5, 21.875, 32.031]),
        ("Unpacked distillation", [87.5, 29.688, 5.469]),
        ("Lossless packed", [71.875, 29.688, 6.25]),
    ]
    parts = [
        '<text x="55" y="36" class="title">The aggregate score hides different game behavior</text>',
        '<text x="55" y="61" class="small">Cell color and label show mean normalized score across four replications.</text>',
    ]
    for j, col in enumerate(cols):
        parts.append(f'<text x="{left+j*cw+cw/2}" y="{top-20}" text-anchor="middle" class="label">{escape(col)}</text>')
    for i, (name, vals) in enumerate(rows):
        y = top + i * ch
        parts.append(f'<text x="{left-14}" y="{y+39}" text-anchor="end" class="label">{escape(name)}</text>')
        for j, val in enumerate(vals):
            x = left + j * cw
            alpha = 0.12 + 0.78 * val / 100
            parts += [
                f'<rect x="{x}" y="{y}" width="{cw-6}" height="{ch-6}" rx="5" fill="{BLUE}" fill-opacity="{alpha:.3f}"/>',
                f'<text x="{x+(cw-6)/2}" y="{y+38}" text-anchor="middle" class="value">{val:.1f}</text>',
            ]
    (OUT / "per_game.svg").write_text(doc(w, h, "\n".join(parts), "Per-game normalized scores", "Mean score by condition on Coin Collector, Map Reader, and TextWorld Commonsense."))


heatmap()


def robustness():
    w, h = 900, 450
    left, top, pw, ph = 100, 90, 700, 250
    scenarios = ["LR 2e-4", "LR 1e-4", "All mappings"]
    direct = [47.135, 45.573, 47.135]
    epd = [40.885, 40.885, 43.75]
    parts = [
        '<text x="65" y="36" class="title">The central ordering was robust</text>',
        '<text x="65" y="61" class="small">Direct SFT remained above unpacked distillation after halving the learning rate and expanding the teacher summary.</text>',
    ]
    for tick in range(0, 71, 10):
        y = top + ph - tick / 70 * ph
        parts += [
            f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="{GRID}"/>',
            f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" class="small">{tick}</text>',
        ]
    for i, scenario in enumerate(scenarios):
        x = left + pw * (i + 0.5) / len(scenarios)
        y1 = top + ph - direct[i] / 70 * ph
        y2 = top + ph - epd[i] / 70 * ph
        parts += [
            f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{GRID}" stroke-width="4"/>',
            f'<circle cx="{x}" cy="{y1}" r="9" fill="{GOLD}"/><text x="{x+14}" y="{y1+5}" class="value">{direct[i]:.1f}</text>',
            f'<circle cx="{x}" cy="{y2}" r="9" fill="{TEAL}"/><text x="{x+14}" y="{y2+5}" class="value">{epd[i]:.1f}</text>',
            f'<text x="{x}" y="{top+ph+32}" text-anchor="middle" class="label">{escape(scenario)}</text>',
        ]
    parts += [
        f'<circle cx="330" cy="400" r="7" fill="{GOLD}"/><text x="345" y="405" class="small">Direct SFT</text>',
        f'<circle cx="480" cy="400" r="7" fill="{TEAL}"/><text x="495" y="405" class="small">Unpacked distillation</text>',
    ]
    (OUT / "robustness.svg").write_text(doc(w, h, "\n".join(parts), "Robustness of method ordering", "Direct SFT and unpacked distillation scores at two learning rates and with a fuller teacher summary."))


robustness()
print(f"Wrote 5 SVG figures to {OUT}")
