# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.14.17",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Experience Distillation on TextWorldExpress

    An agent may act better when its previous attempts remain in the prompt. Experience Distillation asks whether a teacher with that history can generate one-step decisions that teach a student to retain the improvement without seeing the history—or touching the environment—at deployment.

    **Reproduction verdict: partially reproduced.** The experience-conditioned student retained the context gain and the no-experience control collapsed, but direct supervised fine-tuning had the higher point estimate in this small public-benchmark substitute.
    """)
    return


@app.cell
def _():
    results = [
        {"condition": "Zero-shot", "score": 12.50, "sd": 0.00, "retained_gain": 0.0},
        {"condition": "Experience in context", "score": 38.54, "sd": 0.00, "retained_gain": 100.0},
        {"condition": "Direct SFT", "score": 47.14, "sd": 10.19, "retained_gain": 133.0},
        {"condition": "Unpacked distillation", "score": 40.89, "sd": 1.00, "retained_gain": 109.0},
        {"condition": "Lossless packed", "score": 35.94, "sd": 3.56, "retained_gain": 90.0},
        {"condition": "No-experience teacher", "score": 7.03, "sd": 2.99, "retained_gain": -21.0},
    ]
    return (results,)


@app.cell
def _(mo, results):
    colors = ["#7b8087", "#3b6fb6", "#e9a23b", "#2a9d8f", "#7768ae", "#d65f5f"]
    width, height = 820, 360
    left, top, plot_w, plot_h = 55, 45, 735, 235
    bars = []
    slot = plot_w / len(results)
    for i, (row, color) in enumerate(zip(results, colors)):
        x = left + slot * i + 18
        bar_w = slot - 34
        bar_h = plot_h * row["score"] / 70
        y = top + plot_h - bar_h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="4" fill="{color}"/>'
            f'<text x="{x+bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-weight="700">{row["score"]:.1f}</text>'
            f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h+24:.1f}" text-anchor="middle" font-size="11">{row["condition"].replace(" ", "&#160;")}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Headline normalized task scores">
    <rect width="100%" height="100%" fill="#fbfaf7"/>
    <style>text{{font-family:Inter,system-ui,sans-serif;fill:#202124}}</style>
    <text x="28" y="27" font-size="18" font-weight="700">Headline normalized task score</text>
    {''.join(bars)}
    </svg>"""
    mo.Html(svg)
    return


@app.cell
def _(mo, results):
    mo.vstack(
        [
            mo.md(
                """
                ## How to read the evidence

                Zero-shot and experience-in-context are fixed anchors. Retained gain is
                \\((student - zero) / (experience - zero)\\), so 100% matches the in-context
                improvement and values above 100% exceed it.
                """
            ),
            mo.ui.table(results, pagination=False),
        ]
    )
    return


@app.cell
def _():
    frozen_protocol = {
        "benchmark": "TextWorldExpress 1.1.0",
        "games": ["Coin Collector", "Map Reader", "TextWorld Commonsense"],
        "model": "Qwen2.5-0.5B-Instruct",
        "training_seeds_per_game": 6,
        "held_out_seeds_per_game": 8,
        "recorded_environment_interactions": 196,
        "trajectory_sha256": "0907bc8e3ee60412128915f260b56772a2a59d2e22922e332ec77a8519203254",
        "target_generation_environment_interactions": 0,
        "replications": 4,
    }
    return (frozen_protocol,)


@app.cell
def _(frozen_protocol, mo):
    mo.md(
        f"""
        ## Frozen protocol

        The study recorded one exploratory and one successful attempt for each
        training seed, then froze the corpus before teacher target generation.
        The teacher chose one valid next action at each recorded history; the
        student saw the history but not the compressed experience.

        ```json
        {frozen_protocol}
        ```
        """
    )
    return


@app.cell
def _():
    efficiency = [
        {"metric": "Training instances", "unpacked": 196, "lossless_packed": 71, "reduction_pct": 63.8},
        {"metric": "Optimization steps", "unpacked": 125, "lossless_packed": 45, "reduction_pct": 64.0},
        {"metric": "Training seconds", "unpacked": 8.47, "lossless_packed": 6.39, "reduction_pct": 24.6},
        {"metric": "Supervised tokens", "unpacked": 790, "lossless_packed": 790, "reduction_pct": 0.0},
    ]
    return (efficiency,)


@app.cell
def _(efficiency, mo):
    mo.vstack(
        [
            mo.md(
                """
                ## Packing and the negative control

                Lossless packing preserved all target tokens while cutting instances,
                steps, and training time. Packed score was 35.94 versus 40.89 unpacked;
                with four paired replications the 95% interval for the difference includes
                zero. Removing experience from the teacher reduced score to 7.03,
                below the 12.50 zero-shot anchor.
                """
            ),
            mo.ui.table(efficiency, pagination=False),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Claim-by-claim assessment

    | Claim | Paper | This reproduction | Assessment |
    |---|---:|---:|---|
    | Distillation retains more gain than direct SFT | 93.4% vs –2.6% | 109% vs 133% | Reversed point estimate |
    | Packing preserves performance | 43.8 vs 43.1 | 35.94 vs 40.89 | Within four-rep uncertainty |
    | Packing reduces work | 128 vs 4,096 instances | 71 vs 196 | Qualitatively aligned |
    | Teacher needs experience | Gain erased without it | –21% retained gain | Aligned |

    This does not test TaleSuite or the paper's in-house checkpoint. The strongest
    ambiguity is target quality: the 0.5B teacher agreed with recorded actions only
    30.6% of the time, while direct SFT could learn directly from successful gold paths.

    **Compute:** Kubernetes; NVIDIA RTX PRO 6000 Blackwell; four GPUs per job;
    16 GPUs peak concurrent; 0.341342 hours campaign wall time.
    """)
    return


if __name__ == "__main__":
    app.run()
