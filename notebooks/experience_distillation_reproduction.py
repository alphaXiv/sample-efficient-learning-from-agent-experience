# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.15",
#   "matplotlib==3.10.3",
#   "numpy==2.1.2",
#   "pandas==2.2.3",
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
    # Experience Distillation on public text adventures

    Language agents can improve when previous attempts remain in their prompt, but that
    benefit normally vanishes with the prompt. *Experience Distillation* asks a teacher
    with access to the old experience for one improved next action at each recorded state,
    then trains a student that never sees the experience.

    **Verdict: partially reproduced.** On TextWorldExpress, distillation internalized the
    in-context gain and the no-experience control erased it. Direct SFT nevertheless scored
    higher than distillation, while lossless packing reduced work with a statistically
    uncertain performance gap.

    This notebook embeds the completed evidence. It does not rerun training or assume that
    repository-relative result files exist, so it opens directly in Molab.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    score_rows = [
        ("Zero-shot", 0, 12.500000), ("Zero-shot", 1, 12.500000),
        ("Zero-shot", 2, 12.500000), ("Zero-shot", 3, 12.500000),
        ("Experience in context", 0, 38.541667), ("Experience in context", 1, 38.541667),
        ("Experience in context", 2, 38.541667), ("Experience in context", 3, 38.541667),
        ("Direct SFT", 0, 37.500000), ("Direct SFT", 1, 45.833333),
        ("Direct SFT", 2, 61.458333), ("Direct SFT", 3, 43.750000),
        ("Unpacked EPD", 0, 39.583333), ("Unpacked EPD", 1, 41.666667),
        ("Unpacked EPD", 2, 41.666667), ("Unpacked EPD", 3, 40.625000),
        ("Lossless packed EPD", 0, 37.500000), ("Lossless packed EPD", 1, 31.250000),
        ("Lossless packed EPD", 2, 39.583333), ("Lossless packed EPD", 3, 35.416667),
        ("Teacher without experience", 0, 6.250000), ("Teacher without experience", 1, 5.208333),
        ("Teacher without experience", 2, 11.458333), ("Teacher without experience", 3, 5.208333),
        ("Direct SFT, LR 1e-4", 0, 44.791667), ("Direct SFT, LR 1e-4", 1, 50.000000),
        ("Direct SFT, LR 1e-4", 2, 44.791667), ("Direct SFT, LR 1e-4", 3, 42.708333),
        ("Unpacked EPD, LR 1e-4", 0, 39.583333), ("Unpacked EPD, LR 1e-4", 1, 39.583333),
        ("Unpacked EPD, LR 1e-4", 2, 43.750000), ("Unpacked EPD, LR 1e-4", 3, 40.625000),
        ("Unpacked EPD, full mappings", 0, 41.666667), ("Unpacked EPD, full mappings", 1, 47.916667),
        ("Unpacked EPD, full mappings", 2, 43.750000), ("Unpacked EPD, full mappings", 3, 41.666667),
    ]
    score_df = pd.DataFrame(score_rows, columns=["condition", "replicate", "normalized_score"])
    efficiency_df = pd.DataFrame(
        [
            ("Unpacked EPD", 196, 125, 790, 3.426879, 8.473829),
            ("Naïve packed EPD", 41, 105, 567, 3.406593, 6.242887),
            ("Lossless packed EPD", 71, 45, 790, 3.418798, 6.391169),
        ],
        columns=[
            "condition", "training_instances", "training_steps",
            "supervised_tokens", "target_seconds", "train_seconds",
        ],
    )
    return efficiency_df, np, plt, score_df


@app.cell
def _(mo):
    mo.md(r"""
    ## Protocol

    - **Benchmark:** TextWorldExpress 1.1.0 — Coin Collector, Map Reader, and
      TextWorld Commonsense.
    - **Model:** Qwen2.5-0.5B-Instruct with rank-16 LoRA adapters.
    - **Data:** six collection seeds; one exploratory and one successful attempt per
      game/seed; 196 frozen decisions.
    - **Evaluation:** eight disjoint held-out seeds and four adapter seeds.
    - **Isolation:** teacher targets were produced from frozen text with zero additional
      environment interactions.
    - **Compute:** Kubernetes, NVIDIA RTX PRO 6000 Blackwell, 16 GPUs peak concurrent,
      0.341342 elapsed wall-hours through the final scientific result.

    Frozen trajectory SHA-256:
    `0907bc8e3ee60412128915f260b56772a2a59d2e22922e332ec77a8519203254`.
    """)
    return


@app.cell
def _(np, plt, score_df):
    def make_headline():
        order = [
            "Zero-shot", "Experience in context", "Direct SFT",
            "Unpacked EPD", "Lossless packed EPD", "Teacher without experience",
        ]
        grouped = score_df.groupby("condition").normalized_score
        means = grouped.mean().reindex(order)
        sds = grouped.std().reindex(order).fillna(0)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        palette = ["#64748b", "#2563eb", "#ea580c", "#059669", "#7c3aed", "#dc2626"]
        bars = ax.bar(np.arange(len(order)), means, yerr=sds, capsize=4, color=palette)
        ax.set_xticks(
            np.arange(len(order)),
            ["Zero-shot", "Experience\nin context", "Direct SFT", "Unpacked\nEPD",
             "Lossless\npacked EPD", "No-experience\nteacher"],
        )
        ax.set_ylabel("Held-out normalized score")
        ax.set_ylim(0, 65)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, value + 2, f"{value:.1f}", ha="center")
        ax.set_title("Headline result (mean ± replicate SD)")
        fig.tight_layout()
        return fig

    headline_fig = make_headline()
    headline_fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Distillation clearly learned something useful: unpacked EPD reached **40.89**, above
    both zero-shot (**12.50**) and the in-context reference (**38.54**). Yet direct SFT
    reached **47.14**, so the reproduction did not recover the paper's central
    EPD-over-SFT ordering. Keeping fuller successful mappings in the teacher context
    improved EPD to **43.75**, still below the direct-SFT mean.
    """)
    return


@app.cell
def _(mo, score_df):
    condition_picker = mo.ui.dropdown(
        options=sorted(score_df["condition"].unique()),
        value="Unpacked EPD",
        label="Inspect a condition",
    )
    condition_picker
    return (condition_picker,)


@app.cell
def _(condition_picker, mo, score_df):
    selected_table = score_df[
        score_df["condition"] == condition_picker.value
    ].copy()
    selected_mean = selected_table["normalized_score"].mean()
    selected_sd = selected_table["normalized_score"].std()
    mo.vstack(
        [
            mo.md(
                f"**{condition_picker.value}: {selected_mean:.2f} ± "
                f"{selected_sd:.2f}** across four replicates."
            ),
            selected_table,
        ]
    )
    return


@app.cell
def _(efficiency_df, mo):
    mo.md("## Packing: performance and efficiency")
    efficiency_df
    return


@app.cell
def _(np, score_df):
    unpacked_values = (
        score_df[score_df.condition == "Unpacked EPD"]
        .sort_values("replicate").normalized_score.to_numpy()
    )
    packed_values = (
        score_df[score_df.condition == "Lossless packed EPD"]
        .sort_values("replicate").normalized_score.to_numpy()
    )
    paired_differences = unpacked_values - packed_values
    paired_mean = paired_differences.mean()
    paired_se = paired_differences.std(ddof=1) / np.sqrt(len(paired_differences))
    paired_ci = (paired_mean - 3.182 * paired_se, paired_mean + 3.182 * paired_se)
    return paired_ci, paired_differences, paired_mean


@app.cell
def _(mo, paired_ci, paired_differences, paired_mean):
    mo.md(
        f"""
        Lossless packing kept every teacher token and reduced **196→71 instances**,
        **125→45 steps**, and **8.47→6.39 seconds** of optimization. Its paired score
        difference from unpacked EPD was **{paired_mean:.2f} points** (unpacked minus
        packed); with four seeds, the 95% t interval was **[{paired_ci[0]:.2f},
        {paired_ci[1]:.2f}]** and included zero.

        Per-seed differences: `{paired_differences.round(2).tolist()}`.
        """
    )
    return


@app.cell
def _(np, plt, score_df):
    def make_ablation():
        with_exp = (
            score_df[score_df.condition == "Unpacked EPD"]
            .sort_values("replicate").normalized_score.to_numpy()
        )
        no_exp = (
            score_df[score_df.condition == "Teacher without experience"]
            .sort_values("replicate").normalized_score.to_numpy()
        )
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for idx in range(4):
            ax.plot([0, 1], [with_exp[idx], no_exp[idx]], color="#94a3b8")
        ax.scatter(np.zeros(4), with_exp, color="#059669", s=55)
        ax.scatter(np.ones(4), no_exp, color="#dc2626", s=55)
        ax.scatter([0, 1], [with_exp.mean(), no_exp.mean()], marker="D", s=160,
                   color=["#059669", "#dc2626"], edgecolor="white")
        ax.axhline(12.5, color="#64748b", linestyle="--", label="Zero-shot")
        ax.set_xticks([0, 1], ["Teacher sees experience", "Experience removed"])
        ax.set_ylabel("Held-out normalized score")
        ax.set_ylim(0, 50)
        ax.set_title("Removing teacher experience erased the gain")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return fig

    ablation_fig = make_ablation()
    ablation_fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    The no-experience teacher scored **7.03**, below zero-shot and far below the
    experience-conditioned teacher's student at **40.89**. This supports the paper's
    causal mechanism: useful targets came from privileged experience, not merely from
    running another supervised update.

    The divergent SFT comparison is not evidence that the paper is wrong. The public
    corpus contains explicit successful walkthrough actions, which are unusually good
    direct labels; the paper used proprietary TaleSuite tasks, an undisclosed in-house
    model, longer contexts, and 16+ evaluation runs. A full reproduction needs those
    assets and scale.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Claim table

    | Claim | Paper | This reproduction | Assessment |
    |---|---|---|---|
    | EPD retains more gain than direct SFT | 93.4% vs −2.6% | 109% vs 133% | Divergent here |
    | Packing preserves score with less work | 43.8 vs 43.1; >10× time | 35.94 vs 40.89; 1.33× train time | Partially aligned / uncertain |
    | Teacher needs experience | Privileged context drives targets | 40.89 → 7.03 without it | Aligned |

    Formal jobs used the fixed command `bash scripts/run.sh`. Raw experiment and run
    identifiers remain in OpenResearch; the public repository contains the frozen data,
    implementation, result tables, figures, and report.
    """)
    return


if __name__ == "__main__":
    app.run()
