#!/usr/bin/env python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "experience-distillation" / "images"
OUT.mkdir(parents=True, exist_ok=True)
scores = pd.read_csv(ROOT / "results" / "replicate_scores.csv")
eff = pd.read_csv(ROOT / "results" / "efficiency.csv")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfbfd",
    }
)
colors = {
    "blue": "#2563eb",
    "green": "#059669",
    "orange": "#ea580c",
    "red": "#dc2626",
    "gray": "#64748b",
    "purple": "#7c3aed",
}


def stats(names):
    grouped = scores[scores.condition.isin(names)].groupby("condition").normalized_score
    return grouped.mean().reindex(names), grouped.std().reindex(names).fillna(0)


# 1. Headline score
names = [
    "zero_shot",
    "experience_in_context",
    "direct_sft",
    "unpacked_epd",
    "lossless_packed_epd",
    "no_experience_teacher",
]
labels = ["Zero-shot", "Experience\nin context", "Direct SFT", "Unpacked\nEPD", "Lossless\npacked EPD", "Teacher w/o\nexperience"]
means, sds = stats(names)
fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(names))
bars = ax.bar(
    x,
    means,
    yerr=sds,
    capsize=4,
    color=[colors["gray"], colors["blue"], colors["orange"], colors["green"], colors["purple"], colors["red"]],
    alpha=0.92,
)
ax.set_ylabel("Held-out normalized task score")
ax.set_xticks(x, labels)
ax.set_ylim(0, 65)
ax.axhline(means["experience_in_context"], color=colors["blue"], ls="--", lw=1, alpha=0.65)
for bar, value in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 2.0, f"{value:.1f}", ha="center", fontweight="bold")
ax.set_title("Distillation retained the in-context gain, but direct SFT scored higher")
ax.text(0.01, -0.20, "Bars: mean of four adapter seeds; error bars: replicate SD. Same frozen trajectory SHA for trained methods.", transform=ax.transAxes, color="#475569")
fig.tight_layout()
fig.savefig(OUT / "headline_scores.png", dpi=180, bbox_inches="tight")
plt.close(fig)


# 2. Learning-rate robustness
fig, ax = plt.subplots(figsize=(7.5, 4.6))
lr_names = ["direct_sft", "unpacked_epd", "direct_sft_lr1e4", "unpacked_epd_lr1e4"]
lr_means, lr_sds = stats(lr_names)
xs = np.array([0, 1, 3, 4])
bars = ax.bar(
    xs,
    lr_means,
    yerr=lr_sds,
    capsize=4,
    color=[colors["orange"], colors["green"], colors["orange"], colors["green"]],
)
ax.set_xticks([0.5, 3.5], ["Learning rate 2×10⁻⁴", "Learning rate 1×10⁻⁴"])
ax.set_ylabel("Held-out normalized task score")
ax.set_ylim(0, 65)
for bar, value in zip(bars, lr_means):
    ax.text(bar.get_x() + bar.get_width()/2, value + 1.8, f"{value:.1f}", ha="center")
ax.legend([bars[0], bars[1]], ["Direct SFT", "Unpacked EPD"], frameon=False, loc="upper right")
ax.set_title("The SFT advantage persisted across a matched learning-rate check")
fig.tight_layout()
fig.savefig(OUT / "learning_rate_robustness.png", dpi=180, bbox_inches="tight")
plt.close(fig)


# 3. Packing efficiency
fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
order = ["unpacked_epd", "lossless_packed_epd"]
labels2 = ["Unpacked", "Lossless packed"]
sub = eff.set_index("condition").loc[order]
metrics = [
    ("training_instances", "Training instances"),
    ("training_steps", "Optimization steps"),
    ("train_seconds", "Training seconds"),
]
for ax, (metric, title) in zip(axes, metrics):
    vals = sub[metric].to_numpy()
    bars = ax.bar(labels2, vals, color=[colors["green"], colors["purple"]])
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(0, max(vals) * 1.25)
    for bar, value in zip(bars, vals):
        text = f"{value:.2f}" if metric == "train_seconds" else f"{int(value)}"
        ax.text(bar.get_x() + bar.get_width()/2, value * 1.04, text, ha="center", fontweight="bold")
fig.suptitle("Lossless packing reduced instances, steps, and optimization time", y=1.03, fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "packing_efficiency.png", dpi=180, bbox_inches="tight")
plt.close(fig)


# 4. Experience-access mechanism
with_exp = scores[scores.condition == "unpacked_epd"].normalized_score.to_numpy()
without_exp = scores[scores.condition == "no_experience_teacher"].normalized_score.to_numpy()
fig, ax = plt.subplots(figsize=(7.2, 4.6))
positions = [0, 1]
for i in range(4):
    ax.plot(positions, [with_exp[i], without_exp[i]], color="#94a3b8", lw=1.5, alpha=0.85)
ax.scatter(np.zeros(4), with_exp, s=55, color=colors["green"], zorder=3, label="Teacher sees experience")
ax.scatter(np.ones(4), without_exp, s=55, color=colors["red"], zorder=3, label="Experience removed")
ax.scatter(positions, [with_exp.mean(), without_exp.mean()], s=170, marker="D", color=[colors["green"], colors["red"]], edgecolor="white", linewidth=1.5, zorder=4)
ax.axhline(12.5, color=colors["gray"], ls="--", lw=1, label="Zero-shot")
ax.set_xticks(positions, ["Experience-conditioned", "No-experience control"])
ax.set_ylabel("Held-out normalized task score")
ax.set_xlim(-0.35, 1.35)
ax.set_ylim(0, 50)
ax.set_title("Removing experience from the teacher erased the gain")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "teacher_access_ablation.png", dpi=180, bbox_inches="tight")
plt.close(fig)
