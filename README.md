# Experience Distillation on public text adventures

This repository reproduces the central claim of [*Sample-Efficient Learning from Agent Experience* (arXiv:2607.21051)](https://arxiv.org/abs/2607.21051) on three public TextWorldExpress games with Qwen2.5-0.5B-Instruct. We froze 196 environment interactions once, generated every teacher target offline, and trained matched LoRA adapters on four seeds. **Assessment: partially reproduced.** Unpacked Experience Distillation retained 109% of the available in-context gain and removing teacher experience reversed it to −21%, but direct SFT retained 133% and scored 47.14 versus 40.89, so the paper’s EPD-over-SFT ordering did not appear here.

Lossless branch packing reduced 196 examples / 125 optimization steps to 71 / 45 and training time from 8.47s to 6.39s. Its 35.94 score was 4.95 points below unpacked EPD, but the paired four-seed 95% interval included zero. This is a deliberately reduced substitution for the paper’s proprietary six-game TaleSuite study: TextWorldExpress 1.1.0 (Coin Collector, Map Reader, TextWorld Commonsense), six train seeds, eight disjoint held-out seeds, and a 0.5B open instruct model.

[Read the illustrated report](reports/experience-distillation/report.md) · [Explore the self-contained marimo notebook](notebooks/experience_distillation_reproduction.py)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/sample-efficient-learning-from-agent-experience/blob/main/notebooks/experience_distillation_reproduction.py)

Compute: Kubernetes; NVIDIA RTX PRO 6000 Blackwell; 16 GPUs peak concurrent; 0.341342 elapsed wall-hours from the fresh-attempt cutoff to the final scientific result.

## Headline results

| Method | Normalized score, mean ± SD | Retained in-context gain | Instances | Train time |
|---|---:|---:|---:|---:|
| Zero-shot | 12.50 ± 0.00 | 0% | — | — |
| Experience in context | 38.54 ± 0.00 | 100% | — | — |
| Direct SFT | **47.14 ± 10.19** | **133%** | 196 | 8.47s |
| Unpacked one-step EPD | 40.89 ± 0.99 | 109% | 196 | 8.47s |
| Lossless packed EPD | 35.94 ± 3.55 | 90% | 71 | **6.39s** |
| Teacher without experience | 7.03 ± 2.84 | −21% | 196 | 8.58s |

All trained rows used trajectory SHA-256 `0907bc8e3ee60412128915f260b56772a2a59d2e22922e332ec77a8519203254`; target generation made zero environment interactions.

## Experiment log

The exact inherited run command for every experiment below was copied verbatim from `orx exp status`.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public presentation surface | Not run as an experiment (publication surface) | README, report, notebook, figures | — |
| [Compact reference](https://github.com/alphaXiv/sample-efficient-learning-from-agent-experience/tree/orx/compact-same-fold-experience-reference) | Frozen task protocol and in-context reference | `bash scripts/run.sh` | 12.50 → 38.54 with compact experience | Kubernetes, 4× RTX PRO 6000, 57s |
| [Direct SFT](https://github.com/alphaXiv/sample-efficient-learning-from-agent-experience/tree/orx/direct-sft-on-frozen-experience) | Train directly on the same recorded actions | `bash scripts/run.sh` | 47.14; higher than EPD | Kubernetes, 4× RTX PRO 6000, 1m23s |
| [Unpacked EPD](https://github.com/alphaXiv/sample-efficient-learning-from-agent-experience/tree/orx/unpacked-one-step-distillation) | One teacher decision per recorded branch point | `bash scripts/run.sh` | 40.89; 109% retained gain | Kubernetes, 4× RTX PRO 6000, 1m23s |
| [Lossless packed EPD](https://github.com/alphaXiv/sample-efficient-learning-from-agent-experience/tree/orx/lossless-branch-packing) | Pack branches without truncating target tokens | `bash scripts/run.sh` | 35.94; fewer instances/steps; difference uncertain | Kubernetes, 4× RTX PRO 6000, 1m18s |
| [No-experience teacher](https://github.com/alphaXiv/sample-efficient-learning-from-agent-experience/tree/orx/no-experience-teacher-ablation) | Remove privileged experience during target generation | `bash scripts/run.sh` | 7.03; gain erased | Kubernetes, 4× RTX PRO 6000, 1m23s |
| [Full-mapping EPD](https://github.com/alphaXiv/sample-efficient-learning-from-agent-experience/tree/orx/unpacked-epd-full-successful-mappings) | Preserve more successful mappings in teacher context | `bash scripts/run.sh` | 43.75; still below direct SFT mean | Kubernetes, 4× RTX PRO 6000, 1m23s |

## Re-run

The committed configuration selects the lossless packed condition. Every formal run used the fixed command:

```bash
bash scripts/run.sh
```

It installs pinned dependencies, downloads the public model, launches four GPU-local replications, and prints all evidence to the terminal. The Kubernetes manifest requests four GPUs; launch formal experiments through OpenResearch rather than invoking training directly.
