# Experience Distillation on TextWorldExpress

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/sample-efficient-learning-from-agent-experience/blob/main/notebooks/experience_distillation.py)

This is a public, claim-focused reproduction of **Sample-Efficient Learning from Agent Experience** ([arXiv:2607.21051](https://arxiv.org/abs/2607.21051)). We tested whether an experience-conditioned one-step teacher transfers more of an in-context gain than direct supervised fine-tuning (SFT) on the same frozen trajectories, whether branch packing preserves the result more efficiently, and whether removing teacher access to experience erases the gain.

**Assessment: partially reproduced.** The central ordering did not appear: the paper reports 43.8 normalized score for Experience Distillation versus 17.8 for SFT, while this substitute study measured **40.89 ± 0.99 for unpacked distillation versus 47.14 ± 10.19 for SFT** (mean ± sample SD, four replications). Two mechanism claims aligned: the no-experience teacher collapsed to 7.03, and lossless packing cut 196 instances / 125 steps to 71 / 45 while remaining statistically unresolved from unpacked distillation at four paired replications.

We substituted Apache-2.0 [TextWorldExpress](https://github.com/cognitiveailab/TextWorldExpress) (Coin Collector, Map Reader, and TextWorld Commonsense) for unavailable TaleSuite assets and Qwen2.5-0.5B-Instruct LoRA adapters for the paper’s in-house model. Eighteen fixed training episodes (196 recorded interactions; SHA-256 `0907bc8e…203254`) were frozen before target generation. Evaluation used eight disjoint seeds per game from the same benchmark fold. Every formal run used Kubernetes on NVIDIA RTX PRO 6000 Blackwell GPUs; jobs allocated four GPUs and peak concurrency was 16 GPUs. The fresh campaign took 0.44 wall hours from first launch to last completion, including setup diagnostics.

- [Illustrated report](reports/experience-distillation/report.md)
- [Self-contained marimo notebook](notebooks/experience_distillation.py)
- [Open the notebook directly in Molab](https://molab.marimo.io/github/alphaXiv/sample-efficient-learning-from-agent-experience/blob/main/notebooks/experience_distillation.py)
- [Frozen trajectories](data/frozen_trajectories.json) and [compact results](reports/experience-distillation/results/summary.csv)

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public report, notebook, and validated implementation | Not run as an experiment (publication surface) | Presentation only | — |
| [Valid fixed Map Reader seeds](https://github.com/alphaXiv/sample-efficient-learning-from-agent-experience/tree/orx/valid-fixed-map-reader-seeds) | Fresh collection and initial references | `bash scripts/run.sh` | Successful Kubernetes evidence; froze the source corpus later committed as `0907bc8e…` | 4 GPUs, 2m10s |
| [Direct SFT](https://github.com/alphaXiv/sample-efficient-learning-from-agent-experience/tree/orx/direct-sft-on-frozen-experience) | Train on all recorded actions | `bash scripts/run.sh` | 47.14 score; 133% retained gain | 4 GPUs, 1m23s |
| [Unpacked one-step distillation](https://github.com/alphaXiv/sample-efficient-learning-from-agent-experience/tree/orx/unpacked-one-step-distillation) | Experience-conditioned one-step targets | `bash scripts/run.sh` | 40.89 score; 109% retained gain; below SFT point estimate | 4 GPUs, 1m23s |
| [Lossless branch packing](https://github.com/alphaXiv/sample-efficient-learning-from-agent-experience/tree/orx/lossless-branch-packing) | Preserve every target while packing branches | `bash scripts/run.sh` | 35.94 score; 71 instances, 45 steps; paired 95% interval includes no difference | 4 GPUs, 1m18s |
| [No-experience teacher](https://github.com/alphaXiv/sample-efficient-learning-from-agent-experience/tree/orx/no-experience-teacher-ablation) | Remove privileged experience from teacher | `bash scripts/run.sh` | 7.03 score; experience-derived gain erased | 4 GPUs, 1m23s |
| [Lower-LR direct SFT](https://github.com/alphaXiv/sample-efficient-learning-from-agent-experience/tree/orx/direct-sft-lower-learning-rate) | Halve adapter learning rate | `bash scripts/run.sh` | 45.57; ordering remained direct > distillation | 4 GPUs, 1m18s |
| [Lower-LR unpacked distillation](https://github.com/alphaXiv/sample-efficient-learning-from-agent-experience/tree/orx/unpacked-epd-lower-learning-rate) | Halve adapter learning rate | `bash scripts/run.sh` | 40.89; ordering remained direct > distillation | 4 GPUs, 1m23s |

## Re-run

The fixed command is:

```bash
bash scripts/run.sh
```

Each experiment branch changes only committed code/config. Kubernetes shape is defined in `.orx/k8s.yaml`; the script prints the trajectory hash, environment-interaction counts, per-replication scores, training instances/steps, timing, and a terminal summary.
