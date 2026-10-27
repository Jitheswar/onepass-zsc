# Benchmark results

Zero-shot accuracy on **13 public datasets that were never used for training** our model
(300 random test examples each, same examples for every model, all labels of each dataset offered at once).
ECE = expected calibration error of the top prediction (lower is better). Baseline ECEs are on their raw
probabilities; ours uses the shipped calibration, which was fitted on a separate dev set of 8 other unseen
datasets (never on this benchmark). Speeds are CPU texts/second on this cluster and only roughly comparable.

| model | forward passes | mean acc | macro-F1 | ECE | texts/s (CPU) | note |
|---|---|---|---|---|---|---|
| knowledgator/gliclass-v4-edge | 1 | 0.837 | 0.800 | 0.063 | 2.7 | training mix overlaps public benchmarks (see its card): may be inflated |
| MoritzLaurer/deberta-v3-large-zeroshot-v2.0 | 1 per label | 0.762 | 0.717 | 0.104 | 2.7 | non `-c`: trained on 28 public datasets incl. some of these tasks |
| MoritzLaurer/deberta-v3-base-zeroshot-v2.0 | 1 per label | 0.759 | 0.720 | 0.113 | 6.1 | non `-c`: trained on 28 public datasets incl. some of these tasks |
| knowledgator/gliclass-large-v3.0 | 1 | 0.703 | 0.653 | 0.205 | 5.5 |  |
| MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c | 1 per label | 0.697 | 0.659 | 0.123 | 2.4 |  |
| **onepass-zsc-base (ours)** | 1 | 0.682 | 0.622 | 0.134 | 12.8 | calibration fitted on separate dev set |
| MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c | 1 per label | 0.660 | 0.599 | 0.150 | 8.3 |  |
| knowledgator/gliclass-base-v3.0 | 1 | 0.658 | 0.597 | 0.172 | 14.7 |  |
| facebook/bart-large-mnli | 1 per label | 0.637 | 0.554 | 0.117 | 3.7 |  |
| cross-encoder/nli-deberta-v3-xsmall | 1 per label | 0.617 | 0.551 | 0.134 | 5.5 |  |
| knowledgator/gliclass-edge-v3.0 | 1 | 0.526 | 0.480 | 0.208 | 88.3 |  |

## Per-dataset accuracy

| model | 20newsgroups | banking77 | bbc_news | climate | cr | emotion | enron_spam | fin_topic | hate_offensive | massive_scenario | rotten_tomatoes | sms_spam | sst2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knowledgator/gliclass-v4-edge | 0.65 | 0.91 | 0.95 | 0.75 | 0.94 | 0.73 | 0.89 | 0.61 | 0.89 | 0.76 | 0.89 | 0.97 | 0.94 |
| MoritzLaurer/deberta-v3-large-zeroshot-v2.0 | 0.57 | 0.67 | 0.95 | 0.63 | 0.92 | 0.66 | 0.84 | 0.40 | 0.86 | 0.68 | 0.86 | 0.93 | 0.94 |
| MoritzLaurer/deberta-v3-base-zeroshot-v2.0 | 0.55 | 0.65 | 0.94 | 0.57 | 0.92 | 0.72 | 0.74 | 0.51 | 0.87 | 0.67 | 0.85 | 0.95 | 0.93 |
| knowledgator/gliclass-large-v3.0 | 0.61 | 0.56 | 0.91 | 0.90 | 0.94 | 0.51 | 0.57 | 0.45 | 0.71 | 0.58 | 0.84 | 0.65 | 0.91 |
| MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c | 0.55 | 0.55 | 0.92 | 0.60 | 0.92 | 0.49 | 0.77 | 0.44 | 0.73 | 0.53 | 0.79 | 0.91 | 0.87 |
| **onepass-zsc-base (ours)** | 0.54 | 0.57 | 0.90 | 0.58 | 0.92 | 0.47 | 0.50 | 0.53 | 0.60 | 0.63 | 0.84 | 0.88 | 0.91 |
| MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c | 0.55 | 0.51 | 0.89 | 0.53 | 0.92 | 0.43 | 0.51 | 0.48 | 0.66 | 0.54 | 0.80 | 0.89 | 0.88 |
| knowledgator/gliclass-base-v3.0 | 0.60 | 0.47 | 0.89 | 0.78 | 0.91 | 0.48 | 0.59 | 0.39 | 0.76 | 0.56 | 0.82 | 0.44 | 0.86 |
| facebook/bart-large-mnli | 0.38 | 0.44 | 0.71 | 0.82 | 0.92 | 0.41 | 0.41 | 0.47 | 0.73 | 0.45 | 0.77 | 0.87 | 0.89 |
| cross-encoder/nli-deberta-v3-xsmall | 0.42 | 0.37 | 0.82 | 0.78 | 0.90 | 0.38 | 0.53 | 0.32 | 0.60 | 0.46 | 0.77 | 0.83 | 0.85 |
| knowledgator/gliclass-edge-v3.0 | 0.32 | 0.06 | 0.82 | 0.74 | 0.83 | 0.44 | 0.54 | 0.15 | 0.63 | 0.27 | 0.72 | 0.53 | 0.79 |

Held-out datasets: 20newsgroups, banking77, bbc_news, climate, cr, emotion, enron_spam, fin_topic, hate_offensive, massive_scenario, rotten_tomatoes, sms_spam, sst2.
Reproduce: `scripts/evaluate.py`, `scripts/baseline_nli.py`, `scripts/baseline_gliclass.py` (all use `--n 300`, seed 0).

## How the shipped checkpoint was chosen

Run `packed_v4` saved checkpoints every 400 steps. Selection used only the dev set (8 other unseen datasets):
all checkpoints were within 0.5 points of the best dev accuracy (66.9-67.3%), so, by the rule fixed before
looking at the final results, the one with the lowest dev log-loss was shipped (step 800, dev accuracy 67.0%).
Later checkpoints scored up to 68.8% on this benchmark, but the benchmark was never used for selection.
Calibration constants (`results/onepass_final_calibration.json`) were also fitted on dev only.

Speed: 16 CPU threads, one text per call, full label set of each dataset (`scripts/speed.py`). Peak RAM 2.2 GB
(fp32); fp16 weights are ~0.37 GB, well within an RTX 3050 (6 GB).
