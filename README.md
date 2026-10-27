# onepass-zsc

Zero-shot text classification in **one forward pass**: give it any text and any list of labels (or several
label questions at once, Jev-style) and get **calibrated probabilities** back. No task-specific training.

```python
from onepass import ZeroShotClassifier

clf = ZeroShotClassifier.load("path/to/model")     # CUDA + fp16 automatically when a GPU is present

clf("The battery died after two days.", ["positive", "negative", "neutral"])
# {'negative': 0.97, 'neutral': 0.02, 'positive': 0.01}

# several questions about the same text, still one forward pass
clf.ask("My card was charged twice for one coffee.",
        {"topic": ["billing", "delivery", "account access"],
         "sentiment": ["positive", "negative", "neutral"]})

# independent per-label probabilities (multi-label)
clf("Server is down and I was billed twice", ["outage", "billing", "feature request"], multi_label=True)
```

CLI: `python -m onepass --model path/to/model "I loved it" -l positive negative`

## How it works

Classic NLI zero-shot classification runs the model once per label (`text + "This text is about {label}."`),
so 77 labels cost 77 forward passes. onepass packs every hypothesis into **one sequence**:

```
[CLS] text [SEP] | [CLS] This text is about label_1. [SEP] | [CLS] This text is about label_2. [SEP] | ...
```

* a **block attention mask** lets each hypothesis block see the text and itself, but never other labels,
  so labels cannot interfere and the text is encoded once;
* every hypothesis block gets the **same virtual positions** (DeBERTa relative positions), so each block looks
  to the model exactly like an ordinary NLI (premise, hypothesis) pair;
* each block's `[CLS]` goes through the backbone's NLI head; the entailment-vs-rest log-odds is the label's score;
* the model is then fine-tuned in this packed format on ~590k examples (Laurer-style mix): 66% synthetic zero-shot
  tasks (`MoritzLaurer/synthetic_zeroshot_mixtral_v0.1`), 22% MNLI, 13% from 30 classification datasets capped at
  500 examples per class and 10k per dataset, so the model learns the *format* without overfitting any label set;
* **calibration**: single-label probabilities use a temperature and a small mix with the uniform distribution that
  both depend on the number of labels; multi-label uses Platt scaling. Everything is fitted on a separate
  zero-shot dev set of 8 unseen datasets, never on the benchmark.

Backbone: DeBERTa-v3-base initialised from `MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c`, which was trained
only on synthetic data + MNLI/FEVER, so none of the benchmark datasets leak into training.

## Hardware

~185M parameters: ~0.37 GB of weights in fp16. Runs comfortably on an RTX 3050 (6 GB VRAM) / 16 GB RAM laptop,
and on CPU (peak ~2.2 GB RAM in fp32). On 16 CPU threads it classifies 12.8 texts/s with each benchmark's full
label set, 1.5x faster than running the same backbone once per label (8.3 texts/s).

## Evaluation protocol

* **Benchmark** (`data/eval`): 13 public datasets never used for training (SST-2, Rotten Tomatoes, CR, BBC News,
  20 Newsgroups, Banking77, MASSIVE scenario, Enron spam, SMS spam, Emotion, hate/offensive, financial tweet
  topics, climate detection). Every model sees the same 300 examples per dataset with the full label set.
* **Dev set** (`data/dev`): 8 other unseen datasets (ADE, insincere questions, poem sentiment, news categories,
  tweet emotion, arXiv subjects, SNIPS, medical specialty). Used for every modelling decision and for fitting
  the shipped calibration, so the benchmark is never used for selection.
* Benchmark ECE uses the shipped (dev-fitted) calibration; leave-one-dataset-out numbers are in `eval.json`.

See [RESULTS.md](RESULTS.md) for the full comparison with bart-large-mnli, MoritzLaurer zeroshot v2.0 and
GLiClass models.

## Reproduce

```bash
python scripts/build_data.py                    # training data, benchmark, label spaces
python scripts/build_dev.py                     # zero-shot dev set
sbatch -N 8 slurm/train_cpu_multi.sbatch packed_v4 MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c \
    --layout packed --no_mlp --data data/train_v4.jsonl --steps 2800 --bs 8 --accum 2 --lr 1e-5 ...
python scripts/evaluate.py runs/packed_v4/last --split dev --write_calib   # calibration from dev
python scripts/evaluate.py runs/packed_v4/last --n 300                      # benchmark
python scripts/make_results.py --ours runs/packed_v4/last/eval.json
```

## Lessons learned

* `MoritzLaurer/synthetic_zeroshot_mixtral_v0.1` uses the NLI convention **0 = entailment**. Reading 1 as "true"
  silently inverts two thirds of the training data; the model then learns to ignore the text and collapses on
  unseen topics (e.g. BBC News 82% -> 18%).
* A learned residual head on pooled label features picks up label-only priors; scoring with the NLI head alone
  (`--no_mlp`) keeps the model text-conditioned.
* Large supervised datasets (100k+ rows each) cause catastrophic forgetting of zero-shot ability; capping them
  per class and leaning on synthetic + NLI data fixes it.
