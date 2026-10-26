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
* each block's `[CLS]` goes through the NLI entailment head, plus a small learned residual head;
* the model is then fine-tuned in this packed format on ~1.8M examples from 24 classification datasets,
  MNLI and synthetic zero-shot tasks;
* **calibration**: a softmax temperature (single-label) and Platt scaling (multi-label), fitted on a separate
  zero-shot dev set of 8 unseen datasets.

Backbone: DeBERTa-v3-base initialised from `MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c`, which was trained
only on synthetic data + MNLI/FEVER, so none of the benchmark datasets leak into training.

## Hardware

~185M parameters: ~0.4 GB in fp16. Runs comfortably on an RTX 3050 (6 GB VRAM) / 16 GB RAM laptop,
and on CPU.

## Evaluation protocol

* **Benchmark** (`data/eval`): 13 public datasets never used for training (SST-2, Rotten Tomatoes, CR, BBC News,
  20 Newsgroups, Banking77, MASSIVE scenario, Enron spam, SMS spam, Emotion, hate/offensive, financial tweet
  topics, climate detection). Every model sees the same 300 examples per dataset with the full label set.
* **Dev set** (`data/dev`): 8 other unseen datasets (ADE, insincere questions, poem sentiment, news categories,
  tweet emotion, arXiv subjects, SNIPS, medical specialty). Used for every modelling decision and for fitting
  the shipped calibration, so the benchmark is never used for selection.
* Calibration on the benchmark is reported leave-one-dataset-out.

See [RESULTS.md](RESULTS.md) for the full comparison with bart-large-mnli, MoritzLaurer zeroshot v2.0 and
GLiClass models.

## Reproduce

```bash
python scripts/build_data.py                    # training data, benchmark, label spaces
python scripts/build_dev.py                     # zero-shot dev set
sbatch -N 8 slurm/train_cpu_multi.sbatch packed_v1 MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c --layout packed ...
python scripts/evaluate.py runs/packed_v1/last --split dev --write_calib   # calibration from dev
python scripts/evaluate.py runs/packed_v1/last --n 300                      # benchmark
python scripts/make_results.py --ours runs/packed_v1/last/eval.json
```
