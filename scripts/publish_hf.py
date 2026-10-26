"""Package a checkpoint and upload it to the Hugging Face Hub (private), only if it passes the quality gate.

python scripts/publish_hf.py runs/deb_xs/last --repo Jitheswar/onepass-zsc-xsmall [--dry_run]

Gate (on held-out zero-shot benchmarks, see scripts/evaluate.py):
  * mean accuracy above the strongest leak-free baseline (its own multi-pass starting checkpoint)
  * mean ECE with the shipped (dev-fitted) calibration <= 0.08
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import torch
from safetensors.torch import save_file

ROOT = os.path.join(os.path.dirname(__file__), "..")
ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--repo", required=True)
ap.add_argument("--min_gain", type=float, default=0.0)
ap.add_argument("--max_ece", type=float, default=0.08)
ap.add_argument("--baseline", default="results/baseline_MoritzLaurer_deberta-v3-base-zeroshot-v2.0-c.json",
                help="strongest leak-free baseline the model must beat")
ap.add_argument("--dry_run", action="store_true")
a = ap.parse_args()

ev = json.load(open(os.path.join(a.ckpt, "eval.json")))
base = json.load(open(os.path.join(ROOT, a.baseline)))
acc, ece = ev["mean"]["raw"]["acc"], ev["mean"]["shipped"]["ece"]
b_acc = base["mean"]["acc"]
print(f"model acc {acc:.3f} vs {base['model']} {b_acc:.3f} (need > +{a.min_gain}); shipped-calibration ECE {ece:.3f} (need <= {a.max_ece})")
if acc <= b_acc + a.min_gain or ece > a.max_ece:
    sys.exit("QUALITY GATE FAILED - not uploading")
if not os.path.exists(os.path.join(a.ckpt, "calibration.json")):
    sys.exit("missing calibration.json - run evaluate.py --write_calib first")

stage = os.path.join(ROOT, "dist", a.repo.split("/")[-1])
shutil.rmtree(stage, ignore_errors=True)
os.makedirs(stage)
# only the final model files: no logs, eval dumps, training args or code
KEEP = {"config.json", "calibration.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "spm.model", "vocab.txt", "added_tokens.json"}
for f in os.listdir(a.ckpt):
    if f in KEEP:
        shutil.copy(os.path.join(a.ckpt, f), os.path.join(stage, f))
shutil.copytree(os.path.join(a.ckpt, "encoder"), os.path.join(stage, "encoder"))
sd = torch.load(os.path.join(a.ckpt, "model.pt"), map_location="cpu")
save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(stage, "model.safetensors"))

rows = "\n".join(f"| {k} | {v['labels']} | {v['raw']['acc']:.3f} | {base['per_dataset'].get(k, {}).get('acc', float('nan')):.3f} | {v['shipped']['ece']:.3f} |"
                 for k, v in ev["per_dataset"].items())
card = f"""---
license: mit
language: en
pipeline_tag: zero-shot-classification
tags: [zero-shot-classification, text-classification, calibration]
---
# {a.repo.split('/')[-1]}

Zero-shot text classifier: any text + any list of labels -> calibrated probabilities in **one forward pass**.
All candidate labels and the text are encoded together (`[CLS] [L] label1 [L] label2 ... [T] text [SEP]`),
so cost barely grows with the number of labels. Backbone: `{json.load(open(os.path.join(a.ckpt, 'config.json')))['backbone']}`.

## Usage
```python
# code: github.com/Jitheswar/onepass-zsc  (pip install torch transformers safetensors huggingface_hub)
from huggingface_hub import snapshot_download
from onepass import ZeroShotClassifier
path = snapshot_download("{a.repo}")
clf = ZeroShotClassifier.load(path)          # uses CUDA fp16 when available
clf("The battery died after two days.", ["positive", "negative", "neutral"])
clf("Server is down and I was charged twice", ["billing", "outage", "feature request"], multi_label=True)
```

## Held-out zero-shot results
None of these datasets were used in training. Accuracy uses raw scores; ECE uses the shipped temperature,
fitted on a separate dev set of 8 other unseen datasets.

| dataset | #labels | accuracy | starting checkpoint (1 pass/label) | ECE |
|---|---|---|---|---|
{rows}
| **mean** | | **{acc:.3f}** | **{b_acc:.3f}** | **{ece:.3f}** |

Calibration: single-label softmax temperature T={ev['T_all']:.3f}; multi-label sigmoid Platt a={ev['platt'][0]:.3f}, b={ev['platt'][1]:.3f}.
"""
open(os.path.join(stage, "README.md"), "w").write(card)
print("staged at", stage)
if a.dry_run:
    sys.exit(0)
env = {k: v for k, v in os.environ.items() if k != "HF_HOME"}
subprocess.run(["hf", "repos", "create", a.repo, "--private", "--exist-ok"], check=True, env=env)
subprocess.run(["hf", "upload", a.repo, stage, ".", "--private", "--commit-message", "Upload one-pass zero-shot classifier"], check=True, env=env)
print("uploaded https://huggingface.co/" + a.repo)
