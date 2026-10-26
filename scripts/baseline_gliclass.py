"""Baseline: knowledgator GLiClass (also single-pass), same eval data as evaluate.py / baseline_nli.py.

python scripts/baseline_gliclass.py knowledgator/gliclass-base-v3.0 --n 300
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import ROOT, metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
torch.set_num_threads(a.threads)
model = GLiClassModel.from_pretrained(a.model)
tok = AutoTokenizer.from_pretrained(a.model, add_prefix_space=True)
pipe = ZeroShotClassificationPipeline(model, tok, classification_type="multi-label", device=a.device)
spaces = json.load(open(os.path.join(ROOT, "data/spaces.json")))
rng = np.random.RandomState(0)
res, total_t, total_n = {}, 0.0, 0
for fn in sorted(os.listdir(os.path.join(ROOT, "data/eval"))):
    name = fn[:-6]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/eval", fn))]
    if len(rows) > a.n:
        rows = [rows[i] for i in rng.choice(len(rows), a.n, replace=False)]
    labels = spaces["eval/" + name]
    idx = {l: i for i, l in enumerate(labels)}
    y = np.array([idx[r["gold"][0]] for r in rows])
    t0 = time.time()
    out = []
    for i in range(0, len(rows), 8):
        preds = pipe([r["text"] for r in rows[i:i + 8]], labels, threshold=0.0)
        for p in preds:
            sc = {d["label"]: d["score"] for d in p}
            pr = np.clip(np.array([sc.get(l, 1e-6) for l in labels]), 1e-6, 1 - 1e-6)
            out.append(np.log(pr / (1 - pr)))  # back to logits; softmax over them for single-label metrics
    total_t += time.time() - t0
    total_n += len(rows)
    lg = np.stack(out)
    res[name] = metrics(lg, y, 1.0)
    print(f"{name:18s} acc {res[name]['acc']:.3f} f1 {res[name]['f1']:.3f} ece {res[name]['ece']:.3f}", flush=True)
mean = {k: float(np.mean([r[k] for r in res.values()])) for k in ["acc", "f1", "ece", "nll"]}
print("MEAN", mean, f"texts/s {total_n/total_t:.2f}")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump({"model": a.model, "per_dataset": res, "mean": mean, "texts_per_s": total_n / total_t, "device": a.device},
          open(os.path.join(ROOT, "results", "baseline_" + a.model.replace("/", "_") + ".json"), "w"), indent=1)
