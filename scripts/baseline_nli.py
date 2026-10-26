"""Baseline: classic NLI zero-shot (one forward pass per label), same eval data.

python scripts/baseline_nli.py cross-encoder/nli-deberta-v3-xsmall --n 300
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import ROOT, metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--template", default="This example is {}.")
a = ap.parse_args()
torch.set_num_threads(a.threads)
tok = AutoTokenizer.from_pretrained(a.model)
m = AutoModelForSequenceClassification.from_pretrained(a.model, torch_dtype=torch.float32).eval()
ent = [i for i, l in m.config.id2label.items() if l.lower().startswith("entail")][0]
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
    hyps = [a.template.format(l) for l in labels]
    out = []
    t0 = time.time()
    with torch.inference_mode():
        for r in rows:
            sc = []
            for i in range(0, len(hyps), 64):
                h = hyps[i:i + 64]
                b = tok([r["text"]] * len(h), h, truncation="only_first", max_length=256, padding=True, return_tensors="pt")
                sc.append(m(**b).logits[:, ent])
            out.append(torch.cat(sc))
    total_t += time.time() - t0
    total_n += len(rows)
    lg = torch.stack(out).numpy()
    res[name] = metrics(lg, y, 1.0)
    print(f"{name:18s} acc {res[name]['acc']:.3f} f1 {res[name]['f1']:.3f} ece {res[name]['ece']:.3f}", flush=True)
mean = {k: float(np.mean([r[k] for r in res.values()])) for k in ["acc", "f1", "ece", "nll"]}
print("MEAN", mean, f"texts/s {total_n/total_t:.2f}")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump({"model": a.model, "per_dataset": res, "mean": mean, "texts_per_s": total_n / total_t},
          open(os.path.join(ROOT, "results", "baseline_" + a.model.replace("/", "_") + ".json"), "w"), indent=1)
