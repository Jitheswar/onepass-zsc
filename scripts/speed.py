"""Throughput on the benchmark texts with each dataset's full label set (same setup as baseline_nli.py speeds).

python scripts/speed.py CKPT [--threads 16] [--n 50] [--device cpu]
"""
import argparse, json, os, resource, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from onepass import ZeroShotClassifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--bs", type=int, default=1)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
torch.set_num_threads(a.threads)
ROOT = os.path.join(os.path.dirname(__file__), "..")
clf = ZeroShotClassifier.load(a.ckpt, device=a.device)
spaces = json.load(open(os.path.join(ROOT, "data/spaces.json")))
rng = np.random.RandomState(0)
tot_t, tot_n = 0.0, 0
for fn in sorted(f for f in os.listdir(os.path.join(ROOT, "data/eval")) if f.endswith(".jsonl")):
    name = fn[:-6]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/eval", fn))]
    texts = [rows[i]["text"] for i in rng.choice(len(rows), a.n, replace=False)]
    labels = spaces["eval/" + name]
    clf.predict(texts[:2], labels, batch_size=a.bs)  # warm-up
    if a.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    clf.predict(texts, labels, batch_size=a.bs)
    if a.device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    tot_t += dt; tot_n += len(texts)
    print(f"{name:18s} {len(labels):3d} labels  {len(texts)/dt:7.2f} texts/s", flush=True)
print(f"OVERALL {tot_n/tot_t:.2f} texts/s  (bs={a.bs}, threads={a.threads}, device={a.device})  peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
if a.device == "cuda":
    print(f"peak GPU memory {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
