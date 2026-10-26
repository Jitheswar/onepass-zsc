"""Zero-shot evaluation + calibration fitting.

python scripts/evaluate.py CKPT_DIR [--n 500] [--threads 16] [--write_calib]

Reports per held-out dataset: accuracy, macro-F1, ECE and NLL with
 - raw logits (T=1)
 - temperature fitted on held-in calibration rows
 - leave-one-dataset-out temperature (fit on the other held-out sets): honest zero-shot calibration
Also fits multi-label (sigmoid) Platt scaling and reports its binary ECE.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from onepass import ZeroShotClassifier  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def ece(probs, y, bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def binary_ece(p, y, bins=15):
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def fit_T(logit_list, y_list):
    lg = [torch.tensor(l) for l in logit_list]
    ys = [torch.tensor(y) for y in y_list]
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        # average NLL per dataset so big label spaces don't dominate
        loss = sum(F.cross_entropy(l / logT.exp(), y) for l, y in zip(lg, ys)) / len(lg)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp())


def fit_platt(logit_list, y_list):
    xs, ts = [], []
    for l, y in zip(logit_list, y_list):
        oh = np.zeros_like(l)
        oh[np.arange(len(y)), y] = 1
        xs.append(torch.tensor(l.ravel()))
        ts.append(torch.tensor(oh.ravel()))
    a = torch.ones(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = sum(F.binary_cross_entropy_with_logits(a * x + b, t) for x, t in zip(xs, ts)) / len(xs)
        loss.backward()
        return loss
    opt.step(closure)
    return float(a), float(b)


def metrics(logits, y, T):
    p = torch.softmax(torch.tensor(logits) / T, -1).numpy()
    pred = p.argmax(1)
    nll = float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())
    return {"acc": float((pred == y).mean()), "f1": float(f1_score(y, pred, average="macro")),
            "ece": float(ece(p, y)), "nll": nll}


def run(clf, rows, labels, bs):
    idx = {l: i for i, l in enumerate(labels)}
    texts = [r["text"] for r in rows]
    y = np.array([idx[r["gold"][0]] for r in rows])
    out = []
    for i in range(0, len(texts), bs):
        out.append(clf.logits(texts[i: i + bs], labels))
    return torch.cat(out).numpy(), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--write_calib", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="eval", choices=["eval", "dev"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rng = np.random.RandomState(0)
    clf = ZeroShotClassifier.load(a.ckpt, device=a.device, max_len=a.max_len, dtype=torch.float32)
    clf.label_chunk = a.chunk
    spaces = json.load(open(os.path.join(ROOT, "data/spaces.json")))

    t0 = time.time()
    # held-in calibration rows (single-label only), grouped by source
    calib = {}
    for l in open(os.path.join(ROOT, "data/calib.jsonl")):
        r = json.loads(l)
        if len(r["gold"]) == 1:
            calib.setdefault(r["src"], []).append(r)
    cal_logits, cal_y = [], []
    for src, rows in calib.items():
        rows = rows[:100]
        lg, y = run(clf, rows, spaces[src], a.bs)
        cal_logits.append(lg)
        cal_y.append(y)
    T_in = fit_T(cal_logits, cal_y)

    sdir = os.path.join(ROOT, "data", a.split)
    names = sorted(f for f in os.listdir(sdir) if f.endswith(".jsonl"))
    ev = {}
    for fn in names:
        name = fn[:-6]
        rows = [json.loads(l) for l in open(os.path.join(sdir, fn))]
        if len(rows) > a.n:
            rows = [rows[i] for i in rng.choice(len(rows), a.n, replace=False)]
        labels = spaces["eval/" + name] if a.split == "eval" else json.load(open(os.path.join(sdir, name + ".labels.json")))
        ev[name] = run(clf, rows, labels, a.bs)
    el = time.time() - t0

    cpath = os.path.join(a.ckpt, "calibration.json")
    shipped_T = json.load(open(cpath))["softmax_T"] if os.path.exists(cpath) and a.split == "eval" else 1.0
    T_all = fit_T([v[0] for v in ev.values()], [v[1] for v in ev.values()])
    pa, pb = fit_platt([v[0] for v in ev.values()], [v[1] for v in ev.values()])
    res = {}
    for name, (lg, y) in ev.items():
        others = [k for k in ev if k != name]
        T_lodo = fit_T([ev[k][0] for k in others], [ev[k][1] for k in others])
        oa, ob = fit_platt([ev[k][0] for k in others], [ev[k][1] for k in others])
        sig = 1 / (1 + np.exp(-(oa * lg + ob)))
        oh = np.zeros_like(lg)
        oh[np.arange(len(y)), y] = 1
        res[name] = {"n": len(y), "labels": lg.shape[1], "raw": metrics(lg, y, 1.0), "T_in": metrics(lg, y, T_in),
                     "shipped": metrics(lg, y, shipped_T),
                     "lodo": metrics(lg, y, T_lodo), "T_lodo": T_lodo,
                     "sigmoid_ece_lodo": float(binary_ece(sig.ravel(), oh.ravel()))}
    keys = ["raw", "T_in", "lodo", "shipped"]
    print(f"\nckpt={a.ckpt}  T_in={T_in:.3f}  T_all={T_all:.3f}  platt a={pa:.3f} b={pb:.3f}  ({el/60:.1f} min)")
    print(f"{'dataset':18s} {'n':>5s} {'#lab':>4s} {'acc':>6s} {'F1':>6s} | ECE raw / T_in / LODO / shipped | sigECE")
    for name, r in res.items():
        print(f"{name:18s} {r['n']:5d} {r['labels']:4d} {r['raw']['acc']:6.3f} {r['raw']['f1']:6.3f} | "
              + " / ".join(f"{r[k]['ece']:.3f}" for k in keys) + f" | {r['sigmoid_ece_lodo']:.3f}")
    avg = {k: {m: float(np.mean([r[k][m] for r in res.values()])) for m in ["acc", "f1", "ece", "nll"]} for k in keys}
    print(f"{'MEAN':18s} {'':5s} {'':4s} {avg['raw']['acc']:6.3f} {avg['raw']['f1']:6.3f} | "
          + " / ".join(f"{avg[k]['ece']:.3f}" for k in keys)
          + f" | {np.mean([r['sigmoid_ece_lodo'] for r in res.values()]):.3f}")
    summary = {"ckpt": a.ckpt, "shipped_T": shipped_T, "T_in": T_in, "T_all": T_all, "platt": [pa, pb], "per_dataset": res, "mean": avg}
    out = a.out or os.path.join(a.ckpt, "eval.json" if a.split == "eval" else "dev.json")
    json.dump(summary, open(out, "w"), indent=1)
    if a.write_calib:
        json.dump({"softmax_T": T_all, "sigmoid_a": pa, "sigmoid_b": pb,
                   "note": f"fit on {a.split} split zero-shot logits"},
                  open(os.path.join(a.ckpt, "calibration.json"), "w"), indent=1)
        print("wrote calibration.json")


if __name__ == "__main__":
    main()
