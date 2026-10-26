"""Build RESULTS.md from results/*.json (baselines) and our eval.json files.

python scripts/make_results.py --ours runs/X/last/eval.json [--ours_name onepass-base]
"""
import argparse
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
ap = argparse.ArgumentParser()
ap.add_argument("--ours", nargs="*", default=[])
ap.add_argument("--ours_name", nargs="*", default=[])
ap.add_argument("--ours_speed", nargs="*", default=[], help="texts/s per model, same order")
a = ap.parse_args()

NOTES = {
    "knowledgator/gliclass-v4-edge": "training mix overlaps public benchmarks (see its card): may be inflated",
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0": "non `-c`: trained on 28 public datasets incl. some of these tasks",
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0": "non `-c`: trained on 28 public datasets incl. some of these tasks",
}
PASSES = {"knowledgator": "1", "MoritzLaurer": "1 per label", "facebook": "1 per label", "cross-encoder": "1 per label"}

rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "results", "baseline_*.json"))):
    r = json.load(open(f))
    m = r["model"]
    rows.append({"name": m, "acc": r["mean"]["acc"], "f1": r["mean"]["f1"], "ece": r["mean"]["ece"],
                 "speed": r.get("texts_per_s"), "passes": PASSES.get(m.split("/")[0], "?"),
                 "per": {k: v["acc"] for k, v in r["per_dataset"].items()}, "note": NOTES.get(m, ""), "ours": False})
for i, f in enumerate(a.ours):
    r = json.load(open(f))
    name = a.ours_name[i] if i < len(a.ours_name) else f
    rows.append({"name": f"**{name} (ours)**", "acc": r["mean"]["raw"]["acc"], "f1": r["mean"]["raw"]["f1"],
                 "ece": r["mean"]["lodo"]["ece"], "ece_raw": r["mean"]["raw"]["ece"],
                 "speed": float(a.ours_speed[i]) if i < len(a.ours_speed) else None, "passes": "1",
                 "per": {k: v["raw"]["acc"] for k, v in r["per_dataset"].items()}, "note": "calibrated (ECE is leave-one-dataset-out)", "ours": True})
rows.sort(key=lambda r: -r["acc"])
datasets = sorted(rows[0]["per"])

out = ["# Benchmark results", "",
       "Zero-shot accuracy on **13 public datasets that were never used for training** our model",
       "(300 random test examples each, same examples for every model, all labels of each dataset offered at once).",
       "ECE = expected calibration error of the top prediction (lower is better). Baseline ECEs are on their raw",
       "probabilities; ours uses the shipped temperature, measured leave-one-dataset-out so the temperature never",
       "sees the dataset it is scored on. Speeds are CPU texts/second on this cluster and only roughly comparable.", "",
       "| model | forward passes | mean acc | macro-F1 | ECE | texts/s (CPU) | note |", "|---|---|---|---|---|---|---|"]
for r in rows:
    sp = f"{r['speed']:.1f}" if r["speed"] else "-"
    out.append(f"| {r['name']} | {r['passes']} | {r['acc']:.3f} | {r['f1']:.3f} | {r['ece']:.3f} | {sp} | {r['note']} |")
out += ["", "## Per-dataset accuracy", "",
        "| model | " + " | ".join(datasets) + " |", "|---|" + "---|" * len(datasets)]
for r in rows:
    out.append(f"| {r['name']} | " + " | ".join(f"{r['per'].get(d, float('nan')):.2f}" for d in datasets) + " |")
out += ["", "Held-out datasets: " + ", ".join(datasets) + ".",
        "Reproduce: `scripts/evaluate.py`, `scripts/baseline_nli.py`, `scripts/baseline_gliclass.py` (all use `--n 300`, seed 0)."]
open(os.path.join(ROOT, "RESULTS.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out[:20]))
