"""python -m onepass --model checkpoints/onepass-xs "I loved it" -l positive negative [--multi]"""
import argparse
import time

from .classifier import ZeroShotClassifier

ap = argparse.ArgumentParser(prog="onepass")
ap.add_argument("text")
ap.add_argument("-l", "--labels", nargs="+", required=True)
ap.add_argument("--model", default="checkpoints/onepass-xs")
ap.add_argument("--multi", action="store_true", help="independent per-label probabilities")
ap.add_argument("--device", default=None)
a = ap.parse_args()
clf = ZeroShotClassifier.load(a.model, device=a.device)
t = time.time()
res = clf(a.text, a.labels, multi_label=a.multi)
ms = (time.time() - t) * 1000
for k, v in res.items():
    print(f"{v:7.3f}  {k}")
print(f"({ms:.0f} ms, one forward pass, device={clf.device})")
