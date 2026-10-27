"""Uniform weight average of several checkpoints of the same run (model soup).

python scripts/soup.py OUT CKPT1 CKPT2 ...
"""
import os, shutil, sys
import torch
out, cks = sys.argv[1], sys.argv[2:]
sds = [torch.load(os.path.join(c, "model.pt"), map_location="cpu") for c in cks]
avg = {k: sum(sd[k].float() for sd in sds) / len(sds) if sds[0][k].is_floating_point() else sds[0][k] for k in sds[0]}
shutil.rmtree(out, ignore_errors=True)
shutil.copytree(cks[-1], out, ignore=shutil.ignore_patterns("model.pt", "*.json.bak", "eval.json", "dev.json", "calibration.json", ".queued"))
torch.save(avg, os.path.join(out, "model.pt"))
print("soup of", cks, "->", out)
