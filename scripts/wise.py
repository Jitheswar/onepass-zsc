"""WiSE-FT: interpolate fine-tuned encoder weights with the original backbone (head kept from fine-tuned).
python scripts/wise.py <ckpt_dir> <out_dir> <alpha>   (alpha = weight of fine-tuned model)"""
import os, shutil, sys
import torch
from transformers import AutoModel
ck, out, alpha = sys.argv[1], sys.argv[2], float(sys.argv[3])
import json
cfg = json.load(open(os.path.join(ck, "config.json")))
sd = torch.load(os.path.join(ck, "model.pt"), map_location="cpu")
base = AutoModel.from_pretrained(cfg["backbone"], torch_dtype=torch.float32).state_dict()
n = 0
for k, v in base.items():
    kk = "encoder." + k
    if kk in sd and sd[kk].shape == v.shape and v.is_floating_point():
        sd[kk] = alpha * sd[kk] + (1 - alpha) * v
        n += 1
    elif kk in sd and kk.endswith("word_embeddings.weight"):
        m = min(v.shape[0], sd[kk].shape[0])  # new marker tokens keep fine-tuned rows
        sd[kk][:m] = alpha * sd[kk][:m] + (1 - alpha) * v[:m]
        n += 1
shutil.copytree(ck, out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("model.pt", "eval.json", "calibration.json"))
torch.save(sd, os.path.join(out, "model.pt"))
print(f"interpolated {n} tensors alpha={alpha} -> {out}")
