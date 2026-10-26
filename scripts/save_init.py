"""Save an untrained one-pass model (for step-0 evaluation of an initialisation)."""
import json, os, sys
import torch
from transformers import AutoConfig
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from onepass.model import build
bb, out, nli = sys.argv[1], sys.argv[2], "--nli_head" in sys.argv
layout = "packed" if "--packed" in sys.argv else ("text_first" if "--text_first" in sys.argv else "labels_first")
template = "This text is about {}." if "--template" in sys.argv else None
torch.manual_seed(0)
model, tok, _ = build(bb, nli_head=nli, layout=layout)
os.makedirs(out, exist_ok=True)
torch.save(model.state_dict(), os.path.join(out, "model.pt"))
tok.save_pretrained(out)
c = AutoConfig.from_pretrained(bb) if (nli or layout == "packed") else model.encoder.config
c.vocab_size = model.encoder.config.vocab_size
c.save_pretrained(os.path.join(out, "encoder"))
json.dump({"backbone": bb, "max_len": 320, "nli_head": nli, "layout": layout, "template": template}, open(os.path.join(out, "config.json"), "w"))
print("saved", out)
