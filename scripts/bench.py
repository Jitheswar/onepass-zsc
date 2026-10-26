import sys, time, torch
sys.path.insert(0, ".")
from onepass.model import build
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 32)
name = sys.argv[1]
model, tok, enc = build(name, max_len=256)
texts = ["The quick brown fox jumps over the lazy dog and keeps running across the field. " * 6] * 16
labels = [["sports", "politics", "science and technology", "business", "world news", "entertainment", "health", "travel"]] * 16
b = enc(texts, labels)
opt = torch.optim.AdamW(model.parameters(), 1e-5)
n_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6
for i in range(4):
    if i == 1: t = time.time()
    loss = model(**b).mean(); loss.backward(); opt.step(); opt.zero_grad()
train = (time.time() - t) / 3
model.eval()
with torch.no_grad():
    t = time.time()
    for _ in range(3): model(**b)
    inf = (time.time() - t) / 3
print(f"{name}: params={n_params:.0f}M seq={b['input_ids'].shape[1]} train {16/train:.1f} ex/s  infer {16/inf:.1f} ex/s")
