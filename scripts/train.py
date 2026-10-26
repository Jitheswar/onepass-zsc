import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from onepass.model import build  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--backbone", default="cross-encoder/nli-deberta-v3-xsmall")
p.add_argument("--out", required=True)
p.add_argument("--threads", type=int, default=60)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--head_lr", type=float, default=5e-4)
p.add_argument("--max_len", type=int, default=256)
p.add_argument("--max_labels", type=int, default=24)
p.add_argument("--steps", type=int, default=40000)
p.add_argument("--hours", type=float, default=6.0)
p.add_argument("--save_every", type=int, default=2000)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

import torch.distributed as dist
RANK = int(os.environ.get("RANK", 0)); WORLD = int(os.environ.get("WORLD_SIZE", 1))
if WORLD > 1:
    dist.init_process_group("gloo")
args.seed += 1000 * RANK
torch.set_num_threads(args.threads)
torch.manual_seed(0)  # same init on every rank
random.seed(args.seed)
ROOT = os.path.join(os.path.dirname(__file__), "..")
spaces = json.load(open(os.path.join(ROOT, "data/spaces.json")))


def vary(label):
    r = random.random()
    if r < 0.1:
        return label.capitalize()
    if r < 0.15:
        return label.title()
    return label


def make_example(row):
    """Return (text, candidate labels, targets) with randomized label sets."""
    text, gold = row["text"], list(dict.fromkeys(row["gold"]))
    if row["src"] == "mnli":
        cands = row["cands"]
    else:
        space = spaces[row["src"]]
        negs = [l for l in space if l not in gold]
        room = max(1, args.max_labels - len(gold))
        if random.random() < 0.5:
            k = min(len(negs), room)
        else:
            k = random.randint(1, max(1, min(len(negs), room)))
        cands = gold + random.sample(negs, min(k, len(negs)))
        # sometimes drop the gold label(s) so the model learns "none of these"
        if random.random() < 0.1 and len(cands) > len(gold) + 1:
            cands = cands[len(gold):]
    cands = list(dict.fromkeys(cands))[: args.max_labels]
    random.shuffle(cands)
    gs = set(gold)
    y = [1.0 if c in gs else 0.0 for c in cands]
    return text, [vary(c) for c in cands], y


class Stream(IterableDataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path)]

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(args.seed + (info.id if info else 0) + int(time.time()))
        random.seed(rng.random())
        while True:
            idx = list(range(len(self.rows)))
            rng.shuffle(idx)
            for i in idx:
                yield make_example(self.rows[i])


model, tok, enc = build(args.backbone, max_len=args.max_len)


def collate(batch):
    texts, labels, ys = zip(*batch)
    b = enc(list(texts), list(labels))
    N = b["label_valid"].shape[1]
    y = torch.zeros(len(ys), N)
    for i, yy in enumerate(ys):
        y[i, : len(yy)] = torch.tensor(yy)
    b["y"] = y
    return b


loader = DataLoader(Stream(os.path.join(ROOT, "data/train.jsonl")), batch_size=args.bs, collate_fn=collate, num_workers=3, prefetch_factor=8,
                    worker_init_fn=lambda _: torch.set_num_threads(1))

head_params = list(model.head.parameters())
core = model
if WORLD > 1:
    model = torch.nn.parallel.DistributedDataParallel(model)
    core = model.module
opt = torch.optim.AdamW([
    {"params": core.encoder.parameters(), "lr": args.lr},
    {"params": head_params, "lr": args.head_lr},
], weight_decay=0.01)
warm = 500


def lr_mult(step):
    if step < warm:
        return step / warm
    return max(0.05, 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, args.steps - warm))))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
os.makedirs(args.out, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
log = open(os.path.join(args.out, "log.txt"), "a")


def save(tag):
    if RANK != 0:
        return
    d = os.path.join(args.out, tag)
    os.makedirs(d, exist_ok=True)
    torch.save(core.state_dict(), os.path.join(d, "model.pt"))
    tok.save_pretrained(d)
    json.dump({"backbone": args.backbone, "max_len": args.max_len}, open(os.path.join(d, "config.json"), "w"))


model.train()
t0 = time.time()
ema = None
for step, b in enumerate(loader, 1):
    y = b.pop("y")
    valid = b["label_valid"]
    logits = model(**b)
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    bce = (bce * valid).sum() / valid.sum()
    single = (y.sum(1) == 1)
    if single.any():
        ce = F.cross_entropy(logits[single], y[single].argmax(1))
    else:
        ce = torch.zeros(())
    loss = bce + 0.5 * ce
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    seq_sum = seq_sum + b["input_ids"].shape[1] if step > 1 else b["input_ids"].shape[1]
    ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
    if step % 50 == 0 and RANK == 0:
        el = time.time() - t0
        msg = f"step {step} loss {ema:.4f} bce {bce.item():.4f} ce {ce.item():.4f} lr {sched.get_last_lr()[0]:.2e} ex/s {step*args.bs*WORLD/el:.1f} elapsed {el/60:.1f}m seq {seq_sum/step:.0f}"
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
    if step % args.save_every == 0:
        save("last")
    stop = torch.tensor(float(step >= args.steps or time.time() - t0 > args.hours * 3600))
    if WORLD > 1:
        dist.all_reduce(stop)
    if stop.item() > 0:
        break
save("last")
print("done", flush=True)
