import argparse
import contextlib
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
p.add_argument("--data", default="data/train.jsonl")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--workers", type=int, default=3)
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--keep_every", type=int, default=0, help="also keep a copy of every Nth-step checkpoint")
p.add_argument("--alpha", type=float, default=0.5, help="source sampling weight = size^alpha")
p.add_argument("--avg_every", type=int, default=0, help="multi-node: average weights across nodes every N steps (local SGD); 0 = sync every step")
p.add_argument("--log_every", type=int, default=50)
p.add_argument("--accum", type=int, default=1, help="micro-batches per optimizer step")
p.add_argument("--mix_prob", type=float, default=0.15)
p.add_argument("--extra_negs", type=int, default=6, help="max random distractors added to rows with fixed candidates")
args = p.parse_args()

import torch.distributed as dist
if "SLURM_PROCID" in os.environ and "RANK" not in os.environ:  # launched with srun across nodes
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
RANK = int(os.environ.get("RANK", 0)); WORLD = int(os.environ.get("WORLD_SIZE", 1))
if WORLD > 1:
    dist.init_process_group("gloo")
args.seed += 1000 * RANK
torch.set_num_threads(args.threads)
torch.manual_seed(0)  # same init on every rank
random.seed(args.seed)
ROOT = os.path.join(os.path.dirname(__file__), "..")
spaces = json.load(open(os.path.join(ROOT, "data/spaces.json")))
SINGLE_SPACES = sorted(k for k in spaces if not k.startswith("eval/"))  # never mix in held-out eval labels


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
    if "cands" in row:
        cands = list(row["cands"])
        pool = POOL.get(row["src"])
        if pool and random.random() < 0.6:
            # random labels from other texts of the same source: almost always wrong for this text
            extra = random.sample(pool, min(len(pool), random.randint(1, args.extra_negs)))
            cands += [c for c in extra if c not in cands]
        if random.random() < 0.1 and len(cands) > len(gold) + 1:
            cands = [c for c in cands if c not in gold]
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
    unknown = set()
    if random.random() < args.mix_prob and len(cands) < args.max_labels - 2:
        # labels of an unrelated question (target unknown -> excluded from the loss); teaches the
        # model to answer several questions in one pass without them interfering
        other = random.choice(SINGLE_SPACES)
        if other != row["src"]:
            extra = [l for l in random.sample(spaces[other], min(len(spaces[other]), random.randint(2, 6))) if l not in cands]
            extra = extra[: args.max_labels - len(cands)]
            unknown = set(extra)
            cands += extra
    random.shuffle(cands)
    gs = set(gold)
    y = [-1.0 if c in unknown else (1.0 if c in gs else 0.0) for c in cands]
    return text, [vary(c) for c in cands], y


POOL = {}


class Stream(IterableDataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path)]
        pools = {}
        for r in self.rows:
            if "cands" in r and r["src"] == "synth_categories":  # sentence-style labels are too often true for other texts
                pools.setdefault(r["src"], set()).update(r["cands"])
        POOL.update({k: sorted(v) for k, v in pools.items()})
        print("rows", len(self.rows), "pools", {k: len(v) for k, v in POOL.items()}, flush=True)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(args.seed + (info.id if info else 0) + int(time.time()))
        random.seed(rng.random())
        by_src = {}
        for i, r in enumerate(self.rows):
            by_src.setdefault(r["src"], []).append(i)
        srcs = sorted(by_src)
        w = [len(by_src[k]) ** args.alpha for k in srcs]
        while True:
            for k in rng.choices(srcs, weights=w, k=10000):
                yield make_example(self.rows[rng.choice(by_src[k])])


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


loader = DataLoader(Stream(os.path.join(ROOT, args.data)), batch_size=args.bs, collate_fn=collate, num_workers=args.workers, prefetch_factor=8,
                    worker_init_fn=lambda _: torch.set_num_threads(1))

data_iter = iter(loader)  # fork data workers before CUDA is initialised (avoids fork deadlocks)
model.to(args.device)
head_params = list(model.head.parameters())
core = model
local_group = None
if WORLD > 1 and args.avg_every:
    # one DDP group per node (fast shared-memory sync); nodes only meet every avg_every steps
    per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", "1").split("(")[0])
    for n in range(WORLD // per_node):
        g = dist.new_group(list(range(n * per_node, (n + 1) * per_node)))
        if RANK // per_node == n:
            local_group = g
    if per_node > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, process_group=local_group)
        core = model.module
elif WORLD > 1:
    model = torch.nn.parallel.DistributedDataParallel(model)
    core = model.module


def average_weights():
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    ps = [p.data for p in core.parameters()]
    flat = _flatten_dense_tensors(ps)
    dist.all_reduce(flat)
    flat /= WORLD
    for p, f in zip(ps, _unflatten_dense_tensors(flat, ps)):
        p.copy_(f)
opt = torch.optim.AdamW([
    {"params": core.encoder.parameters(), "lr": args.lr},
    {"params": head_params, "lr": args.head_lr},
], weight_decay=0.01)
warm = args.warmup


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
    core.encoder.config.save_pretrained(os.path.join(d, "encoder"))
    json.dump({"backbone": args.backbone, "max_len": args.max_len}, open(os.path.join(d, "config.json"), "w"))


model.train()
t0 = time.time()
ema = None
step = 0
for micro, b in enumerate(data_iter, 1):
    b = {k: v.to(args.device, non_blocking=True) for k, v in b.items()}
    sync = micro % args.accum == 0
    ctx = model.no_sync() if (WORLD > 1 and not sync) else contextlib.nullcontext()
    y = b.pop("y")
    valid = b["label_valid"]
    ctx.__enter__()  # covers forward and backward (exited after backward)
    with torch.autocast(device_type="cuda" if "cuda" in args.device else "cpu", dtype=torch.bfloat16, enabled="cuda" in args.device):
        logits = model(**b)
    logits = logits.float()
    known = valid & (y >= 0)
    yk = y.clamp(min=0)
    bce = F.binary_cross_entropy_with_logits(logits, yk, reduction="none")
    bce = (bce * known).sum() / known.sum()
    single = (yk.sum(1) == 1)
    if single.any():
        ce = F.cross_entropy(logits[single].masked_fill(~known[single], -1e4), yk[single].argmax(1))
    else:
        ce = torch.zeros((), device=logits.device)
    loss = bce + 0.5 * ce
    (loss / args.accum).backward()
    ctx.__exit__(None, None, None)
    if micro % args.accum != 0:
        continue
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    step += 1
    seq_sum = seq_sum + b["input_ids"].shape[1] if step > 1 else b["input_ids"].shape[1]
    ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
    if step % args.log_every == 0 and RANK == 0:
        el = time.time() - t0
        msg = f"step {step} loss {ema:.4f} bce {bce.item():.4f} ce {ce.item():.4f} lr {sched.get_last_lr()[0]:.2e} ex/s {step*args.bs*args.accum*WORLD/el:.1f} elapsed {el/60:.1f}m seq {seq_sum/step:.0f}"
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
    if WORLD > 1 and args.avg_every and step % args.avg_every == 0:
        ta = time.time()
        average_weights()
        if RANK == 0 and step % (args.avg_every * 5) == 0:
            print(f"averaged weights across nodes in {time.time() - ta:.1f}s", flush=True)
    if step % args.save_every == 0:
        save("last")
        if args.keep_every and step % args.keep_every == 0:
            save(f"step{step}")
    stop = torch.tensor(float(step >= args.steps or time.time() - t0 > args.hours * 3600))
    if WORLD > 1:
        dist.all_reduce(stop)
    if stop.item() > 0:
        break
if WORLD > 1 and args.avg_every:
    average_weights()
save("last")
print("done", flush=True)
