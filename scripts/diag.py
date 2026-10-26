import json, sys, os, collections, random
import torch
sys.path.insert(0, ".")
from onepass import ZeroShotClassifier
torch.set_num_threads(16)
clf = ZeroShotClassifier.load(sys.argv[1], device="cpu", max_len=320)
spaces = json.load(open("data/spaces.json"))
for name in sys.argv[2:]:
    rows = [json.loads(l) for l in open(f"data/eval/{name}.jsonl")][:60]
    labels = spaces["eval/" + name]
    pred = collections.Counter(); pos = collections.Counter(); ok = 0; ok_shuf = 0
    for r in rows:
        lg = clf.logits([r["text"]], labels)[0]
        i = int(lg.argmax()); pred[labels[i]] += 1; pos[i] += 1; ok += labels[i] == r["gold"][0]
        sh = labels[:]; random.Random(0).shuffle(sh)
        lg2 = clf.logits([r["text"]], sh)[0]; ok_shuf += sh[int(lg2.argmax())] == r["gold"][0]
    print(name, "acc", ok / len(rows), "acc_shuffled_order", ok_shuf / len(rows))
    print("  top predicted labels:", pred.most_common(5))
    print("  top predicted positions:", pos.most_common(5))
    print("  logits ex0:", [f"{labels[j]}={v:.1f}" for j, v in enumerate(clf.logits([rows[0]['text']], labels)[0].tolist())][:24], "gold", rows[0]["gold"])
