"""Zero-shot DEV set: public datasets used neither for training nor in the 13-dataset benchmark.
Used only for model selection (WiSE-FT ratio, checkpoint choice) and the shipped calibration."""
import json, os, random
from datasets import load_dataset
random.seed(1)
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "dev"); os.makedirs(OUT, exist_ok=True)
spaces = json.load(open(os.path.join(OUT, "..", "spaces.json")))
def take(rows, n=400):
    random.shuffle(rows); return rows[:n]
sets = {}
d = load_dataset("SetFit/ade_corpus_v2_classification", split="test")
m = {"Related": "mentions an adverse drug effect", "Not-Related": "no adverse drug effect"}
sets["ade"] = ([{"text": r["text"], "gold": [m[r["label_text"]]]} for r in d], list(m.values()))
d = load_dataset("SetFit/insincere-questions", split="test")
m = {"insincere question": "insincere question", "valid question": "sincere question"}
sets["insincere"] = ([{"text": r["text"], "gold": [m[r["label_text"]]]} for r in d.select(range(5000))], list(m.values()))
d = load_dataset("google-research-datasets/poem_sentiment", split="train")
n = ["negative", "positive", "no emotional impact", "mixed"]
sets["poem"] = ([{"text": r["verse_text"], "gold": [n[r["label"]]]} for r in d], n)
d = load_dataset("valurank/News_Articles_Categorization", split="train")
cats = sorted(set(d["Category"]))
sets["news_cat"] = ([{"text": r["Text"], "gold": [r["Category"].lower()]} for r in d if r["Text"]], [c.lower() for c in cats])
d = load_dataset("cardiffnlp/tweet_eval", "emotion", split="test")
n = ["anger", "joy", "optimism", "sadness"]
sets["tweet_emotion"] = ([{"text": r["text"], "gold": [n[r["label"]]]} for r in d], n)
AX = {"math.AC": "commutative algebra", "cs.CV": "computer vision", "cs.AI": "artificial intelligence", "cs.SY": "systems and control",
      "math.GR": "group theory", "cs.CE": "computational engineering", "cs.PL": "programming languages", "cs.IT": "information theory",
      "cs.DS": "data structures and algorithms", "cs.NE": "neural and evolutionary computing", "math.ST": "statistics theory"}
d = load_dataset("ccdv/arxiv-classification", split="test")
names = [AX[x] for x in d.features["label"].names]
sets["arxiv"] = ([{"text": r["text"][:3000], "gold": [names[r["label"]]]} for r in d], names)
import re
d = load_dataset("benayas/snips", split="test")
cam = lambda x: re.sub(r"(?<=[a-z])(?=[A-Z])", " ", x).lower()
names = sorted(set(cam(x) for x in d["category"]))
sets["snips"] = ([{"text": r["text"], "gold": [cam(r["category"])]} for r in d], names)
d = load_dataset("rungalileo/medical_transcription_40", split="test")
lab = d.features["label"].names if hasattr(d.features["label"], "names") else None
clean = lambda x: x.strip().replace(" - ", " ").replace("Phy.", "physical").lower()
names = sorted(set(clean(x) for x in lab))
sets["medical_specialty"] = ([{"text": r["text"][:3000], "gold": [clean(lab[r["label"]])]} for r in d], names)
for k, (rows, labels) in sets.items():
    rows = take(rows)
    json.dump(labels, open(os.path.join(OUT, k + ".labels.json"), "w"))
    with open(os.path.join(OUT, k + ".jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(k, len(rows), labels)
