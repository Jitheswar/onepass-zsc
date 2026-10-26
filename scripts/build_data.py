"""Download datasets and write unified jsonl files.

data/train.jsonl   training rows   {"src", "text", "gold": [...], "space": src}
data/calib.jsonl   rows from held-in test splits (for fitting temperature)
data/eval/<name>.jsonl  held-out zero-shot benchmarks (never trained on)
data/spaces.json   label space for every source
"""
import json
import os
import random
import re
from collections import defaultdict

from datasets import load_dataset

random.seed(0)
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(os.path.join(OUT, "eval"), exist_ok=True)
TRAIN_CAP = int(os.environ.get("TRAIN_CAP", 40000))
OUT_NAME = os.environ.get("OUT_NAME", "train.jsonl")
SYNTH_CAP = int(os.environ.get("SYNTH_CAP", 0))
MNLI_CAP = int(os.environ.get("MNLI_CAP", 60000))
CALIB_CAP = 300
EVAL_CAP = 2000

spaces = {}


def clean(s):
    s = s.replace("_", " ").strip()
    return re.sub(r"\s+", " ", s)


def camel(s):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s).lower()


def stars(i):
    return "1 star" if i == 0 else "%d stars" % (i + 1)


def rows_from(ds, text_fn, gold_fn):
    for r in ds:
        t = text_fn(r)
        g = gold_fn(r)
        if t and t.strip() and g:
            yield t.strip(), g


def take(it, cap):
    rows = list(it)
    random.shuffle(rows)
    return rows[:cap]


def names_of(ds, col="label"):
    return ds.features[col].names


# ---------- training sources: (name, loader) where loader returns (train_iter_fn, test_iter_fn, space)
def single(names, col="label"):
    return lambda r: [names[r[col]]]


def T(col="text"):
    return lambda r: r[col]


TRAIN = {}

def reg(name):
    def deco(fn):
        TRAIN[name] = fn
        return fn
    return deco


@reg("ag_news")
def _():
    d = load_dataset("fancyzhx/ag_news")
    n = ["world news", "sports", "business", "science and technology"]
    return d["train"], d["test"], T(), single(n), n


@reg("dbpedia")
def _():
    d = load_dataset("fancyzhx/dbpedia_14")
    n = [camel(x) for x in names_of(d["train"])]
    n[n.index("mean of transportation")] = "means of transportation"
    return d["train"], d["test"], lambda r: r["title"] + ". " + r["content"], single(n), n


@reg("yahoo")
def _():
    d = load_dataset("community-datasets/yahoo_answers_topics")
    n = [x.lower() for x in names_of(d["train"], "topic")]
    tf = lambda r: " ".join(x for x in [r["question_title"], r["question_content"], r["best_answer"]] if x)
    return d["train"], d["test"], tf, single(n, "topic"), n


@reg("go_emotions")
def _():
    d = load_dataset("google-research-datasets/go_emotions", "simplified")
    n = d["train"].features["labels"].feature.names
    return d["train"], d["test"], T(), lambda r: [n[i] for i in r["labels"]], n


@reg("yelp_polarity")
def _():
    d = load_dataset("fancyzhx/yelp_polarity")
    n = ["negative", "positive"]
    return d["train"], d["test"], T(), single(n), n


@reg("amazon_polarity")
def _():
    d = load_dataset("fancyzhx/amazon_polarity")
    n = ["negative", "positive"]
    return d["train"], d["test"], lambda r: r["title"] + ". " + r["content"], single(n), n


@reg("yelp_stars")
def _():
    d = load_dataset("SetFit/yelp_review_full")
    n = [stars(i) for i in range(5)]
    return d["train"], d["test"], T(), single(n), n


@reg("amazon_stars")
def _():
    d = load_dataset("SetFit/amazon_reviews_multi_en")
    n = [stars(i) for i in range(5)]
    return d["train"], d["test"], T(), single(n), n


def tweet(cfg, n):
    def f():
        d = load_dataset("cardiffnlp/tweet_eval", cfg)
        return d["train"], d["test"], T(), single(n), n
    return f

reg("tweet_sentiment")(tweet("sentiment", ["negative", "neutral", "positive"]))
reg("tweet_offensive")(tweet("offensive", ["not offensive", "offensive"]))
reg("tweet_hate")(tweet("hate", ["not hate speech", "hate speech"]))
reg("tweet_irony")(tweet("irony", ["not ironic", "ironic"]))


@reg("clinc")
def _():
    d = load_dataset("clinc/clinc_oos", "plus")
    n = [clean(x) if x != "oos" else "out of scope" for x in names_of(d["train"], "intent")]
    return d["train"], d["test"], T(), single(n, "intent"), n


def setfit(repo, rename=None, text="text", test="test"):
    def f():
        d = load_dataset(repo)
        names = sorted(set(d["train"]["label_text"]))
        rn = lambda x: (rename or {}).get(x, clean(x))
        n = [rn(x) for x in names]
        return d["train"], d[test], T(text), lambda r: [rn(r["label_text"])], n
    return f

reg("imdb")(lambda: (lambda d: (d["train"], d["test"], T(), single(["negative", "positive"]), ["negative", "positive"]))(load_dataset("stanfordnlp/imdb")))
reg("sst5")(setfit("SetFit/sst5"))
reg("toxic")(setfit("SetFit/toxic_conversations"))
reg("subj")(setfit("SetFit/subj"))
reg("massive_intent")(setfit("SetFit/amazon_massive_intent_en-US", {"iot_hue_lightoff": "turn lights off", "iot_hue_lighton": "turn lights on", "iot_hue_lightdim": "dim the lights", "iot_hue_lightup": "brighten the lights", "iot_hue_lightchange": "change light color", "iot_cleaning": "start vacuum cleaner", "iot_coffee": "make coffee", "iot_wemo_on": "turn smart plug on", "iot_wemo_off": "turn smart plug off", "qa_factoid": "factual question", "qa_definition": "definition question", "qa_maths": "math question", "qa_currency": "currency question", "qa_stock": "stock price question"}))
reg("student_questions")(setfit("SetFit/student-question-categories"))
reg("counterfactual")(setfit("SetFit/amazon_counterfactual_en", {"not-counterfactual": "not counterfactual"}))
reg("tweet_sent_extract")(setfit("mteb/tweet_sentiment_extraction"))


@reg("trec_fine")
def _():
    d = load_dataset("SetFit/TREC-QC")
    n = sorted(set(d["train"]["label_text"]))
    return d["train"], d["test"], T(), lambda r: [r["label_text"]], n


@reg("trec_coarse")
def _():
    d = load_dataset("SetFit/TREC-QC")
    m = {"description and abstract concepts": "description or abstract concept", "entities": "entity", "human beings": "person",
         "locations": "location", "numeric values": "number", "abbreviation": "abbreviation"}
    n = sorted(set(m.get(x, x) for x in d["train"]["label_coarse_text"]))
    return d["train"], d["test"], T(), lambda r: [m.get(r["label_coarse_text"], r["label_coarse_text"])], n


@reg("fin_sentiment")
def _():
    d = load_dataset("zeroshot/twitter-financial-news-sentiment")
    n = ["bearish", "bullish", "neutral"]
    return d["train"], d["validation"], T(), single(n), n


# ---------- held-out zero-shot benchmarks
NG = {"alt.atheism": "atheism", "comp.graphics": "computer graphics", "comp.os.ms-windows.misc": "microsoft windows",
      "comp.sys.ibm.pc.hardware": "pc hardware", "comp.sys.mac.hardware": "mac hardware", "comp.windows.x": "x window system",
      "misc.forsale": "items for sale", "rec.autos": "cars", "rec.motorcycles": "motorcycles", "rec.sport.baseball": "baseball",
      "rec.sport.hockey": "hockey", "sci.crypt": "cryptography", "sci.electronics": "electronics", "sci.med": "medicine",
      "sci.space": "space", "soc.religion.christian": "christianity", "talk.politics.guns": "guns",
      "talk.politics.mideast": "middle east politics", "talk.politics.misc": "politics", "talk.religion.misc": "religion"}
FIN_TOPIC = ["analyst update", "central banks", "company and product news", "treasuries and corporate debt", "dividend",
             "earnings", "energy and oil", "financials", "currencies", "general news and opinion", "gold, metals and materials",
             "IPO", "legal and regulation", "mergers, acquisitions and investments", "macroeconomics", "markets", "politics",
             "personnel change", "stock commentary", "stock movement"]
SCEN = {"iot": "smart home", "qa": "general question", "datetime": "date and time", "takeaway": "takeaway food",
        "play": "play media", "social": "social media", "lists": "lists and reminders", "general": "general chat"}


def EV_setfit(repo, rename=None, split="test", text="text"):
    def f():
        d = load_dataset(repo, split=split)
        rn = lambda x: (rename or {}).get(x, clean(x))
        n = sorted(set(rn(x) for x in d["label_text"]))
        return d, T(text), lambda r: [rn(r["label_text"])], n
    return f


def EV_cls(repo, cfg, split, names, text="text", col="label"):
    def f():
        d = load_dataset(repo, cfg, split=split)
        return d, (text if callable(text) else T(text)), single(names, col), names
    return f


EVAL = {
    "sst2": EV_setfit("SetFit/sst2"),
    "rotten_tomatoes": EV_cls("cornell-movie-review-data/rotten_tomatoes", None, "test", ["negative", "positive"]),
    "cr": EV_setfit("SetFit/CR"),
    "bbc_news": EV_setfit("SetFit/bbc-news", {"tech": "technology", "sport": "sports"}),
    "20newsgroups": EV_setfit("SetFit/20_newsgroups", NG),
    "banking77": EV_setfit("mteb/banking77"),
    "massive_scenario": EV_setfit("SetFit/amazon_massive_scenario_en-US", SCEN),
    "enron_spam": EV_setfit("SetFit/enron_spam", {"ham": "not spam"}),
    "sms_spam": EV_cls("ucirvine/sms_spam", None, "train", ["not spam", "spam"], text="sms"),
    "emotion": EV_cls("dair-ai/emotion", "split", "test", ["sadness", "joy", "love", "anger", "fear", "surprise"]),
    "hate_offensive": EV_setfit("SetFit/hate_speech_offensive", {"neither": "neither offensive nor hateful"}),
    "fin_topic": EV_cls("zeroshot/twitter-financial-news-topic", None, "validation", FIN_TOPIC),
    "climate": EV_cls("climatebert/climate_detection", None, "test", ["not about climate", "about climate"]),
}


def synthetic(cap):
    """MoritzLaurer/synthetic_zeroshot_mixtral_v0.1: (text, hypothesis, 0/1) pairs on web text,
    grouped by text into multi-candidate rows. No overlap with the eval benchmarks."""
    if cap <= 0:
        return []
    cfgs = {"mixtral_refinedweb_categories": "category", "mixtral_refinedweb_characteristics": "hypothesis",
            "mixtral_written_texts_for_tasks": "hypothesis", "mixtral_written_texts_for_tasks_v2": "hypothesis",
            "mixtral_written_texts_for_tasks_v3": "hypothesis", "mixtral_written_texts_for_tasks_v4": "hypothesis"}
    out = []
    for cfg, col in cfgs.items():
        d = load_dataset("MoritzLaurer/synthetic_zeroshot_mixtral_v0.1", cfg, split="train")
        groups = defaultdict(dict)
        for t, h, y in zip(d["text"], d[col], d["labels"]):
            if t and h:
                groups[t.strip()][h.strip()] = str(y) == "1"
        rows = [{"src": "synth_" + cfg.split("_")[-1], "text": t, "gold": [h for h, y in hs.items() if y], "cands": list(hs)}
                for t, hs in groups.items() if len(hs) >= 2]
        random.shuffle(rows)
        rows = rows[: cap // len(cfgs)]
        print(f"synthetic {cfg}: {len(groups)} texts -> {len(rows)} rows, avg cands {sum(len(r['cands']) for r in rows)/max(1,len(rows)):.1f}", flush=True)
        out += rows
    return out


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    train, calib = [], []
    for name, fn in TRAIN.items():
        tr, te, tf, gf, n = fn()
        spaces[name] = n
        a = take(rows_from(tr.shuffle(seed=0).select(range(min(len(tr), TRAIN_CAP * 3))), tf, gf), TRAIN_CAP)
        b = take(rows_from(te.shuffle(seed=0).select(range(min(len(te), CALIB_CAP * 3))), tf, gf), CALIB_CAP)
        train += [{"src": name, "text": t, "gold": g} for t, g in a]
        calib += [{"src": name, "text": t, "gold": g} for t, g in b]
        print(f"{name:20s} train={len(a):6d} calib={len(b):4d} labels={len(n)} e.g. {n[:6]}", flush=True)
        bad = [g for _, g in a[:200] for x in g if x not in n]
        assert not bad, (name, bad[:3])

    # MNLI: group hypotheses by premise -> multi-label rows; entailment = positive
    d = load_dataset("nyu-mll/multi_nli", split="train")
    groups = defaultdict(lambda: [None, {}])
    for r in d:
        g = groups[r["premise"]]
        g[0] = r["premise"]
        g[1][r["hypothesis"]] = r["label"] == 0
    rows = list(groups.values())
    random.shuffle(rows)
    nli = [{"src": "mnli", "text": p, "gold": [h for h, y in hs.items() if y], "cands": list(hs.keys())} for p, hs in rows[:MNLI_CAP]]
    train += nli
    print("mnli", len(nli))

    train += synthetic(SYNTH_CAP)
    random.shuffle(train)
    write(os.path.join(OUT, OUT_NAME), train)
    write(os.path.join(OUT, "calib.jsonl"), calib)

    if os.environ.get("SKIP_EVAL"):
        print("train rows", len(train)); return
    for name, fn in EVAL.items():
        d, tf, gf, n = fn()
        rows = take(rows_from(d.shuffle(seed=0).select(range(min(len(d), EVAL_CAP * 2))), tf, gf), EVAL_CAP)
        spaces["eval/" + name] = n
        assert all(g in n for _, gs in rows for g in gs), name
        write(os.path.join(OUT, "eval", name + ".jsonl"), [{"text": t, "gold": g} for t, g in rows])
        print(f"EVAL {name:18s} n={len(rows)} labels={len(n)} {n[:5]}", flush=True)
    json.dump(spaces, open(os.path.join(OUT, "spaces.json"), "w"), indent=1)
    print("train rows", len(train), "calib rows", len(calib))


if __name__ == "__main__":
    main()
