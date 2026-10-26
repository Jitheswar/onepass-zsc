import sys
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
C = [("fancyzhx/ag_news",None),("fancyzhx/dbpedia_14",None),("community-datasets/yahoo_answers_topics",None),
("dair-ai/emotion","split"),("google-research-datasets/go_emotions","simplified"),("fancyzhx/yelp_polarity",None),
("fancyzhx/amazon_polarity",None),("cardiffnlp/tweet_eval","sentiment"),("cardiffnlp/tweet_eval","offensive"),("cardiffnlp/tweet_eval","hate"),("cardiffnlp/tweet_eval","irony"),
("PolyAI/banking77",None),("clinc/clinc_oos","plus"),("CogComp/trec",None),("SetFit/20_newsgroups",None),("stanfordnlp/imdb",None),
("SetFit/sst2",None),("SetFit/sst5",None),("SetFit/enron_spam",None),("SetFit/toxic_conversations",None),("SetFit/subj",None),("SetFit/CR",None),
("SetFit/bbc-news",None),("SetFit/amazon_massive_intent_en-US",None),("SetFit/amazon_massive_scenario_en-US",None),("cornell-movie-review-data/rotten_tomatoes",None),
("takala/financial_phrasebank","sentences_allagree"),("nyu-mll/multi_nli",None),("fancyzhx/yelp_review_full",None),("SetFit/yelp_review_full",None),("SetFit/amazon_reviews_multi_en",None),("ucirvine/sms_spam",None),("SetFit/emotion",None),("climatebert/climate_detection",None),("SetFit/student-question-categories",None),("knowledgator/events_classification_biotech",None),("dvilares/head_qa",None),("SetFit/hate_speech_offensive",None),("legacy-datasets/wikipedia",None)][:-1]
def f(c):
    try:
        d = load_dataset(c[0], c[1])
        s = list(d.keys()); tr = d[s[0]]
        return f"OK  {c} splits={ {k:len(d[k]) for k in s} } feats={str(tr.features)[:300]}"
    except Exception as e:
        return f"ERR {c} {type(e).__name__}: {str(e)[:150]}"
with ThreadPoolExecutor(12) as ex:
    for r in ex.map(f, C): print(r, flush=True)
