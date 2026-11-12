import json, os
from beir.retrieval.evaluation import EvaluateRetrieval

def save_run(run, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(run, open(path, "w"))

def load_run(path):
    return json.load(open(path))

def quick_eval(qrels, run):
    """Sanity-check metrics only. Bharat writes the full evaluation."""
    qrels = {q: qrels[q] for q in run}
    ndcg, _, recall, _ = EvaluateRetrieval().evaluate(
        qrels, run, [10, 100], ignore_identical_ids=False)
    return round(ndcg["NDCG@10"], 4), round(recall["Recall@100"], 4)