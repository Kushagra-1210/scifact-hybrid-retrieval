from beir.retrieval.evaluation import EvaluateRetrieval

KS = [1, 3, 5, 10, 20, 50, 100]

def _ranked(scores):
    return sorted(scores, key=scores.get, reverse=True)

def mrr_at_k(qrels, run, k=10):
    total = 0.0
    for qid, rels in qrels.items():
        relevant = {d for d, s in rels.items() if s > 0}
        for i, d in enumerate(_ranked(run.get(qid, {}))[:k], start=1):
            if d in relevant:
                total += 1.0 / i
                break
    return total / len(qrels)

def evaluate_run(qrels, run):
    ndcg, _map, recall, precision = EvaluateRetrieval().evaluate(
        qrels, run, KS, ignore_identical_ids=False)
    p10, r10 = precision["P@10"], recall["Recall@10"]
    f1 = 2 * p10 * r10 / (p10 + r10) if (p10 + r10) else 0.0
    return {
        "nDCG@10": ndcg["NDCG@10"],
        "MAP@100": _map["MAP@100"],
        "MRR@10": mrr_at_k(qrels, run, 10),
        "P@10": p10,
        "R@10": r10,
        "R@100": recall["Recall@100"],
        "F1@10": f1,
        "recall_curve": {k: recall[f"Recall@{k}"] for k in KS},
    }