import json, os
import numpy as np
import pandas as pd
from src.data import load_scifact
from src.fusion import fuse_runs
from src.utils import load_run, save_run, quick_eval

_, test_q, test_qrels = load_scifact("test")
_, train_q, train_qrels = load_scifact("train")

tr_bm = load_run("results/runs_train/bm25.json")
tr_de = load_run("results/runs_train/dense.json")
te_bm = load_run("results/runs/bm25.json")
te_de = load_run("results/runs/dense.json")

WEIGHTS = [round(float(x), 1) for x in np.arange(0, 1.01, 0.1)]   # BM25 weight; dense = 1 - weight

def ndcg_on_train(qids, w):
    sub_bm = {q: tr_bm[q] for q in qids}
    sub_de = {q: tr_de[q] for q in qids}
    return quick_eval(train_qrels, fuse_runs(sub_bm, sub_de, w_a=w))[0]

# 1) global sweep on train
all_q = list(train_q.keys())
sweep = [(w, ndcg_on_train(all_q, w)) for w in WEIGHTS]
os.makedirs("results", exist_ok=True)
pd.DataFrame(sweep, columns=["bm25_weight", "ndcg10"]).to_csv("results/sweep.csv", index=False)
best_global = max(sweep, key=lambda x: x[1])[0]
print("Sweep on train (BM25 weight, nDCG@10):")
for w, s in sweep:
    print(f"  {w:.1f}  {s}")
print("Best global BM25 weight:", best_global)

# 2) query-length buckets from train (three equal-sized groups)
lens = {q: len(train_q[q].split()) for q in all_q}
e1, e2 = np.percentile(list(lens.values()), [33, 67])
edges = [int(e1), int(e2), 999]

def bucket_of(n):
    return next(i for i, e in enumerate(edges) if n <= e)

buckets = []
for bi, edge in enumerate(edges):
    qids = [q for q in all_q if bucket_of(lens[q]) == bi]
    res = [(w, ndcg_on_train(qids, w)) for w in WEIGHTS]
    top = max(s for _, s in res)
    ties = [w for w, s in res if s == top]
    w_best = min(ties, key=lambda w: abs(w - best_global))   # on a tie, stay near the global weight
    buckets.append({"max_words": edge, "bm25_weight": w_best})
    print(f"Bucket <= {edge} words: {len(qids)} train queries, best BM25 weight {w_best}, nDCG@10 {top}")

cfg = {"global_bm25_weight": best_global, "buckets": buckets}
os.makedirs("config", exist_ok=True)
json.dump(cfg, open("config/fusion_weights.json", "w"), indent=2)

# 3) apply frozen weights to the test split
def weight_for(query):
    n = len(query.split())
    return next(b["bm25_weight"] for b in buckets if n <= b["max_words"])

tuned = fuse_runs(te_bm, te_de, w_a=best_global)
adaptive = fuse_runs(te_bm, te_de, per_query_w={q: weight_for(test_q[q]) for q in te_bm})
save_run(tuned, "results/runs/tuned.json")
save_run(adaptive, "results/runs/adaptive.json")

hybrid = load_run("results/runs/hybrid.json")
print()
print("TEST results (nDCG@10, Recall@100):")
print("  BM25     :", quick_eval(test_qrels, te_bm))
print("  Dense    :", quick_eval(test_qrels, te_de))
print("  Hybrid   :", quick_eval(test_qrels, hybrid))
print("  Tuned    :", quick_eval(test_qrels, tuned))
print("  Adaptive :", quick_eval(test_qrels, adaptive))