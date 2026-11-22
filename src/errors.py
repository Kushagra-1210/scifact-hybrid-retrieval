import json
from src.data import load_scifact
from src.utils import load_run

corpus, queries, qrels = load_scifact("test")
bm25 = load_run("results/runs/bm25.json")
dense = load_run("results/runs/dense.json")
tuned = load_run("results/runs/tuned.json")
tuned_rr = load_run("results/runs/tuned_rerank.json")

def gold_rank(run, qid):
    ranked = sorted(run.get(qid, {}), key=run[qid].get, reverse=True)
    rel = {d for d, s in qrels[qid].items() if s > 0}
    for i, d in enumerate(ranked, 1):
        if d in rel:
            return i
    return 999

rows = [(q, gold_rank(bm25, q), gold_rank(dense, q), gold_rank(tuned, q), gold_rank(tuned_rr, q)) for q in qrels]

cases = {
    "BM25 fails, dense works": sorted([r for r in rows if r[1] > 10 and r[2] <= 3], key=lambda r: -r[1])[:3],
    "Dense fails, BM25 works": sorted([r for r in rows if r[2] > 10 and r[1] <= 3], key=lambda r: -r[2])[:3],
    "Reranker hurts the tuned list": sorted([r for r in rows if r[4] > r[3] and r[3] <= 3], key=lambda r: r[3] - r[4])[:2],
}

with open("results/errors.md", "w", encoding="utf-8") as f:
    for title, rs in cases.items():
        f.write(f"## {title}\n")
        for q, b, d, t, rr in rs:
            f.write(f"- Query {q}: {queries[q]}\n")
            f.write(f"  - gold rank: BM25 {b}, dense {d}, tuned {t}, tuned+rerank {rr}\n")
            f.write("  - why: (write one sentence)\n")

examples = [r[0] for r in rows if r[1] > 3 and r[3] == 1][:5]
json.dump(examples, open("artifacts/examples.json", "w"))
print("example queries for the demo:", examples)