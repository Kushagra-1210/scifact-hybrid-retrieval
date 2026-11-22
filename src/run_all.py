import json
import pandas as pd
from src.data import load_scifact
from src.evaluate import evaluate_run
from src.utils import load_run

SYSTEMS = [
    ("BM25", "bm25"),
    ("BM25 + rerank", "bm25_rerank"),
    ("Dense (BGE-small)", "dense"),
    ("Hybrid RRF 50/50", "hybrid"),
    ("Hybrid + rerank", "hybrid_rerank"),
    ("Tuned hybrid", "tuned"),
    ("Tuned + rerank", "tuned_rerank"),
    ("Adaptive hybrid", "adaptive"),
    ("Adaptive + rerank", "adaptive_rerank"),
]

_, _, qrels = load_scifact("test")
rows, curves = [], {}
for label, name in SYSTEMS:
    m = evaluate_run(qrels, load_run(f"results/runs/{name}.json"))
    curves[label] = m.pop("recall_curve")
    rows.append({"System": label, **{k: round(v, 4) for k, v in m.items()}})

df = pd.DataFrame(rows)
df.to_csv("results/ablation.csv", index=False)
json.dump(curves, open("results/recall_curves.json", "w"))
print(df.to_string(index=False))