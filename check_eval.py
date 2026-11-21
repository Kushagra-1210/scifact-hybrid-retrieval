from src.data import load_scifact
from src.utils import load_run
from src.evaluate import evaluate_run

_, _, qrels = load_scifact("test")
perfect = {q: {d: 1.0 for d, s in r.items() if s > 0} for q, r in qrels.items()}
print("perfect run nDCG@10 (should be 1.0):", round(evaluate_run(qrels, perfect)["nDCG@10"], 3))

for name in ["bm25", "dense", "hybrid", "tuned", "adaptive"]:
    m = evaluate_run(qrels, load_run(f"results/runs/{name}.json"))
    print(name, {k: round(v, 4) for k, v in m.items() if k != "recall_curve"})