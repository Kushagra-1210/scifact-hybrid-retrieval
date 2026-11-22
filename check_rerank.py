from src.data import load_scifact
from src.utils import load_run
from src.evaluate import evaluate_run

_, _, qrels = load_scifact("test")
for name in ["hybrid", "hybrid_rerank", "tuned", "tuned_rerank", "adaptive", "adaptive_rerank"]:
    m = evaluate_run(qrels, load_run(f"results/runs/{name}.json"))
    print(f"{name:16s} nDCG@10 {m['nDCG@10']:.4f}  MRR@10 {m['MRR@10']:.4f}  R@10 {m['R@10']:.4f}  R@100 {m['R@100']:.4f}")