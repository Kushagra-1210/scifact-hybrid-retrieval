from src.data import load_scifact
from src.fusion import fuse_runs
from src.utils import load_run, save_run, quick_eval

_, _, qrels = load_scifact("test")

bm25 = load_run("results/runs/bm25.json")
dense = load_run("results/runs/dense.json")

hybrid = fuse_runs(bm25, dense, w_a=0.5)
save_run(hybrid, "results/runs/hybrid.json")

print("BM25   :", quick_eval(qrels, bm25))
print("Dense  :", quick_eval(qrels, dense))
print("Hybrid :", quick_eval(qrels, hybrid))