from src.data import load_scifact
from src.dense import DenseRetriever
from src.utils import save_run, quick_eval

corpus, test_q, test_qrels = load_scifact("test")
_, train_q, train_qrels = load_scifact("train")

print("loading model and encoding documents (a few minutes the first time)...")
dense = DenseRetriever(corpus)

test_run = dense.run(test_q)
train_run = dense.run(train_q)

save_run(test_run, "results/runs/dense.json")
save_run(train_run, "results/runs_train/dense.json")

print("Dense test  (nDCG@10, Recall@100):", quick_eval(test_qrels, test_run))
print("Dense train (nDCG@10, Recall@100):", quick_eval(train_qrels, train_run))