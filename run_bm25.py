from src.data import load_scifact
from src.bm25 import BM25Retriever
from src.utils import save_run, quick_eval

corpus, test_q, test_qrels = load_scifact("test")
_, train_q, train_qrels = load_scifact("train")

print("building BM25 index...")
bm25 = BM25Retriever(corpus)

test_run = bm25.run(test_q)
train_run = bm25.run(train_q)

save_run(test_run, "results/runs/bm25.json")
save_run(train_run, "results/runs_train/bm25.json")

print("BM25 test  (nDCG@10, Recall@100):", quick_eval(test_qrels, test_run))
print("BM25 train (nDCG@10, Recall@100):", quick_eval(train_qrels, train_run))