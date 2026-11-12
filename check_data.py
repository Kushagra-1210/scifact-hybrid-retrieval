from src.data import load_scifact

corpus, test_q, test_qrels = load_scifact("test")
print("documents:", len(corpus))
print("test queries:", len(test_q))
print("test judgments:", sum(len(v) for v in test_qrels.values()))

_, train_q, train_qrels = load_scifact("train")
print("train queries:", len(train_q))
print("train judgments:", sum(len(v) for v in train_qrels.values()))

avg_len = sum(len((d.get("title", "") + " " + d["text"]).split()) for d in corpus.values()) / len(corpus)
print("avg words per document:", round(avg_len, 1))
print("avg relevant docs per test query:", round(sum(len(v) for v in test_qrels.values()) / len(test_q), 2))