import re
import numpy as np
from rank_bm25 import BM25Okapi
from src.data import doc_text

def tok(s):
    return re.findall(r"\w+", s.lower())

class BM25Retriever:
    def __init__(self, corpus):
        self.doc_ids = list(corpus.keys())
        self.bm25 = BM25Okapi([tok(doc_text(corpus[d])) for d in self.doc_ids])

    def run(self, queries, k=100):
        out = {}
        for qid, q in queries.items():
            s = self.bm25.get_scores(tok(q))
            top = np.argsort(-s)[:k]
            out[qid] = {self.doc_ids[i]: float(s[i]) for i in top}
        return out