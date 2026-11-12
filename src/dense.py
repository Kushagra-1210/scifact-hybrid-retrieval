import os, json
import numpy as np
from sentence_transformers import SentenceTransformer
from src.data import doc_text

MODEL = "BAAI/bge-small-en-v1.5"

class DenseRetriever:
    def __init__(self, corpus, cache="artifacts"):
        self.model = SentenceTransformer(MODEL)
        emb_path = f"{cache}/embeddings.npy"
        ids_path = f"{cache}/doc_ids.json"
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            self.emb = np.load(emb_path)
            self.doc_ids = json.load(open(ids_path))
        else:
            self.doc_ids = list(corpus.keys())
            texts = [doc_text(corpus[d]) for d in self.doc_ids]
            self.emb = self.model.encode(
                texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
            os.makedirs(cache, exist_ok=True)
            np.save(emb_path, self.emb)
            json.dump(self.doc_ids, open(ids_path, "w"))

    def run(self, queries, k=100):
        qids = list(queries.keys())
        qe = self.model.encode(
            [queries[q] for q in qids], batch_size=32, normalize_embeddings=True)
        S = qe @ self.emb.T          # cosine similarity (vectors are normalised)
        out = {}
        for i, qid in enumerate(qids):
            top = np.argsort(-S[i])[:k]
            out[qid] = {self.doc_ids[j]: float(S[i, j]) for j in top}
        return out