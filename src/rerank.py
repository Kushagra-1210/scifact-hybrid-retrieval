import json, os
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from src.data import load_scifact, doc_text
from src.utils import load_run, save_run

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CACHE = "results/ce_cache.json"
TOP_N = 20

def rerank_run(run, queries, texts, ce, cache, top_n=TOP_N):
    out = {}
    for qid, docs in tqdm(run.items()):
        ranked = sorted(docs, key=docs.get, reverse=True)
        head, tail = ranked[:top_n], ranked[top_n:]
        todo = [d for d in head if f"{qid}|{d}" not in cache]
        if todo:
            scores = ce.predict([(queries[qid], texts[d]) for d in todo], batch_size=32)
            for d, s in zip(todo, scores):
                cache[f"{qid}|{d}"] = float(s)
        head = sorted(head, key=lambda d: cache[f"{qid}|{d}"], reverse=True)
        final = head + tail
        out[qid] = {d: float(len(final) - i) for i, d in enumerate(final)}
    return out

if __name__ == "__main__":
    corpus, queries, _ = load_scifact("test")
    texts = {d: doc_text(corpus[d]) for d in corpus}
    ce = CrossEncoder(MODEL)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    for name in ["bm25", "hybrid", "tuned", "adaptive"]:
        run = load_run(f"results/runs/{name}.json")
        save_run(rerank_run(run, queries, texts, ce, cache), f"results/runs/{name}_rerank.json")
        json.dump(cache, open(CACHE, "w"))
        print("saved", name)