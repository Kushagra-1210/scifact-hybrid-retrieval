import os, sys, json, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.data import load_scifact, doc_text
from src.bm25 import tok
from src.fusion import rrf_scores

st.set_page_config(page_title="SciFact claim search", layout="wide")

@st.cache_resource
def load_all():
    corpus, queries, qrels = load_scifact("test")
    doc_ids = json.load(open("artifacts/doc_ids.json"))
    texts = {d: doc_text(corpus[d]) for d in doc_ids}
    bm25 = BM25Okapi([tok(texts[d]) for d in doc_ids])
    emb = np.load("artifacts/embeddings.npy")
    enc = SentenceTransformer("BAAI/bge-small-en-v1.5")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    cfg = json.load(open("config/fusion_weights.json"))
    return corpus, queries, qrels, doc_ids, texts, bm25, emb, enc, ce, cfg

corpus, queries, qrels, doc_ids, texts, bm25, emb, enc, ce, cfg = load_all()

def adaptive_weight(q):
    n = len(q.split())
    for b in cfg["buckets"]:
        if n <= b["max_words"]:
            return b["bm25_weight"]
    return cfg["global_bm25_weight"]

def search(q, w, use_rerank):
    sb = bm25.get_scores(tok(q))
    ib = sb.argsort()[::-1][:100]
    top_b = [doc_ids[i] for i in ib]
    sd = emb @ enc.encode([q], normalize_embeddings=True)[0]
    idd = sd.argsort()[::-1][:100]
    top_d = [doc_ids[i] for i in idd]
    fused = rrf_scores([top_b, top_d], weights=[w, 1 - w])
    ranked = sorted(fused, key=fused.get, reverse=True)
    if use_rerank:
        head = ranked[:20]
        scores = ce.predict([(q, texts[d]) for d in head])
        ours = sorted(zip(head, [float(s) for s in scores]), key=lambda x: -x[1])[:5]
    else:
        ours = [(d, fused[d]) for d in ranked[:5]]
    base = [(doc_ids[i], float(sb[i])) for i in ib[:5]]
    return base, ours

def highlight(text, q):
    for word in set(tok(q)):
        if len(word) > 3:
            text = re.sub(rf"\b({re.escape(word)})\b", r"**\1**", text, flags=re.I)
    return text

st.title("Scientific claim search: BM25 baseline vs hybrid retrieval")

examples = json.load(open("artifacts/examples.json"))
choice = st.sidebar.selectbox("Example queries", [""] + examples,
                              format_func=lambda x: queries.get(x, "(type your own below)"))
q = st.text_input("Claim", queries.get(choice, ""))

mode = st.sidebar.radio("Fusion weight", ["Tuned (global)", "Adaptive (query length)", "Manual"])
if mode == "Tuned (global)":
    w = cfg["global_bm25_weight"]
elif mode.startswith("Adaptive"):
    w = adaptive_weight(q) if q else cfg["global_bm25_weight"]
else:
    w = st.sidebar.slider("BM25 weight", 0.0, 1.0, float(cfg["global_bm25_weight"]), 0.1)
use_rerank = st.sidebar.checkbox("Cross-encoder rerank (top 20)", False)
st.sidebar.write(f"BM25 weight used: {w:.2f}")

if q:
    base, ours = search(q, w, use_rerank)
    gold = {d for d, s in qrels.get(choice, {}).items() if s > 0} if choice else set()
    our_label = "rerank score" if use_rerank else "RRF score"
    c1, c2 = st.columns(2)
    for col, title, res, label in [
        (c1, "BM25 baseline", base, "BM25 score"),
        (c2, "Our system (BM25 + dense, RRF)", ours, our_label),
    ]:
        col.subheader(title)
        for d, s in res:
            mark = "✅ " if d in gold else ""
            col.markdown(f"**{mark}{corpus[d].get('title', d)}**")
            col.caption(f"{label}: {s:.3f}")
            col.markdown(highlight(corpus[d]["text"][:350], q) + "...")
    if gold:
        st.caption("✅ marks the labelled relevant abstract.")