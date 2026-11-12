def _ranked(scores):
    """Doc ids sorted by score, best first."""
    return sorted(scores, key=scores.get, reverse=True)

def rrf_scores(rankings, k=60, weights=None):
    fused = {}
    for i, ranking in enumerate(rankings):
        w = 1.0 if weights is None else weights[i]
        for rank, d in enumerate(ranking, start=1):
            fused[d] = fused.get(d, 0.0) + w / (k + rank)
    return fused

def rrf(rankings, k=60, weights=None):
    """Returns a ranked list of doc ids. Bharat's demo uses this."""
    s = rrf_scores(rankings, k, weights)
    return sorted(s, key=s.get, reverse=True)

def fuse_runs(run_a, run_b, w_a=0.5, k=60, top=100, per_query_w=None):
    """run_a = BM25 run, run_b = dense run. w_a is the BM25 weight.
    per_query_w maps qid -> weight and overrides w_a."""
    out = {}
    for qid in run_a:
        w = per_query_w[qid] if per_query_w else w_a
        s = rrf_scores([_ranked(run_a[qid]), _ranked(run_b[qid])], k, [w, 1 - w])
        out[qid] = {d: s[d] for d in sorted(s, key=s.get, reverse=True)[:top]}
    return out