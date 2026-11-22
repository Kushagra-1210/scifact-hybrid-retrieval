# Hybrid Retrieval for Scientific Claim Search on SciFact

**Course:** CSD358 Information Retrieval, Shiv Nadar University

**Group members**
- Kushagra Bansal (2410110184)
- Bharat Parashar (2310110079)

## Overview

Given a scientific claim, the system retrieves the abstracts that support or refute it. We compare keyword search (BM25), dense retrieval (BGE-small), and a hybrid of the two fused with Reciprocal Rank Fusion (RRF). We also test a cross-encoder reranker and a query-length-adaptive fusion weight. All systems are evaluated on the SciFact test split from the BEIR benchmark, and a Streamlit app shows the BM25 baseline and the hybrid system side by side.

## Dataset

SciFact (BEIR). The data downloads automatically through the `beir` package the first time any script runs, into `data/`.

| Statistic | Value |
|---|---|
| Documents (abstracts) | 5,183 |
| Test queries / judgments | 300 / 339 |
| Train queries / judgments | 809 / 919 |
| Average words per document | 214.6 |
| Average relevant documents per test query | 1.13 |

## Pipeline

1. **BM25** (`rank_bm25`) and **dense retrieval** (`BAAI/bge-small-en-v1.5`, cosine similarity) each return the top 100 abstracts for a claim.
2. **Weighted RRF** merges the two lists: `score(d) = w / (60 + rank_bm25(d)) + (1 - w) / (60 + rank_dense(d))`.
3. The BM25 weight `w` is tuned on the **train** split only (best global value 0.1). An adaptive version picks `w` per query by length bucket (0.2 for up to 10 words, 0.2 for up to 14 words, 0.1 for longer).
4. Optional **cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) of the top 20 results.

## Results (test split)

| System | nDCG@10 | MAP@100 | MRR@10 | P@10 | R@10 | R@100 | F1@10 |
|---|---|---|---|---|---|---|---|
| BM25 | 0.6519 | 0.6132 | 0.6186 | 0.0850 | 0.7740 | 0.8731 | 0.1532 |
| BM25 + rerank | 0.6851 | 0.6449 | 0.6589 | 0.0887 | 0.8023 | 0.8731 | 0.1597 |
| Dense (BGE-small) | 0.7200 | 0.6815 | 0.6845 | 0.0953 | 0.8452 | 0.9533 | 0.1713 |
| Hybrid RRF 50/50 | 0.7102 | 0.6784 | 0.6825 | 0.0913 | 0.8207 | 0.9583 | 0.1644 |
| Hybrid + rerank | 0.6979 | 0.6563 | 0.6662 | 0.0927 | 0.8319 | 0.9583 | 0.1668 |
| **Tuned hybrid** | **0.7229** | 0.6849 | 0.6873 | 0.0953 | 0.8479 | 0.9533 | 0.1714 |
| Tuned + rerank | 0.7065 | 0.6651 | 0.6711 | 0.0943 | 0.8396 | 0.9533 | 0.1696 |
| Adaptive hybrid | 0.7183 | 0.6816 | 0.6829 | 0.0943 | 0.8431 | 0.9533 | 0.1697 |
| Adaptive + rerank | 0.7052 | 0.6640 | 0.6706 | 0.0940 | 0.8374 | 0.9533 | 0.1690 |

P@10 and F1@10 are low because SciFact has about 1.1 relevant documents per query, so P@10 cannot exceed roughly 0.11.

## Key findings

- Dense retrieval is much stronger than BM25 on SciFact (0.7200 vs 0.6519 nDCG@10).
- Equal-weight fusion is slightly worse than dense alone (0.7102). A BM25 weight of 0.1, tuned on the train split, gives the best result (0.7229), but the gain over dense (+0.003) is small for 300 queries.
- Query-length-adaptive weights did not beat the single tuned weight (0.7183 vs 0.7229); the length buckets chose almost the same weights.
- Fusion improves Recall@100 (0.9583 for hybrid vs 0.9533 for dense).
- The cross-encoder reranker helps the weak BM25 list (+0.033) but lowers nDCG@10 for every dense-based list.

For comparison, the BEIR paper reports roughly 0.665 for BM25 and 0.688 for BM25 with a cross-encoder reranker on SciFact (values to be verified against the paper). Ours are 0.6519 and 0.6851. The small gap is expected because BEIR used Elasticsearch BM25, while we use `rank_bm25` with a simple tokenizer.

## Repository structure

```
src/
  data.py          load SciFact, document text
  bm25.py          BM25 retriever
  dense.py         dense retriever (BGE-small), saves embeddings
  fusion.py        RRF and weighted RRF
  rerank.py        cross-encoder reranker (with score cache)
  evaluate.py      nDCG, MAP, MRR, precision, recall, F1
  run_all.py       builds the ablation table
  plots.py         figures
  errors.py        error analysis and demo example queries
  utils.py         save/load run files
app/app.py         Streamlit demo
config/fusion_weights.json   tuned and adaptive fusion weights
artifacts/         document embeddings and demo example queries
results/           run files, ablation table, figures, error analysis
run_bm25.py, run_dense.py, run_hybrid.py, run_tune.py   pipeline scripts
check_data.py, check_eval.py, check_rerank.py           sanity checks
```

## Setup (Windows PowerShell)

```powershell
git clone https://github.com/Kushagra-1210/IR-PROJECT-.git ir-scifact
cd ir-scifact
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproduce the results

Run from the repository root with the virtual environment active:

```powershell
python check_data.py
python run_bm25.py
python run_dense.py
python run_hybrid.py
python run_tune.py
python -m src.rerank
python -m src.run_all
python -m src.plots
python -m src.errors
```

`run_dense.py` encodes all abstracts on the first run (about 12 minutes on CPU) and saves the embeddings to `artifacts/`. Outputs: run files in `results/runs/`, the ablation table in `results/ablation.csv`, figures in `results/figures/`, and error cases in `results/errors.md`.

## Demo

```powershell
python -m streamlit run app/app.py
```

The app opens at `http://localhost:8501`. Pick one of the example claims from the sidebar (a check mark shows the labelled relevant abstract), or type your own claim. The left column shows the BM25 baseline, the right column shows the hybrid system, with scores and matching words in bold. The sidebar switches between the tuned weight, the adaptive weight, and a manual weight, and can turn on the cross-encoder reranker.

## Limitations

- Small corpus (5,183 abstracts) and one domain; claims outside it have no matching paper.
- Small models, no fine-tuning, and a general-purpose reranker not trained on scientific text.
- `rank_bm25` with a simple tokenizer instead of Elasticsearch, so BM25 is slightly below the published figure.
- Adaptive weights are tuned on only about 270 train queries per bucket, and differences of about 0.003 nDCG@10 are not statistically significant.

## References

- Robertson, S. and Zaragoza, H., 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval.
- Karpukhin, V. et al., 2020. Dense passage retrieval for open-domain question answering. EMNLP.
- Thakur, N. et al., 2021. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. NeurIPS Datasets and Benchmarks.
- Cormack, G.V., Clarke, C.L. and Buettcher, S., 2009. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. SIGIR.
- Reimers, N. and Gurevych, I., 2019. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. EMNLP-IJCNLP.
- Nogueira, R. and Cho, K., 2019. Passage re-ranking with BERT. arXiv:1901.04085.
- Wadden, D. et al., 2020. Fact or fiction: Verifying scientific claims. EMNLP.