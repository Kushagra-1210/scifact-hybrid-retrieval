## BM25 fails, dense works
- Query 239: Cellular aging closely links to an older appearance.
  - gold rank: BM25 999, dense 2, tuned 5, tuned+rerank 20
  - why: (write one sentence)
- Query 535: Hypertension is frequently observed in type 1 diabetes patients.
  - gold rank: BM25 999, dense 1, tuned 5, tuned+rerank 15
  - why: (write one sentence)
- Query 800: Modifying the epigenome in the brain affects the normal human aging process by affecting certain genes related to neurogenesis.
  - gold rank: BM25 999, dense 3, tuned 8, tuned+rerank 2
  - why: (write one sentence)
## Dense fails, BM25 works
- Query 128: Arterioles have a larger lumen diameter than venules.
  - gold rank: BM25 3, dense 999, tuned 999, tuned+rerank 999
  - why: (write one sentence)
- Query 1362: Venules have a larger lumen diameter than arterioles.
  - gold rank: BM25 3, dense 999, tuned 999, tuned+rerank 999
  - why: (write one sentence)
- Query 1363: Venules have a thinner or absent smooth layer compared to arterioles.
  - gold rank: BM25 1, dense 999, tuned 999, tuned+rerank 999
  - why: (write one sentence)
## Reranker hurts the tuned list
- Query 70: Activation of PPM1D suppresses p53 function.
  - gold rank: BM25 1, dense 1, tuned 1, tuned+rerank 14
  - why: (write one sentence)
- Query 238: Cells undergoing methionine restriction may activate miRNAs.
  - gold rank: BM25 20, dense 7, tuned 3, tuned+rerank 11
  - why: (write one sentence)
