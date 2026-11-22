import json, os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results/figures", exist_ok=True)
df = pd.read_csv("results/ablation.csv")

# 1. nDCG@10 bar chart
plt.figure(figsize=(10, 5))
plt.bar(df["System"], df["nDCG@10"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("nDCG@10")
plt.ylim(0.6, max(df["nDCG@10"]) + 0.03)
plt.title("nDCG@10 by system on SciFact (test)")
for i, v in enumerate(df["nDCG@10"]):
    plt.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("results/figures/ndcg_bar.png", dpi=200)
plt.close()

# 2. Recall@k curve
curves = json.load(open("results/recall_curves.json"))
plt.figure(figsize=(7, 4.6))
for label in ["BM25", "Dense (BGE-small)", "Hybrid RRF 50/50", "Tuned hybrid"]:
    if label in curves:
        ks = [int(k) for k in curves[label]]
        plt.plot(ks, list(curves[label].values()), marker="o", label=label)
plt.xlabel("k")
plt.ylabel("Recall@k")
plt.title("Recall@k on SciFact (test)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/recall_curve.png", dpi=200)
plt.close()

# 3. Fusion weight sweep (train split)
sw = pd.read_csv("results/sweep.csv")
best = sw.loc[sw["ndcg10"].idxmax()]
plt.figure(figsize=(7, 4.6))
plt.plot(sw["bm25_weight"], sw["ndcg10"], marker="o")
plt.axvline(best["bm25_weight"], ls="--", color="gray")
plt.xlabel("BM25 weight (dense weight = 1 - BM25 weight)")
plt.ylabel("nDCG@10 (train)")
plt.title("Fusion weight sweep on the train split")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/weight_sweep.png", dpi=200)
plt.close()
print("saved 3 figures")