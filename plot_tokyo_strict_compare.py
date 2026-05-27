import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


base = Path("data/raw/baseline")
paper = json.loads((base / "tokyo_strict_paperstyle.json").read_text(encoding="utf-8"))["results"][0]
improved = json.loads((base / "tokyo_strict_with_gamma0.json").read_text(encoding="utf-8"))["results"][0]

labels = ["Baseline", "Paper-style\n($\\gamma\\geq0.5$)", "With $\\gamma=0$\noption"]
recall = [paper["baseline_recall"], paper["best_recall"], improved["best_recall"]]
ndcg = [paper["baseline_ndcg"], paper["best_ndcg"], improved["best_ndcg"]]

x = np.arange(len(labels))
w = 0.36

fig, ax = plt.subplots(figsize=(8.8, 4.8))
ax.bar(x - w / 2, recall, w, label="R@20", color="#4C78A8")
ax.bar(x + w / 2, ndcg, w, label="nDCG@20", color="#F58518")

for i, v in enumerate(recall):
    ax.text(i - w / 2, v + 0.0002, f"{v:.5f}", ha="center", va="bottom", fontsize=9)
for i, v in enumerate(ndcg):
    ax.text(i + w / 2, v + 0.0002, f"{v:.5f}", ha="center", va="bottom", fontsize=9)

ax.set_title("Tokyo: Strict Paper-style vs Gamma=0 Improvement")
ax.set_ylabel("Metric value")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(frameon=False)

fig.tight_layout()
out = Path("tokyo_strict_gamma_compare.png")
fig.savefig(out, dpi=220, bbox_inches="tight")
print(f"Saved: {out.resolve()}")

