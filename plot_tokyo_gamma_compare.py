import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


base_dir = Path("data/raw/baseline")
f_a = base_dir / "tokyo_gamma05plus.json"
f_b = base_dir / "tokyo_gamma0plus.json"

data_a = json.loads(f_a.read_text(encoding="utf-8"))["results"][0]
data_b = json.loads(f_b.read_text(encoding="utf-8"))["results"][0]

baseline_r = data_a["baseline_recall"]
baseline_n = data_a["baseline_ndcg"]

vals_r = [baseline_r, data_b["best_recall"]]
vals_n = [baseline_n, data_b["best_ndcg"]]
labels = ["Baseline", "Gamma >= 0"]

x = np.arange(len(labels))
w = 0.36

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.bar(x - w / 2, vals_r, width=w, label="R@20", color="#4C78A8")
ax.bar(x + w / 2, vals_n, width=w, label="nDCG@20", color="#F58518")

for i, v in enumerate(vals_r):
    ax.text(i - w / 2, v + 0.00025, f"{v:.5f}", ha="center", va="bottom", fontsize=9)
for i, v in enumerate(vals_n):
    ax.text(i + w / 2, v + 0.00025, f"{v:.5f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Metric value")
ax.set_title("Tokyo: Baseline vs Gamma >= 0")
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(frameon=False)

out = Path("tokyo_gamma_compare.png")
fig.tight_layout()
fig.savefig(out, dpi=220, bbox_inches="tight")
print(f"Saved: {out.resolve()}")
