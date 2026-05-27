import matplotlib.pyplot as plt
import numpy as np


cities = ["Chicago", "NYC", "Singapore", "Tokyo"]

# Gains = OTC - Backbone
paper_gain_r20_mf = np.array([0.0175, 0.0064, 0.0144, 0.0049])
repro_gain_r20_mf = np.array([0.0484, 0.0173, 0.0480, 0.0268])
paper_gain_r20_lgcn = np.array([0.0102, 0.0021, 0.0065, 0.0029])
repro_gain_r20_lgcn = np.array([0.0048, 0.0175, 0.0456, 0.0028])

paper_gain_ndcg_mf = np.array([0.0134, 0.0064, 0.0152, 0.0015])
repro_gain_ndcg_mf = np.array([0.0152, 0.0189, 0.0082, 0.0141])
paper_gain_ndcg_lgcn = np.array([0.0114, 0.0021, 0.0086, 0.0032])
repro_gain_ndcg_lgcn = np.array([0.0281, 0.0036, 0.0001, 0.0032])


def build_metric_plot(
    ax,
    paper_mf,
    repro_mf,
    paper_lgcn,
    repro_lgcn,
    title,
    ylabel,
):
    x = np.arange(len(cities))
    w = 0.18

    ax.bar(x - 1.5 * w, paper_mf, width=w, label="Paper MF gain", color="#9ecae1")
    ax.bar(x - 0.5 * w, repro_mf, width=w, label="Repro MF gain", color="#3182bd")
    ax.bar(x + 0.5 * w, paper_lgcn, width=w, label="Paper LightGCN gain", color="#fdae6b")
    ax.bar(x + 1.5 * w, repro_lgcn, width=w, label="Repro LightGCN gain", color="#e6550d")

    ax.set_xticks(x)
    ax.set_xticklabels(cities)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

build_metric_plot(
    axes[0],
    paper_gain_r20_mf,
    repro_gain_r20_mf,
    paper_gain_r20_lgcn,
    repro_gain_r20_lgcn,
    "OTC Gain Comparison (R@20): Paper vs Reproduction",
    "Absolute gain",
)

build_metric_plot(
    axes[1],
    paper_gain_ndcg_mf,
    repro_gain_ndcg_mf,
    paper_gain_ndcg_lgcn,
    repro_gain_ndcg_lgcn,
    "OTC Gain Comparison (nDCG@20): Paper vs Reproduction",
    "Absolute gain",
)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))

out = "paper_vs_repro_gain_comparison.png"
fig.savefig(out, dpi=220, bbox_inches="tight")
print(f"Saved: {out}")

