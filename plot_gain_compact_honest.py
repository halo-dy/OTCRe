import matplotlib.pyplot as plt
import numpy as np


cities = ["Chicago", "NYC", "Singapore", "Tokyo"]
x = np.arange(len(cities))

# Absolute gains (OTC - Backbone)
paper_r20_mf = np.array([0.0175, 0.0064, 0.0144, 0.0049])
repro_r20_mf = np.array([0.0484, 0.0173, 0.0480, 0.0268])
paper_r20_lgcn = np.array([0.0102, 0.0021, 0.0065, 0.0029])
repro_r20_lgcn = np.array([0.0048, 0.0175, 0.0456, 0.0028])

paper_ndcg_mf = np.array([0.0134, 0.0064, 0.0152, 0.0015])
repro_ndcg_mf = np.array([0.0152, 0.089, 0.0082, 0.0141])
paper_ndcg_lgcn = np.array([0.0114, 0.0021, 0.0086, 0.0032])
repro_ndcg_lgcn = np.array([0.0181, 0.0036, 0.0041, 0.0032])


def add_panel(ax, title, p_mf, r_mf, p_lgcn, r_lgcn):
    ax.plot(x, p_mf, marker="o", color="#4C78A8", linewidth=1.8, label="Paper MF")
    ax.plot(x, r_mf, marker="o", color="#4C78A8", linestyle="--", linewidth=1.8, label="Repro MF")
    ax.plot(x, p_lgcn, marker="s", color="#F58518", linewidth=1.8, label="Paper LightGCN")
    ax.plot(x, r_lgcn, marker="s", color="#F58518", linestyle="--", linewidth=1.8, label="Repro LightGCN")

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(cities)
    ax.set_ylim(0, 0.06)  # fixed shared scale to avoid visual exaggeration
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylabel("Absolute gain")


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

add_panel(
    axes[0],
    "OTC Gain on R@20 (Paper vs Repro)",
    paper_r20_mf,
    repro_r20_mf,
    paper_r20_lgcn,
    repro_r20_lgcn,
)

add_panel(
    axes[1],
    "OTC Gain on nDCG@20 (Paper vs Repro)",
    paper_ndcg_mf,
    repro_ndcg_mf,
    paper_ndcg_lgcn,
    repro_ndcg_lgcn,
)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))

out = "paper_vs_repro_gain_compact_honest.png"
fig.savefig(out, dpi=220, bbox_inches="tight")
print(f"Saved: {out}")

