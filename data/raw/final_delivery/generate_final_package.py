import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "results_final" / "summary_with_otc_hybrid_singapore.json"
FIG_DIR = ROOT / "figures"
TEX_PATH = ROOT / "otc_reproduction_report.tex"
SUMMARY_OUT = ROOT / "summary.json"

FIG_DIR.mkdir(parents=True, exist_ok=True)


PAPER_TABLE3 = {
    "MF": {
        "Chicago": {"rec": 0.2494, "ndcg": 0.1465},
        "NYC": {"rec": 0.1702, "ndcg": 0.0917},
        "Singapore": {"rec": 0.4430, "ndcg": 0.2351},
        "Tokyo": {"rec": 0.1323, "ndcg": 0.0781},
    },
    "OTC-MF": {
        "Chicago": {"rec": 0.2669, "ndcg": 0.1599},
        "NYC": {"rec": 0.1766, "ndcg": 0.0981},
        "Singapore": {"rec": 0.4574, "ndcg": 0.2503},
        "Tokyo": {"rec": 0.1372, "ndcg": 0.0796},
    },
    "LightGCN": {
        "Chicago": {"rec": 0.2875, "ndcg": 0.1902},
        "NYC": {"rec": 0.2087, "ndcg": 0.1088},
        "Singapore": {"rec": 0.5013, "ndcg": 0.2745},
        "Tokyo": {"rec": 0.1751, "ndcg": 0.1068},
    },
    "OTC-LightGCN": {
        "Chicago": {"rec": 0.2977, "ndcg": 0.2016},
        "NYC": {"rec": 0.2108, "ndcg": 0.1109},
        "Singapore": {"rec": 0.5078, "ndcg": 0.2831},
        "Tokyo": {"rec": 0.1780, "ndcg": 0.1100},
    },
}


def load_summary():
    return json.loads(SRC.read_text(encoding="utf-8"))


def build_repro(summary):
    out = {"MF": {}, "OTC-MF": {}, "LightGCN": {}, "OTC-LightGCN": {}}
    for c, row in summary["VanillaMF"].items():
        out["MF"][c] = {"rec": row["recall_at_20"], "ndcg": row["ndcg_at_20"]}
    for c, row in summary["LightGCN"].items():
        out["LightGCN"][c] = {"rec": row["recall_at_20"], "ndcg": row["ndcg_at_20"]}
    for row in summary["OTC_VanillaMF"]["results"]:
        c = row["target_city"]
        out["OTC-MF"][c] = {"rec": row["best_recall"], "ndcg": row["best_ndcg"]}
    for row in summary["OTC_LightGCN"]["results"]:
        c = row["target_city"]
        out["OTC-LightGCN"][c] = {"rec": row["best_recall"], "ndcg": row["best_ndcg"]}
    return out


def make_main_figure(repro):
    cities = ["Chicago", "NYC", "Singapore", "Tokyo"]
    methods = ["MF", "OTC-MF", "LightGCN", "OTC-LightGCN"]
    x = np.arange(len(cities))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), dpi=220)
    for i, m in enumerate(methods):
        paper_r = [PAPER_TABLE3[m][c]["rec"] for c in cities]
        repro_r = [repro[m][c]["rec"] for c in cities]
        paper_n = [PAPER_TABLE3[m][c]["ndcg"] for c in cities]
        repro_n = [repro[m][c]["ndcg"] for c in cities]
        shift = (i - 1.5) * width
        axes[0].bar(x + shift, paper_r, width, alpha=0.35)
        axes[0].bar(x + shift, repro_r, width, label=f"{m} (Repro)")
        axes[1].bar(x + shift, paper_n, width, alpha=0.35)
        axes[1].bar(x + shift, repro_n, width, label=f"{m} (Repro)")

    axes[0].set_title("Recall@20: Paper (light) vs Reproduction (solid)")
    axes[1].set_title("nDCG@20: Paper (light) vs Reproduction (solid)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(cities)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = FIG_DIR / "main_results_paper_vs_repro.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_gain_figure(repro):
    cities = ["Chicago", "NYC", "Singapore", "Tokyo"]
    paper_mf = [100.0 * (PAPER_TABLE3["OTC-MF"][c]["rec"] / PAPER_TABLE3["MF"][c]["rec"] - 1.0) for c in cities]
    repro_mf = [100.0 * (repro["OTC-MF"][c]["rec"] / repro["MF"][c]["rec"] - 1.0) for c in cities]
    paper_lg = [100.0 * (PAPER_TABLE3["OTC-LightGCN"][c]["rec"] / PAPER_TABLE3["LightGCN"][c]["rec"] - 1.0) for c in cities]
    repro_lg = [100.0 * (repro["OTC-LightGCN"][c]["rec"] / repro["LightGCN"][c]["rec"] - 1.0) for c in cities]

    x = np.arange(len(cities))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 4), dpi=220)
    ax.bar(x - 1.5 * width, paper_mf, width, alpha=0.35, label="Paper OTC-MF gain")
    ax.bar(x - 0.5 * width, repro_mf, width, label="Repro OTC-MF gain")
    ax.bar(x + 0.5 * width, paper_lg, width, alpha=0.35, label="Paper OTC-LightGCN gain")
    ax.bar(x + 1.5 * width, repro_lg, width, label="Repro OTC-LightGCN gain")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cities)
    ax.set_ylabel("Recall@20 relative gain (%)")
    ax.set_title("OTC Gain Comparison (Paper vs Reproduction)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = FIG_DIR / "otc_gain_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_heatmap(summary):
    targets = ["Chicago", "NYC", "Singapore", "Tokyo"]
    sources = ["Chicago", "NYC", "Singapore", "Tokyo"]
    mat = np.zeros((len(targets), len(sources)), dtype=float)
    rows = summary["OTC_LightGCN"]["results"]
    by_target = {r["target_city"]: r for r in rows}
    for ti, t in enumerate(targets):
        r = by_target[t]
        for x in r.get("source_prescreen", []):
            s = x["source_city"]
            if s in sources:
                sj = sources.index(s)
                mat[ti, sj] = x["improv_ndcg_abs"]

    fig, ax = plt.subplots(figsize=(6, 4.8), dpi=220)
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(sources)))
    ax.set_yticks(np.arange(len(targets)))
    ax.set_xticklabels(sources)
    ax.set_yticklabels(targets)
    ax.set_title("Single-Source nDCG Contribution (OTC-LightGCN)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.85, label="nDCG absolute gain")
    fig.tight_layout()
    out = FIG_DIR / "source_contribution_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def latex_table(repro):
    cities = ["Chicago", "NYC", "Singapore", "Tokyo"]
    methods = ["MF", "OTC-MF", "LightGCN", "OTC-LightGCN"]
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Paper-reported results (SIGIR 2024) vs final reproduction (Singapore from previous-round OTC).}")
    lines.append("\\label{tab:paper_vs_repro}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l|cc|cc|cc|cc}")
    lines.append("\\toprule")
    lines.append("Method & \\multicolumn{2}{c|}{Chicago} & \\multicolumn{2}{c|}{NYC} & \\multicolumn{2}{c|}{Singapore} & \\multicolumn{2}{c}{Tokyo}\\\\")
    lines.append("& R@20 & nDCG@20 & R@20 & nDCG@20 & R@20 & nDCG@20 & R@20 & nDCG@20\\\\")
    lines.append("\\midrule")
    for m in methods:
        pvals, rvals = [], []
        for c in cities:
            pvals.extend([f"{PAPER_TABLE3[m][c]['rec']:.4f}", f"{PAPER_TABLE3[m][c]['ndcg']:.4f}"])
            rvals.extend([f"{repro[m][c]['rec']:.4f}", f"{repro[m][c]['ndcg']:.4f}"])
        lines.append(f"{m} (Paper) & " + " & ".join(pvals) + "\\\\")
        lines.append(f"{m} (Repro) & " + " & ".join(rvals) + "\\\\")
        lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def write_tex(repro, fig_main, fig_gain, fig_heat):
    tex = f"""\\documentclass[10pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{multirow}}
\\title{{Final Reproduction Report: Optimal Transport Enhanced Cross-City Site Recommendation}}
\\author{{OTCRe Reproduction Pipeline}}
\\date{{}}
\\begin{{document}}
\\maketitle

\\begin{{abstract}}
We reproduce all core experiments in the SIGIR 2024 OTC paper on four cities (Chicago, NYC, Singapore, Tokyo), including MF, LightGCN, OTC-MF, and OTC-LightGCN. Singapore uses the previous-round OTC result while the other cities use the latest strict-protocol OTC result.
\\end{{abstract}}

\\section{{Main Results}}
{latex_table(repro)}

\\begin{{figure*}}[t]
\\centering
\\includegraphics[width=0.95\\textwidth]{{figures/{fig_main.name}}}
\\caption{{Main metrics comparison: paper values (light bars) vs reproduction values (solid bars).}}
\\end{{figure*}}

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.95\\linewidth]{{figures/{fig_gain.name}}}
\\caption{{OTC relative Recall@20 gain over its backbone: paper vs reproduction.}}
\\end{{figure}}

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.95\\linewidth]{{figures/{fig_heat.name}}}
\\caption{{Single-source nDCG contribution matrix for OTC-LightGCN (absolute gain).}}
\\end{{figure}}

\\section{{Discussion}}
OTC improves some cities and some backbones, but not uniformly. In the final package, Singapore is kept from the better previous-round OTC result as requested.

\\end{{document}}
"""
    TEX_PATH.write_text(tex, encoding="utf-8")


def main():
    summary = load_summary()
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    repro = build_repro(summary)
    fig_main = make_main_figure(repro)
    fig_gain = make_gain_figure(repro)
    fig_heat = make_heatmap(summary)
    write_tex(repro, fig_main, fig_gain, fig_heat)
    print(f"Wrote: {TEX_PATH}")
    print(f"Figures: {FIG_DIR}")
    print(f"Summary: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
