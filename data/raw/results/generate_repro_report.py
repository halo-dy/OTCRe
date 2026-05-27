import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "fig_repro"
TEX_PATH = ROOT / "otc_reproduction_report.tex"

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


def load_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def load_repro():
    mf = load_json("otc_mf_all.json")
    lgcn = load_json("otc_lightgcn_all.json")

    repro = {"MF": {}, "OTC-MF": {}, "LightGCN": {}, "OTC-LightGCN": {}}
    for x in mf["results"]:
        city = x["target_city"]
        repro["MF"][city] = {"rec": x["baseline_recall"], "ndcg": x["baseline_ndcg"]}
        repro["OTC-MF"][city] = {"rec": x["best_recall"], "ndcg": x["best_ndcg"], "gamma": x["best_gamma"]}
    for x in lgcn["results"]:
        city = x["target_city"]
        repro["LightGCN"][city] = {"rec": x["baseline_recall"], "ndcg": x["baseline_ndcg"]}
        repro["OTC-LightGCN"][city] = {"rec": x["best_recall"], "ndcg": x["best_ndcg"], "gamma": x["best_gamma"]}
    return repro, mf, lgcn


def make_main_bar(repro):
    cities = ["Chicago", "NYC", "Singapore", "Tokyo"]
    models = ["MF", "OTC-MF", "LightGCN", "OTC-LightGCN"]
    x = np.arange(len(cities))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=200)
    for i, m in enumerate(models):
        rec = [repro[m][c]["rec"] for c in cities]
        nd = [repro[m][c]["ndcg"] for c in cities]
        axes[0].bar(x + (i - 1.5) * width, rec, width, label=m)
        axes[1].bar(x + (i - 1.5) * width, nd, width, label=m)

    axes[0].set_title("Recall@20 on OpenSiteRec (Reproduced)")
    axes[1].set_title("nDCG@20 on OpenSiteRec (Reproduced)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(cities)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "main_results_repro.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_improv_bar(repro):
    cities = ["Chicago", "NYC", "Singapore", "Tokyo"]
    improv_mf = [(repro["OTC-MF"][c]["rec"] / repro["MF"][c]["rec"] - 1.0) * 100.0 for c in cities]
    improv_lgcn = [(repro["OTC-LightGCN"][c]["rec"] / repro["LightGCN"][c]["rec"] - 1.0) * 100.0 for c in cities]

    x = np.arange(len(cities))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=200)
    ax.bar(x - width / 2, improv_mf, width, label="OTC-MF vs MF")
    ax.bar(x + width / 2, improv_lgcn, width, label="OTC-LightGCN vs LightGCN")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cities)
    ax.set_title("Relative Improvement of Recall@20 (Reproduced)")
    ax.set_ylabel("Improvement (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "improv_recall_repro.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_gamma_curve(otc_json, model_name, city):
    rows = [r for r in otc_json["results"] if r["target_city"] == city][0]["gamma_results"]
    gam = [r["gamma"] for r in rows]
    rec = [r["recall"] for r in rows]
    nd = [r["ndcg"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.0, 3.8), dpi=200)
    ax1.plot(gam, rec, marker="o", label="Recall@20")
    ax1.plot(gam, nd, marker="s", label="nDCG@20")
    ax1.set_xlabel("gamma")
    ax1.set_title(f"Gamma Sensitivity on {city} ({model_name}, Reproduced)")
    ax1.grid(linestyle="--", alpha=0.35)
    ax1.legend()
    fig.tight_layout()
    out = FIG_DIR / f"gamma_{model_name.lower()}_{city.lower()}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def latex_table_paper_vs_repro(repro):
    cities = ["Chicago", "NYC", "Singapore", "Tokyo"]
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Paper-reported results vs reproduced results on OpenSiteRec.}")
    lines.append("\\label{tab:paper_vs_repro}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l|cc|cc|cc|cc}")
    lines.append("\\toprule")
    lines.append("Method & \\multicolumn{2}{c|}{Chicago} & \\multicolumn{2}{c|}{NYC} & \\multicolumn{2}{c|}{Singapore} & \\multicolumn{2}{c}{Tokyo}\\\\")
    lines.append("& R@20 & nDCG@20 & R@20 & nDCG@20 & R@20 & nDCG@20 & R@20 & nDCG@20\\\\")
    lines.append("\\midrule")
    for m in ["MF", "OTC-MF", "LightGCN", "OTC-LightGCN"]:
        paper_vals = []
        repro_vals = []
        for c in cities:
            paper_vals.append(f"{PAPER_TABLE3[m][c]['rec']:.4f}")
            paper_vals.append(f"{PAPER_TABLE3[m][c]['ndcg']:.4f}")
        for c in cities:
            repro_vals.append(f"{repro[m][c]['rec']:.4f}")
            repro_vals.append(f"{repro[m][c]['ndcg']:.4f}")
        lines.append(f"{m} (Paper) & " + " & ".join(paper_vals) + "\\\\")
        lines.append(f"{m} (Repro) & " + " & ".join(repro_vals) + "\\\\")
        lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def write_latex(repro, fig_main, fig_improv, fig_g1, fig_g2):
    tex = f"""\\documentclass[10pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{multirow}}
\\title{{Reproduction Report: Optimal Transport Enhanced Cross-City Site Recommendation}}
\\author{{Reproduction in OTCRe Repository}}
\\date{{}}
\\begin{{document}}
\\maketitle

\\begin{{abstract}}
This report reproduces the full experimental workflow of ``Optimal Transport Enhanced Cross-City Site Recommendation'' on all four target cities (Chicago, NYC, Singapore, Tokyo), including MF-BPR, LightGCN, OTC-MF, and OTC-LightGCN. We follow the same evaluation metrics (Recall@20, nDCG@20), 5-core filtering, and cross-city optimal transport setting, and provide direct paper-vs-reproduction comparisons in table and figure form.
\\end{{abstract}}

\\section{{Introduction}}
The target paper proposes OTC, which uses Gromov-Wasserstein optimal transport to project brand/region representations from source cities to a target city and fuse projected inference with the target-city recommender output.

\\section{{Methodology}}
We implement the pipeline in the same structure as the paper:
\\begin{{enumerate}}
\\item Train backbone models (MF-BPR, LightGCN) per city.
\\item Compute GW transport plans for brand and region embeddings across city pairs.
\\item Perform barycentric projection from source to target embeddings.
\\item Fuse source-inferred scores with target scores by a weight $\\gamma$.
\\item Grid-search $\\gamma$ and report best Recall@20 and nDCG@20.
\\end{{enumerate}}

\\section{{Experimental Settings}}
We use OpenSiteRec data from the repository, 5-core brand filtering, all-ranking evaluation protocol, and train/val/test split of 70\\%/10\\%/20\\%. Models are trained with Adam and early stopping on validation nDCG@20.

\\section{{Main Results}}
{latex_table_paper_vs_repro(repro)}

\\begin{{figure*}}[t]
\\centering
\\includegraphics[width=0.95\\textwidth]{{{fig_main.as_posix()}}}
\\caption{{Reproduced main results on four cities, matching paper-style grouped comparison across methods.}}
\\label{{fig:main_repro}}
\\end{{figure*}}

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.9\\linewidth]{{{fig_improv.as_posix()}}}
\\caption{{Relative Recall@20 improvement of OTC over its backbone methods in reproduction.}}
\\label{{fig:improv_repro}}
\\end{{figure}}

\\section{{Gamma Sensitivity}}
\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.9\\linewidth]{{{fig_g1.as_posix()}}}
\\caption{{Gamma sensitivity curve (OTC-LightGCN on Chicago).}}
\\label{{fig:gamma_lgcn_chicago}}
\\end{{figure}}

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.9\\linewidth]{{{fig_g2.as_posix()}}}
\\caption{{Gamma sensitivity curve (OTC-MF on Chicago).}}
\\label{{fig:gamma_mf_chicago}}
\\end{{figure}}

\\section{{Discussion}}
The reproduction recovers the full multi-city workflow and partially matches the trend of OTC gains in some cities (e.g., Chicago/Tokyo), while deviating on others (notably Singapore under current implementation). This indicates remaining implementation or setting gaps compared with the original authors' internal code path.

\\section{{Conclusion}}
We complete end-to-end reproduction code and reporting artifacts for all cities and all core methods in the paper, with structured comparison to paper-reported numbers.

\\end{{document}}
"""
    TEX_PATH.write_text(tex, encoding="utf-8")


def main():
    repro, mf_json, lg_json = load_repro()
    fig_main = make_main_bar(repro)
    fig_improv = make_improv_bar(repro)
    fig_g1 = make_gamma_curve(lg_json, "LightGCN", "Chicago")
    fig_g2 = make_gamma_curve(mf_json, "VanillaMF", "Chicago")
    write_latex(repro, fig_main, fig_improv, fig_g1, fig_g2)
    print(f"Wrote LaTeX report to: {TEX_PATH}")
    print(f"Figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
