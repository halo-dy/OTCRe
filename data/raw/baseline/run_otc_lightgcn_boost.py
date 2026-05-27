import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "raw" / "baseline"
OUT = ROOT / "data" / "raw" / "results_otc_boost"
OUT.mkdir(parents=True, exist_ok=True)


SEARCH = [
    # paper-style baseline
    {
        "name": "classic_sigmoid_zscore_coord",
        "gw_solver": "classic",
        "score_space": "sigmoid",
        "score_norm": "zscore",
        "per_source_gamma": "1",
        "allow_zero_gamma": "0",
        "max_coord_iter": "16",
    },
    # global gamma, may generalize better when coord search overfits
    {
        "name": "classic_sigmoid_zscore_global",
        "gw_solver": "classic",
        "score_space": "sigmoid",
        "score_norm": "zscore",
        "per_source_gamma": "0",
        "allow_zero_gamma": "0",
        "max_coord_iter": "1",
    },
    # raw-space fallback
    {
        "name": "classic_raw_none_coord",
        "gw_solver": "classic",
        "score_space": "raw",
        "score_norm": "none",
        "per_source_gamma": "1",
        "allow_zero_gamma": "0",
        "max_coord_iter": "16",
    },
]


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print("RUN:", " ".join(cmd))
    print(p.stdout[-1600:])
    if p.returncode != 0:
        print(p.stderr[-1600:])
        raise RuntimeError("command failed")


def run_one(cfg):
    out_file = OUT / f"otc_lightgcn_{cfg['name']}.json"
    cmd = [
        "python", str(BASE / "otc_lightgcn.py"),
        "--all_targets", "1",
        "--model", "LightGCN",
        "--topk", "20",
        "--gammas", "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0",
        "--prescreen_sources", "0",
        "--min_improv_ndcg", "0.0",
        "--gw_solver", cfg["gw_solver"],
        "--score_space", cfg["score_space"],
        "--score_norm", cfg["score_norm"],
        "--per_source_gamma", cfg["per_source_gamma"],
        "--allow_zero_gamma", cfg["allow_zero_gamma"],
        "--max_coord_iter", cfg["max_coord_iter"],
        "--metrics_out", str(out_file),
    ]
    run(cmd)
    return json.loads(out_file.read_text(encoding="utf-8"))


def aggregate_gain(payload):
    rows = payload["results"]
    rec_gain = sum(r["improv_recall_pct"] for r in rows) / len(rows)
    nd_gain = sum(r["improv_ndcg_pct"] for r in rows) / len(rows)
    return rec_gain, nd_gain


def main():
    summary = {}
    best_name = None
    best_score = None

    for cfg in SEARCH:
        payload = run_one(cfg)
        rec_gain, nd_gain = aggregate_gain(payload)
        summary[cfg["name"]] = {
            "avg_improv_recall_pct": rec_gain,
            "avg_improv_ndcg_pct": nd_gain,
            "results": payload["results"],
        }
        score = (nd_gain, rec_gain)
        if best_score is None or score > best_score:
            best_score = score
            best_name = cfg["name"]

    out = {
        "best_config": best_name,
        "best_score": {"avg_improv_ndcg_pct": best_score[0], "avg_improv_recall_pct": best_score[1]},
        "all": summary,
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("WROTE", OUT / "summary.json")


if __name__ == "__main__":
    main()
