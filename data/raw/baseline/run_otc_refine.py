import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "raw" / "baseline"
OUT = ROOT / "data" / "raw" / "results_opt6"
OUT.mkdir(parents=True, exist_ok=True)


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print("RUN:", " ".join(cmd))
    print(p.stdout[-1500:])
    if p.returncode != 0:
        print(p.stderr[-1500:])
        raise RuntimeError("command failed")


def run_otc(model):
    out_file = OUT / f"otc_{model.lower()}_refined.json"
    cmd = [
        "python", str(BASE / "otc_lightgcn.py"),
        "--all_targets", "1",
        "--model", model,
        "--topk", "20",
        "--gammas", "0.0,0.05,0.1,0.15,0.2,0.3,0.5,0.8,1.0,1.5,2.0",
        "--per_source_gamma", "1",
        "--allow_zero_gamma", "1",
        "--max_coord_iter", "20",
        "--prescreen_sources", "1",
        "--min_improv_ndcg", "0.0",
        "--score_norm", "zscore",
        "--metrics_out", str(out_file),
    ]
    run(cmd)
    return json.loads(out_file.read_text(encoding="utf-8"))


def main():
    summary = {
        "OTC_VanillaMF_refined": run_otc("VanillaMF"),
        "OTC_LightGCN_refined": run_otc("LightGCN"),
    }
    (OUT / "otc_refined_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE", OUT / "otc_refined_summary.json")


if __name__ == "__main__":
    main()
