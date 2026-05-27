import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "raw" / "baseline"
OUT = ROOT / "data" / "raw" / "results_otc_readd"
OUT.mkdir(parents=True, exist_ok=True)


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print("RUN:", " ".join(cmd))
    print(p.stdout[-1800:])
    if p.returncode != 0:
        print(p.stderr[-1800:])
        raise RuntimeError("command failed")


def run_otc(model):
    out_file = OUT / f"otc_{model.lower()}_paperstyle.json"
    cmd = [
        "python", str(BASE / "otc_lightgcn.py"),
        "--all_targets", "1",
        "--model", model,
        "--topk", "20",
        "--gammas", "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0",
        "--per_source_gamma", "1",
        "--allow_zero_gamma", "0",
        "--max_coord_iter", "12",
        "--prescreen_sources", "0",
        "--score_norm", "zscore",
        "--score_space", "sigmoid",
        "--gw_solver", "classic",
        "--metrics_out", str(out_file),
    ]
    run(cmd)
    return json.loads(out_file.read_text(encoding="utf-8"))


def main():
    summary = {
        "OTC_VanillaMF": run_otc("VanillaMF"),
        "OTC_LightGCN": run_otc("LightGCN"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE", OUT / "summary.json")


if __name__ == "__main__":
    main()
