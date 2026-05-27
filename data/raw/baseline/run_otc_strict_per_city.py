import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "raw" / "baseline"
OUT = ROOT / "data" / "raw" / "results_otc_strict"
OUT.mkdir(parents=True, exist_ok=True)

CITIES = ["Chicago", "NYC", "Singapore", "Tokyo"]


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print("RUN:", " ".join(cmd))
    print(p.stdout[-1600:])
    if p.returncode != 0:
        print(p.stderr[-1600:])
        raise RuntimeError("command failed")


def run_one(model, target):
    out_file = OUT / f"{model}_{target}.json"
    cmd = [
        "python", str(BASE / "otc_lightgcn.py"),
        "--target_city", target,
        "--model", model,
        "--topk", "20",
        "--gammas", "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0",
        "--per_source_gamma", "1",
        "--allow_zero_gamma", "0",
        "--max_coord_iter", "20",
        "--prescreen_sources", "0",
        "--score_norm", "none",
        "--score_space", "raw",
        "--gw_solver", "classic",
        "--metrics_out", str(out_file),
    ]
    run(cmd)
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    return payload["results"][0]


def main():
    out = {"OTC_VanillaMF": {"results": []}, "OTC_LightGCN": {"results": []}}
    for city in CITIES:
        out["OTC_VanillaMF"]["results"].append(run_one("VanillaMF", city))
    for city in CITIES:
        out["OTC_LightGCN"]["results"].append(run_one("LightGCN", city))

    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("WROTE", OUT / "summary.json")


if __name__ == "__main__":
    main()
