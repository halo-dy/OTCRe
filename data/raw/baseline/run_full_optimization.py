import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "data" / "raw" / "baseline"
RESULTS = ROOT / "data" / "raw" / "results_opt"
RESULTS.mkdir(parents=True, exist_ok=True)

CITIES = ["Chicago", "NYC", "Singapore", "Tokyo"]


def run_cmd(cmd):
    print("RUN:", " ".join(cmd))
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(p.stdout[-4000:])
    if p.returncode != 0:
        print(p.stderr[-4000:])
        raise RuntimeError("Command failed")


def train_grid(model, city):
    # Keep grid compact but meaningful; paper uses lr/batch grid search.
    grid = [
        {"lr": "0.01", "batch_size": "128"},
        {"lr": "0.005", "batch_size": "128"},
        {"lr": "0.001", "batch_size": "128"},
        {"lr": "0.0005", "batch_size": "256"},
    ]
    # LightGCN paper baseline typically no dropout in vanilla setting.
    dropout = "0.0" if model == "LightGCN" else "0.2"
    out_rows = []
    for i, g in enumerate(grid):
        out_file = RESULTS / f"{city}_{model}_grid{i}.json"
        cmd = [
            "python", str(BASELINE / "main.py"),
            "--city", city,
            "--model", model,
            "--epochs", "220",
            "--eval_freq", "5",
            "--patience", "12",
            "--batch_size", g["batch_size"],
            "--lr", g["lr"],
            "--dropout", dropout,
            "--weight_decay", "1e-4",
            "--cuda", "-1",
            "--save", "1",
            "--metrics_out", str(out_file),
        ]
        run_cmd(cmd)
        row = json.loads(out_file.read_text(encoding="utf-8"))
        row["grid_id"] = i
        out_rows.append(row)
    # choose by nDCG@20, then Recall@20
    out_rows.sort(key=lambda x: (x["ndcg_at_20"], x["recall_at_20"]), reverse=True)
    best = out_rows[0]
    (RESULTS / f"{city}_{model}_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best


def run_otc(model):
    out_file = RESULTS / f"otc_{model.lower()}_all.json"
    cmd = [
        "python", str(BASELINE / "otc_lightgcn.py"),
        "--all_targets", "1",
        "--model", model,
        "--topk", "20",
        "--gammas", "0.1,0.2,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0",
        "--per_source_gamma", "1",
        "--allow_zero_gamma", "1",
        "--max_coord_iter", "12",
        "--metrics_out", str(out_file),
    ]
    run_cmd(cmd)
    return json.loads(out_file.read_text(encoding="utf-8"))


def main():
    summary = {"MF": {}, "LightGCN": {}}
    for city in CITIES:
        summary["MF"][city] = train_grid("VanillaMF", city)
        summary["LightGCN"][city] = train_grid("LightGCN", city)

    otc_mf = run_otc("VanillaMF")
    otc_lg = run_otc("LightGCN")
    summary["OTC_MF"] = otc_mf
    summary["OTC_LightGCN"] = otc_lg

    (RESULTS / "optimization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Wrote:", RESULTS / "optimization_summary.json")


if __name__ == "__main__":
    main()
