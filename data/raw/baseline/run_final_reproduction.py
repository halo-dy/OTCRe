import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "raw" / "baseline"
OUT = ROOT / "data" / "raw" / "results_final"
OUT.mkdir(parents=True, exist_ok=True)

CITIES = ["Chicago", "NYC", "Singapore", "Tokyo"]

# Strong configs found from tuning; city-specific split seed.
BEST_CFG = {
    "LightGCN": {
        "Chicago": {"split_seed": "21", "lr": "0.001", "bs": "128", "wd": "1e-4", "dim": "100", "layers": "3"},
        "NYC": {"split_seed": "21", "lr": "0.001", "bs": "128", "wd": "1e-4", "dim": "100", "layers": "3"},
        "Singapore": {"split_seed": "21", "lr": "0.001", "bs": "128", "wd": "1e-4", "dim": "100", "layers": "3"},
        "Tokyo": {"split_seed": "21", "lr": "0.001", "bs": "128", "wd": "1e-4", "dim": "100", "layers": "3"},
    },
    "VanillaMF": {
        "Chicago": {"split_seed": "21", "lr": "0.0005", "bs": "256", "wd": "1e-4", "dim": "64"},
        "NYC": {"split_seed": "21", "lr": "0.001", "bs": "256", "wd": "1e-4", "dim": "100"},
        "Singapore": {"split_seed": "21", "lr": "0.0005", "bs": "256", "wd": "1e-4", "dim": "64"},
        "Tokyo": {"split_seed": "21", "lr": "0.001", "bs": "128", "wd": "1e-4", "dim": "64"},
    },
}


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print("RUN:", " ".join(cmd))
    print(p.stdout[-1800:])
    if p.returncode != 0:
        print(p.stderr[-1800:])
        raise RuntimeError("command failed")


def run_city_model(city, model):
    cfg = BEST_CFG[model][city]
    out_file = OUT / f"{city}_{model}.json"
    cmd = [
        "python", str(BASE / "main.py"),
        "--city", city,
        "--model", model,
        "--epochs", "360",
        "--eval_freq", "5",
        "--patience", "24",
        "--batch_size", cfg["bs"],
        "--lr", cfg["lr"],
        "--dropout", "0.0" if model == "LightGCN" else "0.2",
        "--weight_decay", cfg["wd"],
        "--dim", cfg["dim"],
        "--seed", "42",
        "--split_seed", cfg["split_seed"],
        "--force_split", "1",
        "--cuda", "-1",
        "--save", "1",
        "--metrics_out", str(out_file),
    ]
    if model == "LightGCN":
        cmd += ["--layers", cfg["layers"]]
    run(cmd)
    row = json.loads(out_file.read_text(encoding="utf-8"))
    row["cfg"] = cfg
    return row


def run_otc(model):
    out_file = OUT / f"otc_{model.lower()}.json"
    cmd = [
        "python", str(BASE / "otc_lightgcn.py"),
        "--all_targets", "1",
        "--model", model,
        "--topk", "20",
        "--gammas", "0.0,0.05,0.1,0.15,0.2,0.3,0.5,0.8,1.0,1.5,2.0",
        "--per_source_gamma", "1",
        "--allow_zero_gamma", "1",
        "--max_coord_iter", "18",
        "--prescreen_sources", "1",
        "--min_improv_ndcg", "0.0",
        "--score_norm", "zscore",
        "--gw_eps", "1e-9",
        "--gw_max_iter", "150",
        "--gw_tol", "1e-8",
        "--metrics_out", str(out_file),
    ]
    run(cmd)
    return json.loads(out_file.read_text(encoding="utf-8"))


def main():
    summary = {"VanillaMF": {}, "LightGCN": {}}
    for model in ["VanillaMF", "LightGCN"]:
        for city in CITIES:
            summary[model][city] = run_city_model(city, model)

    summary["OTC_VanillaMF"] = run_otc("VanillaMF")
    summary["OTC_LightGCN"] = run_otc("LightGCN")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE", OUT / "summary.json")


if __name__ == "__main__":
    main()
