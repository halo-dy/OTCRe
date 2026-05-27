import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "raw" / "baseline"
OUT = ROOT / "data" / "raw" / "results_opt4"
OUT.mkdir(parents=True, exist_ok=True)

CITIES = ["Chicago", "NYC", "Singapore", "Tokyo"]
SPLIT_SEEDS = [7, 21, 42, 84, 168]

LIGHTGCN_GRID = [
    {"lr": "0.005", "bs": "128", "wd": "1e-4", "dim": "64", "layers": "2"},
    {"lr": "0.001", "bs": "128", "wd": "1e-4", "dim": "64", "layers": "2"},
    {"lr": "0.0005", "bs": "256", "wd": "1e-4", "dim": "64", "layers": "2"},
    {"lr": "0.005", "bs": "128", "wd": "5e-5", "dim": "100", "layers": "3"},
    {"lr": "0.001", "bs": "128", "wd": "5e-5", "dim": "100", "layers": "3"},
    {"lr": "0.0005", "bs": "256", "wd": "5e-5", "dim": "100", "layers": "3"},
]

MF_GRID = [
    {"lr": "0.005", "bs": "128", "wd": "1e-4", "dim": "64"},
    {"lr": "0.001", "bs": "128", "wd": "1e-4", "dim": "64"},
    {"lr": "0.0005", "bs": "256", "wd": "1e-4", "dim": "64"},
    {"lr": "0.005", "bs": "128", "wd": "5e-5", "dim": "100"},
    {"lr": "0.001", "bs": "128", "wd": "5e-5", "dim": "100"},
]


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print("RUN:", " ".join(cmd))
    print(p.stdout[-1800:])
    if p.returncode != 0:
        print(p.stderr[-1800:])
        raise RuntimeError("command failed")


def pick_best(rows):
    rows.sort(key=lambda x: (x["ndcg_at_20"], x["recall_at_20"]), reverse=True)
    return rows[0]


def train_one(city, model, split_seed, cfg_id, cfg):
    out_file = OUT / f"{city}_{model}_split{split_seed}_cfg{cfg_id}.json"
    cmd = [
        "python", str(BASE / "main.py"),
        "--city", city,
        "--model", model,
        "--epochs", "340",
        "--eval_freq", "5",
        "--patience", "24",
        "--batch_size", cfg["bs"],
        "--lr", cfg["lr"],
        "--dropout", "0.0" if model == "LightGCN" else "0.2",
        "--weight_decay", cfg["wd"],
        "--dim", cfg["dim"],
        "--seed", "42",
        "--split_seed", str(split_seed),
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
    row["cfg_id"] = cfg_id
    row["split_seed"] = split_seed
    return row


def run_city_model(city, model):
    grid = LIGHTGCN_GRID if model == "LightGCN" else MF_GRID
    rows = []
    for split_seed in SPLIT_SEEDS:
        for i, cfg in enumerate(grid):
            rows.append(train_one(city, model, split_seed, i, cfg))
    best = pick_best(rows)
    (OUT / f"{city}_{model}_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best


def run_otc(model):
    out_file = OUT / f"otc_{model.lower()}_all.json"
    cmd = [
        "python", str(BASE / "otc_lightgcn.py"),
        "--all_targets", "1",
        "--model", model,
        "--topk", "20",
        "--gammas", "0.0,0.05,0.1,0.15,0.2,0.3,0.5,0.8,1.0,1.5,2.0,2.5",
        "--per_source_gamma", "1",
        "--allow_zero_gamma", "1",
        "--max_coord_iter", "18",
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
