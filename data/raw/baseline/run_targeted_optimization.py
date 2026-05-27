import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "raw" / "baseline"
OUT = ROOT / "data" / "raw" / "results_opt2"
OUT.mkdir(parents=True, exist_ok=True)

CITIES = ["Chicago", "NYC", "Singapore", "Tokyo"]

# City-specific search spaces to avoid one-size-fits-all failure.
SEARCH = {
    "VanillaMF": {
        "Chicago": [
            {"lr": "0.005", "bs": "128", "dropout": "0.2", "wd": "1e-4"},
            {"lr": "0.001", "bs": "128", "dropout": "0.2", "wd": "1e-4"},
        ],
        "NYC": [
            {"lr": "0.0005", "bs": "256", "dropout": "0.2", "wd": "5e-5"},
            {"lr": "0.001", "bs": "256", "dropout": "0.2", "wd": "5e-5"},
            {"lr": "0.0005", "bs": "128", "dropout": "0.2", "wd": "1e-4"},
        ],
        "Singapore": [
            {"lr": "0.001", "bs": "128", "dropout": "0.2", "wd": "1e-4"},
            {"lr": "0.0005", "bs": "128", "dropout": "0.2", "wd": "1e-4"},
        ],
        "Tokyo": [
            {"lr": "0.001", "bs": "128", "dropout": "0.2", "wd": "1e-4"},
            {"lr": "0.0005", "bs": "256", "dropout": "0.2", "wd": "1e-4"},
        ],
    },
    "LightGCN": {
        "Chicago": [
            {"lr": "0.001", "bs": "128", "dropout": "0.0", "wd": "1e-4"},
            {"lr": "0.005", "bs": "128", "dropout": "0.0", "wd": "1e-4"},
        ],
        "NYC": [
            {"lr": "0.0005", "bs": "256", "dropout": "0.0", "wd": "5e-5"},
            {"lr": "0.001", "bs": "256", "dropout": "0.0", "wd": "5e-5"},
            {"lr": "0.0005", "bs": "128", "dropout": "0.0", "wd": "1e-4"},
        ],
        "Singapore": [
            {"lr": "0.001", "bs": "128", "dropout": "0.0", "wd": "1e-4"},
            {"lr": "0.0005", "bs": "128", "dropout": "0.0", "wd": "1e-4"},
        ],
        "Tokyo": [
            {"lr": "0.001", "bs": "128", "dropout": "0.0", "wd": "1e-4"},
            {"lr": "0.0005", "bs": "256", "dropout": "0.0", "wd": "1e-4"},
        ],
    },
}


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print("RUN:", " ".join(cmd))
    print(p.stdout[-2500:])
    if p.returncode != 0:
        print(p.stderr[-2500:])
        raise RuntimeError("command failed")


def train_one(model, city, cfg, idx):
    out_file = OUT / f"{city}_{model}_cfg{idx}.json"
    cmd = [
        "python", str(BASE / "main.py"),
        "--city", city,
        "--model", model,
        "--epochs", "260",
        "--eval_freq", "5",
        "--patience", "16",
        "--batch_size", cfg["bs"],
        "--lr", cfg["lr"],
        "--dropout", cfg["dropout"],
        "--weight_decay", cfg["wd"],
        "--cuda", "-1",
        "--save", "1",
        "--metrics_out", str(out_file),
    ]
    run(cmd)
    obj = json.loads(out_file.read_text(encoding="utf-8"))
    obj["cfg"] = cfg
    obj["cfg_id"] = idx
    return obj


def pick_best(rows):
    rows.sort(key=lambda x: (x["ndcg_at_20"], x["recall_at_20"]), reverse=True)
    return rows[0]


def run_otc(model):
    out_file = OUT / f"otc_{model.lower()}_all.json"
    cmd = [
        "python", str(BASE / "otc_lightgcn.py"),
        "--all_targets", "1",
        "--model", model,
        "--topk", "20",
        "--gammas", "0.1,0.2,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0",
        "--per_source_gamma", "1",
        "--allow_zero_gamma", "1",
        "--max_coord_iter", "16",
        "--metrics_out", str(out_file),
    ]
    run(cmd)
    return json.loads(out_file.read_text(encoding="utf-8"))


def main():
    summary = {"VanillaMF": {}, "LightGCN": {}}
    for model in ["VanillaMF", "LightGCN"]:
        for city in CITIES:
            rows = []
            for i, cfg in enumerate(SEARCH[model][city]):
                rows.append(train_one(model, city, cfg, i))
            best = pick_best(rows)
            summary[model][city] = best
            (OUT / f"{city}_{model}_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    summary["OTC_VanillaMF"] = run_otc("VanillaMF")
    summary["OTC_LightGCN"] = run_otc("LightGCN")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE", OUT / "summary.json")


if __name__ == "__main__":
    main()
