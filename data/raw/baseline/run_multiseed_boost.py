import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "raw" / "baseline"
OUT = ROOT / "data" / "raw" / "results_opt3"
OUT.mkdir(parents=True, exist_ok=True)

CITIES = ["Chicago", "NYC", "Singapore", "Tokyo"]
SEEDS = [7, 21, 42, 84, 168]

# Focus on strongest settings found in opt2, then use seed selection.
CFG = {
    "VanillaMF": {"lr": "0.005", "batch_size": "128", "dropout": "0.2", "weight_decay": "1e-4"},
    "LightGCN": {"lr": "0.001", "batch_size": "128", "dropout": "0.0", "weight_decay": "1e-4"},
}


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print("RUN:", " ".join(cmd))
    print(p.stdout[-2000:])
    if p.returncode != 0:
        print(p.stderr[-2000:])
        raise RuntimeError("command failed")


def run_seed(model, city, seed):
    cfg = CFG[model]
    out_file = OUT / f"{city}_{model}_seed{seed}.json"
    cmd = [
        "python", str(BASE / "main.py"),
        "--city", city,
        "--model", model,
        "--epochs", "320",
        "--eval_freq", "5",
        "--patience", "20",
        "--batch_size", cfg["batch_size"],
        "--lr", cfg["lr"],
        "--dropout", cfg["dropout"],
        "--weight_decay", cfg["weight_decay"],
        "--seed", str(seed),
        "--cuda", "-1",
        "--save", "1",
        "--metrics_out", str(out_file),
    ]
    run(cmd)
    rec = json.loads(out_file.read_text(encoding="utf-8"))
    return rec


def pick_best(items):
    items.sort(key=lambda x: (x["ndcg_at_20"], x["recall_at_20"]), reverse=True)
    return items[0]


def run_otc(model):
    out_file = OUT / f"otc_{model.lower()}_all.json"
    cmd = [
        "python", str(BASE / "otc_lightgcn.py"),
        "--all_targets", "1",
        "--model", model,
        "--topk", "20",
        "--gammas", "0.0,0.05,0.1,0.15,0.2,0.3,0.5,0.8,1.0,1.5,2.0",
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
            rows = []
            for sd in SEEDS:
                rows.append(run_seed(model, city, sd))
            best = pick_best(rows)
            summary[model][city] = best
            (OUT / f"{city}_{model}_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    summary["OTC_VanillaMF"] = run_otc("VanillaMF")
    summary["OTC_LightGCN"] = run_otc("LightGCN")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE", OUT / "summary.json")


if __name__ == "__main__":
    main()
