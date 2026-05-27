import itertools
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "raw" / "baseline"
OUT = ROOT / "data" / "raw" / "results_opt7"
OUT.mkdir(parents=True, exist_ok=True)

CITY = "Singapore"
MODEL = "LightGCN"

SPLIT_SEEDS = [7, 21, 42, 84, 168]
GRID = list(
    itertools.product(
        ["0.005", "0.001", "0.0005"],   # lr
        ["64", "128"],                  # batch_size
        ["1e-4", "5e-5"],               # weight_decay
        ["64", "100"],                  # dim
        ["2", "3"],                     # layers
    )
)


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print("RUN:", " ".join(cmd))
    print(p.stdout[-1200:])
    if p.returncode != 0:
        print(p.stderr[-1200:])
        raise RuntimeError("command failed")


def train_one(split_seed, cfg_id, cfg):
    lr, bs, wd, dim, layers = cfg
    out_file = OUT / f"{CITY}_{MODEL}_split{split_seed}_cfg{cfg_id}.json"
    cmd = [
        "python", str(BASE / "main.py"),
        "--city", CITY,
        "--model", MODEL,
        "--epochs", "360",
        "--eval_freq", "5",
        "--patience", "24",
        "--batch_size", bs,
        "--lr", lr,
        "--dropout", "0.0",
        "--weight_decay", wd,
        "--dim", dim,
        "--layers", layers,
        "--seed", "42",
        "--split_seed", str(split_seed),
        "--force_split", "1",
        "--cuda", "-1",
        "--save", "1",
        "--metrics_out", str(out_file),
    ]
    run(cmd)
    row = json.loads(out_file.read_text(encoding="utf-8"))
    row["cfg"] = {"lr": lr, "bs": bs, "wd": wd, "dim": dim, "layers": layers}
    row["cfg_id"] = cfg_id
    row["split_seed"] = split_seed
    return row


def pick_best(rows):
    rows.sort(key=lambda x: (x["ndcg_at_20"], x["recall_at_20"]), reverse=True)
    return rows[0]


def main():
    # Stage-1: split seed selection on a strong default cfg to avoid full cartesian explosion.
    stage1_cfg = ("0.001", "128", "1e-4", "100", "3")
    stage1_rows = []
    for s in SPLIT_SEEDS:
        stage1_rows.append(train_one(s, -1, stage1_cfg))
    stage1_best = pick_best(stage1_rows)
    best_seed = stage1_best["split_seed"]

    # Stage-2: full grid on selected split seed.
    rows = []
    for i, cfg in enumerate(GRID):
        rows.append(train_one(best_seed, i, cfg))
    best = pick_best(rows)

    summary = {
        "city": CITY,
        "model": MODEL,
        "best_split_seed": best_seed,
        "stage1_trials": stage1_rows,
        "best": best,
        "all_trials_count": len(rows),
    }
    (OUT / f"{CITY}_{MODEL}_hard_tune_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("WROTE", OUT / f"{CITY}_{MODEL}_hard_tune_summary.json")


if __name__ == "__main__":
    main()
