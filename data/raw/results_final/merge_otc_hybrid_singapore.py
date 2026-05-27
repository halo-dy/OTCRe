import json
from pathlib import Path


strict_path = Path("data/raw/results_final/summary_with_otc_strict.json")
prev_path = Path("data/raw/results_final/summary_with_otc_readd.json")
out_path = Path("data/raw/results_final/summary_with_otc_hybrid_singapore.json")

strict = json.loads(strict_path.read_text(encoding="utf-8"))
prev = json.loads(prev_path.read_text(encoding="utf-8"))


def replace_city(results_strict, results_prev, city):
    prev_row = None
    for r in results_prev:
        if r["target_city"] == city:
            prev_row = r
            break
    if prev_row is None:
        return results_strict
    out = []
    for r in results_strict:
        if r["target_city"] == city:
            out.append(prev_row)
        else:
            out.append(r)
    return out


strict["OTC_VanillaMF"]["results"] = replace_city(
    strict["OTC_VanillaMF"]["results"],
    prev["OTC_VanillaMF"]["results"],
    "Singapore",
)
strict["OTC_LightGCN"]["results"] = replace_city(
    strict["OTC_LightGCN"]["results"],
    prev["OTC_LightGCN"]["results"],
    "Singapore",
)

out_path.write_text(json.dumps(strict, indent=2), encoding="utf-8")
print(f"Wrote {out_path}")
