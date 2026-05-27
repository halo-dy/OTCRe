import json
from pathlib import Path


final_summary = Path("data/raw/results_final/summary.json")
strict_summary = Path("data/raw/results_otc_strict/summary.json")
out_path = Path("data/raw/results_final/summary_with_otc_strict.json")

base = json.loads(final_summary.read_text(encoding="utf-8"))
strict = json.loads(strict_summary.read_text(encoding="utf-8"))

base["OTC_VanillaMF"] = strict["OTC_VanillaMF"]
base["OTC_LightGCN"] = strict["OTC_LightGCN"]

out_path.write_text(json.dumps(base, indent=2), encoding="utf-8")
print(f"Wrote {out_path}")
