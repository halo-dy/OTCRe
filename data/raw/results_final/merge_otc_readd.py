import json
from pathlib import Path


final_summary = Path("data/raw/results_final/summary.json")
otc_readd_summary = Path("data/raw/results_otc_readd/summary.json")
out_path = Path("data/raw/results_final/summary_with_otc_readd.json")


base = json.loads(final_summary.read_text(encoding="utf-8"))
otc = json.loads(otc_readd_summary.read_text(encoding="utf-8"))

base["OTC_VanillaMF"] = otc["OTC_VanillaMF"]
base["OTC_LightGCN"] = otc["OTC_LightGCN"]

out_path.write_text(json.dumps(base, indent=2), encoding="utf-8")
print(f"Wrote {out_path}")
