"""Quick benchmark; run `python eval_phase4.py`"""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

import requests

settings = get_settings()


def _one(q):
    resp = requests.post(f"http://{settings.host}:{settings.port}/answer", json={"query": q["question"]})
    latency = resp.json().get("latency_ms", {}).get("total", 0)
    return latency


def main():
    eval_path = Path("eval_set.jsonl")
    if not eval_path.exists():
        print("Eval set not found. Skipping.")
        return
    items = [json.loads(l) for l in eval_path.read_text().splitlines()]
    with ThreadPoolExecutor(max_workers=8) as ex:
        latencies = list(ex.map(_one, items))
    print(f"p95 latency: {sorted(latencies)[int(len(latencies)*0.95)]} ms | avg {mean(latencies):.1f} ms")


if __name__ == "__main__":
    main()

# =========================================================
# End of single‑file Phase 4 package
# To split into real files, divide by the '# --- filename ---' markers.
# =========================================================
