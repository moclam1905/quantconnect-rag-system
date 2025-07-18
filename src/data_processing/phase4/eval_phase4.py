"""
Quick‑Eval Phase 4
------------------
• Đọc bộ câu hỏi `eval_set.jsonl` (mỗi dòng: {"question": "...", "answers": ["..."]})
• Gửi song song tới /answer
• Tính:
    – p95 latency   (ms)
    – avg latency   (ms)
    – Recall@5      (có ít nhất 1 câu trả lời vàng xuất hiện trong answer, ignore‑case)
    – tokens_used   (tổng)
"""

from __future__ import annotations
import json, time, concurrent.futures, statistics as st, requests, os, sys, pathlib, argparse

# ---------- config ----------
HOST   = os.getenv("HOST",   "127.0.0.1")
PORT   = os.getenv("PORT",   "8001")
APIKEY = os.getenv("API_KEY_HEADER", "1905")   # phase‑3 key
EVAL   = pathlib.Path("eval_set.jsonl")         # chỉnh nếu để folder khác
WORKERS = 8

# ---------- helper ----------
def _one(item):
    q = item["question"]
    tgt = [a.lower() for a in item.get("answers", [])]
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"http://{HOST}:{PORT}/answer",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": APIKEY,
            },
            json={"query": q},
            timeout=60,
        ).json()
    except Exception as e:
        print("ERR:", q[:60], e, file=sys.stderr)
        return None
    lat = r.get("latency_ms", {}).get("total", 0)
    ok  = any(ans in r["answer"].lower() for ans in tgt) if tgt else None
    toks = r.get("metadata", {}).get("tokens_used", 0)
    return lat, ok, toks

# ---------- main ----------
def main():
    if not EVAL.exists():
        print("⚠️  eval_set.jsonl not found.")
        return
    items = [json.loads(l) for l in EVAL.read_text().splitlines()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(filter(None, ex.map(_one, items)))

    lats, oks, toks = zip(*results)
    p95 = sorted(lats)[int(len(lats) * 0.95)]
    recall = sum(1 for x in oks if x) / len([x for x in oks if x is not None]) * 100

    print(f"\n=== Phase 4 quick eval ===")
    print(f"total queries : {len(items)}")
    print(f"avg latency   : {st.mean(lats):.0f} ms")
    print(f"p95 latency   : {p95:.0f} ms")
    print(f"recall@5      : {recall:.1f} %  (queries có đáp án vàng)")
    print(f"tokens used   : {sum(toks)}")

if __name__ == "__main__":
    main()
