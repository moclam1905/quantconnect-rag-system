"""
qc_structure_summarizer.py
--------------------------
Tóm lược nhanh cấu trúc của các file HTML QuantConnect đã qua Phase 0.

Chạy ví dụ:
    python qc_structure_summarizer.py \
        --input  data/processed_html/Quantconnect-Lean-Engine.html \
        --output data/snapshots/Lean-Engine.struct_snapshot.json
"""

import argparse, json, statistics as stats
from collections import Counter, defaultdict
from pathlib import Path
from lxml import etree, html   # pip install lxml

HEADING_TAGS = {f"h{i}" for i in range(1, 7)}
SKIP_TAGS    = {"script", "style", "meta", "link", "iframe", "noscript"}

def summarise(file_path: Path, limit_nodes: int | None = None) -> dict:
    """Trả về dict chứa các thống kê quan trọng của file HTML."""
    tag_counter, class_counter = Counter(), Counter()
    heading_counter           = Counter()
    para_lengths              = []  # Đo chiều dài text <p> để tính median/P95
    code_blocks = table_blocks = 0
    longest_text, longest_len = "", 0

    # iterparse: duyệt tuần tự, giữ tối thiểu node trong RAM
    context = etree.iterparse(
        file_path,
        html=True,
        events=("end",),
        tag="*",
        encoding="utf-8"
    )

    for idx, (event, elem) in enumerate(context):
        tag = elem.tag.lower() if isinstance(elem.tag, str) else ""

        if tag in SKIP_TAGS:
            elem.clear()
            continue

        tag_counter[tag] += 1

        # Heading
        if tag in HEADING_TAGS:
            heading_counter[tag] += 1

        # Code / Table
        if tag in ("pre", "code"):
            code_blocks += 1
        if tag == "table":
            table_blocks += 1

        # Class
        cls = elem.get("class")
        if cls:
            for c in cls.split():
                class_counter[c] += 1

        # Đếm & lấy mẫu <p>
        if tag == "p":
            text = (elem.text or "").strip()
            if text:
                para_lengths.append(len(text))
                if len(text) > longest_len:
                    longest_len, longest_text = len(text), text[:200]

        # Giới hạn nếu muốn test nhanh
        if limit_nodes and idx >= limit_nodes:
            elem.clear()
            break

        elem.clear()  # tiết kiệm RAM

    # Thống kê độ dài đoạn <p>
    median_len = stats.median(para_lengths) if para_lengths else 0
    p95_len    = stats.quantiles(para_lengths, n=20)[18] if para_lengths else 0

    return {
        "file_name": file_path.name,
        "file_size_bytes": file_path.stat().st_size,
        "total_nodes": sum(tag_counter.values()),
        "headings": heading_counter,
        "tags_top20": tag_counter.most_common(300),
        "classes_top20": class_counter.most_common(300),
        "code_blocks": code_blocks,
        "table_blocks": table_blocks,
        "paragraph_len_median": median_len,
        "paragraph_len_p95": p95_len,
        "longest_text_len": longest_len,
        "longest_text_sample": longest_text,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="Path tới file HTML đã pre-process")
    ap.add_argument("--output", required=True, help="Path JSON snapshot xuất ra")
    ap.add_argument("--limit_nodes", type=int, default=None,
                    help="Giới hạn số node (tuỳ chọn test nhanh)")
    args = ap.parse_args()

    snapshot = summarise(Path(args.input), args.limit_nodes)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
    print(f"✅  Snapshot saved → {out_path} ({snapshot['total_nodes']:,} nodes)")

if __name__ == "__main__":
    main()
