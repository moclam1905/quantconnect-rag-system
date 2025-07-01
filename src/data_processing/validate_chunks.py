#!/usr/bin/env python
"""
validate_chunks.py – Đối chiếu chunks.parquet với dữ liệu section gốc.

Chạy:
    python src/data_processing/validate_chunks.py \
        --parquet      data/chunks_data/chunks.parquet \
        --sections-dir data/parsed_content
"""
import argparse
import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict

import tiktoken
import pyarrow.parquet as pq

from chunking_phase3 import ChunkGenerator

ENC = tiktoken.get_encoding("cl100k_base")
GEN = ChunkGenerator()  # chỉ dùng để gọi _section_to_text


# ---------- helpers ----------
def encode_len(text: str) -> int:
    return len(ENC.encode(text))


def load_origin_tokens(sections_dir: Path) -> Dict[str, int]:
    """
    Trả về số token (đếm theo cl100k_base) của TỪNG SECTION GỐC,
    sau khi đã chuẩn hoá giống hệt chunker (gồm bảng Markdown, ký tự đặc biệt).
    """
    tokens = {}
    for p in sections_dir.rglob("sections/*.json"):
        data = json.loads(p.read_text())
        full_text, _ = GEN._section_to_text(data)   # lấy chuỗi, bỏ content_types
        tokens[str(p)] = encode_len(full_text)
    return tokens



def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Gộp các đoạn [start, end] chồng lấn để bỏ overlap."""
    if not intervals:
        return []
    intervals.sort()
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        cur_s, cur_e = merged[-1]
        if s <= cur_e + 1:
            merged[-1][1] = max(cur_e, e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def load_chunk_tokens(parquet_path: Path) -> Dict[str, int]:
    """Tổng token (không trùng) của mỗi section sau khi chunk."""
    tbl = pq.read_table(
        parquet_path, columns=["source_path", "token_start", "token_end"]
    )
    by_section: Dict[str, List[Tuple[int, int]]] = {}
    for src, s, e in zip(tbl.column(0), tbl.column(1), tbl.column(2)):
        by_section.setdefault(src.as_py(), []).append((s.as_py(), e.as_py()))

    unique = {}
    for src, intervals in by_section.items():
        merged = merge_intervals(intervals)
        unique[src] = sum(e - s + 1 for s, e in merged)
    return unique


def count_duplicates(parquet_path: Path) -> int:
    """Số chunk trùng nội dung (MD5)."""
    tbl = pq.read_table(parquet_path, columns=["text"])
    seen, dup = set(), 0
    for txt in tbl.column(0):
        md5 = hashlib.md5(txt.as_py().encode()).hexdigest()
        if md5 in seen:
            dup += 1
        seen.add(md5)
    return dup


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--sections-dir", required=True, type=Path)
    args = ap.parse_args()

    origin_tokens = load_origin_tokens(args.sections_dir)
    chunk_tokens = load_chunk_tokens(args.parquet)

    missing = [
        p for p, tok in origin_tokens.items() if chunk_tokens.get(p, -1) != tok
    ]
    dup = count_duplicates(args.parquet)

    if missing:
        print(f"❌ TOKEN MISMATCH cho {len(missing)} section (đã gộp overlap):")
        for p in missing[:10]:
            print(" ", p)
    else:
        print("✅ Bao phủ token 100 % (đã tính overlap)")

    if dup:
        print(f"❌ Phát hiện {dup} chunk bị trùng nội dung")
    else:
        print("✅ Không có chunk trùng")


if __name__ == "__main__":
    main()
