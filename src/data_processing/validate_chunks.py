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
        full_text, _ = GEN._section_to_text(data)  # lấy chuỗi, bỏ content_types
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


def analyze_duplicates(parquet_path: Path, limit: int = 10):
    """Phân tích chi tiết chunks trùng."""
    tbl = pq.read_table(parquet_path)

    # Group by MD5
    hash_to_chunks = {}
    for i in range(len(tbl)):
        text = tbl.column("text")[i].as_py()
        md5 = hashlib.md5(text.encode()).hexdigest()

        if md5 not in hash_to_chunks:
            hash_to_chunks[md5] = []

        hash_to_chunks[md5].append({
            "chunk_id": tbl.column("chunk_id")[i].as_py(),
            "source_path": tbl.column("source_path")[i].as_py(),
            "section_order": tbl.column("section_order")[i].as_py(),
            "section_title": tbl.column("section_title")[i].as_py(),
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "text_length": len(text),
            "token_count": tbl.column("token_end")[i].as_py() - tbl.column("token_start")[i].as_py() + 1
        })

    # Analyze patterns
    print("\n" + "=" * 60)
    print("DUPLICATE ANALYSIS")
    print("=" * 60)

    # Summary stats
    total_unique = len(hash_to_chunks)
    duplicated_hashes = [h for h, chunks in hash_to_chunks.items() if len(chunks) > 1]
    total_duplicated_chunks = sum(len(chunks) - 1 for h, chunks in hash_to_chunks.items() if len(chunks) > 1)

    print(f"\nSummary:")
    print(f"  - Total unique content hashes: {total_unique}")
    print(f"  - Hashes with duplicates: {len(duplicated_hashes)}")
    print(f"  - Total duplicated chunks: {total_duplicated_chunks}")

    # Show duplicates
    print(f"\nShowing first {limit} duplicate groups:")
    print("-" * 60)

    shown = 0
    for md5, chunks in sorted(hash_to_chunks.items(), key=lambda x: -len(x[1])):
        if len(chunks) > 1:
            print(f"\n[Duplicate Group {shown + 1}]")
            print(f"MD5: {md5}")
            print(f"Occurrences: {len(chunks)}")
            print(f"Text length: {chunks[0]['text_length']} chars, {chunks[0]['token_count']} tokens")
            print(f"Preview: {repr(chunks[0]['text_preview'])}")

            print("\nFound in:")
            # Group by source document
            by_doc = {}
            for c in chunks:
                doc = c['source_path'].split('/')[-3]  # Extract doc name
                if doc not in by_doc:
                    by_doc[doc] = []
                by_doc[doc].append(c)

            for doc, doc_chunks in by_doc.items():
                print(f"  📄 {doc}:")
                for c in doc_chunks[:5]:  # Show first 5 per doc
                    print(f"     - {c['chunk_id']} | Section {c['section_order']}: {c['section_title']}")
                if len(doc_chunks) > 5:
                    print(f"     ... and {len(doc_chunks) - 5} more")

            shown += 1
            if shown >= limit:
                break

    # Analyze patterns
    print("\n" + "-" * 60)
    print("Pattern Analysis:")

    # Check for empty/minimal content
    empty_or_minimal = [h for h, chunks in hash_to_chunks.items()
                        if len(chunks) > 1 and chunks[0]['text_length'] < 50]
    if empty_or_minimal:
        print(f"\n⚠️  Found {len(empty_or_minimal)} duplicate groups with <50 chars")
        for i, h in enumerate(empty_or_minimal[:3]):
            chunks = hash_to_chunks[h]
            print(f"   Example {i + 1}: {repr(chunks[0]['text_preview'])} ({len(chunks)} occurrences)")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--sections-dir", required=True, type=Path)
    ap.add_argument("--analyze-duplicates", action="store_true",
                    help="Show detailed duplicate analysis")
    ap.add_argument("--duplicate-limit", type=int, default=10,
                    help="Number of duplicate groups to show (default: 10)")
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

    # Run detailed analysis if requested or if duplicates found
    if (args.analyze_duplicates or dup > 0) and args.parquet.exists():
        analyze_duplicates(args.parquet, args.duplicate_limit)


if __name__ == "__main__":
    main()