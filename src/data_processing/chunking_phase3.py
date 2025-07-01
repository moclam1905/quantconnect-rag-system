"""
Chunking Phase 3.1 – QuantConnect RAG
====================================
Transform section‑level JSON (sinh từ Phase 2) thành lát RAG‑friendly cố định 500 token, cửa sổ trượt 400 token, kèm metadata đầy đủ. Kết quả được ghi vào một file Parquet duy nhất (`chunks.parquet`, nén Snappy) và file log JSON.

Chuẩn tương thích:
  • chunk_size  = 500
  • overlap     = 100  (=> step = 400)
  • min_tail    = 150  (nối phần đuôi <150 vào lát trước)
  • tokenizer   = tiktoken cl100k_base

CLI
---
    python -m qc_rag.chunking.cli \
        --sections-dir data/parsed_content \
        --output-path data/chunks_data/chunks.parquet \
        --stats-path data/chunks_data/chunking_stats.json \
        --chunk-size 500 --overlap 100 --min-tail 150 \
        --tokenizer cl100k_base --num-workers 4

"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Any, Generator, Tuple

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken

LOGGER = logging.getLogger("qc_rag.chunking")
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO
)

def table_to_markdown(table: Any) -> str:
    """
    Convert list-of-rows table → Markdown *đầy đủ*.
    Không còn nhánh rút gọn theo kích thước.
    """
    if not isinstance(table, list) or not table:
        return ""
    header, *rows = table
    if not all(isinstance(r, list) for r in [header, *rows]):
        return ""
    sep_row = ["---"] * len(header)
    md_lines = [
        "| " + " | ".join(str(c) for c in header) + " |",
        "| " + " | ".join(sep_row) + " |",
    ]
    for r in rows:
        md_lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(md_lines)


###############################################################################
# Tokenisation & window slicing
###############################################################################

def get_encoder(name: str = "cl100k_base"):
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        # fallback to default GPT‑2 encoder
        return tiktoken.get_encoding("gpt2")

def slide_window(tokens: List[int], size: int, overlap: int, min_tail: int):
    """
    Trả về list (start_idx, slice_tokens) – đảm bảo:
      • Các lát chuẩn có length == size (500)
      • Phần đuôi < min_tail chỉ thêm phần CHƯA có trong lát cuối.
    """
    step = size - overlap
    n = len(tokens)
    slices: List[Tuple[int, List[int]]] = []
    i = 0

    # lát chuẩn
    while i + size <= n:
        slices.append((i, tokens[i : i + size]))
        i += step

    # phần đuôi
    if i < n:
        tail_start = i
        tail_len = n - i
        if tail_len < min_tail and slices:
            # chỉ nối những token CHƯA có trong lát cuối
            last_start, last_tokens = slices[-1]
            last_end = last_start + len(last_tokens) - 1
            if last_end < n - 1:
                last_tokens.extend(tokens[last_end + 1 : n])
            # không tạo lát mới
        else:
            slices.append((tail_start, tokens[tail_start:n]))

    return slices




###############################################################################
# Core generator
###############################################################################

class ChunkGenerator:
    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
        min_tail: int = 150,
        tokenizer_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_tail = min_tail
        self.encoder = get_encoder(tokenizer_name)

    def _load_section(self, path: Path) -> Dict:
        with path.open("rb") as f:
            return orjson.loads(f.read())

    def _section_to_text(self, section: Dict) -> Tuple[str, List[str]]:
        """
        Trả về chuỗi văn bản gốc của cả section, bảo toàn ranh giới:
          • 2 newline giữa các đoạn thường
          • Gói code block vào ``` để LLM hiểu ngữ cảnh
        Đồng thời trả về tập content_types duy nhất trong section.
        """
        texts: List[str] = []
        types: set[str] = set()

        for ch in section.get("chunks", []):
            semantic = ch.get("semantic_type", "unknown")
            types.add(semantic)

            txt = ch.get("text", "")
            tbl = ch.get("table_content")

            if semantic.startswith("code"):
                # bọc trong ``` để giữ nguyên định dạng
                texts.append(f"```{txt}```")
            else:
                texts.append(txt)

            if tbl:
                texts.append(table_to_markdown(tbl))  # bảng đầy đủ

        joined = "\n\n".join(texts)
        return joined, sorted(types)

    # ---------------------------------------------------------------------
    # ---------------------------------------------
    def process_section(self, path: Path) -> List[Dict[str, Any]]:
        sec = self._load_section(path)

        order: int = sec.get("order", 0)
        title: str = sec.get("title", "")
        hierarchy = sec.get("hierarchy_path", [])
        doc_name = path.parts[-3]

        # nhận cả text + list content_types
        raw_text, section_types = self._section_to_text(sec)
        if not raw_text.strip():
            return []

        tokens = self.encoder.encode(raw_text)
        slices = slide_window(tokens, self.chunk_size, self.overlap, self.min_tail)

        results: List[Dict[str, Any]] = []
        for local_idx, (start_idx, slice_tokens) in enumerate(slices):
            results.append(
                {
                    "chunk_id": f"{doc_name}/{order}_{local_idx:03d}",
                    "doc": doc_name,
                    "section_order": order,
                    "section_title": title,
                    "hierarchy_path": hierarchy,
                    "token_start": start_idx,
                    "token_end": start_idx + len(slice_tokens) - 1,
                    "text": self.encoder.decode(slice_tokens),
                    "source_path": str(path),
                    "content_types": section_types,  # dùng list tổng hợp
                }
            )
        return results


###############################################################################
# Parquet writer (append‑safe)
###############################################################################

class ParquetWriter:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.schema = pa.schema(
            [
                ("chunk_id", pa.string()),
                ("doc", pa.string()),
                ("section_order", pa.int32()),
                ("section_title", pa.string()),
                ("hierarchy_path", pa.list_(pa.string())),
                ("token_start", pa.int32()),
                ("token_end", pa.int32()),
                ("text", pa.string()),
                ("source_path", pa.string()),
                ("content_types", pa.list_(pa.string())),
            ]
        )
        self._writer: pq.ParquetWriter | None = None

    def _ensure_writer(self):
        """Create ParquetWriter lazily (and dir if missing)."""
        # 👉 NEW: make sure parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.output_path, self.schema, compression="snappy", write_statistics=False
            )

    def append(self, rows: List[Dict[str, Any]]):
        if not rows:
            return
        self._ensure_writer()
        batch = pa.Table.from_pylist(rows, schema=self.schema)
        self._writer.write_table(batch)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

###############################################################################
# Worker entrypoint
###############################################################################

def _worker_init(generator_kwargs):
    global _CHUNK_GENERATOR
    _CHUNK_GENERATOR = ChunkGenerator(**generator_kwargs)


def _process_path(path_str: str) -> List[Dict[str, Any]]:
    path = Path(path_str)
    return _CHUNK_GENERATOR.process_section(path)

###############################################################################
# CLI
###############################################################################

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("QuantConnect Phase 3.1 – Chunking")
    p.add_argument("--sections-dir", type=Path, default=Path("data/parsed_content"))
    p.add_argument("--output-path", type=Path, default=Path("data/chunks_data/chunks.parquet"))
    p.add_argument("--stats-path", type=Path, default=Path("data/chunks_data/chunking_stats.json"))
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--overlap", type=int, default=100)
    p.add_argument("--min-tail", type=int, default=150)
    p.add_argument("--tokenizer", type=str, default="cl100k_base")
    p.add_argument("--num-workers", type=int, default=os.cpu_count() or 4)
    return p


def iter_section_paths(sections_root: Path) -> Generator[Path, None, None]:
    for doc_dir in sections_root.iterdir():
        sec_dir = doc_dir / "sections"
        if sec_dir.is_dir():
            yield from sec_dir.glob("*.json")


def main(argv: List[str] | None = None):
    args = build_arg_parser().parse_args(argv)
    start_time = time.time()

    generator_kwargs = dict(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_tail=args.min_tail,
        tokenizer_name=args.tokenizer,
    )

    section_paths = list(iter_section_paths(args.sections_dir))
    LOGGER.info("Found %d section files", len(section_paths))

    writer = ParquetWriter(args.output_path)

    total_chunks = 0
    longest_chunk_tokens = 0
    total_tokens = 0

    if args.num_workers > 1:
        with mp.Pool(args.num_workers, initializer=_worker_init, initargs=(generator_kwargs,)) as pool:
            for chunk_list in pool.imap_unordered(_process_path, map(str, section_paths)):
                writer.append(chunk_list)
                total_chunks += len(chunk_list)
                for c in chunk_list:
                    len_tokens = c["token_end"] - c["token_start"] + 1
                    total_tokens += len_tokens
                    longest_chunk_tokens = max(longest_chunk_tokens, len_tokens)
    else:
        _worker_init(generator_kwargs)
        for path in section_paths:
            chunk_list = _process_path(str(path))
            writer.append(chunk_list)
            total_chunks += len(chunk_list)
            for c in chunk_list:
                len_tokens = c["token_end"] - c["token_start"] + 1
                total_tokens += len_tokens
                longest_chunk_tokens = max(longest_chunk_tokens, len_tokens)

    writer.close()

    runtime = time.time() - start_time
    avg_tokens = total_tokens / total_chunks if total_chunks else 0

    stats = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sections_dir": str(args.sections_dir),
        "output_path": str(args.output_path),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "min_tail": args.min_tail,
        "tokenizer": args.tokenizer,
        "num_workers": args.num_workers,
        "total_sections": len(section_paths),
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
        "avg_tokens_per_chunk": round(avg_tokens, 2),
        "longest_chunk_tokens": longest_chunk_tokens,
        "runtime_sec": round(runtime, 2),
    }

    with args.stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    LOGGER.info("Chunking completed: %d chunks → %s (%.1fs)", total_chunks, args.output_path, runtime)
    LOGGER.info("Stats saved to %s", args.stats_path)


if __name__ == "__main__":
    main()
