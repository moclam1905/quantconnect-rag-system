#!/usr/bin/env python3
"""build_bm25.py

Build a BM25 index for QuantConnect RAG chunks.parquet with a code‑aware tokenizer
and write both the compressed pickle index and a detailed metadata JSON file.

Usage::

    python build_bm25.py \
        --input-parquet data/chunks_data/chunks.parquet \
        --output-index data/bm25_index.pkl.gz \
        --output-meta  data/bm25_meta.json

The script implements:
• Custom stop‑word list tuned for technical docs (see ``CUSTOM_STOPWORDS``).
• Code‑aware tokenization: splits CamelCase & snake_case, keeps alphanumerics.
• SHA‑256 hash over the parquet file (with progress bar) for traceability.
• BM25Okapi index from ``rank_bm25`` (k1=1.2, b=0.75, epsilon=0.25).
• Stats calculation (unique token count, average length, RAM footprint).
• Outputs:
    – ``bm25_index.pkl.gz``  : compressed pickle of the BM25Okapi object.
    – ``bm25_meta.json``     : metadata describing tokenizer, params, hash, env.

This script must be run once after Phase 3.1 chunking. Loading the resulting
index in the Hybrid Retrieval API typically takes <2 s.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Set

import polars as pl
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Custom stop‑word setup (tuned for QuantConnect technical docs)
# ---------------------------------------------------------------------------
CUSTOM_STOPWORDS: Set[str] = {
    # articles & determiners
    "the",
    "a",
    "an",
    # conjunctions
    "and",
    "or",
    "but",
    # prepositions (most common only)
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    # be verbs & auxiliaries
    "is",
    "are",
    "was",
    "were",
    "been",
    "be",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
}

# Words we *never* drop even if they appear in stop list
PRESERVE_WORDS: Set[str] = {
    # programming keywords
    "class",
    "def",
    "function",
    "method",
    "return",
    "self",
    "this",
    "import",
    "from",
    "using",
    "namespace",
    "public",
    "private",
    # QuantConnect specific
    "algorithm",
    "initialize",
    "ondata",
    "schedule",
    "liquidate",
    # trading terms
    "order",
    "buy",
    "sell",
    "long",
    "short",
    "position",
    # question words
    "how",
    "what",
    "when",
    "where",
    "why",
    "which",
}

STOPWORDS = CUSTOM_STOPWORDS - PRESERVE_WORDS

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
import re
_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")  # split CamelCase
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")      # extract alphanumerics

def code_aware_tokenizer(text: str) -> List[str]:
    """Tokenize *text* for BM25.

    Steps:
    1. Replace CamelCase → Camel Case.
    2. Replace snake_case '_' with space.
    3. Find alphanumerics; lowercase.
    4. Drop stopwords.
    """
    spaced = _CAMEL_RE.sub(" ", text)
    spaced = spaced.replace("_", " ")
    tokens = [tok.lower() for tok in _TOKEN_RE.findall(spaced)]
    return [t for t in tokens if t not in STOPWORDS]

# ---------------------------------------------------------------------------
# SHA‑256 helper with progress bar
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 8192) -> str:
    hash_ = hashlib.sha256()
    total = path.stat().st_size
    with path.open("rb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc="Hashing") as pbar:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hash_.update(chunk)
            pbar.update(len(chunk))
    return hash_.hexdigest()

# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_bm25(parquet_path: Path, index_path: Path, meta_path: Path) -> None:
    t0 = time.perf_counter()

    # Load parquet lazily with Polars (text only)
    print("📥 Loading parquet → memory (chunk_id, text)...")
    df = pl.scan_parquet(str(parquet_path)).select(["chunk_id", "text"]).collect()

    chunk_ids: List[str] = df["chunk_id"].to_list()
    texts: List[str] = df["text"].to_list()
    del df  # free Polars frame

    # Tokenize corpus
    print("🔪 Tokenizing corpus with code‑aware tokenizer...")
    corpus_tokens: List[List[str]] = [code_aware_tokenizer(txt) for txt in tqdm(texts, unit="chunk")]
    avg_len = sum(len(toks) for toks in corpus_tokens) / len(corpus_tokens)

    # Build BM25 index
    print("📚 Building BM25Okapi index (k1=1.2, b=0.75, eps=0.25)...")
    bm25 = BM25Okapi(corpus_tokens, k1=1.2, b=0.75, epsilon=0.25)

    # Dump compressed pickle
    print(f"💾 Saving index → {index_path} (gzip pickle)...")
    with gzip.open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Stats
    unique_tokens = len(bm25.idf)
    total_tokens = sum(len(toks) for toks in corpus_tokens)

    # SHA‑256 parquet
    print("🔑 Computing SHA‑256 of parquet for provenance...")
    parquet_sha = sha256_file(parquet_path)

    # Meta JSON
    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "build_time_seconds": round(time.perf_counter() - t0, 2),
        "source": {
            "parquet_path": str(parquet_path),
            "parquet_sha256": parquet_sha,
            "parquet_size_bytes": parquet_path.stat().st_size,
            "total_chunks": len(chunk_ids),
        },
        "tokenizer": {
            "type": "code_aware_v1",
            "stopwords_count": len(STOPWORDS),
            "case_sensitive": False,
            "split_camelcase": True,
            "split_snake_case": True,
        },
        "bm25_params": {
            "k1": 1.2,
            "b": 0.75,
            "epsilon": 0.25,
        },
        "index_stats": {
            "unique_tokens": unique_tokens,
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": round(avg_len, 2),
            "index_size_bytes": index_path.stat().st_size if index_path.exists() else None,
        },
        "environment": {
            "python_version": platform.python_version(),
            "rank_bm25_version": getattr(sys.modules.get("rank_bm25"), "__version__", "unknown"),
            "platform": platform.system().lower(),
        },
    }

    print(f"📝 Writing meta → {meta_path}")
    meta_path.write_text(json.dumps(meta, indent=2))

    print("✅ BM25 build completed in", meta["build_time_seconds"], "seconds.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BM25 index for QuantConnect chunks.parquet")
    parser.add_argument("--input-parquet", required=True, type=Path)
    parser.add_argument("--output-index", required=True, type=Path)
    parser.add_argument("--output-meta", required=True, type=Path)
    args = parser.parse_args()

    if not args.input_parquet.exists():
        parser.error(f"Input parquet {args.input_parquet} not found")

    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    args.output_meta.parent.mkdir(parents=True, exist_ok=True)

    build_bm25(args.input_parquet, args.output_index, args.output_meta)
