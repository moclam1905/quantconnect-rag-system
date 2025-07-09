#!/usr/bin/env python3
"""push_vectors.py – Upsert existing embeddings into Qdrant **with raw text payload**.

Usage:
    python push_vectors.py \
        --embeddings-parquet data/embeddings_data/2025-07-03_embeddings.parquet \
        --chunks-parquet     data/chunks_data/chunks.parquet \
        --collection         quantconnect_chunks \
        --host               http://localhost:6333 \
        --batch-size         100

* Không gọi OpenAI, KHÔNG tốn phí.
* Nếu cột `text` không có trong embeddings parquet, script tự join sang chunks.parquet để lấy.
* Upsert theo lô nhỏ (mặc định 100) và timeout cao để tránh lỗi.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl
from qdrant_client import QdrantClient, models as qdr
from tqdm import tqdm


def load_embeddings(epath: Path, chunks_path: Path):
    """Return frame with all metadata + vector + text column."""
    df = pl.read_parquet(str(epath))

    if "text" not in df.columns:
        print("⚠️  'text' column missing – joining with chunks.parquet to fetch raw text…")
        chunks_df = (
            pl.read_parquet(str(chunks_path))
            .select(["chunk_id", "text"])
            .unique("chunk_id")
        )
        df = df.join(chunks_df, on="chunk_id", how="left")

    if "text" not in df.columns:
        raise ValueError("Could not populate 'text' column; check input paths.")

    return df


def upsert_batches(df: pl.DataFrame, client: QdrantClient, collection: str, batch: int):
    vectors = np.vstack(df["vector"].to_list())
    payload_cols = [c for c in df.columns if c not in ("vector",)]
    ids = list(range(len(df)))

    for start in tqdm(range(0, len(df), batch), desc="Upserting", unit="batch"):
        end = min(start + batch, len(df))
        payloads: List[Dict] = df[start:end].select(payload_cols).to_dicts()
        client.upsert(
            collection_name=collection,
            points=qdr.Batch(
                ids=ids[start:end],
                vectors=vectors[start:end].tolist(),
                payloads=payloads,
            ),
        )


def main():
    ap = argparse.ArgumentParser(description="Upsert existing embeddings to Qdrant with raw text payload")
    ap.add_argument("--embeddings-parquet", required=True, type=Path)
    ap.add_argument("--chunks-parquet", required=True, type=Path)
    ap.add_argument("--collection", default="quantconnect_chunks")
    ap.add_argument("--host", default="http://localhost:6333")
    ap.add_argument("--batch-size", type=int, default=100)
    args = ap.parse_args()

    df = load_embeddings(args.embeddings_parquet, args.chunks_parquet)
    print("✅ DataFrame ready – rows:", len(df))

    client = QdrantClient(url=args.host, timeout=300)
    print("↺ Recreating collection", args.collection)
    client.recreate_collection(
        collection_name=args.collection,
        vectors_config=qdr.VectorParams(size=3072, distance="Cosine"),
    )

    upsert_batches(df, client, args.collection, args.batch_size)
    print("✅ Done – vectors with text payload pushed to Qdrant")


if __name__ == "__main__":
    main()
