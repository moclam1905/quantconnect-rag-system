#!/usr/bin/env python
"""Phase 3.2 – Embedding & Vector DB for QuantConnect RAG

Usage (macOS / Linux / Windows):
    python phase3_embed.py \
        --input-parquet data/chunks_data/chunks.parquet \
        --output-parquet data/embeddings_data/2025-07-02_embeddings.parquet

Flags:
    --batch-size N         : number of chunks per OpenAI request (default 256)
    --qdrant-host HOST     : Qdrant host (default from env QDRANT_HOST or "localhost")
    --qdrant-port PORT     : Qdrant port (default 6333)
    --collection NAME      : Qdrant collection name (default "quantconnect_chunks")
    --skip-upsert          : embed & write parquet only, do not upsert to Qdrant

Environment variables (loaded from .env):
    OPENAI_API_KEY         : required
    QDRANT_HOST / PORT     : optional overrides for flags

The script will:
 1. Load chunks from Parquet.
 2. Compute embeddings via OpenAI text-embedding-3-large in batches with retry.
 3. Save vectors & metadata to Parquet (timestamped file).
 4. Create‑or‑update a Qdrant collection and upsert vectors in batches of 1 000.

Tested on macOS 13 (M1 Pro, Python 3.11).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm import tqdm

# --- Third‑party clients
from openai import OpenAI, OpenAIError
from qdrant_client import QdrantClient, models as qdr

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    today_str = _dt.date.today().isoformat()
    parser = argparse.ArgumentParser("Embed QuantConnect chunks & upload to Qdrant")
    parser.add_argument(
        "--input-parquet",
        default="data/chunks_data/chunks.parquet",
        help="Path to chunks.parquet produced by Phase 3.1",
    )
    parser.add_argument(
        "--output-parquet",
        default=f"data/embeddings_data/{today_str}_embeddings.parquet",
        help="Path to save embeddings.parquet (default timestamped)",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    parser.add_argument("--collection", default="quantconnect_chunks", help="Qdrant collection name")
    parser.add_argument(
        "--qdrant-host",
        default=os.getenv("QDRANT_HOST", "localhost"),
        help="Qdrant host (env QDRANT_HOST)",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=int(os.getenv("QDRANT_PORT", 6333)),
        help="Qdrant port (env QDRANT_PORT)",
    )
    parser.add_argument("--skip-upsert", action="store_true", help="Do not upsert to Qdrant")
    return parser.parse_args()


def _load_chunks(path: str | Path) -> pd.DataFrame:
    """Load chunk metadata & text for embedding."""
    if not Path(path).exists():
        sys.exit(f"❌  Input parquet not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    required_cols = {"chunk_id", "text"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        sys.exit(f"❌  Missing columns in chunks.parquet: {missing}")
    return df


def _embed_batch(texts: List[str], client: OpenAI, max_retries: int = 5) -> List[List[float]]:
    """Return list of 3072‑dim embeddings for a batch of N texts."""
    attempt = 0
    while True:
        try:
            rsp = client.embeddings.create(
                model="text-embedding-3-large",
                input=texts,
                encoding_format="float",
            )
            # OpenAI returns in the same order
            return [d.embedding for d in rsp.data]
        except OpenAIError as e:  # covers RateLimitError, APIError, etc.
            if attempt >= max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"⚠️  OpenAI error: {e}. Retrying in {wait}s…", file=sys.stderr)
            time.sleep(wait)
            attempt += 1


def _write_embeddings_parquet(df: pd.DataFrame, out_path: str | Path):
    """Write embeddings & metadata to Parquet efficiently using PyArrow."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build Arrow columns
    arrays = {
        "vector_id": pa.array(df["vector_id"], type=pa.uint64()),
        "chunk_id": pa.array(df["chunk_id"], type=pa.string()),
        "vector": pa.array(df["vector"].tolist(), type=pa.list_(pa.float32())),
    }
    # Include useful payload columns if present
    for col in [
        "doc",
        "hierarchy_path",
        "token_start",
        "token_end",
        "content_types",
    ]:
        if col in df.columns:
            if col == "content_types":
                arrays[col] = pa.array(df[col].tolist(), type=pa.list_(pa.string()))
            else:
                arrays[col] = pa.array(df[col].astype(str), type=pa.string())

    table = pa.Table.from_pydict(arrays)
    pq.write_table(table, out_path, compression="zstd")


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------

def main():
    load_dotenv()
    args = _parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("❌  OPENAI_API_KEY not set (check .env)")

    print("📥  Loading chunks parquet…")
    df = _load_chunks(args.input_parquet)
    n_chunks = len(df)
    print(f"   → {n_chunks:,} chunks loaded")

    # Assign uint64 ids once for consistency
    df = df.reset_index(drop=True)
    df["vector_id"] = df.index.astype(np.uint64)

    client = OpenAI()

    vectors: List[List[float]] = []
    batch_size = args.batch_size

    print("🧮  Computing embeddings…")
    for start in tqdm(range(0, n_chunks, batch_size)):
        batch_texts = df.loc[start : start + batch_size - 1, "text"].tolist()
        embs = _embed_batch(batch_texts, client)
        vectors.extend(embs)

    if len(vectors) != n_chunks:
        sys.exit("❌  Embedding count mismatch. Abort.")

    df["vector"] = vectors  # list[float] 3072‑dim each

    print("💾  Writing embeddings Parquet…")
    _write_embeddings_parquet(df, args.output_parquet)
    print(f"   → Saved to {args.output_parquet}")

    # ---------------------- Qdrant Upload ----------------------
    if args.skip_upsert:
        print("⏭️  --skip-upsert set → Finished without uploading to Qdrant.")
        return

    print("🚀  Uploading vectors to Qdrant…")
    qclient = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)

    # Create‑or‑update collection
    dim = 3072
    coll = args.collection
    if not qclient.collection_exists(coll):
        print(f"   → Collection '{coll}' not found. Creating…")
        qclient.recreate_collection(
            collection_name=coll,
            vectors_config=qdr.VectorParams(size=dim, distance=qdr.Distance.COSINE),
        )
    else:
        info = qclient.get_collection(coll)
        if info.vectors_config.size != dim:
            sys.exit(
                f"❌  Existing collection dimension {info.vectors_config.size} ≠ {dim}. Delete or rename collection."
            )

    batch_upsert = 1000
    for start in tqdm(range(0, n_chunks, batch_upsert)):
        sub = df.iloc[start : start + batch_upsert]
        id_list = sub["vector_id"].tolist()
        vecs_np = np.vstack(sub["vector"].to_numpy()).astype(np.float32)
        payload_cols = [
            c
            for c in [
                "chunk_id",
                "doc",
                "hierarchy_path",
                "token_start",
                "token_end",
                "content_types",
            ]
            if c in sub.columns
        ]
        payloads = sub[payload_cols].to_dict(orient="records")
        qclient.upload_collection(
            collection_name=coll,
            ids=id_list,
            vectors=vecs_np,
            payload=payloads,
            batch_size=batch_upsert,
        )

    print("✅  All done – embeddings stored & vectors upserted.")


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")
