#!/usr/bin/env python
"""Phase 3.2 – Embedding & Vector DB for QuantConnect RAG (fixed ndarray upload bug)

Usage:
  source .venv/bin/activate
  python phase3_embed.py \\
      --input-parquet data/chunks_data/chunks.parquet \\
      --output-parquet data/embeddings_data/2025-07-03_embeddings.parquet

Notes:
- Default model: text‑embedding‑3‑small (1536‑dim). Override with --model.
- Batch size default 512; for "large" model recommend 256.
- Fix: use list‑of‑list for `vectors` when calling `upload_collection` (numpy ndarray was not JSON‑serialisable in 1.14 qdrant‑client).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI, OpenAIError
from qdrant_client import QdrantClient, models as qdr

# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    today = dt.date.today().isoformat()
    p = argparse.ArgumentParser("Embed QuantConnect chunks & upload to Qdrant")

    p.add_argument("--input-parquet", default="data/chunks_data/chunks.parquet")
    p.add_argument("--output-parquet", default=f"data/embeddings_data/{today}_embeddings.parquet")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--model", default="text-embedding-3-small")
    p.add_argument("--collection", default="quantconnect_chunks")
    p.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"))
    p.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", 6333)))
    p.add_argument("--skip-upsert", action="store_true")
    return p.parse_args()

# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------

def _load_chunks(parquet_path: str | Path) -> pd.DataFrame:
    if not Path(parquet_path).exists():
        sys.exit(f"❌  Parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    if {"chunk_id", "text"} - set(df.columns):
        sys.exit("❌  Parquet missing required columns 'chunk_id' & 'text'")
    return df


def _embed_batch(texts: List[str], client: OpenAI, model: str, retries: int = 5) -> List[List[float]]:
    attempt = 0
    while True:
        try:
            resp = client.embeddings.create(model=model, input=texts, encoding_format="float")
            return [d.embedding for d in resp.data]
        except OpenAIError as e:
            if attempt >= retries - 1:
                raise
            wait = 2 ** attempt
            print(f"⚠️  OpenAI error: {e}. retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
            attempt += 1


def _write_parquet(df: pd.DataFrame, out_path: str | Path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "vector_id": pa.array(df["vector_id"], type=pa.uint64()),
        "chunk_id": pa.array(df["chunk_id"], type=pa.string()),
        "vector": pa.array(df["vector"].tolist(), type=pa.list_(pa.float32())),
    }
    for col in ["doc", "hierarchy_path", "token_start", "token_end", "content_types"]:
        if col in df.columns:
            if col == "content_types":
                arrays[col] = pa.array(df[col].tolist(), type=pa.list_(pa.string()))
            else:
                arrays[col] = pa.array(df[col].astype(str), type=pa.string())
    pq.write_table(pa.Table.from_pydict(arrays), out, compression="zstd")

# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------

def main():
    load_dotenv()
    args = _parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("❌  OPENAI_API_KEY missing (see .env)")

    print("📥  Reading chunks…")
    df = _load_chunks(args.input_parquet).reset_index(drop=True)
    n = len(df)
    df["vector_id"] = df.index.astype(np.uint64)
    print(f"   → {n:,} chunks")

    client = OpenAI()
    vectors: List[List[float]] = []
    print("🧮  Embedding…")
    for start in tqdm(range(0, n, args.batch_size)):
        texts = df.loc[start : start + args.batch_size - 1, "text"].tolist()
        vectors.extend(_embed_batch(texts, client, args.model))
    if len(vectors) != n:
        sys.exit("❌  Embedding count mismatch")
    df["vector"] = vectors

    print("💾  Writing embeddings Parquet…")
    _write_parquet(df, args.output_parquet)
    print(f"   → {args.output_parquet}")

    if args.skip_upsert:
        print("⏭️  Skip-upsert done.")
        return

    dim = 3072 if "large" in args.model else 1536
    qc = QdrantClient(
        host=args.qdrant_host,
        port=args.qdrant_port,
        timeout=60  # 60 s cho mỗi request
    )

    if not qc.collection_exists(args.collection):
        print(f"   → Create collection '{args.collection}' ({dim}‑dim)…")
        qc.create_collection(args.collection, vectors_config=qdr.VectorParams(size=dim, distance=qdr.Distance.COSINE))
    else:
        if qc.get_collection(args.collection).vectors_config.size != dim:
            sys.exit("❌  Collection dim mismatch")

    print("🚀  Upserting to Qdrant…")
    batch_upsert = 500

    def ensure_python_type(x):
        """Convert mọi numpy type → Python native (list / int / float / str)."""
        if isinstance(x, np.ndarray):
            return x.tolist()  # ndarray → list
        if isinstance(x, np.generic):  # mọi scalar numpy (int32, float32…)
            return x.item()  # → int / float Python
        if isinstance(x, list):
            # bảo đảm phần tử trong list cũng là python scalar
            return [ensure_python_type(i) for i in x]
        return x  # đã là python thuần

    for start in tqdm(range(0, n, batch_upsert)):
        sub = df.iloc[start: start + batch_upsert]

        # ---------- ids & vectors ----------
        ids = [int(i) for i in sub["vector_id"].values]
        vectors = [ensure_python_type(v) for v in sub["vector"].values]

        # ---------- payload ----------
        payload_cols = ["chunk_id", "doc", "hierarchy_path",
                        "token_start", "token_end", "content_types"]
        payload_cols = [c for c in payload_cols if c in sub.columns]

        payloads_df = sub[payload_cols].copy()
        list_cols = ["hierarchy_path", "content_types"]
        for col in list_cols:
            if col in payloads_df.columns:
                payloads_df[col] = payloads_df[col].map(ensure_python_type)

        payloads = payloads_df.to_dict("records")

        # ---------- upsert ----------
        qc.upload_collection(
            collection_name=args.collection,
            ids=ids,
            vectors=vectors,
            payload=payloads,
            batch_size=batch_upsert,
        )

    print("✅  Done – vectors upserted.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
