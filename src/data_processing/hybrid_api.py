#!/usr/bin/env python3
"""hybrid_api.py – FastAPI service for hybrid retrieval (Dense via Qdrant + BM25).
     • Fixes validation error by normalising `hierarchy_path` & `content_types` to list.
     • Works out‑of‑the‑box once BM25 index & Qdrant collection are ready.
"""
from __future__ import annotations

import asyncio
import gzip
import os
import pickle
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Sequence

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from qdrant_client import QdrantClient, models as qdr
from rank_bm25 import BM25Okapi

load_dotenv()

# ---------------------------------------------------------------------------
# Config (from .env with sensible defaults)
# ---------------------------------------------------------------------------
API_KEY             = os.getenv("API_KEY", "dev-key")
BM25_INDEX_PATH     = Path(os.getenv("BM25_INDEX_PATH", "data/bm25_index.pkl.gz"))
QDRANT_HOST         = os.getenv("QDRANT_HOST", "http://localhost:6333")
QDRANT_COLLECTION   = os.getenv("QDRANT_COLLECTION", "quantconnect_chunks")
QDRANT_EXPECTED_DIM = int(os.getenv("QDRANT_EXPECTED_DIM", "3072"))
RRF_K               = int(os.getenv("RRF_K", "60"))
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# Tokenizer (must match build_bm25.py)
# ---------------------------------------------------------------------------
_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
CUSTOM_STOP = {"the","a","an","and","or","but","in","on","at","to","for","of","with","by","is","are","was","were","been","be","have","has","had","do","does","did"}
PRESERVE   = {"class","def","function","method","return","self","this","import","from","using","namespace","public","private","algorithm","initialize","ondata","schedule","liquidate","order","buy","sell","long","short","position","how","what","when","where","why","which"}
STOPWORDS  = CUSTOM_STOP - PRESERVE

def tokenize(text:str)->List[str]:
    text = _CAMEL_RE.sub(" ", text).replace("_"," ")
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
from functools import lru_cache

def as_list(val:Any)->List[Any]:
    """Ensure *val* is a Python list for safe Pydantic validation."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, (set, tuple, Sequence)) and not isinstance(val,(str,bytes)):
        return list(val)
    # handle string like "['A' 'B']" or "['A','B']"
    if isinstance(val,str) and val.startswith("[") and val.endswith("]"):
        inner=val[1:-1]
        parts=[p.strip(" ' \"") for p in re.split(r"[;,]",inner) if p.strip()]
        return parts if parts else [val]
    return [val]

# ---------------------------------------------------------------------------
# Load BM25
# ---------------------------------------------------------------------------
print("📥 Loading BM25 index →", BM25_INDEX_PATH)
with gzip.open(BM25_INDEX_PATH,'rb') as fh:
    obj = pickle.load(fh)
BM25:BM25Okapi = obj['bm25']
CHUNK_IDS:List[str] = obj['chunk_ids']
print(f"✅ BM25 loaded ({len(CHUNK_IDS)} chunks)")

# ---------------------------------------------------------------------------
# Qdrant client
# ---------------------------------------------------------------------------
client = QdrantClient(url=QDRANT_HOST)
info = client.get_collection(QDRANT_COLLECTION)
if info.config.params.vectors.size != QDRANT_EXPECTED_DIM:
    raise RuntimeError("Qdrant vector size mismatch")
print("✅ Qdrant collection verified")

# ---------------------------------------------------------------------------
# Cache Embedding & BM25 results
# ---------------------------------------------------------------------------
@lru_cache(maxsize=512)
def embed_cached(query:str)->List[float]:
    headers={"Authorization":f"Bearer {OPENAI_API_KEY}","Content-Type":"application/json"}
    data={"model":"text-embedding-3-large","input":query}
    r=httpx.post("https://api.openai.com/v1/embeddings",headers=headers,json=data,timeout=30)
    r.raise_for_status()
    return r.json()['data'][0]['embedding']

@lru_cache(maxsize=512)
def bm25_cached(key:str,k:int)->List[int]:
    scores=BM25.get_scores(key.split())
    return [i for i,_ in sorted(enumerate(scores),key=lambda t:-t[1])[:k]]

# ---------------------------------------------------------------------------
# Reciprocal‑Rank Fusion
# ---------------------------------------------------------------------------

def rrf(ranks:List[List[int]],k:int=RRF_K)->List[int]:
    score:dict[int,float]={}
    for lst in ranks:
        for r,cid in enumerate(lst):
            score[cid]=score.get(cid,0)+1/(k+r)
    return [cid for cid,_ in sorted(score.items(),key=lambda x:-x[1])]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query:str
    k_dense:int=8
    k_sparse:int=32
    k_final:int=5
    doc:Optional[List[str]]=None

class ChunkResult(BaseModel):
    chunk_id:str
    score:float
    doc:str
    section_title:str
    hierarchy:List[str]
    text:str
    content_types:List[str]

class SearchResponse(BaseModel):
    latency_ms:int
    results:List[ChunkResult]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="QuantConnect Hybrid Retrieval API",version="0.2")

async def verify_key(x_api_key:str=Header(...,alias="X-API-Key")):
    if x_api_key!=API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Invalid API key")

FIELDS=["chunk_id","doc","section_title","hierarchy_path","text","content_types"]

@app.post("/search",response_model=SearchResponse,dependencies=[Depends(verify_key)])
async def search(req:SearchRequest):
    t0=time.perf_counter()

    # Sparse
    tok_key=" ".join(tokenize(req.query))
    sparse=bm25_cached(tok_key,req.k_sparse)

    # Dense
    qvec=await asyncio.get_event_loop().run_in_executor(None,embed_cached,req.query)
    qfilter=qdr.Filter(must=[qdr.FieldCondition(key="doc",match=qdr.MatchAny(any=req.doc))]) if req.doc else None
    dense_pts=client.search(collection_name=QDRANT_COLLECTION,query_vector=qvec,limit=req.k_dense,with_payload=FIELDS,query_filter=qfilter)
    dense=[int(pt.id) for pt in dense_pts]

    # Fusion
    fused=rrf([dense,sparse],k=RRF_K)[:req.k_final]
    payload_map={int(pt.id):pt for pt in dense_pts}

    results:List[ChunkResult]=[]
    for cid in fused:
        pload=payload_map.get(cid)
        if not pload:
            results.append(ChunkResult(chunk_id=CHUNK_IDS[cid],score=0.0,doc="",section_title="",hierarchy=[],text="",content_types=[]))
            continue
        p=pload.payload
        results.append(ChunkResult(
            chunk_id=p.get("chunk_id",CHUNK_IDS[cid]),
            score=float(pload.score) if pload.score is not None else 0.0,
            doc=p.get("doc",""),
            section_title=p.get("section_title",""),
            hierarchy=as_list(p.get("hierarchy_path")),
            text=p.get("text",""),
            content_types=as_list(p.get("content_types")),
        ))

    return SearchResponse(latency_ms=int((time.perf_counter()-t0)*1000),results=results)

@app.get("/health")
async def health():
    return {"status":"ok","time":datetime.utcnow().isoformat()+"Z"}
