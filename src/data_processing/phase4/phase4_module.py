# =========================================================
# QuantConnect RAG – Phase 4 implementation (single‑file package)
# This file contains **all** components for the Answering/RAG layer
# Each logical module is delimited by "# --- <filename> ---" so you can
# later split into real files if desired.
# =========================================================

# ---------------------------------------------------------
# settings.py
# ---------------------------------------------------------
"""Centralised configuration & helpers (read from .env if present)."""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from langchain.prompts import ChatPromptTemplate
from pydantic import Field
from pydantic_settings import BaseSettings


class _Settings(BaseSettings):
    # --- core API keys & endpoints
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")
    bm25_endpoint: str = Field("http://localhost:8000/search", env="BM25_ENDPOINT")
    qdrant_collection: str = Field("quantconnect_chunks", env="QDRANT_COLLECTION")
    embedding_model: str = Field("text-embedding-3-large", env="EMBEDDING_MODEL")

    # --- cache behaviour
    cache_ttl_hours: int = Field(24, env="CACHE_TTL_HOURS")  # 0 = infinite
    index_version: str = Field("2025-07-17", env="INDEX_VERSION")

    # --- model hierarchy
    llm_primary: str = Field("gpt-4o", env="LLM_PRIMARY")
    llm_secondary: str = Field("gpt-4o-mini", env="LLM_SECONDARY")
    llm_fallback: str = Field("gpt-3.5-turbo-0125", env="LLM_FALLBACK")

    model_timeout: int = Field(25, env="LLM_TIMEOUT_SECONDS")
    max_retries: int = Field(2, env="LLM_RETRIES")

    # --- server
    host: str = Field("127.0.0.1", env="HOST")
    port: int = Field(8001, env="PORT")
    api_key_header: Optional[str] = Field(None, env="API_KEY_HEADER")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

@lru_cache() # type: ignore[call-arg]
def get_settings() -> _Settings:
    return _Settings()


# Utility: project root
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------
# retriever.py
# ---------------------------------------------------------
"""HybridRetriever combines BM25 HTTP + Qdrant vector search."""
from typing import List, Dict, Any

from qdrant_client import QdrantClient

from langchain.schema import Document


class HybridRetriever:
    """Fuse BM25 & vector results using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, *, bm25_endpoint: str, qdrant_client: QdrantClient,
                 collection_name: str | None = None):
        self.bm25_endpoint = bm25_endpoint.rstrip("/")
        self.qdrant = qdrant_client
        # nếu không truyền tên, dùng giá trị .env (QDRANT_COLLECTION)
        self.collection = collection_name or settings.qdrant_collection

    # ----------------------------- BM25 search
    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if settings.api_key_header:
            headers["X-API-Key"] = settings.api_key_header
        resp = requests.post(
            self.bm25_endpoint,
            json={"query": query, "limit": top_k},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    # ----------------------------- vector search via Qdrant
    def _vector_search(self, query_embedding: List[float], top_k: int):
        # dùng API cũ 'search' – Qdrant nào cũng hỗ trợ
        return self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_embedding,  # ← trường chuẩn cho bản cũ
            with_payload=True,
            with_vectors=False,
            limit=top_k,
        )

    # ----------------------------- public interface
    def get_relevant_documents(self, query: str, embedding_fn, *, top_k: int = 6) -> List[Document]:
        """Return LangChain Documents fused from both sources."""
        # Step 1: get embedding once
        query_emb = embedding_fn(query)
        bm25_raw = self._bm25_search(query, top_k=top_k)
        vec_raw = self._vector_search(query_emb, top_k=top_k)

        # Build mapping id -> (score, source)
        scores = {}

        def _rrf(rank: int, k: int = 60):
            return 1.0 / (k + rank)

        # BM25 part
        for rank, item in enumerate(bm25_raw):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0) + _rrf(rank)

        # Vector part
        for rank, item in enumerate(vec_raw):
            cid = item.payload["chunk_id"]
            scores[cid] = scores.get(cid, 0) + _rrf(rank)

        # Sort by combined score desc
        ranked_ids = sorted(scores.items(), key=lambda t: t[1], reverse=True)[:top_k]

        max_score = max(scores.values() or [1.0])

        docs: List[Document] = []
        for cid, sc in ranked_ids:
            payload = next(
                (p.payload for p in vec_raw if hasattr(p, "payload") and p.payload["chunk_id"] == cid),
                None,
            ) or next((b for b in bm25_raw if isinstance(b, dict) and b["chunk_id"] == cid), None)
            if payload:
                rel = round(sc / max_score, 3)  # chuẩn hóa 0‑1 và làm tròn
                docs.append(
                    Document(
                        page_content=payload["text"],
                        metadata={"chunk_id": cid, "relevance": rel},
                    )
                )
        return docs


# ---------------------------------------------------------
# cache.py
# ---------------------------------------------------------
"""Redis Semantic Cache (cos‑sim based)."""
import hashlib
from datetime import timedelta
from typing import Optional

import redis
import numpy as np

settings = get_settings()


class SemanticCache:
    def __init__(self, redis_url: str):
        self.r = redis.Redis.from_url(redis_url, decode_responses=False)
        self.ttl = None if settings.cache_ttl_hours == 0 else timedelta(hours=settings.cache_ttl_hours)

    def _make_key(self, query: str) -> str:
        base = hashlib.sha256((query + settings.index_version).encode()).hexdigest()
        return f"ragcache:{base}"

    def get(self, query: str, query_emb: List[float]) -> Optional[Dict[str, Any]]:
        key = self._make_key(query)
        raw = self.r.get(key)
        if raw is None:
            return None
        try:
            raw = self.r.get(key)
            if raw is None:
                return None
            entry = json.loads(raw.decode())

            cached_emb = np.array(entry["embedding"], dtype=np.float32)
            sim = self._cosine(np.array(query_emb, dtype=np.float32), cached_emb)
            if sim >= 0.95:
                return entry["response"]
        except Exception:
            return None
        return None

    def set(self, query: str, query_emb: List[float], response: Dict[str, Any]):
        key = self._make_key(query)
        data = json.dumps({"embedding": query_emb, "response": response}).encode()
        if self.ttl:
            self.r.setex(key, self.ttl, data)
        else:
            self.r.set(key, data)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------------------------------------------------------
# table_processor.py
# ---------------------------------------------------------
"""Detect table queries & summarise Markdown tables into small DataFrame markdown."""
import re
from io import StringIO

import pandas as pd

_TABLE_REGEX = re.compile(r"\b(bảng|table|so sánh|compare|columns?)\b", re.I)


def detect_table_query(query: str, context_text: str) -> bool:
    if not _TABLE_REGEX.search(query):
        return False
    if context_text.count("|") >= 3:
        return True
    return False


def summarise_markdown_table(md: str, max_cols: int = 3, max_rows: int = 10) -> str:
    """Return a reduced Markdown table."""
    try:
        # convert md -> html -> df via pandas
        df_list = pd.read_html(StringIO(md), flavor="bs4")
        if not df_list:
            return ""
        df = df_list[0]
        # drop columns with low variance
        for col in df.columns.tolist():
            if df[col].nunique() <= 1:
                df.drop(columns=col, inplace=True)
        if len(df.columns) > max_cols:
            df = df.iloc[:, :max_cols]
        df = df.head(max_rows)
        return df.to_markdown(index=False)
    except Exception:
        return ""


# ---------------------------------------------------------
# analyze_query.py
# ---------------------------------------------------------
"""Simple rule‑based template selector; LLM fallback when needed."""
import re

_RULES = [
    (re.compile(r"\b(how|cách|steps?)\b", re.I), "how_to"),
    (re.compile(r"\b(code|ví dụ|example)\b", re.I), "code_explain"),
    (re.compile(r"\b(tham số|parameter|argument)\b", re.I), "api_reference"),
]


def choose_template(query: str) -> str:
    for rx, name in _RULES:
        if rx.search(query):
            return name
    if len(query.split()) > 25:
        # fallback: treat as general but could call mini‑LLM classifier later
        return "general"
    return "general"


# ---------------------------------------------------------
# prompt_templates.md (in‑code fallback, will load external file if exists)
# ---------------------------------------------------------
_DEFAULT_TEMPLATES = {
    "general": """You are a knowledgeable assistant. Answer the question briefly and cite sources like [Lean-Cli/00042].\n\nQuestion:\n{question}\n\nContext:\n{context}\n""",
    "how_to": """You are an expert tutor. Provide step‑by‑step guidance and cite chunks.\n\n{question}\n---\n{context}\n""",
    "code_explain": """Explain the following code snippet clearly, provide a runnable example, and cite sources.\n\n{question}\n---\n{context}\n""",
    "api_reference": """Describe the API method, its parameters, and usage. Cite docs.\n\n{question}\n---\n{context}\n""",
}


# ---------------------------------------------------------
# answer_chain.py
# ---------------------------------------------------------
"""Central pipeline: retrieve → build prompt → call LLM → cache → return."""
import time
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.schema import LLMResult
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.messages import BaseMessage
settings = get_settings()


class _LatencyTracer(BaseCallbackHandler):
    def __init__(self):
        self.start = time.perf_counter()
        self.first_token_time: float | None = None
        self.total_time: float | None = None

    def on_llm_new_token(self, token: str, **kwargs):
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def on_llm_end(self, response: LLMResult, **kwargs):
        self.total_time = time.perf_counter() - self.start


class AnswerChain:
    def __init__(self, retriever: HybridRetriever, cache: SemanticCache, embedding_fn):
        self.retriever = retriever
        self.cache = cache
        self.embed = embedding_fn

    # -----------------------------------------------------
    def _call_llm(self, messages: list[BaseMessage], tracer: _LatencyTracer, level: int = 0) -> tuple[str, int]:
        llm_name = [settings.llm_primary, settings.llm_secondary, settings.llm_fallback][level]
        llm = ChatOpenAI(
            model=llm_name,
            temperature=0.1,
            timeout=settings.model_timeout,
            max_tokens=512,
            callbacks=[tracer],
        )
        res = llm.invoke(messages)
        tokens = res.response_metadata.get("token_usage", {}).get("total_tokens", 0)
        return res.content, tokens

    # -----------------------------------------------------
    def __call__(self, query: str) -> Dict[str, Any]:
        # 1) check cache -------------------------------------------------------
        q_emb = self.embed(query)
        cached = self.cache.get(query, q_emb)
        if cached:
            cached["metadata"]["cache_hit"] = True
            return cached

        # 2) retrieve context --------------------------------------------------
        t0 = time.perf_counter()
        docs = self.retriever.get_relevant_documents(query, self.embed, top_k=6)
        retrieval_ms = int((time.perf_counter() - t0) * 1000)

        # 3) build chat‑prompt --------------------------------------------------
        template_name = choose_template(query)
        template_text = _DEFAULT_TEMPLATES.get(template_name,
                                               _DEFAULT_TEMPLATES["general"])

        prompt_tpl = ChatPromptTemplate.from_messages(
            [("system", template_text)]
        )
        messages = prompt_tpl.format_messages(
            question=query,
            context="\n\n".join(d.page_content for d in docs),
        )

        # 4) call LLM with fallback & tracer -----------------------------------
        tracer = _LatencyTracer()
        content, tokens = None, 0
        for level in range(3):
            try:
                content, tokens = self._call_llm(messages, tracer, level)
                fallback_level = level
                break
            except Exception:
                if level == 2:
                    raise

        # 5) craft response & cache -------------------------------------------
        total_ms = retrieval_ms + int((tracer.total_time or 0) * 1000)

        response = {
            "answer": content,
            "citations": [
                {
                    "chunk_id": d.metadata["chunk_id"],
                    "relevance": d.metadata.get("relevance", 0.0),
                    "snippet": d.page_content[:200] + "…",
                } for d in docs
            ],
            "metadata": {
                "model": [settings.llm_primary,
                          settings.llm_secondary,
                          settings.llm_fallback][fallback_level],
                "tokens_used": tokens,
                "confidence": 1.0,
                "template": template_name,
                "cache_hit": False,
            },
            "latency_ms": {
                "retrieval": retrieval_ms,
                "generation": int((tracer.total_time or 0) * 1000),
                "total": total_ms,
            },
        }
        self.cache.set(query, q_emb, response)
        return response


# ---------------------------------------------------------
# citation_validator.py
# ---------------------------------------------------------
import re
from typing import List

_CHUNK_PATTERN = re.compile(r"\[([\w\-/]+?)]")


def validate_citations(answer: str, retrieved_ids: List[str]):
    ids_in_answer = _CHUNK_PATTERN.findall(answer)
    missing = [cid for cid in ids_in_answer if cid not in retrieved_ids]
    if missing:
        raise ValueError(f"Answer cites chunks not in retrieval: {missing}")


# ---------------------------------------------------------
# answer_api.py
# ---------------------------------------------------------
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
import uvicorn

settings = get_settings()

app = FastAPI(title="QuantConnect RAG – Answer API")

# instantiate heavy stuff once
from langchain_openai import OpenAIEmbeddings


embedding_model = OpenAIEmbeddings(
    model=settings.embedding_model,
)
embedding_fn = embedding_model.embed_query


qdrant_client = QdrantClient(url=settings.qdrant_url)
retriever = HybridRetriever(bm25_endpoint=settings.bm25_endpoint,
                            qdrant_client=qdrant_client,
                            collection_name=settings.qdrant_collection)

cache = SemanticCache(redis_url=settings.redis_url)
chain = AnswerChain(retriever, cache, embedding_fn)

# middleware for latency


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    import time

    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(duration)
    return response


@app.post("/answer")
async def answer_endpoint(body: Dict[str, str], x_api_key: str | None = Header(default=None)):
    if settings.api_key_header and x_api_key != settings.api_key_header:
        raise HTTPException(status_code=401, detail="Invalid API key")
    query = body.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    try:
        response = chain(query)
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)

# ---------------------------------------------------------
# eval_phase4.py
# ---------------------------------------------------------
"""Quick benchmark; run `python eval_phase4.py`"""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

import requests

settings = get_settings()


def _one(q):
    resp = requests.post(f"http://{settings.host}:{settings.port}/answer", json={"query": q["question"]})
    latency = resp.json().get("latency_ms", {}).get("total", 0)
    return latency


def main():
    eval_path = Path("eval_set.jsonl")
    if not eval_path.exists():
        print("Eval set not found. Skipping.")
        return
    items = [json.loads(l) for l in eval_path.read_text().splitlines()]
    with ThreadPoolExecutor(max_workers=8) as ex:
        latencies = list(ex.map(_one, items))
    print(f"p95 latency: {sorted(latencies)[int(len(latencies)*0.95)]} ms | avg {mean(latencies):.1f} ms")


if __name__ == "__main__":
    main()

# =========================================================
# End of single‑file Phase 4 package
# To split into real files, divide by the '# --- filename ---' markers.
# =========================================================
