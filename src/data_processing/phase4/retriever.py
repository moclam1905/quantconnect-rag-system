"""HybridRetriever combines BM25 HTTP + Qdrant vector search."""
from typing import List, Dict, Any

from qdrant_client import QdrantClient

from langchain.schema import Document
import requests

from .settings import get_settings
settings = get_settings()

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


