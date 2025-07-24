"""Redis Semantic Cache (cos‑sim based)."""
import hashlib
from datetime import timedelta
from typing import Optional

import redis
import numpy as np
from typing import List, Dict, Any
import json

from .settings import get_settings
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


