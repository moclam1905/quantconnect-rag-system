"""Central pipeline: retrieve → build prompt → call LLM → cache → return."""
import time
from typing import Dict, Any

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import LLMResult
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.messages import BaseMessage

from .analyze_query import choose_template
from .retriever import HybridRetriever
from .cache import SemanticCache

from .settings import get_settings
settings = get_settings()

_DEFAULT_TEMPLATES = {
    "general": """You are a knowledgeable assistant. Answer the question briefly and cite sources like [Lean-Cli/00042].\n\nQuestion:\n{question}\n\nContext:\n{context}\n""",
    "how_to": """You are an expert tutor. Provide step‑by‑step guidance and cite chunks.\n\n{question}\n---\n{context}\n""",
    "code_explain": """Explain the following code snippet clearly, provide a runnable example, and cite sources.\n\n{question}\n---\n{context}\n""",
    "api_reference": """Describe the API method, its parameters, and usage. Cite docs.\n\n{question}\n---\n{context}\n""",
}

# ---------- override templates from prompt_template.md ----------
import pathlib, re, logging

def _load_external_templates() -> None:
    # giả sử file đặt cạnh answer_chain.py → src/phase4/prompt_template.md
    md = pathlib.Path(__file__).parent / "prompt_template.md"
    if not md.exists():
        logging.info("prompt_template.md not found – dùng template mặc định")
        return

    text = md.read_text(encoding="utf-8")
    # Bóc theo định dạng --- name: template_id ---
    blocks = re.split(r"^---\s*name:\s*(\w+)\s*---\s*$",
                      text, flags=re.M)[1:]
    for name, body in zip(blocks[0::2], blocks[1::2]):
        _DEFAULT_TEMPLATES[name.strip()] = body.strip()
    logging.info("Loaded %d templates from %s",
                 len(blocks)//2, md.name)

_load_external_templates()

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


