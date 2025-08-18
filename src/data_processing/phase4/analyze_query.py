"""Simple rule‑based template selector; LLM fallback when needed."""
import re

_RULES = [
    (re.compile(r"\bhow to\b|\bsteps?\b", re.I), "how_to"),
    (re.compile(r"\bexample\b|\bcode\b", re.I), "code_explain"),
    (re.compile(r"\bparameter\b|\bargument\b|\bapi\b", re.I), "api_reference"),
    (re.compile(r"\btable\b|\brow\b|\bcolumn\b", re.I), "table_query"),
    (re.compile(r"\berror\b|\bexception\b", re.I), "debug_error"),
    (re.compile(r"\bcompare\b|\bversus\b|\bvs\.?\b", re.I), "comparison"),
    (re.compile(r"\bstep[- ]by[- ]step\b", re.I), "step_by_step"),
]


def choose_template(query: str) -> str:
    for rx, name in _RULES:
        if rx.search(query):
            return name
    if len(query.split()) > 25:
        # fallback: treat as general but could call mini‑LLM classifier later
        return "general"
    return "general"


