"""Simple rule‑based template selector; LLM fallback when needed."""
import re

_RULES = [
    (re.compile(r"\b(how|cách|steps?)\b", re.I), "how_to"),
    (re.compile(r"\b(code|ví dụ|example)\b", re.I), "code_explain"),
    (re.compile(r"\b(schedule|rebalance|consolidator)\b", re.I), "code_explain"),
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


