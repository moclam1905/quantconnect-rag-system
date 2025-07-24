import re
from typing import List

_CHUNK_PATTERN = re.compile(r"\[([\w\-/]+?)]")


def validate_citations(answer: str, retrieved_ids: List[str]):
    ids_in_answer = _CHUNK_PATTERN.findall(answer)
    missing = [cid for cid in ids_in_answer if cid not in retrieved_ids]
    if missing:
        raise ValueError(f"Answer cites chunks not in retrieval: {missing}")


