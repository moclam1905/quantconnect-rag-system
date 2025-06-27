#!/usr/bin/env python3
"""Phase 2 – Parse *_structured.json* → Section JSON

Usage:
    python phase2_parser.py \
        --input data/structured_analysis/Lean-Engine_structured.json \
        [--rules pattern_rules.yaml] [--debug]

This script **supersedes** the old HTML‑based Phase 1 parser.  It reads the
*_structured.json* produced by **DocumentStructureAnalyzer**, reconstructs the
heading tree, classifies each non‑heading block with the help of
`pattern_rules.yaml`, and writes one JSON file per section to:

    data/parsed_content/<doc>/sections/{order:02d}_{slug}.json

If any block falls back to the generic type `documentation_text`, it will be
recorded in `unclassified_blocks.log` for later rule refinement.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # PyYAML
from slugify import slugify  # python‑slugify

# ─────────────────────────────── Logging ──────────────────────────────── #
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s", stream=sys.stdout)
logger = logging.getLogger("Phase2Parser")

# ═════════════════════════════ BlockClassifier ═══════════════════════════ #
class BlockClassifier:
    """Classify a block based on *pattern_rules.yaml*.

    The YAML structure is expected to be:

    ```yaml
    skip_content:
      - navigation_content          # by content_type string
      - {class: "some‑class"}       # (class not used – kept for compat)
    api_reference:
      - code_content                # override content_type
    tutorial_content:
      - {tag: "example_content"}
    ```
    Anything under the top‑level key **skip_content** triggers skipping the
    block.  For the remaining groups, any entry (string or dict with `tag` or
    `class`) will map to the group name as the final *semantic_type*.
    """

    def __init__(self, rules_path: str) -> None:
        try:
            with open(rules_path, "r", encoding="utf-8") as fh:
                rules: Dict[str, Any] = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            logger.warning("Rules file '%s' not found – all blocks keep their original content_type", rules_path)
            rules = {}

        # Skip rules ------------------------------------------------------ #
        self.skip_types: set[str] = set()
        for item in rules.get("skip_content", []):
            if isinstance(item, str):
                self.skip_types.add(item)
            elif isinstance(item, dict):
                if "tag" in item:
                    self.skip_types.add(item["tag"])
                if "class" in item:
                    # structured JSON no longer holds original class, keep for compat
                    pass

        # Forward/override rules ----------------------------------------- #
        self.override_map: Dict[str, str] = {}
        for group, entries in rules.items():
            if group == "skip_content":
                continue
            for entry in entries or []:
                if isinstance(entry, str):
                    self.override_map[entry] = group
                elif isinstance(entry, dict):
                    if "tag" in entry:
                        self.override_map[entry["tag"]] = group
                    if "class" in entry:
                        self.override_map[entry["class"]] = group  # kept for compat

    # ------------------------------------------------------------------ #
    def classify(self, content_type: str) -> Optional[str]:
        """Return final semantic_type or *None* to skip the block."""
        if content_type in self.skip_types:
            return None
        if content_type in self.override_map:
            return self.override_map[content_type]
        # Fallback: keep original content_type or use documentation_text
        return content_type if content_type else "documentation_text"

# ═════════════════════════════ SectionBuilder ════════════════════════════ #
class SectionBuilder:
    """Build a nested section tree using a heading stack."""

    def __init__(self, max_level: int = 3, min_chunks: int = 2):
        self._stack: deque[Dict[str, Any]] = deque()
        self.sections: List[Dict[str, Any]] = []
        self._serial = 0  # incremental section id & order
        self.max_level  = max_level
        self.min_chunks = min_chunks


    # ------------------------------------------------------------------ #
    def _push_heading(self, block: Dict[str, Any]) -> None:
        level: int = block["metadata"]["heading_level"]
        title: str = block["metadata"]["heading_text"]

        if level > self.max_level:
            level = self.max_level

        # pop until parent level found
        while self._stack and self._stack[-1]["level"] >= level:
            self._stack.pop()

        parent_id = self._stack[-1]["id"] if self._stack else None
        self._serial += 1
        sec_id = f"section_{self._serial}"
        path = self._stack[-1]["hierarchy_path"] + [title] if self._stack else [title]

        section_obj: Dict[str, Any] = {
            "id": sec_id,
            "parent_id": parent_id,
            "level": level,
            "title": title,
            "order": self._serial,
            "hierarchy_path": path,
            "chunks": [],
        }

        self._stack.append(section_obj)
        self.sections.append(section_obj)

    # ------------------------------------------------------------------ #
    def _add_chunk(self, block: Dict[str, Any], semantic_type: str) -> None:
        if not self._stack:
            # ensure at least one section exists (preamble)
            self._push_heading({
                "content_type": "heading_1",
                "metadata": {"heading_level": 1, "heading_text": "Preamble"},
                "order": 0,
            })
        current = self._stack[-1]
        chunk_id = f"chunk_{block['order']:05d}"
        current["chunks"].append({
            "chunk_id": chunk_id,
            "semantic_type": semantic_type,
            "text": block["text"],
            "token_count": block["metadata"].get("length", 0),
        })

    # ------------------------------------------------------------------ #
    def ingest(self, block: Dict[str, Any], semantic_type: Optional[str]) -> None:
        if block["content_type"].startswith("heading_"):
            self._push_heading(block)
        elif semantic_type is not None:
            self._add_chunk(block, semantic_type)
        # else: skipped

    # ------------------------------------------------------------------ #
    def flush(self, out_dir: Path) -> None:
        # ── gộp section có ít chunk ────────────────────────────────
        id2sec = {s["id"]: s for s in self.sections}
        for sec in reversed(self.sections):
            if len(sec["chunks"]) < self.min_chunks:
                parent = id2sec.get(sec["parent_id"])
                if parent:
                    parent["chunks"].extend(sec["chunks"])
                    sec["chunks"].clear()

        # ── ghi file ───────────────────────────────────────────────
        out_dir.mkdir(parents=True, exist_ok=True)
        for sec in self.sections:
            if not sec["chunks"]:
                continue            # bỏ section rỗng
            filename = f"{sec['order']:02d}_{slugify(sec['title']) or 'untitled'}.json"
            with open(out_dir / filename, "w", encoding="utf-8") as fh:
                json.dump(sec, fh, ensure_ascii=False, indent=2)


# ═════════════════════════════ Phase2Parser ══════════════════════════════ #
class Phase2Parser:
    def __init__(self, rules_path: str):
        self.classifier = BlockClassifier(rules_path)
        self.stats = {"total": 0, "skipped": 0, "unclassified": 0}
        self.unclassified: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def parse(
            self,
            structured_path: Path,
            max_level: int = 3,
            min_chunks: int = 2,
            debug: bool = False,
    ) -> Dict[str, Any]:

        if not structured_path.exists():
            raise FileNotFoundError(structured_path)

        doc_name = structured_path.stem.replace("_structured", "")
        out_base = Path("data/parsed_content") / doc_name
        sections_dir = out_base / "sections"
        log_path = out_base / "unclassified_blocks.log"

        with open(structured_path, "r", encoding="utf-8") as fh:
            doc_json = json.load(fh)

        builder = SectionBuilder(max_level=max_level, min_chunks=min_chunks)

        for block in doc_json.get("content_blocks", []):
            self.stats["total"] += 1

            if block["content_type"].startswith("heading_"):
                # headings – ensure section tree
                builder.ingest(block, None)
                continue

            semantic_type = self.classifier.classify(block["content_type"])
            if semantic_type is None:
                self.stats["skipped"] += 1
                continue

            if semantic_type == "documentation_text" and block["content_type"] != "documentation_text":
                self.stats["unclassified"] += 1
                preview = block["text"][:120].replace("\n", " ")
                builder_sec_title = builder._stack[-1]["title"] if builder._stack else None
                self.unclassified.append({
                    "section_title": builder_sec_title,
                    "block_order": block["order"],
                    "content_type": block["content_type"],
                    "preview": preview,
                })

            builder.ingest(block, semantic_type)

        # write files ----------------------------------------------------- #
        builder.flush(sections_dir)
        if self.unclassified:
            with open(log_path, "w", encoding="utf-8") as fh:
                for rec in self.unclassified:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # KPI ------------------------------------------------------------- #
        pct_uncl = (self.stats["unclassified"] / self.stats["total"] * 100) if self.stats["total"] else 0
        logger.info(
            "Parsed %d blocks | Skipped: %d | Unclassified: %d (%.1f%%)",
            self.stats["total"], self.stats["skipped"], self.stats["unclassified"], pct_uncl,
        )
        if self.unclassified:
            logger.info("Unclassified log → %s", log_path)

        return {
            "section_count": len(builder.sections),
            "output_directory": str(out_base),
        }

# ═════════════════════════════════ Main ══════════════════════════════════ #

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 2 – Structured JSON parser")
    ap.add_argument("--input", required=True, help="Path to *_structured.json")
    ap.add_argument("--rules", default="pattern_rules.yaml", help="YAML rules file")
    ap.add_argument("--debug", action="store_true", help="Verbose debug")
    ap.add_argument("--max-level", type=int, default=3,
                    help="Chỉ tách heading tới cấp này (mặc định 3)")
    ap.add_argument("--min-chunks", type=int, default=2,
                    help="Gộp section có ít hơn N chunk (mặc định 2)")

    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    parser = Phase2Parser(args.rules)
    result = parser.parse(
        Path(args.input),
        max_level=args.max_level,
        min_chunks=args.min_chunks,
        debug=args.debug,
    )

    print("\n✔️  Phase 2 completed.")
    print(f"   Sections created : {result['section_count']}")
    print(f"   Output directory : {result['output_directory']}")


if __name__ == "__main__":
    main()
