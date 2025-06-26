# src/data_processing/document_structure_analyzer.py
from __future__ import annotations

import csv
import json
import re
import statistics as stats
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterator, Optional

from bs4 import BeautifulSoup, NavigableString, Tag
from lxml import etree
from tqdm import tqdm

from src.utils.constants import should_skip

HEADING_TAGS = {f"h{i}" for i in range(1, 7)}

DEFAULT_LABEL_BY_TAG = {
    "pre": "code_content",

    "table": "table_content",
    "nav": "navigation_content",
    "li": "list_item",
}


def load_class_label_map(csv_paths: List[Path]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)

            # nếu có header, bỏ qua
            if not (header and len(header) >= 2 and "selector" in header[1].lower()):
                fh.seek(0)          # không có header → đọc lại từ đầu
                reader = csv.reader(fh)

            for row in reader:
                if not row:
                    continue

                # ── phân biệt 2 định dạng ───────────────────────
                if len(row) >= 6:                           # discover_patterns.csv
                    class_cell = row[2].strip()
                    label_cell = row[5].strip()
                else:                                       # pattern_review.csv
                    class_cell = row[1].strip()
                    label_cell = row[2].strip() if len(row) > 2 else ""

                if not class_cell:
                    continue

                # ô Human-Decision trống ⇒ mặc định skip_content
                label = label_cell or "skip_content"

                for token in class_cell.split():            # tách "blue-text-action language-buttons"
                    mapping[token] = label
    return mapping



class DocumentStructureAnalyzer:
    """
    Phase 1 – Phân tích cấu trúc tài liệu đã qua pre-process & discover.
    """

    def __init__(
        self,
        class_label_map: Dict[str, str],
        max_block_len: int = 5000,  # ký tự
    ) -> None:
        self.class_label_map = class_label_map
        self.max_block_len = max_block_len

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #

    def analyze_document_structure(
        self,
        html_path: Path,
        discovered_patterns: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw_blocks = list(self._identify_content_blocks_streaming(html_path))
        logical_blocks = self._extract_logical_units(raw_blocks)

        return {
            "document_metadata": self._extract_doc_meta(html_path),
            "content_blocks": logical_blocks,
            "discovered_patterns": discovered_patterns or {},
        }

    # ------------------------------------------------------------------ #
    # STEP 1 : STREAM-PARSE & GROUP BLOCKS
    # ------------------------------------------------------------------ #

    def _identify_content_blocks_streaming(
        self, html_path: Path
    ) -> Iterator[Dict[str, Any]]:
        """
        Duyệt node bằng lxml.iterparse để tiết kiệm RAM. Trả iterator
        block: {"nodes": List[str], "label": str, "heading_level": int|None}
        """
        BLOCK_TAGS = {"p", "pre", "table", "div", "li", *HEADING_TAGS}

        context = etree.iterparse(
            html_path,
            html=True,
            events=("end",),
            tag="*",
            encoding="utf-8",
        )

        current_block: Dict[str, Any] = {
            "nodes": [],
            "label": None,
            "heading_level": None,
            "heading_text": None,
        }
        current_len = 0

        for _, elem in tqdm(context, desc=f"Parse {html_path.name}"):
            tag = elem.tag.lower() if isinstance(elem.tag, str) else ""

            # ── BỎ khối <pre class="csharp"> nếu liền sau là <pre class="python"> ──
            if tag == "pre":
                cls = elem.get("class") or ""
                if "csharp" in cls:
                    next_sib = elem.getnext()
                    if next_sib is not None and next_sib.tag == "pre":
                        if "python" in (next_sib.get("class") or ""):
                            elem.clear()
                            continue
            # ───────────────────────────────────────────────────────────────────────
            # ─── BỎ <li> khi nó nằm trong bảng ─────────────────────
            if tag == "li":
                if any(anc.tag.lower() == "table" for anc in elem.iterancestors()):
                    continue
            # ───────────────────────────────────────────────────────

            # — BỎ <p class="property-description"> KHI NẰM TRONG BẢNG —
            if tag == "p" and "property-description" in (elem.get("class") or []):
                if any(anc.tag.lower() == "table" for anc in elem.iterancestors()):
                    # KHÔNG clear, chỉ không tạo block; text vẫn còn trong <td>
                    continue
            # ------------------------------------------------------------
            # ── KHÔNG tạo block cho <div class="error-messages"> bên trong <table> ──
            if tag == "div" and "error-messages" in (elem.get("class") or []):
                if any(anc.tag.lower() == "table" for anc in elem.iterancestors()):
                    continue  # text vẫn còn, nhưng block code_content không sinh
            # -----------------------------------------------------------------------

            if should_skip(elem):
                elem.clear()
                continue

            if tag not in BLOCK_TAGS:
                continue

            label = self._classify(elem, tag)

            if label == "skip_content":
                elem.clear()
                continue

            # detect heading
            if tag in HEADING_TAGS:
                # flush current block
                if current_block["nodes"]:
                    yield current_block
                heading_text = (elem.text or "").strip()
                current_block = {
                    "nodes": [etree.tostring(elem, encoding="unicode")],
                    "label": f"heading_{tag[-1]}",
                    "heading_level": int(tag[-1]),
                    "heading_text": heading_text,
                }
                current_len = len(heading_text)
                elem.clear()
                yield current_block
                current_block = {"nodes": [], "label": None, "heading_level": None}
                current_len = 0
                continue

            # start new block when label change or len exceed
            if (
                current_block["label"] is not None
                and label != current_block["label"]
            )or tag == "li"  or current_len >= self.max_block_len:
                yield current_block
                current_block = {
                    "nodes": [],
                    "label": label,
                    "heading_level": None,
                    "heading_text": None,
                }
                current_len = 0

            # init label
            if current_block["label"] is None:
                current_block["label"] = label

            # append node HTML string
            html_str = etree.tostring(elem, encoding="unicode")
            current_block["nodes"].append(html_str)
            current_len += len(html_str)

            elem.clear()

        # flush last
        if current_block["nodes"]:
            yield current_block

    # ------------------------------------------------------------------ #
    # STEP 2 : CLEAN & ADD METADATA
    # ------------------------------------------------------------------ #

    def _extract_logical_units(
        self, raw_blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Chuyển raw_blocks thành schema chuẩn."""
        logical: List[Dict[str, Any]] = []
        for order, blk in enumerate(raw_blocks):
            text = self._combine_text(blk["nodes"])
            if not text.strip():
                continue  # bỏ block rỗng
            logical.append(
                {
                    "order": order,
                    "content_type": blk["label"],
                    "text": text,
                    "metadata": {
                        "heading_level": blk.get("heading_level"),
                        "heading_text": blk.get("heading_text"),
                        "length": len(text),
                    },
                }
            )
        return logical

    # ------------------------------------------------------------------ #
    # HELPERS
    # ------------------------------------------------------------------ #

    def _classify(self, elem: etree._Element, tag: str) -> str:
        """Gán nhãn cho element dựa trên class → label, rồi fallback theo tag."""

        cls_attr = elem.get("class") or []  # list | str | []
        tokens = cls_attr.split() if isinstance(cls_attr, str) else cls_attr

        # Giữ div.code-snippet thành code_content
        if "code-snippet" in tokens:
            return "code_content"

        # tra mapping class → label
        for tok in tokens:
            if tok in self.class_label_map:
                return self.class_label_map[tok]

        # fallback theo tag
        return DEFAULT_LABEL_BY_TAG.get(tag, "generic_text")

    # ── helper: chuyển bảng HTML thành chuỗi văn bản ──────────────────────────
    def _table_to_text(self, table_soup) -> str:
        rows = []
        for tr in table_soup.find_all("tr"):
            cells_txt = []

            for cell in tr.find_all(["th", "td"]):
                #Kiểm tra ô có bản Python không?
                has_python = cell.find("code", class_="python") is not None
                if has_python:
                    for csh in cell.find_all("code", class_="csharp"):
                        csh.decompose()

                # 1️⃣  Nếu ô chứa danh sách <li> trực tiếp  → gom từng li trước rồi sang ô kế
                lis = cell.find_all("li")
                if lis:
                    li_texts = []
                    for li in lis:
                        parts = []
                        for node in li.descendants:
                            if isinstance(node, Tag) and node.name == "a" and node.get("href"):
                                parts.append(f"{node.get_text(strip=True)} ({node['href'].strip()})")
                            elif isinstance(node, str):
                                parent_tag = getattr(node, "parent", None)
                                # bỏ text nếu nó nằm trực tiếp trong <a>
                                if isinstance(parent_tag, Tag) and parent_tag.name == "a":
                                    continue
                                txt = node.strip()
                                if txt:
                                    parts.append(txt)

                        li_texts.append(" ".join(parts))
                    cells_txt.append(" • ".join(li_texts))  # bullet giữa LI
                    continue  # ⬅️  sang ô kế

                # 2️⃣  Ô KHÔNG có <li>  → logic cũ
                parts = []
                for node in cell.descendants:
                    if isinstance(node, Tag) and node.name == "a" and node.get("href"):
                        parts.append(f"{node.get_text(strip=True)} ({node['href'].strip()})")
                    elif isinstance(node, Tag) and node.name == "img":
                        alt = (node.get("alt") or "").lower()
                        if "check" in alt or not alt:
                            parts.append("✅")
                        elif "x" in alt or "cross" in alt:
                            parts.append("❌")
                        else:
                            parts.append(f"[{alt or 'icon'}]")
                    elif isinstance(node, str):
                        parent = getattr(node, "parent", None)
                        if isinstance(parent, Tag) and parent.name in {"a", "img"}:
                            continue
                        txt = node.strip()
                        if txt:
                            parts.append(txt)

                cells_txt.append(" ".join(parts))  # 3️⃣  ghép ô thường

            rows.append(" | ".join(cells_txt))

        return "\n".join(rows)

    # ─────────────────────────────────────────────────────────────────────────

    def _combine_text(self, node_html_list: List[str]) -> str:
        cleaned_parts: List[str] = []
        has_code_block = False  # NEW

        for html_frag in node_html_list:
            soup_frag = BeautifulSoup(html_frag, "html.parser")

            # Giữ cấu trúc bảng (table gốc)
            if soup_frag.name == "table":
                cleaned_parts.append(self._table_to_text(soup_frag))
                has_code_block = True
                continue

            # Giữ cấu trúc bảng khi gốc là [document] và có 1 table con
            table_child = None
            if soup_frag.name == "[document]":
                # duyệt con trực tiếp, lấy phần tử đầu tiên là <table>
                for child in soup_frag.contents:
                    if getattr(child, "name", None) == "table":
                        table_child = child  # kiểu Tag nên IDE chấp nhận
                        break

            if table_child is not None:
                cleaned_parts.append(self._table_to_text(table_child))
                has_code_block = True
                continue

            # ----- LI xử lý gọn 1 dòng -----
            li_tag = soup_frag if soup_frag.name == "li" else soup_frag.find("li", recursive=False)
            if li_tag is not None:
                pieces = []
                for node in li_tag.descendants:
                    # 1) anchor → text + URL
                    if isinstance(node, Tag) and node.name == "a" and node.get("href"):
                        link_text = node.get_text(" ", strip=True)
                        href = node["href"].strip()
                        pieces.append(f"{link_text} ({href})")
                    # 2) text thuần → chỉ lấy nếu không phải text trực tiếp bên trong <a>
                    elif isinstance(node, str):
                        # chỉ lấy text nếu KHÔNG nằm trực tiếp trong <a>
                        parent_tag = getattr(node, "parent", None)
                        if not (isinstance(parent_tag, Tag) and parent_tag.name == "a"):
                            pieces.append(node.strip())

                line = " ".join(filter(None, pieces))
                cleaned_parts.append("• " + line)
                continue
            # --------------------------------

            # Giữ nguyên <pre> đa dòng
            if soup_frag.name in {"pre"} or soup_frag.find("pre"):
                cleaned_parts.append(soup_frag.get_text("\n"))
                has_code_block = True
                continue
            # Giữ div.code-snippet đa dòng
            if "code-snippet" in (soup_frag.get("class") or []):
                cleaned_parts.append(soup_frag.get_text("\n"))
                has_code_block = True
                continue

            # --- xử lý inline <code> ------------------------------------------------
            if soup_frag.find("code"):
                pieces, seen = [], set()
                for node in soup_frag.descendants:
                    if isinstance(node, Tag) and node.name == "code":
                        text = node.get_text(strip=True)
                        cls = node.get("class") or []
                        lang = "C#" if "csharp" in cls else "Py" if "python" in cls else ""
                        key = (text, lang)
                        if key in seen and pieces and pieces[-1].rstrip().endswith(text):
                            continue
                        seen.add(key)
                        pieces.append(f"{text} ({lang})" if lang else text)

                    elif isinstance(node, str):
                        parent = getattr(node, "parent", None)
                        if not (isinstance(parent, Tag) and parent.name == "code"):
                            pieces.append(node.strip())

                merged = " ".join(filter(None, pieces))
                if merged:
                    cleaned_parts.append(merged)
                continue
            # -----------------------------------------------------------------------

            # ----- thêm URL sau anchor text -----
            if soup_frag.find("a"):
                pieces = []
                for node in soup_frag.descendants:
                    if isinstance(node, Tag) and node.name == "a":
                        href = node.get("href")
                        if href:
                            link_text = node.get_text(" ", strip=True)
                            pieces.append(f"{link_text} ({href.strip()})")
                    elif isinstance(node, str):
                        parent_tag = getattr(node, "parent", None)
                        if not (isinstance(parent_tag, Tag) and parent_tag.name == "a"):
                            pieces.append(node.strip())

                merged = " ".join(filter(None, pieces))
                if merged:
                    cleaned_parts.append(merged)
                continue
            # ------------------------------------

            # Các phần còn lại (p, code inline, a, …)
            stripped = soup_frag.get_text(" ", strip=True)
            if stripped:
                cleaned_parts.append(stripped)

        joined_raw = " ".join(cleaned_parts)

        if has_code_block:  # giữ xuống dòng trong code-block
            return joined_raw.strip()

        return re.sub(r"\s+", " ", joined_raw).strip()

    @staticmethod
    def _extract_doc_meta(html_path: Path) -> Dict[str, Any]:
        return {
            "source_file": html_path.name,
            "source_size_bytes": html_path.stat().st_size,
        }


# ---------------------------------------------------------------------- #
# CLI tiện dụng
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Phase-1 Document Structure Analyzer")
    ap.add_argument("--html", required=True, help="HTML đã qua preprocess")
    ap.add_argument("--patterns", help="JSON discovered_patterns (tuỳ chọn)")
    ap.add_argument("--class_map_csv", nargs="+", default=[
        "data/pattern_review.csv",
        "data/discover_patterns.csv",
    ])
    ap.add_argument("--out", required=True, help="File *.structured.json")
    ap.add_argument("--max_len", type=int, default=5000)
    args = ap.parse_args()

    class_map = load_class_label_map([Path(p) for p in args.class_map_csv])

    analyzer = DocumentStructureAnalyzer(
        class_label_map=class_map,
        max_block_len=args.max_len,
    )

    discovered = {}
    if args.patterns and Path(args.patterns).is_file():
        discovered = json.loads(Path(args.patterns).read_text())

    result = analyzer.analyze_document_structure(
        Path(args.html),
        discovered_patterns=discovered,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"✅  Saved structured JSON → {out_path}")
