#!/usr/bin/env python3
"""
Phase 0: Pattern Discovery & Lightweight HTML Parser cho QuantConnect docs
Systematic approach để discover ALL patterns và tạo structured blocks

Usage:
------
1. **Discover patterns từ folder HTML files**
   python discover_patterns.py discover \
       --html_dir data/processed_html \
       --yaml pattern_rules.yaml \
       --out_dir data/pattern_analysis \
       --top_k 50

2. **Parse single HTML file thành structured JSON**
   python discover_patterns.py parse \
       --html_path data/processed_html/Quantconnect-Lean-Engine.html \
       --yaml pattern_rules.yaml \
       --discovered_json data/pattern_analysis/discovered_global.json \
       --out_file data/structured_analysis/Lean-Engine_structured.json

3. **Parse multiple HTML files in directory (BATCH)**
   python discover_patterns.py parse_dir \
       --html_dir data/processed_html \
       --yaml pattern_rules.yaml \
       --discovered_json data/pattern_analysis/discovered_global.json \
       --out_dir data/structured_analysis \
       --glob_pattern "*.html"

Key Features:
- Flexible wildcard pattern matching
- Unknown pattern logging và export
- BeautifulSoup parsing (no hang issues)
- Batch processing support
- Comprehensive quality reporting

Dependencies: beautifulsoup4, PyYAML
"""

from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import yaml
from bs4 import BeautifulSoup, Tag
from src.utils.constants import should_skip


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type definitions
Signature = Tuple[str, Tuple[str, ...], Tuple[str, ...]]  # (tag, classes, data_attrs)


###############################################################################
# Core utility functions
###############################################################################

def signature_of(el: Tag) -> Signature:
    """
    Tạo hashable signature cho element (tag, sorted classes, sorted data-* attrs)

    Returns:
        Tuple[tag, classes, data_attributes]
    """
    tag = (el.name or '').lower()

    # Extract và sort classes
    classes = el.get("class", [])
    if isinstance(classes, str):
        classes = classes.split()
    classes = tuple(sorted(classes)) if classes else tuple()

    # Extract data-* attributes
    data_attrs = tuple(sorted(attr for attr in el.attrs if attr.startswith("data-")))

    return tag, classes, data_attrs


def stream_parse_html(html_path: Path) -> Tuple[Counter[Signature], Dict[Signature, str]]:
    """
    Parse HTML file với BeautifulSoup và collect signature frequencies + samples

    Returns:
        Tuple[Counter với frequencies, Dict với sample content]
    """
    logger.info(f"🔍 Parsing with BeautifulSoup: {html_path}")

    counter: Counter[Signature] = Counter()
    samples: Dict[Signature, str] = {}
    elements_processed = 0

    try:
        # Load entire file content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Parse với BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all elements
        all_elements = soup.find_all()
        logger.info(f"📊 Found {len(all_elements)} total elements to process")

        for el in all_elements:
            elements_processed += 1

            # Generate signature
            sig = signature_of(el)
            counter[sig] += 1

            # Collect first sample for each signature
            if sig not in samples:
                try:
                    snippet = str(el)[:200]  # Get first 200 chars of element HTML
                    samples[sig] = html_lib.escape(snippet)
                except Exception as e:
                    samples[sig] = f"[Error extracting sample: {str(e)}]"

            # Progress logging
            if elements_processed % 5000 == 0:
                logger.info(f"   Progress: {elements_processed:,}/{len(all_elements)} elements processed")

    except Exception as e:
        logger.error(f"❌ Error parsing {html_path}: {e}")
        raise

    logger.info(f"✅ Completed: {elements_processed:,} elements → {len(counter)} unique signatures")
    return counter, samples


def load_yaml_rules(yaml_path: Path) -> Dict[str, List[str]]:
    """Load pattern rules từ YAML file"""
    if not yaml_path.exists():
        logger.warning(f"⚠️ YAML rules file not found: {yaml_path}")
        return {}

    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            rules = yaml.safe_load(f) or {}
        logger.info(f"📋 Loaded {len(rules)} rule categories from {yaml_path}")
        return rules
    except Exception as e:
        logger.error(f"❌ Error loading YAML rules: {e}")
        return {}


def auto_label_signature(sig: Signature, yaml_rules: Dict[str, List[str]], default: str = "generic_text") -> str:
    """
    Auto-label signature dựa trên YAML rules và built-in patterns

    Priority:
    1. Heading tags (h1-h6)
    2. List tags (ul, ol, li)
    3. YAML-defined rules
    4. Default fallback
    """
    tag, classes, data_attrs = sig

    # Built-in patterns - high priority
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return f"heading_{tag[1]}"

    if tag in {"li", "ol", "ul"}:
        return "list_content"

    if tag in {"table", "tr", "td", "th", "thead", "tbody"}:
        return "table_content"

    if tag in {"pre", "code"}:
        return "code_content"

    # YAML-driven classification
    for label, class_patterns in yaml_rules.items():
        if label == "skip_content":
            continue  # Handle skip separately

        # Check if any class matches patterns
        for pattern in class_patterns:
            if isinstance(pattern, dict):
                # Handle dict format: {"class": "value"}
                if "class" in pattern and pattern["class"] in classes:
                    return label
                if "tag" in pattern and pattern["tag"] == tag:
                    return label
                if "attr" in pattern and pattern["attr"] in data_attrs:
                    return label
            elif isinstance(pattern, str):
                # Handle string format: direct class name
                if pattern in classes:
                    return label

    return default


###############################################################################
# Command: discover
###############################################################################

def cmd_discover(args: argparse.Namespace) -> None:
    """
    Discovery command: scan all HTML files và generate global signature database
    """
    html_dir = Path(args.html_dir)
    out_dir = Path(args.out_dir)
    yaml_path = Path(args.yaml)

    # Validation
    if not html_dir.exists():
        logger.error(f"❌ HTML directory not found: {html_dir}")
        return

    # Setup output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {out_dir}")

    # Load YAML rules
    yaml_rules = load_yaml_rules(yaml_path)

    # Global aggregation
    global_counter: Counter[Signature] = Counter()
    global_samples: Dict[Signature, str] = {}
    files_processed = 0

    # Process all HTML files
    html_files = list(html_dir.rglob("*.html"))
    logger.info(f"🔍 Found {len(html_files)} HTML files to process")

    for html_path in html_files:
        try:
            logger.info(f"📄 Processing: {html_path.name}")
            counter, samples = stream_parse_html(html_path)

            # Merge results
            global_counter.update(counter)

            # Merge samples (first-seen wins)
            for sig, sample in samples.items():
                if sig not in global_samples:
                    global_samples[sig] = sample

            files_processed += 1

        except Exception as e:
            logger.error(f"❌ Failed to process {html_path}: {e}")
            continue

    if files_processed == 0:
        logger.error("❌ No files were successfully processed")
        return

    logger.info(f"✅ Processed {files_processed} files, found {len(global_counter)} unique signatures")

    # Auto-label all signatures
    discovered_signatures = []
    label_stats = Counter()

    for sig, freq in global_counter.most_common():
        label = auto_label_signature(sig, yaml_rules)
        label_stats[label] += 1

        discovered_signatures.append({
            "tag": sig[0],
            "classes": list(sig[1]),
            "data_attrs": list(sig[2]),
            "frequency": freq,
            "label": label,
            "sample": global_samples.get(sig, "")
        })

    # Generate comprehensive output
    discovery_data = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "files_processed": files_processed,
            "total_signatures": len(discovered_signatures),
            "yaml_rules_used": str(yaml_path),
            "label_distribution": dict(label_stats)
        },
        "signatures": discovered_signatures
    }

    # Save complete JSON
    json_path = out_dir / "discovered_global.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(discovery_data, f, ensure_ascii=False, indent=2)

    # Generate Top-K CSV for human review (optional)
    if args.top_k > 0:
        csv_path = out_dir / "pattern_review.csv"
        top_k = min(args.top_k, len(discovered_signatures))

        fieldnames = ["id", "tag", "classes", "data_attrs", "frequency", "label", "sample"]

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for idx, sig_data in enumerate(discovered_signatures[:top_k], 1):
                row = {
                    "id": idx,
                    "tag": sig_data["tag"],
                    "classes": " ".join(sig_data["classes"]),
                    "data_attrs": ",".join(sig_data["data_attrs"]),
                    "frequency": sig_data["frequency"],
                    "label": sig_data["label"],
                    "sample": sig_data["sample"][:100]  # Truncate for CSV
                }
                writer.writerow(row)

        logger.info(f"📋 Top-{top_k} patterns CSV: {csv_path}")

    # Summary report
    logger.info("=" * 60)
    logger.info("🎯 DISCOVERY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"📁 HTML files processed: {files_processed}")
    logger.info(f"📊 Unique signatures found: {len(discovered_signatures):,}")
    logger.info(f"💾 Global database: {json_path}")

    logger.info("\n📋 Label Distribution:")
    for label, count in label_stats.most_common(10):
        logger.info(f"   {label}: {count:,} signatures")

    if label_stats.get("generic_text", 0) > 0:
        logger.warning(f"⚠️ {label_stats['generic_text']} signatures labeled as 'generic_text'")
        logger.warning("   Consider reviewing và adding rules to pattern_rules.yaml")


###############################################################################
# Command: parse
###############################################################################

def load_signature_mapping(discovered_json_path: Path) -> Dict[Signature, str]:
    """Load signature→label mapping từ discovered JSON"""
    with discovered_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for sig_data in data["signatures"]:
        key = (
            sig_data["tag"],
            tuple(sig_data["classes"]),
            tuple(sig_data["data_attrs"])
        )
        mapping[key] = sig_data["label"]

    logger.info(f"📋 Loaded {len(mapping)} signature mappings")
    return mapping


def flexible_label_lookup(sig: Signature, mapping: Dict[Signature, str], default: str = "generic_text") -> str:
    """
    Flexible signature matching với wildcard support

    Matching priority:
    1. Exact match (tag, classes, data_attrs)
    2. Wildcard tag match (*, classes, data_attrs)
    3. Subset class match (tag có subset của classes)
    4. Tag-only match
    5. Default fallback
    """
    tag, classes, data_attrs = sig

    # 1. Exact match
    if sig in mapping:
        return mapping[sig]

    # 2. Wildcard tag match
    wildcard_key = ("*", classes, data_attrs)
    if wildcard_key in mapping:
        return mapping[wildcard_key]

    # 3. Subset class matching - check if any mapped signature's classes are subset of current
    for mapped_sig, label in mapping.items():
        mapped_tag, mapped_classes, mapped_data_attrs = mapped_sig

        # Skip if different tag (unless wildcard)
        if mapped_tag != "*" and mapped_tag != tag:
            continue

        # Check if mapped classes are subset of current classes
        if mapped_classes and set(mapped_classes).issubset(set(classes)):
            return label

    # 4. Tag-only match (no classes/data_attrs)
    tag_only_key = (tag, tuple(), tuple())
    if tag_only_key in mapping:
        return mapping[tag_only_key]

    return default


def build_logical_blocks(html_path: Path, signature_mapping: Dict[Signature, str]) -> Tuple[List[Dict], Counter]:
    """
    Build logical content blocks từ HTML với signature-based labeling

    Strategy:
    - Parse HTML với BeautifulSoup
    - Apply flexible signature mapping để get labels
    - Group adjacent elements với same label
    - Create boundary khi label changes hoặc encounter headings
    - Track unknown patterns for debugging

    Returns:
        Tuple[blocks, unknown_patterns_counter]
    """
    logger.info(f"🔧 Building logical blocks from: {html_path}")

    blocks: List[Dict] = []
    current_block: Optional[Dict] = None
    elements_processed = 0
    unknown_patterns: Counter[Signature] = Counter()

    try:
        # Load và parse HTML file
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        all_elements = soup.find_all()
        logger.info(f"📊 Processing {len(all_elements)} elements for logical blocks")
        toc_map = {}  # {"section_id": {"title": str, "level": int, "order": "1.2"}}

        for el in all_elements:
            if should_skip(el):
                continue
            if el.name == "paramref":
                param = el.get("name", "").strip()
                if param and current_block is not None:
                    current_block["text"] += f" {param}"
                continue
            # Giữ tên chỉ báo trong <see cref="..."/>
            if el.name == "see":
                cref = el.get("cref", "").split(":")[-1]      # lấy phần sau dấu :
                if cref and current_block is not None:
                    current_block["text"] += f" {cref}"
                continue
            if el.name == "sup" and el.find("a"):
                text = el.get_text(strip=True)
                if current_block is not None and text:
                    current_block["text"] += f" [{text}]"
                continue
            if el.name == "remark":
                text = el.get_text(strip=True)
                if current_block is not None and text:
                    current_block["text"] += f" [{text}]"
                continue
            if el.name == "nav":
                # Lấy mọi anchor trong ToC
                for a in el.find_all("a"):
                    # Tìm class "toc-h1", "toc-h2", ...
                    m = re.search(r"toc-h(\d)", " ".join(a.get("class", [])))
                    if not m:
                        continue
                    level = int(m.group(1))  # 1, 2, 3, ...
                    order, title = a.get_text(strip=True).split(" ", 1)
                    sec_id = a["href"].lstrip("#")  # "#1" → "1"
                    toc_map[sec_id] = {"title": title,
                                       "level": level,
                                       "order": order}
                continue  # 💡 KHÔNG cho nav/anchors thành chunk

            # ── giữ inline <c> (hằng số/code) ─────────────────
            if el.name == "c":
                token = el.get_text(strip=True)
                if token and current_block is not None:
                    current_block["text"] += f" {token}"
                continue


            elements_processed += 1

            # Get signature và label với flexible matching
            sig = signature_of(el)
            label = flexible_label_lookup(sig, signature_mapping)

            # Track unknown patterns
            if label == "generic_text":
                unknown_patterns[sig] += 1

            # Skip unwanted content
            if label == "skip_content":
                continue

            # Extract text content
            text = el.get_text(strip=True) if el.get_text else ""
            if not text:
                continue

            # Determine if we need new block
            should_create_new_block = (
                    current_block is None or
                    label.startswith("heading") or  # Always start new block for headings
                    label != current_block["content_type"] or  # Different content type
                    len(current_block["text"]) > 1000  # Prevent overly long blocks
            )

            if should_create_new_block:
                # Finalize current block
                if current_block is not None:
                    blocks.append(current_block)

                # Start new block
                current_block = {
                    "block_id": f"block_{len(blocks) + 1:04d}",
                    "content_type": label,
                    "text": text,
                    "metadata": {
                        "element_count": 1,
                        "signature": f"{sig[0]}.{'.'.join(sig[1])}",
                        "start_element": elements_processed
                    }
                }

                # Special handling cho headings
                if label.startswith("heading"):
                    current_block["metadata"]["heading_level"] = int(label[-1]) if label[-1].isdigit() else 0
                    current_block["metadata"]["heading_text"] = text
                    # ─── PATCH: gắn info từ ToC ───
                    sec_id = el.get("id")
                    if sec_id and sec_id in toc_map:
                        current_block["metadata"].update(toc_map[sec_id])
                    # ──────────────────────────────
            else:
                # Append to current block
                current_block["text"] += " " + text
                current_block["metadata"]["element_count"] += 1

            # Progress logging
            if elements_processed % 2000 == 0:
                logger.info(
                    f"   Progress: {elements_processed:,}/{len(all_elements)} elements, {len(blocks)} blocks created")

    except Exception as e:
        logger.error(f"❌ Error building blocks: {e}")
        raise

    # Finalize last block
    if current_block is not None:
        blocks.append(current_block)

    logger.info(f"✅ Built {len(blocks)} logical blocks from {elements_processed:,} elements")

    if unknown_patterns:
        logger.warning(f"⚠️ Found {len(unknown_patterns)} unknown signature patterns")

    return blocks, unknown_patterns


def cmd_parse(args: argparse.Namespace) -> None:
    """
    Parse command: convert single HTML file thành structured JSON blocks
    """
    html_path = Path(args.html_path)
    discovered_json_path = Path(args.discovered_json)
    yaml_path = Path(args.yaml)
    out_file = Path(args.out_file)

    # Validation
    if not html_path.exists():
        logger.error(f"❌ HTML file not found: {html_path}")
        return

    if not discovered_json_path.exists():
        logger.error(f"❌ Discovered JSON not found: {discovered_json_path}")
        logger.info("💡 Run 'discover' command first to generate signature database")
        return

    # Setup output
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load dependencies
    yaml_rules = load_yaml_rules(yaml_path)
    signature_mapping = load_signature_mapping(discovered_json_path)

    # Ensure essential patterns are available (fallback)
    for tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        sig = (tag, tuple(), tuple())
        signature_mapping.setdefault(sig, f"heading_{tag[1]}")

    # Build logical blocks với unknown pattern tracking
    blocks, unknown_patterns = build_logical_blocks(html_path, signature_mapping)

    # Export unknown patterns if significant
    unknown_threshold = 5  # Export if frequency >= 5
    significant_unknowns = {sig: count for sig, count in unknown_patterns.items() if count >= unknown_threshold}

    if significant_unknowns:
        unknown_csv_path = out_file.with_suffix('.unknown.csv')

        with unknown_csv_path.open('w', newline='', encoding='utf-8') as f:
            fieldnames = ['tag', 'classes', 'data_attrs', 'frequency', 'sample_signature']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for sig, freq in Counter(significant_unknowns).most_common():
                tag, classes, data_attrs = sig
                writer.writerow({
                    'tag': tag,
                    'classes': ' '.join(classes),
                    'data_attrs': ','.join(data_attrs),
                    'frequency': freq,
                    'sample_signature': f"{tag}.{'.'.join(classes)}"
                })

        logger.warning(f"📋 Unknown patterns exported: {unknown_csv_path}")
        logger.warning(f"   Consider adding these patterns to {yaml_path}")

    # Analyze results
    content_type_stats = Counter(block["content_type"] for block in blocks)
    total_text_length = sum(len(block["text"]) for block in blocks)

    # Calculate generic_text ratio
    total_unknown_elements = sum(unknown_patterns.values())
    total_elements = sum(block["metadata"]["element_count"] for block in blocks) + total_unknown_elements
    unknown_ratio = (total_unknown_elements / total_elements * 100) if total_elements > 0 else 0

    # Prepare output data
    output_data = {
        "document_metadata": {
            "source_file": html_path.name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_blocks": len(blocks),
            "total_text_length": total_text_length,
            "content_type_distribution": dict(content_type_stats),
            "extraction_method": "signature_based_logical_blocks",
            "unknown_patterns_count": len(unknown_patterns),
            "unknown_elements_ratio": round(unknown_ratio, 2)
        },
        "content_blocks": blocks
    }

    # Save structured JSON
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Summary report
    logger.info("=" * 60)
    logger.info("🎯 PARSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"📄 Source: {html_path}")
    logger.info(f"📊 Blocks created: {len(blocks)}")
    logger.info(f"📝 Total text: {total_text_length:,} characters")
    logger.info(f"💾 Output: {out_file}")

    logger.info("\n📋 Content Type Distribution:")
    for content_type, count in content_type_stats.most_common():
        percentage = (count / len(blocks)) * 100
        logger.info(f"   {content_type}: {count} blocks ({percentage:.1f}%)")

    # Quality warnings
    if unknown_ratio > 0:
        logger.warning(f"⚠️ Unknown elements: {unknown_ratio:.1f}% of total elements")
        if unknown_ratio > 20:
            logger.warning("   HIGH: Consider updating pattern_rules.yaml with missing patterns")
        elif unknown_ratio > 5:
            logger.warning("   MEDIUM: Some patterns may be missing from rules")
    else:
        logger.info("✅ All elements successfully classified!")

    generic_blocks = content_type_stats.get("generic_text", 0)
    if generic_blocks > 0:
        percentage = (generic_blocks / len(blocks)) * 100
        logger.info(f"📊 Generic blocks: {generic_blocks} ({percentage:.1f}%)")


def cmd_parse_dir(args: argparse.Namespace) -> None:
    """
    Parse multiple HTML files in directory to structured JSON blocks
    """
    html_dir = Path(args.html_dir)
    discovered_json_path = Path(args.discovered_json)
    yaml_path = Path(args.yaml)
    out_dir = Path(args.out_dir)

    # Validation
    if not html_dir.exists():
        logger.error(f"❌ HTML directory not found: {html_dir}")
        return

    if not discovered_json_path.exists():
        logger.error(f"❌ Discovered JSON not found: {discovered_json_path}")
        logger.info("💡 Run 'discover' command first to generate signature database")
        return

    # Setup output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find HTML files
    html_files = list(html_dir.glob(args.glob_pattern))
    if not html_files:
        logger.error(f"❌ No HTML files found matching pattern: {args.glob_pattern}")
        return

    logger.info(f"🔍 Found {len(html_files)} HTML files to parse")

    # Process each file
    successful_files = 0
    total_blocks = 0
    total_unknown_patterns = Counter()

    for html_file in html_files:
        try:
            # Generate output filename
            out_file = out_dir / f"{html_file.stem}_structured.json"

            # Create temporary args for individual file parsing
            file_args = argparse.Namespace(
                html_path=str(html_file),
                discovered_json=str(discovered_json_path),
                yaml=str(yaml_path),
                out_file=str(out_file)
            )

            logger.info(f"📄 Processing: {html_file.name}")

            # Reuse cmd_parse logic but capture results
            yaml_rules = load_yaml_rules(yaml_path)
            signature_mapping = load_signature_mapping(discovered_json_path)

            # Add essential patterns
            for tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                sig = (tag, tuple(), tuple())
                signature_mapping.setdefault(sig, f"heading_{tag[1]}")

            # Build blocks
            blocks, unknown_patterns = build_logical_blocks(html_file, signature_mapping)

            # Save individual file
            output_data = {
                "document_metadata": {
                    "source_file": html_file.name,
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "total_blocks": len(blocks),
                    "extraction_method": "signature_based_logical_blocks_batch"
                },
                "content_blocks": blocks
            }

            with out_file.open("w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            # Aggregate stats
            successful_files += 1
            total_blocks += len(blocks)
            total_unknown_patterns.update(unknown_patterns)

            logger.info(f"✅ {html_file.name} → {out_file.name} ({len(blocks)} blocks)")

        except Exception as e:
            logger.error(f"❌ Failed to process {html_file.name}: {e}")
            continue

    # Export aggregated unknown patterns
    if total_unknown_patterns:
        unknown_csv_path = out_dir / "batch_unknown_patterns.csv"

        # Filter significant patterns
        significant_unknowns = {sig: count for sig, count in total_unknown_patterns.items() if count >= 10}

        if significant_unknowns:
            with unknown_csv_path.open('w', newline='', encoding='utf-8') as f:
                fieldnames = ['tag', 'classes', 'data_attrs', 'frequency', 'sample_signature']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for sig, freq in Counter(significant_unknowns).most_common():
                    tag, classes, data_attrs = sig
                    writer.writerow({
                        'tag': tag,
                        'classes': ' '.join(classes),
                        'data_attrs': ','.join(data_attrs),
                        'frequency': freq,
                        'sample_signature': f"{tag}.{'.'.join(classes)}"
                    })

            logger.warning(f"📋 Batch unknown patterns: {unknown_csv_path}")

    # Final summary
    logger.info("=" * 60)
    logger.info("🎯 BATCH PARSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"📁 Source directory: {html_dir}")
    logger.info(f"📄 Files processed: {successful_files}/{len(html_files)}")
    logger.info(f"📊 Total blocks created: {total_blocks:,}")
    logger.info(f"💾 Output directory: {out_dir}")

    if total_unknown_patterns:
        total_unknown_count = sum(total_unknown_patterns.values())
        logger.warning(
            f"⚠️ Total unknown patterns: {len(total_unknown_patterns)} types, {total_unknown_count:,} instances")
    else:
        logger.info("✅ All patterns successfully classified!")


###############################################################################
# CLI entry point
###############################################################################

def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Phase 0: Pattern Discovery & Structured Parser for QuantConnect",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover patterns từ folder HTML files  
  python discover_patterns.py discover --html_dir data/processed_html --yaml pattern_rules.yaml

  # Parse single file thành structured blocks
  python discover_patterns.py parse --html_path data/processed_html/file.html --yaml pattern_rules.yaml \\
      --discovered_json discovered_global.json --out_file structured.json

  # Parse multiple files in directory
  python discover_patterns.py parse_dir --html_dir data/processed_html --yaml pattern_rules.yaml \\
      --discovered_json discovered_global.json --out_dir data/structured_analysis
        """
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Discover subcommand
    discover_parser = subparsers.add_parser(
        "discover",
        help="Discover patterns across HTML directory"
    )
    discover_parser.add_argument(
        "--html_dir",
        required=True,
        help="Directory containing processed HTML files (e.g., data/processed_html)"
    )
    discover_parser.add_argument(
        "--yaml",
        required=True,
        help="Path to pattern_rules.yaml file"
    )
    discover_parser.add_argument(
        "--out_dir",
        default="data/pattern_analysis",
        help="Output directory (default: data/pattern_analysis)"
    )
    discover_parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Generate Top-K patterns CSV for review (0 to skip, default: 50)"
    )
    discover_parser.set_defaults(func=cmd_discover)

    # Parse subcommand
    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse single HTML into structured JSON blocks"
    )
    parse_parser.add_argument(
        "--html_path",
        required=True,
        help="Path to processed HTML file to parse (e.g., data/processed_html/file.html)"
    )
    parse_parser.add_argument(
        "--yaml",
        required=True,
        help="Path to pattern_rules.yaml file"
    )
    parse_parser.add_argument(
        "--discovered_json",
        required=True,
        help="Path to discovered_global.json from discover command"
    )
    parse_parser.add_argument(
        "--out_file",
        required=True,
        help="Output structured JSON file path"
    )
    parse_parser.set_defaults(func=cmd_parse)

    # Parse directory subcommand (NEW)
    parse_dir_parser = subparsers.add_parser(
        "parse_dir",
        help="Parse multiple HTML files in directory to structured JSON blocks"
    )
    parse_dir_parser.add_argument(
        "--html_dir",
        required=True,
        help="Directory containing processed HTML files to parse"
    )
    parse_dir_parser.add_argument(
        "--yaml",
        required=True,
        help="Path to pattern_rules.yaml file"
    )
    parse_dir_parser.add_argument(
        "--discovered_json",
        required=True,
        help="Path to discovered_global.json from discover command"
    )
    parse_dir_parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for structured JSON files"
    )
    parse_dir_parser.add_argument(
        "--glob_pattern",
        default="*.html",
        help="File glob pattern (default: *.html)"
    )
    parse_dir_parser.set_defaults(func=cmd_parse_dir)

    # Parse arguments và execute
    args = parser.parse_args(argv)

    try:
        args.func(args)
    except Exception as e:
        logger.error(f"❌ Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()