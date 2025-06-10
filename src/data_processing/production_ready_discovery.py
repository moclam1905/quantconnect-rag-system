#!/usr/bin/env python3
"""
Production-Ready Pattern Discovery System
Based on "Machines discover, Humans decide, Tools execute" philosophy
Implements reservoir sampling + human-friendly review + dynamic rules
"""

import csv
import json
import logging
import random
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import yaml
from lxml import etree
from html_preprocessor import HTMLPreprocessor

from enhanced_pattern_discovery import EnhancedPatternDiscovery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PatternCandidate:
    """Represents a pattern candidate with evidence scoring"""
    identifier: str
    selector_type: str  # 'class', 'tag', 'attr', 'id_pattern'
    selector_value: str
    frequency: int
    evidence_score: float
    characteristics: Dict
    sample_content: str
    depth_stats: Dict
    text_stats: Dict


class ReservoirSampler:
    """Efficient reservoir sampling for large HTML files"""

    def __init__(self, sample_size: int = 10000):
        self.sample_size = sample_size
        self.reservoir = []
        self.count = 0

    def add_element(self, element_data: Dict) -> bool:
        """Add element to reservoir, return True if added"""
        self.count += 1

        if len(self.reservoir) < self.sample_size:
            self.reservoir.append(element_data)
            return True
        else:
            # Replace with decreasing probability
            replace_idx = random.randint(0, self.count - 1)
            if replace_idx < self.sample_size:
                self.reservoir[replace_idx] = element_data
                return True
        return False


class EvidenceScorer:
    """Calculate evidence scores for pattern candidates"""

    def __init__(self):
        self.code_indicators = {
            'keywords': ['def ', 'class ', 'import ', 'using ', 'public ', 'private', '$ ', '#!/bin/bash', 'sudo '],
            'symbols': ['{', '}', ';', '()', '=>', '==', '!='],
            'patterns': [r'\w+\.\w+\(', r'^\s*#', r'^\s*//', r'@\w+', r'^\s*\$ ']
        }

        self.content_type_keywords = {
            'code': ['code', 'highlight', 'syntax', 'python', 'csharp', 'example', 'cli', 'bash', 'shell'],
            'api': ['api', 'reference', 'method', 'property', 'class', 'namespace'],
            'tutorial': ['tutorial', 'step', 'guide', 'walkthrough', 'lesson'],
            'navigation': ['nav', 'toc', 'breadcrumb', 'menu', 'sidebar'],
            'table': ['table', 'grid', 'data', 'responsive'],
            'interactive': ['widget', 'calculator', 'tool', 'interactive']
        }

    def calculate_evidence_score(self, candidate: Dict) -> float:
        """Calculate comprehensive evidence score"""
        score = 0.0

        # Estimated Frequency score (log scale to prevent dominance)
        freq_score = min(0.3, 0.1 * (candidate['frequency'] / 100))
        score += freq_score

        # Content type clustering score
        content_score = self._calculate_content_type_score(candidate)
        score += content_score * 0.4

        # Text analysis score
        text_score = self._calculate_text_analysis_score(candidate)
        score += text_score * 0.2

        # Structural score
        structural_score = self._calculate_structural_score(candidate)
        score += structural_score * 0.1

        return min(score, 1.0)

    def _calculate_content_type_score(self, candidate: Dict) -> float:
        """Score based on content type indicators"""
        selector_value = candidate['selector_value'].lower()
        max_score = 0.0

        for content_type, keywords in self.content_type_keywords.items():
            type_score = sum(0.2 for keyword in keywords if keyword in selector_value)
            max_score = max(max_score, min(type_score, 1.0))

        return max_score

    def _calculate_text_analysis_score(self, candidate: Dict) -> float:
        """Score based on text content analysis"""
        sample_text = candidate.get('sample_content', '').lower()
        if not sample_text:
            return 0.0

        # Code indicators
        code_score = 0.0
        for keyword in self.code_indicators['keywords']:
            if keyword in sample_text:
                code_score += 0.15

        for symbol in self.code_indicators['symbols']:
            if symbol in sample_text:
                code_score += 0.1

        for pattern in self.code_indicators['patterns']:
            if re.search(pattern, sample_text):
                code_score += 0.2

        return min(code_score, 1.0)

    def _calculate_structural_score(self, candidate: Dict) -> float:
        """Score based on structural characteristics"""
        chars = candidate.get('characteristics', {})

        # Balanced frequency (not too rare, not too common)
        freq = candidate['frequency']
        if 10 <= freq <= 1000:
            freq_balance = 0.5
        elif 5 <= freq <= 2000:
            freq_balance = 0.3
        else:
            freq_balance = 0.1

        # Text length consistency
        avg_length = chars.get('avg_text_length', 0)
        if 50 <= avg_length <= 500:  # Good content length
            length_score = 0.3
        elif avg_length > 20:
            length_score = 0.2
        else:
            length_score = 0.0

        return freq_balance + length_score


class ProductionPatternDiscovery:
    """Main production-ready pattern discovery system"""

    def __init__(self, sample_ratio: float = 0.1, max_memory_gb: float = 3.0):
        self.sample_ratio = sample_ratio
        self.max_memory_gb = max_memory_gb
        self.sampler = ReservoirSampler(sample_size=int(100000 * sample_ratio))
        self.scorer = EvidenceScorer()

    def run_discovery_pipeline(self, html_file_path: str, output_dir: str = "discovery_output"):
        """Complete discovery pipeline"""
        logger.info(f"Starting production discovery pipeline for {html_file_path}")

        # Step 0: Streaming Structural Snapshot with Reservoir Sampling
        logger.info("Step 0: Streaming structural snapshot...")
        snapshot_data = self._streaming_snapshot_with_sampling(html_file_path)

        # Step 1: Auto-scoring Pattern Candidates
        logger.info("Step 1: Auto-scoring pattern candidates...")
        pattern_candidates = self._auto_score_pattern_candidates(snapshot_data)

        # Step 2: Prepare Human Review Materials
        logger.info("Step 2: Preparing human review materials...")
        self._prepare_human_review_materials(pattern_candidates, output_dir)

        # Save intermediate results
        self._save_discovery_results(snapshot_data, pattern_candidates, output_dir)

        logger.info(f"Discovery complete. Review materials saved to {output_dir}/")
        logger.info("Next: Review patterns.csv and create reviewed_patterns.csv")

        return {
            'snapshot': snapshot_data,
            'candidates': pattern_candidates,
            'output_dir': output_dir
        }

    def _streaming_snapshot_with_sampling(self, html_file_path: str) -> Dict:
        """Streaming snapshot with reservoir sampling - FIXED for multiple HTML docs"""

        # NEW: Preprocess file to handle multiple HTML documents
        logger.info("Preprocessing multiple HTML documents...")
        try:
            preprocessor = HTMLPreprocessor()
            processed_file_path = preprocessor.preprocess_file(Path(html_file_path).name)
        except ImportError:
            print("❌ Cannot import HTMLPreprocessor - using file directly")
            processed_file_path = html_file_path

        try:
            # Basic counters
            tag_counter = Counter()
            class_counter = Counter()
            attr_counter = Counter()
            id_pattern_counter = Counter()

            # Advanced analysis
            depth_stats = defaultdict(list)
            text_length_stats = defaultdict(list)

            # FIXED: Parse from temporary processed file
            context = etree.iterparse(
                processed_file_path,  # Use processed file
                html=True,
                events=("end",),
                encoding="utf-8"
            )

            nodes_processed = 0
            nodes_sampled = 0

            for event, elem in context:
                nodes_processed += 1

                # Debug logging (remove this line after testing)
                if nodes_processed <= 50 or nodes_processed % 1000 == 0:
                    logger.info(f"Processing node {nodes_processed}: {elem.tag}")

                # Reservoir sampling decision
                if random.random() < self.sample_ratio:
                    nodes_sampled += 1

                    # Basic stats
                    tag_counter[elem.tag] += 1
                    depth = len(list(elem.iterancestors()))

                    # Class analysis
                    if (classes := elem.get("class")):
                        for cls in classes.split():
                            class_counter[cls] += 1
                            depth_stats[cls].append(depth)

                    # Attribute analysis
                    for attr_name in elem.attrib:
                        attr_counter[attr_name] += 1

                    # ID pattern analysis
                    if (elem_id := elem.get("id")):
                        # Generalize ID pattern
                        id_pattern = re.sub(r'\d+', 'N', elem_id)
                        id_pattern = re.sub(r'[a-f0-9]{8,}', 'HASH', id_pattern)
                        id_pattern_counter[id_pattern] += 1

                    # Text analysis for sampled elements
                    elem_text = elem.text_content() if hasattr(elem, 'text_content') else (elem.text or "")
                    text_length = len(elem_text.strip())

                    if classes:
                        for cls in classes.split():
                            text_length_stats[cls].append(text_length)

                    # Collect detailed data for reservoir
                    element_data = {
                        'tag': elem.tag,
                        'classes': classes.split() if classes else [],
                        'attributes': list(elem.attrib.keys()),
                        'text_length': text_length,
                        'text_sample': elem_text[:200].strip(),
                        'depth': depth,
                        'has_children': len(elem) > 0
                    }

                    self.sampler.add_element(element_data)

                # Memory management
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

                # Progress logging
                if nodes_processed % 5000 == 0:
                    logger.info(f"Processed {nodes_processed} nodes, sampled {nodes_sampled}")

            logger.info(f"FINAL: Processed {nodes_processed} nodes, sampled {nodes_sampled}")

        finally:
            logger.info(f"FINAL: Processed")

        # Calculate statistics from sampled data
        return {
            'processing_stats': {
                'total_nodes_processed': nodes_processed,
                'nodes_sampled': nodes_sampled,
                'sampling_ratio': nodes_sampled / nodes_processed if nodes_processed > 0 else 0,
                'reservoir_size': len(self.sampler.reservoir)
            },
            'element_stats': {
                'top_tags': tag_counter.most_common(30),
                'top_classes': class_counter.most_common(100),
                'top_attributes': attr_counter.most_common(50),
                'id_patterns': id_pattern_counter.most_common(30)
            },
            'depth_analysis': {cls: {
                'avg_depth': sum(depths) / len(depths),
                'min_depth': min(depths),
                'max_depth': max(depths)
            } for cls, depths in depth_stats.items() if len(depths) >= 3},
            'text_analysis': {cls: {
                'avg_length': sum(lengths) / len(lengths),
                'max_length': max(lengths),
                'non_empty_ratio': sum(1 for l in lengths if l > 10) / len(lengths)
            } for cls, lengths in text_length_stats.items() if len(lengths) >= 3},
            'reservoir_sample': self.sampler.reservoir
        }

    def _auto_score_pattern_candidates(self, snapshot_data: Dict) -> List[PatternCandidate]:
        """Auto-score pattern candidates using evidence-based approach"""
        candidates = []

        # Process class patterns
        for class_name, frequency in snapshot_data['element_stats']['top_classes']:
            if frequency >= 3:  # Minimum threshold
                candidate_data = {
                    'selector_value': class_name,
                    'frequency': frequency,
                    'sample_content': self._get_sample_content_for_class(class_name, snapshot_data),
                    'characteristics': self._get_class_characteristics(class_name, snapshot_data)
                }

                evidence_score = self.scorer.calculate_evidence_score(candidate_data)

                candidate = PatternCandidate(
                    identifier=f"class_{class_name}",
                    selector_type="class",
                    selector_value=class_name,
                    frequency=frequency,
                    evidence_score=evidence_score,
                    characteristics=candidate_data['characteristics'],
                    sample_content=candidate_data['sample_content'],
                    depth_stats=snapshot_data['depth_analysis'].get(class_name, {}),
                    text_stats=snapshot_data['text_analysis'].get(class_name, {})
                )
                candidates.append(candidate)

        # Process attribute patterns
        for attr_name, frequency in snapshot_data['element_stats']['top_attributes']:
            if frequency >= 5 and attr_name.startswith('data-'):
                candidate_data = {
                    'selector_value': attr_name,
                    'frequency': frequency,
                    'sample_content': "",  # Attributes don't have content
                    'characteristics': {'type': 'data_attribute'}
                }

                evidence_score = self.scorer.calculate_evidence_score(candidate_data)

                candidate = PatternCandidate(
                    identifier=f"attr_{attr_name}",
                    selector_type="attr",
                    selector_value=attr_name,
                    frequency=frequency,
                    evidence_score=evidence_score,
                    characteristics=candidate_data['characteristics'],
                    sample_content="",
                    depth_stats={},
                    text_stats={}
                )
                candidates.append(candidate)

        # Sort by evidence score
        candidates.sort(key=lambda c: c.evidence_score, reverse=True)

        return candidates

    def _get_sample_content_for_class(self, class_name: str, snapshot_data: Dict) -> str:
        """Get sample content for a specific class"""
        for element in snapshot_data['reservoir_sample']:
            if class_name in element['classes']:
                return element['text_sample']
        return ""

    def _get_class_characteristics(self, class_name: str, snapshot_data: Dict) -> Dict:
        """Get characteristics for a specific class"""
        characteristics = {
            'avg_text_length': 0,
            'common_tags': [],
            'has_children_ratio': 0
        }

        matching_elements = [
            elem for elem in snapshot_data['reservoir_sample']
            if class_name in elem['classes']
        ]

        if matching_elements:
            characteristics['avg_text_length'] = sum(
                elem['text_length'] for elem in matching_elements
            ) / len(matching_elements)

            tag_counter = Counter(elem['tag'] for elem in matching_elements)
            characteristics['common_tags'] = tag_counter.most_common(3)

            characteristics['has_children_ratio'] = sum(
                elem['has_children'] for elem in matching_elements
            ) / len(matching_elements)

        return characteristics

    def _prepare_human_review_materials(self, candidates: List[PatternCandidate], output_dir: str):
        """MODIFIED: Use SmartCSVManager với incremental logic"""
        Path(output_dir).mkdir(exist_ok=True)

        # NEW: Use SmartCSVManager instead of direct CSV creation
        from smart_csv_manager import SmartCSVManager

        csv_manager = SmartCSVManager(output_dir)

        # Take top candidates only (same as before)
        top_candidates = candidates[:200]  # Review top 200

        # Create incremental review CSV
        review_csv_path = csv_manager.create_incremental_review_csv(top_candidates)

        # Keep existing review instructions generation
        instructions_file = Path(output_dir) / "review_instructions.md"
        with open(instructions_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_review_instructions())

        # Get và log statistics
        stats = csv_manager.get_statistics()

        logger.info(f"📋 Smart CSV management complete:")
        logger.info(f"   Review file: {review_csv_path}")
        logger.info(f"   Instructions: {instructions_file}")
        logger.info(f"   Total existing decisions: {stats.get('total_decisions', 0)}")
        logger.info(f"   Latest version: v{stats.get('latest_version', 0)}")
        logger.info(f"   🎯 Focus on rows with empty 'Human Decision' or ⚠️ markers")

    def _suggest_content_type(self, candidate: PatternCandidate) -> str:
        """Suggest content type based on evidence"""
        selector_lower = candidate.selector_value.lower()
        sample_lower = candidate.sample_content.lower()

        # Strong indicators
        if any(ind in selector_lower for ind in ['code', 'highlight', 'python', 'csharp', 'cli', 'bash']):
            return 'code_content'

        if 'data-tree' in selector_lower or any(ind in selector_lower for ind in ['api', 'reference']):
            return 'api_reference'

        if any(ind in selector_lower for ind in ['table', 'grid']):
            return 'table_content'

        if any(ind in selector_lower for ind in ['nav', 'toc', 'breadcrumb']):
            return 'navigation_content'

        # Content-based suggestions
        if any(ind in sample_lower for ind in ['def ', 'class ', 'import']):
            return 'code_content'

        if sample_lower.startswith('$ ') or '#!/bin/bash' in sample_lower:
            return 'code_content'

        if candidate.evidence_score > 0.7:
            return 'documentation_text'

        return 'unknown'

    def _generate_review_instructions(self) -> str:
        """Generate human review instructions"""
        return """# Pattern Review Instructions

## Overview
Please review the patterns_for_review.csv file and fill in the "Human Decision" column.

## Content Types to Choose From:
- **code_content**: Code examples, syntax highlighting, programming snippets
- **api_reference**: API documentation, method signatures, class references  
- **tutorial_content**: Step-by-step guides, tutorials, learning materials
- **table_content**: Data tables, comparison tables, structured data
- **navigation_content**: Navigation menus, breadcrumbs, table of contents
- **documentation_text**: General documentation text, explanations
- **skip**: Patterns that should be ignored (ads, social media, etc.)

## Review Process:
1. Open patterns_for_review.csv in Excel/Google Sheets
2. Look at each pattern's:
   - Selector Value (the HTML class/attribute name)
   - Evidence Score (higher = more likely to be important)
   - Sample Content (what content looks like)
   - Suggested Type (algorithm's best guess)
3. Fill in "Human Decision" column with your choice
4. Add notes in "Notes" column if needed
5. Focus on top ~50 patterns first (these cover 80%+ of content)

## Tips:
- Trust high evidence scores (>0.7) - these are usually important
- Look for patterns in naming (e.g., "section-example-*" = code)
- When in doubt, choose "documentation_text" rather than "skip"
- Pay special attention to "data-*" attributes - often API-related

## After Review:
Save the file as "reviewed_patterns.csv" in the same directory.
"""

    def _save_discovery_results(self, snapshot_data: Dict, candidates: List[PatternCandidate], output_dir: str):
        """Save discovery results for reference"""
        Path(output_dir).mkdir(exist_ok=True)

        # Save snapshot data
        with open(Path(output_dir) / "snapshot_data.json", 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, indent=2, ensure_ascii=False)

        # Save candidates data
        candidates_data = [asdict(candidate) for candidate in candidates]
        with open(Path(output_dir) / "pattern_candidates.json", 'w', encoding='utf-8') as f:
            json.dump(candidates_data, f, indent=2, ensure_ascii=False)

    # ===============================
    # REPLACE generate_rule_file_from_human_review method trong ProductionPatternDiscovery class:
    # ===============================

    def generate_rule_file_from_human_review(self, reviewed_csv_path: str, output_path: str = "pattern_rules.yaml"):
        """Generate YAML rule file from human review - FIXED với skip support"""
        logger.info(f"Generating rule file from {reviewed_csv_path}")

        rules = defaultdict(list)
        skip_patterns = []  # ← NEW: Track skip patterns

        with open(reviewed_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                human_decision = row['Human Decision'].strip()
                selector_type = row['Selector Type']
                selector_value = row['Selector Value']

                # *** NEW LOGIC: Handle empty decisions as SKIP ***
                if not human_decision:
                    # Empty decision = SKIP
                    if selector_type == 'class':
                        skip_patterns.append({'class': selector_value})
                    elif selector_type == 'attr':
                        skip_patterns.append({'attr': selector_value})
                    elif selector_type == 'tag':
                        skip_patterns.append({'tag': selector_value})
                    continue

                # Handle explicit 'skip' decisions
                if human_decision.lower() == 'skip':
                    if selector_type == 'class':
                        skip_patterns.append({'class': selector_value})
                    elif selector_type == 'attr':
                        skip_patterns.append({'attr': selector_value})
                    elif selector_type == 'tag':
                        skip_patterns.append({'tag': selector_value})
                    continue

                # Handle positive decisions (existing logic)
                if selector_type == 'class':
                    rules[human_decision].append({'class': selector_value})
                elif selector_type == 'attr':
                    rules[human_decision].append({'attr': selector_value})
                elif selector_type == 'tag':
                    rules[human_decision].append({'tag': selector_value})

        # *** BUILD COMPREHENSIVE RULE DICT ***
        rule_dict = {}

        # 1. Add SKIP section first (highest priority)
        if skip_patterns:
            rule_dict['skip_content'] = skip_patterns

            # *** ADD COMPREHENSIVE MEDIA SKIP RULES ***
            rule_dict['skip_content'].extend([
                # Media tags
                {'tag': 'img'},  # Images
                {'tag': 'video'},  # Videos
                {'tag': 'audio'},  # Audio
                {'tag': 'iframe'},  # Embedded content
                {'tag': 'embed'},  # Flash/plugins
                {'tag': 'object'},  # Objects
                {'tag': 'canvas'},  # Canvas elements

                # Script/Style tags
                {'tag': 'script'},  # JavaScript
                {'tag': 'style'},  # CSS
                {'tag': 'link'},  # External resources
                {'tag': 'meta'},  # Metadata

                # Media-related classes (common patterns)
                {'class': 'image'},
                {'class': 'img'},
                {'class': 'video'},
                {'class': 'audio'},
                {'class': 'media'},
                {'class': 'gif'},
                {'class': 'animation'},
                {'class': 'player'},
                {'class': 'embed'},

                # QuantConnect specific media patterns
                {'class': 'cover-icon'},
                {'class': 'cover-image'},
                {'class': 'chart'},
                {'class': 'graph'},
                {'class': 'diagram'}
            ])

        # 2. Add content rules
        for content_type, patterns in rules.items():
            rule_dict[content_type] = patterns

        # *** SAVE WITH ORGANIZED STRUCTURE ***
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(rule_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        # *** ENHANCED LOGGING ***
        logger.info(f"Rule file generated: {output_path}")
        logger.info(f"📊 Rule summary:")
        logger.info(f"  🚫 Skip patterns: {len(rule_dict.get('skip_content', []))}")
        for content_type, patterns in rule_dict.items():
            if content_type != 'skip_content':
                logger.info(f"  ✅ {content_type}: {len(patterns)} patterns")

        return rule_dict

def load_pattern_rules(rule_file_path: str = "pattern_rules.yaml") -> Dict:
    """Load pattern rules from YAML file"""
    try:
        with open(rule_file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Rule file {rule_file_path} not found. Using default rules.")
        return {}


# ===============================
# UPDATED classify_content_type function (same as before but with media rules)
# ===============================

def classify_content_type(elem, rules: dict = None) -> str:
    """
    Dynamic content type classifier với comprehensive skip support
    Returns 'SKIP' for all media và unwanted content
    """
    if rules is None:
        rules = load_pattern_rules()

    if not rules:
        return 'documentation_text'

    classes = elem.get('class', '').split()
    attrs = elem.attrib.keys()
    tag = elem.tag.lower()

    # === PRIORITY 1: Check SKIP patterns first ===
    if 'skip_content' in rules:
        for rule in rules['skip_content']:
            # Class-based skip
            if 'class' in rule and rule['class'] in classes:
                return 'SKIP'

            # Tag-based skip (img, video, script, etc.)
            if 'tag' in rule and rule['tag'] == tag:
                return 'SKIP'

            # Attribute-based skip
            if 'attr' in rule and rule['attr'] in attrs:
                return 'SKIP'

    # === PRIORITY 2: Content classification ===
    for content_type, type_rules in rules.items():
        if content_type == 'skip_content':
            continue

        for rule in type_rules:
            if 'class' in rule and rule['class'] in classes:
                return content_type
            if 'attr' in rule and rule['attr'] in attrs:
                return content_type
            if 'tag' in rule and rule['tag'] == tag:
                return content_type

    # === PRIORITY 3: Default ===
    return 'documentation_text'


def run_enhanced_discovery_pipeline(html_file_path: str, output_dir: str = "discovery_output"):
    """Enhanced version với coverage guarantee"""

    discovery = EnhancedPatternDiscovery(target_coverage=0.95)
    return discovery.run_adaptive_discovery(html_file_path, output_dir)

def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Production Pattern Discovery')
    parser.add_argument('--input', required=True, help='HTML file path')
    parser.add_argument('--output', default='discovery_output', help='Output directory')
    parser.add_argument('--sample-ratio', type=float, default=0.1, help='Sampling ratio')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced discovery với coverage >95%')
    parser.add_argument('--generate-rules', help='Generate rules from reviewed CSV')

    args = parser.parse_args()

    if args.generate_rules:
        discovery = ProductionPatternDiscovery()
        discovery.generate_rule_file_from_human_review(args.generate_rules)
    elif args.enhanced:
        # NEW: Enhanced discovery
        results = run_enhanced_discovery_pipeline(args.input, args.output)
        print(f"✅ Enhanced discovery complete - Coverage: {results['coverage_metrics'].overall_coverage:.1%}")
    else:
        # Original discovery
        discovery = ProductionPatternDiscovery()
        results = discovery.run_discovery_pipeline(args.input, args.output)
        print("=== ORIGINAL DISCOVERY COMPLETE ===")
        print(f"Processed {results['snapshot']['processing_stats']['total_nodes_processed']} nodes")
        print(f"Sampled {results['snapshot']['processing_stats']['nodes_sampled']} nodes")
        print(f"Found {len(results['candidates'])} pattern candidates")
        print(f"Review materials saved to {args.output}/")
        print("\nNext steps:")
        print(f"1. Open {args.output}/patterns_for_review.csv")
        print("2. Fill in 'Human Decision' column")
        print("3. Save as 'reviewed_patterns.csv'")
        print("4. Run: python script.py --generate-rules reviewed_patterns.csv")


if __name__ == "__main__":
    main()