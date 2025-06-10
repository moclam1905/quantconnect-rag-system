#!/usr/bin/env python3
"""
Enhanced Pattern Discovery với Coverage >95% Support
Extends ProductionPatternDiscovery với hybrid scanning và adaptive coverage
"""

import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from lxml import etree

logger = logging.getLogger(__name__)


@dataclass
class CoverageMetrics:
    """Coverage analysis results"""
    overall_coverage: float
    high_value_coverage: float
    medium_value_coverage: float
    low_value_coverage: float
    missing_patterns: List[str]
    total_patterns_found: int
    total_patterns_baseline: int
    coverage_by_content_type: Dict[str, float]


class CoverageAnalyzer:
    """Analyze pattern coverage và identify gaps"""

    def __init__(self, target_coverage: float = 0.95):
        self.target_coverage = target_coverage

        # High-value pattern indicators
        self.high_value_indicators = {
            'content_keywords': ['code', 'api', 'reference', 'example', 'table', 'data'],
            'structural_keywords': ['section', 'container', 'content', 'main'],
            'semantic_keywords': ['python', 'csharp', 'algorithm', 'tutorial']
        }

    def measure_coverage(self, discovered_patterns: Dict, baseline_patterns: Dict) -> CoverageMetrics:
        """
        Calculate comprehensive coverage metrics

        Args:
            discovered_patterns: {pattern_id: frequency} từ discovery
            baseline_patterns: {pattern_id: frequency} từ full scan

        Returns:
            CoverageMetrics object với detailed analysis
        """
        logger.info("🔍 Calculating coverage metrics...")

        # Classify patterns into value tiers
        high_value, medium_value, low_value = self._classify_patterns_by_value(baseline_patterns)

        # Calculate frequency-weighted coverage
        overall_coverage = self._calculate_frequency_weighted_coverage(
            discovered_patterns, baseline_patterns
        )

        # Calculate coverage by value tier
        high_value_coverage = self._calculate_tier_coverage(
            discovered_patterns, {p: baseline_patterns[p] for p in high_value}
        )

        medium_value_coverage = self._calculate_tier_coverage(
            discovered_patterns, {p: baseline_patterns[p] for p in medium_value}
        )

        low_value_coverage = self._calculate_tier_coverage(
            discovered_patterns, {p: baseline_patterns[p] for p in low_value}
        )

        # Identify missing patterns
        missing_patterns = self._identify_missing_patterns(discovered_patterns, baseline_patterns)

        # Coverage by content type
        coverage_by_type = self._analyze_coverage_by_content_type(
            discovered_patterns, baseline_patterns
        )

        metrics = CoverageMetrics(
            overall_coverage=overall_coverage,
            high_value_coverage=high_value_coverage,
            medium_value_coverage=medium_value_coverage,
            low_value_coverage=low_value_coverage,
            missing_patterns=missing_patterns,
            total_patterns_found=len(discovered_patterns),
            total_patterns_baseline=len(baseline_patterns),
            coverage_by_content_type=coverage_by_type
        )

        logger.info(f"📊 Coverage Analysis Results:")
        logger.info(f"   Overall: {overall_coverage:.1%}")
        logger.info(f"   High-value: {high_value_coverage:.1%}")
        logger.info(f"   Medium-value: {medium_value_coverage:.1%}")
        logger.info(f"   Low-value: {low_value_coverage:.1%}")
        logger.info(f"   Missing patterns: {len(missing_patterns)}")

        return metrics

    def identify_high_value_patterns(self, all_patterns: Dict) -> Set[str]:
        """
        Identify patterns requiring full scan coverage

        High-value criteria:
        1. Frequency >= 10 (statistically significant)
        2. Contains semantic importance keywords
        3. Likely to contain code/API/structured content
        """
        high_value_patterns = set()

        for pattern_id, frequency in all_patterns.items():
            # Frequency-based classification
            if frequency >= 10:
                high_value_patterns.add(pattern_id)
                continue

            # Semantic importance classification
            pattern_lower = pattern_id.lower()

            # Check for high-value keywords
            for keyword_type, keywords in self.high_value_indicators.items():
                if any(keyword in pattern_lower for keyword in keywords):
                    high_value_patterns.add(pattern_id)
                    break

            # Special patterns (data attributes, specific structures)
            if pattern_id.startswith('attr_data-') or 'tree' in pattern_lower:
                high_value_patterns.add(pattern_id)

        logger.info(f"🎯 Identified {len(high_value_patterns)} high-value patterns")
        return high_value_patterns

    def _classify_patterns_by_value(self, patterns: Dict) -> Tuple[Set, Set, Set]:
        """Classify patterns into high/medium/low value tiers"""
        high_value = set()
        medium_value = set()
        low_value = set()

        for pattern_id, frequency in patterns.items():
            if frequency >= 10 or self._is_semantically_important(pattern_id):
                high_value.add(pattern_id)
            elif frequency >= 3:
                medium_value.add(pattern_id)
            else:
                low_value.add(pattern_id)

        return high_value, medium_value, low_value

    def _is_semantically_important(self, pattern_id: str) -> bool:
        """Check if pattern is semantically important regardless of frequency"""
        pattern_lower = pattern_id.lower()

        # Check all high-value indicators
        for keywords in self.high_value_indicators.values():
            if any(keyword in pattern_lower for keyword in keywords):
                return True

        # Special cases
        if pattern_id.startswith('attr_data-') or 'tree' in pattern_lower:
            return True

        return False

    def _calculate_frequency_weighted_coverage(self, discovered: Dict, baseline: Dict) -> float:
        """Calculate coverage weighted by pattern frequency"""
        if not baseline:
            return 0.0

        total_baseline_weight = sum(baseline.values())
        covered_weight = sum(discovered.get(pattern, 0) for pattern in baseline.keys())

        return covered_weight / total_baseline_weight if total_baseline_weight > 0 else 0.0

    def _calculate_tier_coverage(self, discovered: Dict, tier_patterns: Dict) -> float:
        """Calculate coverage for specific pattern tier"""
        if not tier_patterns:
            return 1.0  # 100% if no patterns in tier

        return self._calculate_frequency_weighted_coverage(discovered, tier_patterns)

    def _identify_missing_patterns(self, discovered: Dict, baseline: Dict) -> List[str]:
        """Identify patterns missing from discovery results"""
        missing = []

        for pattern_id in baseline.keys():
            if pattern_id not in discovered:
                missing.append(pattern_id)

        # Sort by frequency (most important missing patterns first)
        missing.sort(key=lambda p: baseline[p], reverse=True)

        return missing[:50]  # Return top 50 missing patterns

    def _analyze_coverage_by_content_type(self, discovered: Dict, baseline: Dict) -> Dict[str, float]:
        """Analyze coverage by inferred content type"""
        content_types = {
            'code': ['code', 'highlight', 'python', 'csharp', 'example'],
            'api': ['api', 'reference', 'method', 'tree'],
            'table': ['table', 'grid', 'data', 'responsive'],
            'navigation': ['nav', 'toc', 'breadcrumb', 'menu'],
            'content': ['content', 'section', 'container', 'main']
        }

        coverage_by_type = {}

        for content_type, keywords in content_types.items():
            type_baseline = {p: f for p, f in baseline.items()
                             if any(keyword in p.lower() for keyword in keywords)}

            if type_baseline:
                type_coverage = self._calculate_frequency_weighted_coverage(discovered, type_baseline)
                coverage_by_type[content_type] = type_coverage

        return coverage_by_type


class HybridScanStrategy:
    """Execute hybrid scanning strategy for optimal coverage"""

    def __init__(self, high_value_sample_ratio: float = 1.0, common_sample_ratio: float = 0.1):
        self.high_value_sample_ratio = high_value_sample_ratio  # 100% for high-value
        self.common_sample_ratio = common_sample_ratio  # 10% for common patterns

    def execute_hybrid_scan(self, file_path: str, high_value_patterns: Set) -> Dict:
        """
        Execute hybrid scanning strategy

        Strategy:
        1. Full scan for high-value patterns (100% coverage)
        2. Sample scan for remaining patterns (configurable coverage)
        3. Combine results with proper weighting
        """
        logger.info("🔄 Executing hybrid scan strategy...")
        logger.info(f"   High-value patterns: {len(high_value_patterns)} (100% scan)")
        logger.info(f"   Common patterns: Remaining ({self.common_sample_ratio:.1%} scan)")
        logger.info(f"   📁 Scanning file: {file_path}")

        # VALIDATION: Check file exists and size
        import os
        if not os.path.exists(file_path):
            logger.error(f"❌ File not found: {file_path}")
            return {'patterns': {}, 'sample_content': {}, 'total_nodes_processed': 0}

        file_size = os.path.getsize(file_path)
        logger.info(f"   📊 File size: {file_size:,} bytes")

        # Execute high-value full scan
        high_value_results = self._targeted_full_scan(file_path, high_value_patterns)

        # Execute common pattern sampling
        common_results = self._targeted_sample_scan(file_path, high_value_patterns)

        # Combine results
        combined_results = self._combine_scan_results(high_value_results, common_results)

        logger.info(f"✅ Hybrid scan complete:")
        logger.info(f"   High-value patterns found: {len(high_value_results.get('patterns', {}))}")
        logger.info(f"   Common patterns found: {len(common_results.get('patterns', {}))}")
        logger.info(f"   Total unique patterns: {len(combined_results.get('patterns', {}))}")
        logger.info(f"   📊 Nodes processed: {combined_results.get('total_nodes_processed', 0):,}")

        return combined_results

    def _targeted_full_scan(self, file_path: str, target_patterns: Set) -> Dict:
        """Full scan nhưng chỉ collect specified patterns"""
        logger.info("🎯 Executing targeted full scan for high-value patterns...")
        logger.info(f"   📁 Scanning file: {file_path}")
        logger.info(f"   🎯 Target patterns: {len(target_patterns)}")

        pattern_counts = Counter()
        sample_content = {}
        nodes_processed = 0
        patterns_found = 0

        try:
            context = etree.iterparse(file_path, html=True, events=("end",), encoding="utf-8")

            for event, elem in context:
                nodes_processed += 1

                # Extract patterns from this element
                element_patterns = self._extract_element_patterns(elem)

                # Only collect target patterns
                for pattern_id in element_patterns:
                    if pattern_id in target_patterns:
                        pattern_counts[pattern_id] += 1
                        patterns_found += 1

                        # Collect sample content
                        if pattern_id not in sample_content:
                            elem_text = elem.text_content() if hasattr(elem, 'text_content') else (elem.text or "")
                            sample_content[pattern_id] = elem_text[:200].strip()

                # Memory management
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

                # Progress logging (more frequent for debugging)
                if nodes_processed % 50000 == 0:
                    logger.info(
                        f"   Progress: {nodes_processed:,} nodes processed, {len(pattern_counts)} unique patterns found")

        except Exception as e:
            logger.error(f"❌ Error in targeted full scan: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")

        logger.info(f"📊 Targeted full scan results:")
        logger.info(f"   Total nodes processed: {nodes_processed:,}")
        logger.info(f"   Unique patterns found: {len(pattern_counts)}")
        logger.info(f"   Total pattern instances: {patterns_found}")
        logger.info(f"   Sample content collected: {len(sample_content)}")

        return {
            'patterns': dict(pattern_counts),
            'sample_content': sample_content,
            'nodes_processed': nodes_processed,
            'scan_type': 'full'
        }

    def _targeted_sample_scan(self, file_path: str, exclude_patterns: Set) -> Dict:
        """Sample scan cho patterns không thuộc high-value"""
        logger.info("📊 Executing sample scan for common patterns...")
        logger.info(f"   📁 Scanning file: {file_path}")
        logger.info(f"   🚫 Excluding patterns: {len(exclude_patterns)}")
        logger.info(f"   📊 Sample ratio: {self.common_sample_ratio:.1%}")

        pattern_counts = Counter()
        sample_content = {}
        nodes_processed = 0
        nodes_sampled = 0
        patterns_found = 0

        try:
            context = etree.iterparse(file_path, html=True, events=("end",), encoding="utf-8")

            for event, elem in context:
                nodes_processed += 1

                # Sample decision
                if random.random() < self.common_sample_ratio:
                    nodes_sampled += 1

                    # Extract patterns from this element
                    element_patterns = self._extract_element_patterns(elem)

                    # Only collect non-high-value patterns
                    for pattern_id in element_patterns:
                        if pattern_id not in exclude_patterns:
                            pattern_counts[pattern_id] += 1
                            patterns_found += 1

                            # Collect sample content
                            if pattern_id not in sample_content:
                                elem_text = elem.text_content() if hasattr(elem, 'text_content') else (elem.text or "")
                                sample_content[pattern_id] = elem_text[:200].strip()

                # Memory management
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

                # Progress logging (more frequent for debugging)
                if nodes_processed % 50000 == 0:
                    logger.info(
                        f"   Progress: {nodes_processed:,} total, {nodes_sampled:,} sampled, {len(pattern_counts)} patterns")

        except Exception as e:
            logger.error(f"❌ Error in sample scan: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")

        logger.info(f"📊 Sample scan results:")
        logger.info(f"   Total nodes processed: {nodes_processed:,}")
        logger.info(f"   Nodes sampled: {nodes_sampled:,} ({nodes_sampled / nodes_processed * 100:.1f}%)")
        logger.info(f"   Unique patterns found: {len(pattern_counts)}")
        logger.info(f"   Total pattern instances: {patterns_found}")
        logger.info(f"   Sample content collected: {len(sample_content)}")

        return {
            'patterns': dict(pattern_counts),
            'sample_content': sample_content,
            'nodes_processed': nodes_processed,
            'nodes_sampled': nodes_sampled,
            'scan_type': 'sample'
        }

    def _extract_element_patterns(self, elem) -> Set[str]:
        """Extract all patterns from HTML element"""
        patterns = set()

        # Class patterns
        if classes := elem.get("class"):
            for cls in classes.split():
                patterns.add(f"class_{cls}")

        # Attribute patterns
        for attr_name in elem.attrib:
            if attr_name.startswith('data-'):
                patterns.add(f"attr_{attr_name}")

        # Tag patterns (if needed)
        patterns.add(f"tag_{elem.tag}")

        return patterns

    def _combine_scan_results(self, high_value_results: Dict, common_results: Dict) -> Dict:
        """Combine results from high-value and common scans"""
        combined_patterns = {}
        combined_sample_content = {}

        # Add high-value patterns (full counts)
        combined_patterns.update(high_value_results.get('patterns', {}))
        combined_sample_content.update(high_value_results.get('sample_content', {}))

        # Add common patterns (scaled counts based on sampling ratio)
        for pattern_id, count in common_results.get('patterns', {}).items():
            if pattern_id not in combined_patterns:
                # Scale up the sampled count to estimate full count
                estimated_count = int(count / self.common_sample_ratio) if self.common_sample_ratio > 0 else count
                combined_patterns[pattern_id] = estimated_count

        # Add sample content for common patterns
        for pattern_id, content in common_results.get('sample_content', {}).items():
            if pattern_id not in combined_sample_content:
                combined_sample_content[pattern_id] = content

        return {
            'patterns': combined_patterns,
            'sample_content': combined_sample_content,
            'high_value_results': high_value_results,
            'common_results': common_results,
            'total_nodes_processed': max(
                high_value_results.get('nodes_processed', 0),
                common_results.get('nodes_processed', 0)
            )
        }


# Add missing import
import random


class EnhancedPatternDiscovery:
    """
    Enhanced Pattern Discovery với Coverage >95% Support
    Extends functionality của ProductionPatternDiscovery
    """

    def __init__(self, target_coverage: float = 0.95, sample_ratio: float = 0.1,
                 max_memory_gb: float = 3.0):
        self.target_coverage = target_coverage
        self.sample_ratio = sample_ratio
        self.max_memory_gb = max_memory_gb

        # Initialize components
        self.coverage_analyzer = CoverageAnalyzer(target_coverage)
        self.hybrid_strategy = HybridScanStrategy(
            high_value_sample_ratio=1.0,  # 100% for high-value
            common_sample_ratio=sample_ratio  # Configurable for common
        )

        # For fallback to original discovery
        from production_ready_discovery import ProductionPatternDiscovery
        self.fallback_discovery = ProductionPatternDiscovery(sample_ratio, max_memory_gb)

    def run_adaptive_discovery(self, html_file_path: str, output_dir: str = "discovery_output") -> Dict:
        """
        Main adaptive discovery để đạt target coverage

        Workflow:
        1. Quick baseline scan → get all patterns baseline
        2. Identify high-value patterns
        3. Hybrid collection → targeted data collection
        4. Coverage analysis → measure actual coverage
        5. Adaptive tuning → adjust parameters if needed
        6. Final validation → ensure target met
        """
        logger.info(f"🚀 Starting adaptive discovery with {self.target_coverage:.1%} coverage target")

        # PHASE 0: Get processed file path (CRITICAL FIX)
        processed_file_path = self._get_processed_file_path(html_file_path)
        logger.info(f"📁 Using processed file: {processed_file_path}")

        # Phase 1: Baseline Pattern Discovery
        logger.info("Phase 1: Baseline pattern discovery...")
        baseline_patterns = self._quick_baseline_scan(processed_file_path)

        # Phase 2: Identify High-Value Patterns
        logger.info("Phase 2: Identifying high-value patterns...")
        high_value_patterns = self.coverage_analyzer.identify_high_value_patterns(baseline_patterns)

        # Phase 3: Hybrid Collection (FIX: Use processed file)
        logger.info("Phase 3: Hybrid data collection...")
        collection_results = self.hybrid_strategy.execute_hybrid_scan(processed_file_path, high_value_patterns)

        # Phase 4: Coverage Analysis
        logger.info("Phase 4: Coverage analysis...")
        coverage_metrics = self.coverage_analyzer.measure_coverage(
            collection_results['patterns'], baseline_patterns
        )

        # Phase 5: Adaptive Tuning (if needed)
        final_results = collection_results
        attempts = 1
        max_attempts = 3

        while coverage_metrics.overall_coverage < self.target_coverage and attempts < max_attempts:
            logger.warning(f"⚠️ Coverage {coverage_metrics.overall_coverage:.1%} < target {self.target_coverage:.1%}")
            logger.info(f"🔄 Adaptive tuning attempt {attempts + 1}/{max_attempts}")

            # Adjust strategy parameters
            self._adapt_strategy_parameters(coverage_metrics)

            # Re-run collection with adjusted parameters (FIX: Use processed file)
            final_results = self.hybrid_strategy.execute_hybrid_scan(processed_file_path, high_value_patterns)
            coverage_metrics = self.coverage_analyzer.measure_coverage(
                final_results['patterns'], baseline_patterns
            )

            attempts += 1

        # Phase 6: Final Validation & Results
        logger.info("Phase 6: Final validation...")

        if coverage_metrics.overall_coverage >= self.target_coverage:
            logger.info(f"✅ Target coverage achieved: {coverage_metrics.overall_coverage:.1%}")
        else:
            logger.warning(f"⚠️ Target coverage not achieved: {coverage_metrics.overall_coverage:.1%}")

        # Convert to format compatible with existing pipeline
        pattern_candidates = self._convert_to_pattern_candidates(final_results, baseline_patterns)

        # Use existing human review preparation
        self.fallback_discovery._prepare_human_review_materials(pattern_candidates, output_dir)

        # Save enhanced results
        self._save_enhanced_results(coverage_metrics, final_results, baseline_patterns, output_dir)

        return {
            'coverage_metrics': coverage_metrics,
            'collection_results': final_results,
            'baseline_patterns': baseline_patterns,
            'pattern_candidates': pattern_candidates,
            'output_dir': output_dir,
            'target_achieved': coverage_metrics.overall_coverage >= self.target_coverage
        }

    def _get_processed_file_path(self, html_file_path: str) -> str:
        """
        Get processed file path, handling preprocessing if needed
        CRITICAL: Both baseline và hybrid scan must use same processed file!
        """
        try:
            from html_preprocessor import HTMLPreprocessor
            preprocessor = HTMLPreprocessor()
            processed_file_path = preprocessor.preprocess_file(Path(html_file_path).name)

            # VALIDATION: Check file sizes
            import os
            original_size = os.path.getsize(html_file_path) if os.path.exists(html_file_path) else 0
            processed_size = os.path.getsize(processed_file_path) if os.path.exists(processed_file_path) else 0

            logger.info(f"✅ File preprocessing validation:")
            logger.info(f"   Original file: {html_file_path} ({original_size:,} bytes)")
            logger.info(f"   Processed file: {processed_file_path} ({processed_size:,} bytes)")

            if processed_size == 0:
                logger.error(f"❌ Processed file is empty! Using original file.")
                return html_file_path

            return processed_file_path

        except ImportError:
            logger.warning("❌ HTMLPreprocessor not available, using original file")
            return html_file_path
        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {e}")
            logger.warning("⚠️ Falling back to original file")
            return html_file_path

    def _quick_baseline_scan(self, processed_file_path: str) -> Dict[str, int]:
        """
        Quick full scan để get baseline của all patterns
        Lightweight scan chỉ để count patterns, không collect detailed data

        UPDATED: Now takes processed_file_path directly
        """
        logger.info("🔍 Quick baseline scan for all patterns...")

        pattern_counts = Counter()
        nodes_processed = 0

        try:
            context = etree.iterparse(processed_file_path, html=True, events=("end",), encoding="utf-8")

            for event, elem in context:
                nodes_processed += 1

                # Extract all patterns from element
                element_patterns = self._extract_all_patterns(elem)
                pattern_counts.update(element_patterns)

                # Memory management
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

                # Progress logging
                if nodes_processed % 20000 == 0:
                    logger.debug(f"Baseline scan progress: {nodes_processed:,} nodes")

        except Exception as e:
            logger.error(f"Error in baseline scan: {e}")

        # Filter patterns (remove very rare ones)
        filtered_patterns = {p: c for p, c in pattern_counts.items() if c >= 1}

        logger.info(f"📊 Baseline scan complete:")
        logger.info(f"   Total nodes: {nodes_processed:,}")
        logger.info(f"   Unique patterns: {len(filtered_patterns):,}")
        logger.info(f"   Top pattern frequency: {max(filtered_patterns.values()) if filtered_patterns else 0}")

        return dict(filtered_patterns)

    def _extract_all_patterns(self, elem) -> List[str]:
        """Extract all possible patterns from HTML element"""
        patterns = []

        # Class patterns
        if classes := elem.get("class"):
            for cls in classes.split():
                patterns.append(f"class_{cls}")

        # Attribute patterns (focus on data- attributes)
        for attr_name in elem.attrib:
            if attr_name.startswith('data-'):
                patterns.append(f"attr_{attr_name}")

        # ID patterns (generalized)
        if elem_id := elem.get("id"):
            # Generalize ID pattern
            id_pattern = re.sub(r'\d+', 'N', elem_id)
            id_pattern = re.sub(r'[a-f0-9]{8,}', 'HASH', id_pattern)
            patterns.append(f"id_{id_pattern}")

        return patterns

    def _adapt_strategy_parameters(self, coverage_metrics: CoverageMetrics):
        """
        Adapt strategy parameters based on coverage analysis
        Multi-strategy approach for different coverage gaps
        """
        logger.info("🔧 Adapting strategy parameters...")

        # Strategy 1: Increase sampling ratio for common patterns
        if coverage_metrics.low_value_coverage < 0.8:
            old_ratio = self.hybrid_strategy.common_sample_ratio
            self.hybrid_strategy.common_sample_ratio = min(old_ratio * 1.5, 0.3)
            logger.info(
                f"   📈 Increased common sampling ratio: {old_ratio:.1%} → {self.hybrid_strategy.common_sample_ratio:.1%}")

        # Strategy 2: Expand high-value pattern definition
        if coverage_metrics.high_value_coverage < 0.95:
            # Lower the frequency threshold for high-value patterns
            logger.info("   🎯 Expanding high-value pattern criteria")
            # This would require re-identifying high-value patterns with relaxed criteria

        # Strategy 3: Target specific missing patterns
        if len(coverage_metrics.missing_patterns) > 0:
            logger.info(f"   🔍 Targeting {len(coverage_metrics.missing_patterns)} missing patterns")
            # Could implement targeted collection for specific missing patterns

    def _convert_to_pattern_candidates(self, collection_results: Dict, baseline_patterns: Dict):
        """
        Convert collection results to PatternCandidate format
        Compatible với existing human review pipeline
        """
        from production_ready_discovery import PatternCandidate, EvidenceScorer

        scorer = EvidenceScorer()
        candidates = []

        for pattern_id, frequency in collection_results['patterns'].items():
            # Extract pattern information
            if pattern_id.startswith('class_'):
                selector_type = 'class'
                selector_value = pattern_id[6:]  # Remove 'class_' prefix
            elif pattern_id.startswith('attr_'):
                selector_type = 'attr'
                selector_value = pattern_id[5:]  # Remove 'attr_' prefix
            elif pattern_id.startswith('id_'):
                selector_type = 'id_pattern'
                selector_value = pattern_id[3:]  # Remove 'id_' prefix
            else:
                continue  # Skip unknown pattern types

            # Get sample content
            sample_content = collection_results.get('sample_content', {}).get(pattern_id, '')

            # Calculate evidence score
            candidate_data = {
                'selector_value': selector_value,
                'frequency': frequency,
                'sample_content': sample_content,
                'characteristics': {'baseline_frequency': baseline_patterns.get(pattern_id, frequency)}
            }

            evidence_score = scorer.calculate_evidence_score(candidate_data)

            # Create PatternCandidate
            candidate = PatternCandidate(
                identifier=pattern_id,
                selector_type=selector_type,
                selector_value=selector_value,
                frequency=frequency,
                evidence_score=evidence_score,
                characteristics=candidate_data['characteristics'],
                sample_content=sample_content,
                depth_stats={},  # Could be enhanced later
                text_stats={}  # Could be enhanced later
            )

            candidates.append(candidate)

        # Sort by evidence score
        candidates.sort(key=lambda c: c.evidence_score, reverse=True)

        logger.info(f"📋 Converted {len(candidates)} patterns to candidates")

        return candidates

    def _save_enhanced_results(self, coverage_metrics: CoverageMetrics,
                               collection_results: Dict, baseline_patterns: Dict,
                               output_dir: str):
        """Save enhanced discovery results với coverage analysis"""
        Path(output_dir).mkdir(exist_ok=True)

        # Save coverage analysis
        coverage_data = {
            'metrics': {
                'overall_coverage': coverage_metrics.overall_coverage,
                'high_value_coverage': coverage_metrics.high_value_coverage,
                'medium_value_coverage': coverage_metrics.medium_value_coverage,
                'low_value_coverage': coverage_metrics.low_value_coverage,
                'total_patterns_found': coverage_metrics.total_patterns_found,
                'total_patterns_baseline': coverage_metrics.total_patterns_baseline,
                'target_achieved': coverage_metrics.overall_coverage >= self.target_coverage
            },
            'missing_patterns': coverage_metrics.missing_patterns[:20],  # Top 20 missing
            'coverage_by_content_type': coverage_metrics.coverage_by_content_type
        }

        with open(Path(output_dir) / "coverage_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(coverage_data, f, indent=2, ensure_ascii=False)

        # Save baseline patterns for reference
        with open(Path(output_dir) / "baseline_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(baseline_patterns, f, indent=2, ensure_ascii=False)

        # Save collection results
        with open(Path(output_dir) / "collection_results.json", 'w', encoding='utf-8') as f:
            # Remove non-serializable parts
            serializable_results = {
                'patterns': collection_results['patterns'],
                'sample_content': collection_results.get('sample_content', {}),
                'total_nodes_processed': collection_results.get('total_nodes_processed', 0)
            }
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Enhanced results saved to {output_dir}/")
        logger.info(f"   - coverage_analysis.json")
        logger.info(f"   - baseline_patterns.json")
        logger.info(f"   - collection_results.json")


def main():
    """Test function cho Enhanced Pattern Discovery"""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Pattern Discovery với Coverage >95%')
    parser.add_argument('--input', required=True, help='HTML file path')
    parser.add_argument('--output', default='discovery_output', help='Output directory')
    parser.add_argument('--target-coverage', type=float, default=0.95, help='Target coverage (default: 0.95)')
    parser.add_argument('--sample-ratio', type=float, default=0.1, help='Sample ratio for common patterns')

    args = parser.parse_args()

    # Run enhanced discovery
    discovery = EnhancedPatternDiscovery(
        target_coverage=args.target_coverage,
        sample_ratio=args.sample_ratio
    )

    results = discovery.run_adaptive_discovery(args.input, args.output)

    # Print results summary
    print("=" * 60)
    print("🚀 ENHANCED DISCOVERY COMPLETE")
    print("=" * 60)
    print(f"📊 Coverage Achieved: {results['coverage_metrics'].overall_coverage:.1%}")
    print(f"🎯 Target Coverage: {discovery.target_coverage:.1%}")
    print(f"✅ Target Met: {results['target_achieved']}")
    print(f"📁 Results saved to: {args.output}/")
    print()
    print("📋 Coverage Breakdown:")
    print(f"   High-value patterns: {results['coverage_metrics'].high_value_coverage:.1%}")
    print(f"   Medium-value patterns: {results['coverage_metrics'].medium_value_coverage:.1%}")
    print(f"   Low-value patterns: {results['coverage_metrics'].low_value_coverage:.1%}")
    print()
    print("🔍 Pattern Statistics:")
    print(f"   Baseline patterns: {len(results['baseline_patterns']):,}")
    print(f"   Discovered patterns: {len(results['collection_results']['patterns']):,}")
    print(f"   Missing patterns: {len(results['coverage_metrics'].missing_patterns)}")
    print()
    print("📝 Next Steps:")
    print(f"1. Review {args.output}/patterns_for_review.csv")
    print("2. Fill in 'Human Decision' column")
    print("3. Save as 'reviewed_patterns.csv'")
    print("4. Generate rules using existing workflow")


if __name__ == "__main__":
    main()