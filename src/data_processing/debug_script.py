# ===============================
# DEBUG VERSION - Parser Fixes
# ===============================

import logging
from bs4 import BeautifulSoup
from lxml import etree, html
from collections import Counter, defaultdict
import random
from typing import Dict, List, Tuple, Set, Optional

from production_ready_discovery import ReservoirSampler

logger = logging.getLogger(__name__)


class DebugHTMLAnalyzer:
    """Debug analyzer để compare different parsing methods"""

    def __init__(self):
        self.results = {}

    def analyze_file_basic(self, html_file_path: str):
        """Basic file analysis"""
        print("=" * 50)
        print("BASIC FILE ANALYSIS")
        print("=" * 50)

        with open(html_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Basic stats
        file_size_mb = len(content) / (1024 * 1024)
        line_count = content.count('\n')
        html_tag_count = content.count('<')

        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Line count: {line_count:,}")
        print(f"HTML tags (rough): {html_tag_count:,}")

        # Sample content
        print(f"\nFirst 500 chars:")
        print(content[:500])
        print(f"\nLast 500 chars:")
        print(content[-500:])

        return {
            'file_size_mb': file_size_mb,
            'line_count': line_count,
            'html_tag_count': html_tag_count
        }

    def analyze_with_beautifulsoup(self, html_file_path: str):
        """Analyze với BeautifulSoup"""
        print("=" * 50)
        print("BEAUTIFULSOUP ANALYSIS")
        print("=" * 50)

        with open(html_file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

        # Count elements
        all_elements = soup.find_all()
        tag_counter = Counter(elem.name for elem in all_elements)
        class_counter = Counter()

        # Class analysis
        for elem in all_elements:
            if elem.get('class'):
                for cls in elem.get('class'):
                    class_counter[cls] += 1

        print(f"Total elements found: {len(all_elements):,}")
        print(f"Top 10 tags: {tag_counter.most_common(10)}")
        print(f"Top 10 classes: {class_counter.most_common(10)}")

        return {
            'total_elements': len(all_elements),
            'tag_distribution': tag_counter,
            'class_distribution': class_counter
        }

    def analyze_with_lxml_fixed(self, html_file_path: str, sample_ratio: float = 0.1):
        """Fixed lxml parsing với enhanced events"""
        print("=" * 50)
        print("LXML FIXED ANALYSIS")
        print("=" * 50)

        tag_counter = Counter()
        class_counter = Counter()
        nodes_processed = 0
        nodes_sampled = 0

        try:
            # Method 1: Use both start and end events
            context = etree.iterparse(
                html_file_path,
                html=True,
                events=("start", "end"),  # Both start and end
                encoding="utf-8",
                recover=True  # Handle malformed HTML
            )

            for event, elem in context:
                if event == "end":  # Only process on end event
                    nodes_processed += 1

                    # Sample decision
                    if random.random() < sample_ratio:
                        nodes_sampled += 1

                        # Collect stats
                        tag_counter[elem.tag] += 1

                        if classes := elem.get("class"):
                            for cls in classes.split():
                                class_counter[cls] += 1

                        # Progress logging
                        if nodes_processed % 1000 == 0:
                            print(f"Processed: {nodes_processed:,}, Sampled: {nodes_sampled:,}")

                    # Memory management - more careful
                    if event == "end":
                        elem.clear()
                        # Less aggressive cleanup
                        if elem.getprevious() is not None:
                            try:
                                del elem.getparent()[0]
                            except:
                                pass  # Skip cleanup errors

        except Exception as e:
            print(f"LXML parsing error: {e}")
            return None

        print(f"Final - Processed: {nodes_processed:,}, Sampled: {nodes_sampled:,}")
        print(f"Top 10 tags: {tag_counter.most_common(10)}")
        print(f"Top 10 classes: {class_counter.most_common(10)}")

        return {
            'nodes_processed': nodes_processed,
            'nodes_sampled': nodes_sampled,
            'tag_distribution': tag_counter,
            'class_distribution': class_counter
        }


# ===============================
# ENHANCED ReservoirSampler
# ===============================

class DebugReservoirSampler(ReservoirSampler):
    """Enhanced reservoir sampler với debug logging"""

    def __init__(self, sample_size: int = 10000):
        super().__init__(sample_size)
        self.debug_stats = {
            'elements_seen': 0,
            'elements_added': 0,
            'elements_replaced': 0
        }

    def add_element(self, element_data: Dict) -> bool:
        """Enhanced add với debug tracking"""
        self.debug_stats['elements_seen'] += 1
        result = super().add_element(element_data)

        if result:
            if len(self.reservoir) <= self.sample_size:
                self.debug_stats['elements_added'] += 1
            else:
                self.debug_stats['elements_replaced'] += 1

        # Debug logging every 1000 elements
        if self.debug_stats['elements_seen'] % 1000 == 0:
            logger.info(f"Sampling stats: {self.debug_stats}")

        return result


# ===============================
# FIXED _streaming_snapshot_with_sampling
# ===============================

def debug_streaming_snapshot_with_sampling(self, html_file_path: str) -> Dict:
    """Debug version của streaming snapshot"""
    print("=" * 50)
    print("DEBUG STREAMING SNAPSHOT")
    print("=" * 50)

    # Step 1: Basic file verification
    analyzer = DebugHTMLAnalyzer()
    basic_stats = analyzer.analyze_file_basic(html_file_path)

    # Step 2: BeautifulSoup comparison
    bs_stats = analyzer.analyze_with_beautifulsoup(html_file_path)

    # Step 3: Fixed lxml parsing
    lxml_stats = analyzer.analyze_with_lxml_fixed(html_file_path, self.sample_ratio)

    # Step 4: Compare results
    print("=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    print(f"BeautifulSoup elements: {bs_stats['total_elements']:,}")
    if lxml_stats:
        print(f"LXML processed: {lxml_stats['nodes_processed']:,}")
        print(f"LXML sampled: {lxml_stats['nodes_sampled']:,}")

    # Return enhanced results
    return {
        'basic_analysis': basic_stats,
        'beautifulsoup_analysis': bs_stats,
        'lxml_analysis': lxml_stats,
        'processing_stats': {
            'comparison_complete': True,
            'parsing_method': 'debug_multi_parser'
        }
    }


# ===============================
# ALTERNATIVE: BeautifulSoup-based Sampling
# ===============================

def beautifulsoup_sampling_alternative(html_file_path: str, sample_ratio: float = 0.1):
    """Alternative implementation using BeautifulSoup"""
    print("=" * 50)
    print("BEAUTIFULSOUP SAMPLING ALTERNATIVE")
    print("=" * 50)

    with open(html_file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    all_elements = soup.find_all()
    total_elements = len(all_elements)

    # Reservoir sampling on BeautifulSoup elements
    sampler = DebugReservoirSampler(sample_size=int(10000 * sample_ratio))

    tag_counter = Counter()
    class_counter = Counter()

    for i, elem in enumerate(all_elements):
        # Sample decision
        if random.random() < sample_ratio:
            element_data = {
                'tag': elem.name,
                'classes': elem.get('class', []),
                'attributes': list(elem.attrs.keys()),
                'text_length': len(elem.get_text(strip=True)),
                'text_sample': elem.get_text(strip=True)[:200],
                'has_children': len(elem.find_all()) > 0
            }

            sampler.add_element(element_data)

            # Basic stats
            tag_counter[elem.name] += 1
            if elem.get('class'):
                for cls in elem.get('class'):
                    class_counter[cls] += 1

        # Progress
        if (i + 1) % 5000 == 0:
            print(f"Processed {i + 1:,}/{total_elements:,} elements")

    print(f"Final sampling: {len(sampler.reservoir):,} elements in reservoir")
    print(f"Top tags: {tag_counter.most_common(10)}")
    print(f"Top classes: {class_counter.most_common(10)}")

    return {
        'total_elements': total_elements,
        'sampled_elements': len(sampler.reservoir),
        'tag_distribution': tag_counter,
        'class_distribution': class_counter,
        'reservoir_data': sampler.reservoir
    }


# ===============================
# DEBUG RUNNER
# ===============================

def run_debug_analysis(html_file_path: str):
    """Run complete debug analysis"""
    print("Starting comprehensive debug analysis...")

    results = {}

    # Method 1: Basic analysis
    analyzer = DebugHTMLAnalyzer()
    results['basic'] = analyzer.analyze_file_basic(html_file_path)

    # Method 2: BeautifulSoup
    results['beautifulsoup'] = analyzer.analyze_with_beautifulsoup(html_file_path)

    # Method 3: Fixed LXML
    results['lxml_fixed'] = analyzer.analyze_with_lxml_fixed(html_file_path)

    # Method 4: BeautifulSoup sampling
    results['bs_sampling'] = beautifulsoup_sampling_alternative(html_file_path)

    return results