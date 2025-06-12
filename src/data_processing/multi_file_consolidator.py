#!/usr/bin/env python3
"""
Multi-File CSV Consolidator
Consolidates multiple pattern CSV files từ different HTML sources
Handles overlaps, conflicts, và generates unified review file
"""

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


class MultiFileConsolidator:
    """Consolidate multiple CSV files từ different HTML sources"""

    def __init__(self, input_dir: str = "discovery_output"):
        self.input_dir = Path(input_dir)
        self.consolidated_patterns = {}
        self.source_mapping = {}  # Track which file each pattern came from
        self.overlap_analysis = defaultdict(list)

    def consolidate_all_csvs(self, output_filename: str = "patterns_for_review_CONSOLIDATED.csv") -> str:
        """
        Main function: Consolidate all CSV files in directory
        
        Returns: Path to consolidated CSV file
        """
        logger.info("🔄 Starting multi-file CSV consolidation...")

        # 1. Find all CSV files to consolidate
        csv_files = self._find_csv_files()
        logger.info(f"📁 Found {len(csv_files)} CSV files to consolidate")

        if not csv_files:
            logger.error("❌ No CSV files found for consolidation")
            return ""

        # 2. Load patterns from all files
        all_patterns = {}
        
        for csv_file in csv_files:
            patterns = self._load_patterns_from_csv(csv_file)
            source_name = self._get_source_name(csv_file)
            
            logger.info(f"   📄 {csv_file.name}: {len(patterns)} patterns")
            
            # Track source mapping và detect overlaps
            for pattern_id, pattern_data in patterns.items():
                if pattern_id in all_patterns:
                    # Pattern overlap detected
                    self.overlap_analysis[pattern_id].append({
                        'source': source_name,
                        'frequency': pattern_data['frequency'],
                        'evidence_score': pattern_data['evidence_score']
                    })
                    
                    # Keep pattern với higher evidence score
                    if pattern_data['evidence_score'] > all_patterns[pattern_id]['evidence_score']:
                        all_patterns[pattern_id] = pattern_data
                        self.source_mapping[pattern_id] = source_name
                else:
                    all_patterns[pattern_id] = pattern_data
                    self.source_mapping[pattern_id] = source_name
                    self.overlap_analysis[pattern_id] = [{
                        'source': source_name,
                        'frequency': pattern_data['frequency'],
                        'evidence_score': pattern_data['evidence_score']
                    }]

        # 3. Analyze consolidation results
        self._analyze_consolidation_results(all_patterns, csv_files)

        # 4. Create consolidated CSV
        output_path = self._create_consolidated_csv(all_patterns, output_filename)

        # 5. Generate consolidation report
        self._generate_consolidation_report(all_patterns, csv_files, output_path)

        logger.info(f"✅ Consolidation complete: {output_path}")
        return str(output_path)

    def _find_csv_files(self) -> List[Path]:
        """Find all pattern CSV files in directory"""
        csv_files = []
        
        # Look for patterns_for_review files
        for pattern in ["*patterns_for_review*.csv", "*patterns*.csv"]:
            csv_files.extend(self.input_dir.glob(pattern))
        
        # Remove duplicates và filter out reviewed files
        unique_files = []
        seen_names = set()
        
        for csv_file in csv_files:
            # Skip reviewed files (human output)
            if 'reviewed' in csv_file.name.lower():
                continue
                
            # Skip consolidated files (our output)
            if 'CONSOLIDATED' in csv_file.name:
                continue
                
            if csv_file.name not in seen_names:
                unique_files.append(csv_file)
                seen_names.add(csv_file.name)

        return sorted(unique_files)

    def _get_source_name(self, csv_file: Path) -> str:
        """Extract source name từ CSV filename"""
        name = csv_file.stem
        
        # Remove version numbers
        import re
        name = re.sub(r'_v\d+$', '', name)
        
        # Clean up name
        name = name.replace('patterns_for_review', 'source')
        name = name.replace('patterns', 'source')
        
        return name if name != 'source' else csv_file.stem

    def _load_patterns_from_csv(self, csv_file: Path) -> Dict:
        """Load patterns từ single CSV file"""
        patterns = {}
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Extract pattern identity
                    selector_type = row.get('Selector Type', '').strip()
                    selector_value = row.get('Selector Value', '').strip()
                    
                    if not selector_type or not selector_value:
                        continue
                    
                    pattern_id = f"{selector_type}|{selector_value}"
                    
                    patterns[pattern_id] = {
                        'pattern_id': row.get('Pattern ID', ''),
                        'selector_type': selector_type,
                        'selector_value': selector_value,
                        'frequency': self._safe_int(row.get('Estimated Frequency', '0')),
                        'evidence_score': self._safe_float(row.get('Evidence Score', '0')),
                        'sample_content': row.get('Sample Content', ''),
                        'suggested_type': row.get('Suggested Type', ''),
                        'human_decision': row.get('Human Decision', '').strip(),
                        'notes': row.get('Notes', '').strip(),
                        'source_file': csv_file.name
                    }
                    
        except Exception as e:
            logger.error(f"❌ Error loading {csv_file}: {e}")
            
        return patterns

    def _safe_int(self, value: str) -> int:
        """Safely convert string to int"""
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    def _safe_float(self, value: str) -> float:
        """Safely convert string to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _analyze_consolidation_results(self, all_patterns: Dict, csv_files: List[Path]):
        """Analyze consolidation results và log statistics"""
        total_patterns = len(all_patterns)
        
        # Count patterns by source
        source_counts = defaultdict(int)
        for pattern_id in all_patterns:
            source = self.source_mapping[pattern_id]
            source_counts[source] += 1
        
        # Count overlaps
        overlapped_patterns = sum(1 for patterns in self.overlap_analysis.values() 
                                if len(patterns) > 1)
        
        logger.info(f"📊 Consolidation Analysis:")
        logger.info(f"   Input files: {len(csv_files)}")
        logger.info(f"   Total unique patterns: {total_patterns}")
        logger.info(f"   Overlapped patterns: {overlapped_patterns}")
        logger.info(f"   Patterns by source:")
        
        for source, count in sorted(source_counts.items()):
            logger.info(f"     {source}: {count} patterns")

    def _create_consolidated_csv(self, all_patterns: Dict, output_filename: str) -> Path:
        """Create consolidated CSV file"""
        output_path = self.input_dir / output_filename
        
        # CSV headers
        headers = [
            'Pattern ID',
            'Selector Type', 
            'Selector Value',
            'Estimated Frequency',
            'Evidence Score',
            'Sample Content',
            'Suggested Type',
            'Human Decision',
            'Notes',
            'Source Files'  # NEW: Track which files contributed
        ]
        
        # Sort patterns by evidence score (highest first)
        sorted_patterns = sorted(all_patterns.items(), 
                               key=lambda x: x[1]['evidence_score'], reverse=True)
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                
                for pattern_id, pattern_data in sorted_patterns:
                    # Build source files list
                    source_files = []
                    for source_info in self.overlap_analysis[pattern_id]:
                        source_files.append(source_info['source'])
                    source_files_str = '; '.join(sorted(set(source_files)))
                    
                    # Enhance notes với overlap info if needed
                    notes = pattern_data['notes']
                    if len(self.overlap_analysis[pattern_id]) > 1:
                        overlap_info = []
                        for source_info in self.overlap_analysis[pattern_id]:
                            overlap_info.append(f"{source_info['source']}({source_info['frequency']})")
                        overlap_str = f"MULTI-SOURCE: {', '.join(overlap_info)}"
                        notes = f"{overlap_str}; {notes}" if notes else overlap_str
                    
                    writer.writerow({
                        'Pattern ID': pattern_data['pattern_id'],
                        'Selector Type': pattern_data['selector_type'],
                        'Selector Value': pattern_data['selector_value'],
                        'Estimated Frequency': pattern_data['frequency'],
                        'Evidence Score': f"{pattern_data['evidence_score']:.3f}",
                        'Sample Content': pattern_data['sample_content'][:100],
                        'Suggested Type': pattern_data['suggested_type'],
                        'Human Decision': pattern_data['human_decision'],
                        'Notes': notes,
                        'Source Files': source_files_str
                    })
                    
        except Exception as e:
            logger.error(f"❌ Error writing consolidated CSV: {e}")
            raise
            
        return output_path

    def _generate_consolidation_report(self, all_patterns: Dict, csv_files: List[Path], output_path: Path):
        """Generate detailed consolidation report"""
        report_path = output_path.parent / "consolidation_report.json"
        
        # Count patterns with human decisions
        decided_patterns = sum(1 for p in all_patterns.values() if p['human_decision'])
        
        # Evidence score distribution
        evidence_scores = [p['evidence_score'] for p in all_patterns.values()]
        
        report = {
            'consolidation_summary': {
                'input_files': [f.name for f in csv_files],
                'total_patterns': len(all_patterns),
                'overlapped_patterns': sum(1 for patterns in self.overlap_analysis.values() 
                                         if len(patterns) > 1),
                'patterns_with_decisions': decided_patterns,
                'patterns_needing_review': len(all_patterns) - decided_patterns
            },
            'evidence_score_stats': {
                'min': min(evidence_scores) if evidence_scores else 0,
                'max': max(evidence_scores) if evidence_scores else 0,
                'avg': sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0
            },
            'source_breakdown': dict(defaultdict(int)),
            'top_overlaps': []
        }
        
        # Source breakdown
        for pattern_id in all_patterns:
            source = self.source_mapping[pattern_id]
            report['source_breakdown'][source] = report['source_breakdown'].get(source, 0) + 1
        
        # Top overlaps
        for pattern_id, sources in self.overlap_analysis.items():
            if len(sources) > 1:
                report['top_overlaps'].append({
                    'pattern': pattern_id,
                    'sources': len(sources),
                    'details': sources
                })
        
        # Sort top overlaps by number of sources
        report['top_overlaps'].sort(key=lambda x: x['sources'], reverse=True)
        report['top_overlaps'] = report['top_overlaps'][:20]  # Top 20
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"📋 Consolidation report saved: {report_path}")


def main():
    """Main function cho consolidation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-File CSV Consolidator')
    parser.add_argument('--input-dir', default='discovery_output', 
                       help='Directory containing CSV files to consolidate')
    parser.add_argument('--output', default='patterns_for_review_CONSOLIDATED.csv',
                       help='Output consolidated CSV filename')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🔄 MULTI-FILE CSV CONSOLIDATOR")
    print("=" * 50)
    
    try:
        # Run consolidation
        consolidator = MultiFileConsolidator(args.input_dir)
        output_path = consolidator.consolidate_all_csvs(args.output)
        
        if output_path:
            print(f"\n✅ CONSOLIDATION COMPLETE")
            print(f"📁 Consolidated file: {output_path}")
            print(f"📋 Report: {Path(output_path).parent / 'consolidation_report.json'}")
            print(f"\n📝 Next steps:")
            print(f"1. Review {args.output} và fill 'Human Decision' column")
            print(f"2. Save as 'reviewed_patterns_CONSOLIDATED.csv'") 
            print(f"3. Generate rules: --generate-rules reviewed_patterns_CONSOLIDATED.csv")
        else:
            print("❌ Consolidation failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()