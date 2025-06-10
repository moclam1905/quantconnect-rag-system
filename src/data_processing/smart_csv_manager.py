#!/usr/bin/env python3
"""
Smart CSV Manager với Incremental Review Support
Handles versioning, change detection, và preservation của human decisions
"""

import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VersionManager:
    """Manages CSV file versioning với auto-increment logic"""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_next_version(self, base_name: str = "patterns_for_review") -> str:
        """
        Auto-increment version: v1 → v2 → v3
        Returns: "patterns_for_review_v2.csv"
        """
        existing_versions = self._find_existing_versions(base_name)

        if not existing_versions:
            next_version = 1
        else:
            highest_version = max(existing_versions)
            next_version = highest_version + 1

        filename = f"{base_name}_v{next_version}.csv"
        logger.info(f"📋 Next version: {filename}")

        return filename

    def _find_existing_versions(self, base_name: str) -> List[int]:
        """Find all existing version numbers"""
        versions = []
        pattern = f"{base_name}_v*.csv"

        for file_path in self.base_dir.glob(pattern):
            # Extract version number từ filename
            match = re.search(f"{base_name}_v(\d+)\.csv", file_path.name)
            if match:
                versions.append(int(match.group(1)))

        logger.debug(f"Found existing versions: {versions}")
        return versions

    def list_all_files(self, pattern: str = "*patterns*v*.csv") -> List[Path]:
        """List all versioned files matching pattern"""
        return list(self.base_dir.glob(pattern))

    def backup_file(self, file_path: Path) -> Optional[Path]:
        """Backup existing file với timestamp suffix"""
        if not file_path.exists():
            return None

        backup_dir = self.base_dir / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Add timestamp to backup filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name

        # Copy file to backup
        import shutil
        shutil.copy2(file_path, backup_path)

        logger.info(f"💾 Backed up: {file_path.name} → {backup_path.name}")
        return backup_path


class SmartCSVManager:
    """Smart CSV Manager với incremental review support"""

    def __init__(self, output_dir: str = "discovery_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.version_manager = VersionManager(output_dir)

        # Thresholds cho "significant change" detection
        self.frequency_change_threshold = 0.5  # 50% change
        self.evidence_change_threshold = 0.2  # 0.2 change

    def create_incremental_review_csv(self, new_candidates: List) -> str:
        """
        Main function: Smart merge với change detection

        Args:
            new_candidates: List of PatternCandidate objects

        Returns:
            Path to new versioned CSV file
        """
        logger.info("🔄 Creating incremental review CSV...")

        # 1. Load existing decisions từ all previous versions
        existing_decisions = self._load_all_existing_decisions()
        logger.info(f"📂 Loaded {len(existing_decisions)} existing decisions")

        # 2. Process each candidate: existing vs new vs changed
        processed_patterns = []
        stats = {'existing': 0, 'new': 0, 'changed': 0}

        for candidate in new_candidates:
            pattern_id = self._get_pattern_identity(candidate)

            if pattern_id in existing_decisions:
                # Pattern exists - check for significant changes
                old_data = existing_decisions[pattern_id]
                change_note = self._detect_significant_changes(candidate, old_data)

                if change_note:
                    # Significant change → need re-review với warning
                    processed_patterns.append(self._create_changed_pattern_row(candidate, old_data, change_note))
                    stats['changed'] += 1
                else:
                    # No significant change → keep existing decision
                    processed_patterns.append(self._create_existing_pattern_row(candidate, old_data))
                    stats['existing'] += 1
            else:
                # New pattern → needs human review
                processed_patterns.append(self._create_new_pattern_row(candidate))
                stats['new'] += 1

        # 3. Sort patterns: new first, changed second, existing last
        processed_patterns.sort(key=self._pattern_sort_key)

        # 4. Create new versioned CSV
        version_filename = self.version_manager.get_next_version()
        csv_path = self._write_csv_file(processed_patterns, version_filename)

        # 5. Log summary
        logger.info(f"✅ Incremental CSV created: {csv_path}")
        logger.info(f"📊 Pattern summary:")
        logger.info(f"   🆕 New patterns: {stats['new']}")
        logger.info(f"   ⚠️  Changed patterns: {stats['changed']}")
        logger.info(f"   ✅ Existing patterns: {stats['existing']}")
        logger.info(f"   📋 Total: {len(processed_patterns)}")

        return str(csv_path)

    def _get_pattern_identity(self, candidate) -> str:
        """Unique pattern ID: selector_type|selector_value"""
        return f"{candidate.selector_type}|{candidate.selector_value}"

    def _load_all_existing_decisions(self) -> Dict[str, Dict]:
        """
        Load ALL previous human decisions từ all versions
        Returns: {pattern_identity: {human_decision, notes, frequency, evidence_score, ...}, ...}
        """
        existing_decisions = {}

        # Find all versioned CSV files
        csv_files = self.version_manager.list_all_files("*patterns*v*.csv")

        # Also check for non-versioned files (legacy support)
        legacy_files = list(self.output_dir.glob("*patterns*.csv"))
        legacy_files = [f for f in legacy_files if "_v" not in f.name]  # Exclude versioned files

        all_files = csv_files + legacy_files

        logger.info(f"📂 Scanning {len(all_files)} files for existing decisions...")

        for csv_file in all_files:
            try:
                decisions = self._load_decisions_from_csv(csv_file)
                logger.debug(f"   Loaded {len(decisions)} decisions from {csv_file.name}")

                # Merge decisions (latest file wins in case of conflicts)
                existing_decisions.update(decisions)

            except Exception as e:
                logger.warning(f"⚠️ Error loading {csv_file.name}: {e}")

        return existing_decisions

    def _load_decisions_from_csv(self, csv_file: Path) -> Dict[str, Dict]:
        """Load decisions from single CSV file"""
        decisions = {}

        if not csv_file.exists():
            return decisions

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Skip rows without human decision (unless it's explicitly empty)
                    if 'Human Decision' not in row:
                        continue

                    selector_type = row.get('Selector Type', '').strip()
                    selector_value = row.get('Selector Value', '').strip()

                    if not selector_type or not selector_value:
                        continue

                    pattern_id = f"{selector_type}|{selector_value}"

                    decisions[pattern_id] = {
                        'human_decision': row.get('Human Decision', '').strip(),
                        'notes': row.get('Notes', '').strip(),
                        'suggested_type': row.get('Suggested Type', '').strip(),
                        'frequency': self._safe_int(row.get('Estimated Frequency', '0')),
                        'evidence_score': self._safe_float(row.get('Evidence Score', '0')),
                        'sample_content': row.get('Sample Content', '').strip()
                    }

        except Exception as e:
            logger.error(f"Error reading {csv_file}: {e}")

        return decisions

    def _safe_int(self, value: str) -> int:
        """Safely convert string to int"""
        try:
            return int(float(value))  # Handle "123.0" format
        except (ValueError, TypeError):
            return 0

    def _safe_float(self, value: str) -> float:
        """Safely convert string to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _detect_significant_changes(self, new_candidate, old_data: Dict) -> str:
        """
        Detect significant changes between old và new pattern data
        Returns: change description string hoặc empty if no significant change
        """
        changes = []

        # Check frequency change
        old_freq = old_data.get('frequency', 0)
        new_freq = new_candidate.frequency
        if old_freq > 0:
            freq_change_ratio = abs(new_freq - old_freq) / old_freq
            if freq_change_ratio > self.frequency_change_threshold:
                changes.append(f"freq {old_freq}→{new_freq}")

        # Check evidence score change
        old_evidence = old_data.get('evidence_score', 0)
        new_evidence = new_candidate.evidence_score
        if abs(new_evidence - old_evidence) > self.evidence_change_threshold:
            changes.append(f"evidence {old_evidence:.2f}→{new_evidence:.2f}")

        if changes:
            return f"⚠️ CHANGED: {', '.join(changes)}"
        return ""

    def _create_changed_pattern_row(self, candidate, old_data: Dict, change_note: str) -> Dict:
        """Create CSV row cho pattern cần re-review do significant change"""
        return {
            'Pattern ID': candidate.identifier,
            'Selector Type': candidate.selector_type,
            'Selector Value': candidate.selector_value,
            'Estimated Frequency': candidate.frequency,
            'Evidence Score': f"{candidate.evidence_score:.3f}",
            'Sample Content': candidate.sample_content[:100],
            'Suggested Type': self._suggest_content_type(candidate),
            'Human Decision': '',  # Empty - needs new review
            'Notes': change_note,  # Mark the change clearly
            '_sort_priority': 2  # Medium priority for sorting
        }

    def _create_existing_pattern_row(self, candidate, old_data: Dict) -> Dict:
        """Create CSV row cho pattern giữ nguyên existing decision"""
        return {
            'Pattern ID': candidate.identifier,
            'Selector Type': candidate.selector_type,
            'Selector Value': candidate.selector_value,
            'Estimated Frequency': candidate.frequency,
            'Evidence Score': f"{candidate.evidence_score:.3f}",
            'Sample Content': candidate.sample_content[:100],
            'Suggested Type': old_data.get('suggested_type', ''),
            'Human Decision': old_data.get('human_decision', ''),  # Keep existing
            'Notes': old_data.get('notes', ''),  # Keep existing notes
            '_sort_priority': 3  # Lowest priority for sorting
        }

    def _create_new_pattern_row(self, candidate) -> Dict:
        """Create CSV row cho completely new pattern"""
        return {
            'Pattern ID': candidate.identifier,
            'Selector Type': candidate.selector_type,
            'Selector Value': candidate.selector_value,
            'Estimated Frequency': candidate.frequency,
            'Evidence Score': f"{candidate.evidence_score:.3f}",
            'Sample Content': candidate.sample_content[:100],
            'Suggested Type': self._suggest_content_type(candidate),
            'Human Decision': '',  # Empty - needs review
            'Notes': '',  # Empty
            '_sort_priority': 1  # Highest priority for sorting
        }

    def _suggest_content_type(self, candidate) -> str:
        """Suggest content type based on evidence (simplified version)"""
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

    def _pattern_sort_key(self, pattern_row: Dict) -> Tuple[int, float]:
        """
        Sort key cho patterns:
        1. Priority (1=new, 2=changed, 3=existing)
        2. Evidence score (descending)
        """
        priority = pattern_row.get('_sort_priority', 3)
        evidence_score = float(pattern_row.get('Evidence Score', 0))

        return (priority, -evidence_score)  # Negative for descending

    def _write_csv_file(self, patterns: List[Dict], filename: str) -> Path:
        """Write patterns to CSV file"""
        csv_path = self.output_dir / filename

        # Backup existing file if it exists
        if csv_path.exists():
            self.version_manager.backup_file(csv_path)

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
            'Notes'
        ]

        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

                for pattern in patterns:
                    # Remove internal fields before writing
                    clean_pattern = {k: v for k, v in pattern.items() if not k.startswith('_')}
                    writer.writerow(clean_pattern)

            logger.info(f"📝 CSV written: {csv_path}")

        except Exception as e:
            logger.error(f"❌ Error writing CSV: {e}")
            raise

        return csv_path

    def get_statistics(self) -> Dict:
        """Get statistics về existing decisions và files"""
        stats = {
            'total_files': 0,
            'total_decisions': 0,
            'decision_breakdown': defaultdict(int),
            'latest_version': 0
        }

        try:
            # Count files
            csv_files = self.version_manager.list_all_files("*patterns*.csv")
            stats['total_files'] = len(csv_files)

            # Find latest version
            versions = self.version_manager._find_existing_versions("patterns_for_review")
            if versions:
                stats['latest_version'] = max(versions)

            # Load all decisions và count by type
            all_decisions = self._load_all_existing_decisions()
            stats['total_decisions'] = len(all_decisions)

            for decision_data in all_decisions.values():
                decision_type = decision_data.get('human_decision', 'empty')
                if not decision_type:
                    decision_type = 'empty'
                stats['decision_breakdown'][decision_type] += 1

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")

        return dict(stats)


def main():
    """Test function cho SmartCSVManager"""
    print("🧪 Testing SmartCSVManager")
    print("=" * 50)

    # Create test manager
    manager = SmartCSVManager("test_output")

    # Get statistics
    stats = manager.get_statistics()
    print(f"📊 Current statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Test version manager
    vm = VersionManager("test_output")
    next_version = vm.get_next_version()
    print(f"\n📋 Next version would be: {next_version}")

    # List existing files
    existing_files = vm.list_all_files()
    print(f"\n📂 Existing files: {len(existing_files)}")
    for file_path in existing_files:
        print(f"   {file_path.name}")

    print("\n✅ SmartCSVManager test complete")


if __name__ == "__main__":
    main()