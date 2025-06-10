#!/usr/bin/env python3
"""
Cache Manager với Timestamp-based Validation
Handles caching logic cho HTML preprocessor
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """Cache management với timestamp validation và metadata tracking"""

    def __init__(self, cache_dir="data/processed_html"):
        self.cache_dir = Path(cache_dir)
        self.metadata_file = self.cache_dir / ".cache_metadata.json"

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing metadata
        self.metadata = self._load_metadata()

    def is_cache_valid(self, original_file: str, cached_file: str) -> bool:
        """
        Check if cached file is valid (newer than original)

        Args:
            original_file: Path to original file
            cached_file: Path to cached file

        Returns:
            True if cache is valid và newer than original
        """
        try:
            # Check if cached file exists
            if not os.path.exists(cached_file):
                logger.debug(f"Cache miss: {cached_file} does not exist")
                return False

            # Check if original file exists
            if not os.path.exists(original_file):
                logger.warning(f"Original file missing: {original_file}")
                return False

            # Compare timestamps
            original_mtime = os.path.getmtime(original_file)
            cached_mtime = os.path.getmtime(cached_file)

            # Cache is valid if cached file is newer
            is_valid = cached_mtime > original_mtime

            if is_valid:
                logger.debug(f"Cache hit: {cached_file} is newer than {original_file}")
            else:
                logger.debug(f"Cache stale: {cached_file} is older than {original_file}")

            return is_valid

        except Exception as e:
            logger.error(f"Error checking cache validity: {e}")
            return False

    def get_cached_file_path(self, original_file: str) -> str:
        """
        Generate cached file path với date-based naming

        Args:
            original_file: Path to original file

        Returns:
            Path to cached file với format: filename_YYYYMMDD.ext
        """
        original_path = Path(original_file)

        # Get current date for filename
        date_str = datetime.now().strftime("%Y%m%d")

        # Build cached filename: Writing-Algorithms_20250609.html
        stem = original_path.stem  # filename without extension
        suffix = original_path.suffix  # .html

        cached_filename = f"{stem}_{date_str}{suffix}"
        cached_file_path = self.cache_dir / cached_filename

        return str(cached_file_path)

    def save_cache_metadata(self, original_file: str, processed_file: str):
        """
        Save cache metadata for tracking

        Args:
            original_file: Path to original file
            processed_file: Path to processed/cached file
        """
        try:
            # Prepare metadata entry
            original_path = Path(original_file)
            processed_path = Path(processed_file)

            metadata_entry = {
                'original_file': str(original_path.absolute()),
                'processed_file': str(processed_path.absolute()),
                'original_size': original_path.stat().st_size if original_path.exists() else 0,
                'processed_size': processed_path.stat().st_size if processed_path.exists() else 0,
                'original_mtime': os.path.getmtime(original_file) if os.path.exists(original_file) else 0,
                'processed_mtime': os.path.getmtime(processed_file) if os.path.exists(processed_file) else 0,
                'created_at': datetime.now().isoformat(),
            }

            # Use original filename as key
            key = original_path.name
            self.metadata[key] = metadata_entry

            # Save metadata
            self._save_metadata()

            logger.debug(f"Metadata saved for: {key}")

        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def invalidate_cache(self, original_file: str):
        """
        Remove cached file và metadata entry

        Args:
            original_file: Path to original file
        """
        try:
            original_path = Path(original_file)
            key = original_path.name

            # Remove metadata entry
            if key in self.metadata:
                # Get cached file path from metadata
                cached_file = self.metadata[key].get('processed_file')

                # Remove cached file if exists
                if cached_file and os.path.exists(cached_file):
                    os.remove(cached_file)
                    logger.info(f"Removed cached file: {cached_file}")

                # Remove metadata entry
                del self.metadata[key]
                self._save_metadata()

                logger.info(f"Cache invalidated for: {key}")
            else:
                logger.warning(f"No cache metadata found for: {key}")

        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")

    def clean_old_caches(self, max_age_days: int = 7) -> int:
        """
        Clean up cached files older than max_age_days

        Args:
            max_age_days: Maximum age của cached files

        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        try:
            # Get all cached files in directory
            cached_files = list(self.cache_dir.glob("*.html"))

            for cached_file in cached_files:
                try:
                    # Check file age
                    file_mtime = datetime.fromtimestamp(cached_file.stat().st_mtime)

                    if file_mtime < cutoff_date:
                        # Remove old cached file
                        cached_file.unlink()
                        cleaned_count += 1
                        logger.debug(f"Removed old cache: {cached_file}")

                        # Clean up metadata for removed file
                        self._cleanup_orphaned_metadata()

                except Exception as e:
                    logger.error(f"Error removing cached file {cached_file}: {e}")

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old cached files")

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")

        return cleaned_count

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics for monitoring

        Returns:
            Dict với cache statistics
        """
        try:
            stats = {
                'cache_dir': str(self.cache_dir),
                'total_cached_files': len(list(self.cache_dir.glob("*.html"))),
                'metadata_entries': len(self.metadata),
                'cache_dir_size_mb': self._get_directory_size_mb(),
                'oldest_cache': self._get_oldest_cache_date(),
                'newest_cache': self._get_newest_cache_date(),
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}

    def list_cached_files(self) -> List[Dict]:
        """
        List all cached files với metadata

        Returns:
            List của cached file information
        """
        cached_files = []

        try:
            for key, metadata in self.metadata.items():
                cached_files.append({
                    'original_name': key,
                    'cached_file': metadata.get('processed_file', ''),
                    'original_size_mb': round(metadata.get('original_size', 0) / 1024 / 1024, 2),
                    'processed_size_mb': round(metadata.get('processed_size', 0) / 1024 / 1024, 2),
                    'created_at': metadata.get('created_at', ''),
                    'exists': os.path.exists(metadata.get('processed_file', ''))
                })

        except Exception as e:
            logger.error(f"Error listing cached files: {e}")

        return cached_files

    def _load_metadata(self) -> Dict:
        """Load metadata từ JSON file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}

    def _save_metadata(self):
        """Save metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def _cleanup_orphaned_metadata(self):
        """Remove metadata entries cho files that no longer exist"""
        orphaned_keys = []

        for key, metadata in self.metadata.items():
            cached_file = metadata.get('processed_file', '')
            if cached_file and not os.path.exists(cached_file):
                orphaned_keys.append(key)

        for key in orphaned_keys:
            del self.metadata[key]
            logger.debug(f"Removed orphaned metadata: {key}")

        if orphaned_keys:
            self._save_metadata()

    def _get_directory_size_mb(self) -> float:
        """Calculate total size của cache directory in MB"""
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
            return round(total_size / 1024 / 1024, 2)
        except Exception:
            return 0.0

    def _get_oldest_cache_date(self) -> Optional[str]:
        """Get date của oldest cached file"""
        try:
            cached_files = list(self.cache_dir.glob("*.html"))
            if not cached_files:
                return None

            oldest_file = min(cached_files, key=lambda f: f.stat().st_mtime)
            oldest_date = datetime.fromtimestamp(oldest_file.stat().st_mtime)
            return oldest_date.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return None

    def _get_newest_cache_date(self) -> Optional[str]:
        """Get date của newest cached file"""
        try:
            cached_files = list(self.cache_dir.glob("*.html"))
            if not cached_files:
                return None

            newest_file = max(cached_files, key=lambda f: f.stat().st_mtime)
            newest_date = datetime.fromtimestamp(newest_file.stat().st_mtime)
            return newest_date.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return None


def main():
    """Test function cho CacheManager"""

    print("🧪 Testing CacheManager")
    print("=" * 50)

    cache_manager = CacheManager()

    # Test file paths
    test_original = "data/raw_html/test.html"
    test_cached = cache_manager.get_cached_file_path(test_original)

    print(f"📁 Cache directory: {cache_manager.cache_dir}")
    print(f"📄 Test original: {test_original}")
    print(f"📄 Test cached: {test_cached}")

    # Test cache validity
    print(f"\n🔍 Cache validity check:")
    is_valid = cache_manager.is_cache_valid(test_original, test_cached)
    print(f"   Cache valid: {is_valid}")

    # Show cache stats
    print(f"\n📊 Cache Statistics:")
    stats = cache_manager.get_cache_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # List cached files
    print(f"\n📋 Cached Files:")
    cached_files = cache_manager.list_cached_files()
    if cached_files:
        for file_info in cached_files:
            print(f"   {file_info['original_name']} → {file_info['processed_size_mb']}MB")
    else:
        print("   No cached files found")


if __name__ == "__main__":
    main()