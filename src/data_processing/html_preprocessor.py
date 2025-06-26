#!/usr/bin/env python3
"""
HTML Preprocessor với Caching Support
Tách logic preprocess từ production_ready_discovery.py và thêm caching layer
"""

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

from cache_manager import CacheManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HTMLPreprocessor:
    """HTML Preprocessor với intelligent caching"""

    def __init__(self, input_dir="data/raw_html", cache_dir="data/processed_html"):
        self.input_dir = Path(input_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_manager = CacheManager(cache_dir)

        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_file(self, filename: str, force_refresh: bool = False) -> str:
        """
        Main preprocess function với caching logic

        Args:
            filename: Name of HTML file to preprocess
            force_refresh: Force reprocessing even if cache is valid

        Returns:
            Path to processed file (cached hoặc newly processed)
        """
        logger.info(f"🔄 Preprocessing file: {filename}")

        # Build file paths
        original_file_path = self.input_dir / filename

        if not original_file_path.exists():
            raise FileNotFoundError(f"Original file not found: {original_file_path}")

        # Generate cached file path với date
        cached_file_path = self.cache_manager.get_cached_file_path(str(original_file_path))

        # Check cache validity
        if not force_refresh and self.cache_manager.is_cache_valid(str(original_file_path), cached_file_path):
            logger.info(f"✅ Using cached file: {cached_file_path}")
            return cached_file_path

        # Cache invalid hoặc force refresh → preprocess
        logger.info(f"🔧 Preprocessing and caching: {filename}")

        try:
            # Read original content
            logger.info(f"📖 Reading original file: {original_file_path}")
            with open(original_file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            logger.info(f"📊 Original file size: {len(original_content):,} characters")

            # Preprocess content
            processed_content = self._preprocess_multi_html_content(original_content)

            logger.info(f"📊 Processed file size: {len(processed_content):,} characters")

            # Save to cache
            with open(cached_file_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)

            # Update cache metadata
            self.cache_manager.save_cache_metadata(str(original_file_path), cached_file_path)

            logger.info(f"✅ Cached processed file: {cached_file_path}")

            return cached_file_path

        except Exception as e:
            logger.error(f"❌ Preprocessing failed for {filename}: {e}")
            logger.error(f"💡 Consider checking file format or memory availability")
            raise

    def _preprocess_multi_html_content(self, content: str) -> str:
        """
        Core preprocess logic (moved từ production_ready_discovery.py)

        Xử lý QuantConnect HTML files chứa multiple HTML documents
        Combines them into single valid HTML document for parsing

        Args:
            content: Raw HTML content với multiple documents

        Returns:
            Processed HTML content as single valid document
        """
        logger.info("🔧 Processing multi-HTML content...")

        # Count patterns trước khi xử lý (for debugging)
        doctype_count = len(re.findall(r'<!DOCTYPE[^>]*>', content, flags=re.IGNORECASE))
        html_open_count = len(re.findall(r'<html[^>]*>', content, flags=re.IGNORECASE))

        logger.info(f"📊 Found {doctype_count} DOCTYPE declarations, {html_open_count} HTML tags")

        # Step 1: Remove all DOCTYPE declarations
        content = re.sub(r'<!DOCTYPE[^>]*>', '', content, flags=re.IGNORECASE)

        # Step 2: Remove opening html tags (preserve attributes if any)
        content = re.sub(r'<html[^>]*>', '', content, flags=re.IGNORECASE)

        # Step 3: Remove opening body tags (preserve attributes if any)
        content = re.sub(r'<body[^>]*>', '', content, flags=re.IGNORECASE)

        # Step 4: Remove closing html/body tags
        content = re.sub(r'</html>', '', content, flags=re.IGNORECASE)
        content = re.sub(r'</body>', '', content, flags=re.IGNORECASE)

        # NEW: Escape generic types với negative lookahead
        def escape_generic_types(match):
            full_match = match.group(0)
            escaped = full_match.replace('<', '&lt;').replace('>', '&gt;')
            return escaped

        # Pattern với negative lookahead để tránh closing tags
        generic_pattern = r'\b([A-Z]\w*)<(?!/)([^<>]*(?:<[^<>]+>[^<>]*)*)>'
        content = re.sub(generic_pattern, escape_generic_types, content)

        # Step 5: Wrap everything in single HTML structure
        processed_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>QuantConnect Documentation - Processed</title>
    <meta name="processed-by" content="HTMLPreprocessor">
    <meta name="processed-date" content="{datetime.now().isoformat()}">
</head>
<body>
{content}
</body>
</html>'''

        logger.info("✅ Multi-HTML processing complete")

        return processed_content

    def clean_old_caches(self, max_age_days: int = 7):
        """
        Clean up old cached files

        Args:
            max_age_days: Maximum age của cached files (default: 7 days)
        """
        logger.info(f"🧹 Cleaning caches older than {max_age_days} days...")

        try:
            cleaned_count = self.cache_manager.clean_old_caches(max_age_days)
            logger.info(f"✅ Cleaned {cleaned_count} old cached files")

        except Exception as e:
            logger.error(f"❌ Cache cleanup failed: {e}")

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics for monitoring

        Returns:
            Dict với cache statistics
        """
        try:
            return self.cache_manager.get_cache_stats()
        except Exception as e:
            logger.error(f"❌ Failed to get cache stats: {e}")
            return {}

    def invalidate_cache(self, filename: str):
        """
        Invalidate cache cho specific file

        Args:
            filename: Name của file cần invalidate cache
        """
        logger.info(f"🗑️ Invalidating cache for: {filename}")

        try:
            original_file_path = self.input_dir / filename
            self.cache_manager.invalidate_cache(str(original_file_path))
            logger.info(f"✅ Cache invalidated for: {filename}")

        except Exception as e:
            logger.error(f"❌ Cache invalidation failed for {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess HTML files with intelligent caching"
    )
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
        help="Đường dẫn tới file HTML (hoặc danh sách file), ví dụ: data/raw_html/Quanconnect-*.html"
    )
    parser.add_argument(
        "-f", "--force-refresh",
        action="store_true",
        help="Bỏ qua cache, luôn preprocess lại"
    )
    args = parser.parse_args()

    preprocessor = HTMLPreprocessor()

    for path_str in args.input:
        # lấy tên file, để phù hợp với logic self.input_dir
        filename = Path(path_str).name
        full_path = Path(path_str)
        print("\n" + "="*50)
        print(f"🔧 Xử lý file: {full_path}")
        start = datetime.now()
        try:
            out_path = preprocessor.preprocess_file(
                filename,
                force_refresh=args.force_refresh
            )
            took = (datetime.now() - start).total_seconds()
            print(f"✅ Hoàn thành trong {took:.2f}s → {out_path}")
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {filename}: {e}")


if __name__ == "__main__":
    main()