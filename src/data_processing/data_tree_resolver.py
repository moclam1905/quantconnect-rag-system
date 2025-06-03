"""
DataTreeResolver - Fetch and format content for data-tree elements
Calls QuantConnect API to get real content for data-tree placeholders
"""

import requests
import json
import time
from typing import Dict, Optional, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import logger


class DataTreeResolver:
    """
    Resolves data-tree values by calling QuantConnect inspector API
    """

    def __init__(self):
        self.api_base_url = "https://www.quantconnect.com/services/inspector"
        self.cache = {}  # Cache API responses
        self.session = requests.Session()

        # Set headers to mimic browser request
        self.session.headers.update({
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'x-requested-with': 'XMLHttpRequest'
        })

    def resolve_data_tree(self, data_tree_value: str, language: str = "python") -> Optional[str]:
        """
        Resolve data-tree value to formatted HTML content

        Args:
            data_tree_value: e.g., "QuantConnect.Resolution"
            language: "python" or "csharp"

        Returns:
            Formatted HTML content or None if failed
        """
        # Normalize the data tree name first
        normalized_name = self._normalize_data_tree_name(data_tree_value)

        cache_key = f"{normalized_name}_{language}"

        # Check cache first
        if cache_key in self.cache:
            logger.debug(f"Using cached result for {normalized_name}")
            return self.cache[cache_key]

        try:
            # Call API with normalized name
            api_data = self._call_api(normalized_name, language)

            if not api_data or not api_data.get('success'):
                error_msg = api_data.get('error', 'Unknown error') if api_data else 'No response'
                logger.warning(f"API call failed for {normalized_name}: {error_msg}")
                self.cache[cache_key] = None
                return None

            # Format response to HTML
            formatted_content = self._format_api_response(api_data, language)

            # Cache result
            self.cache[cache_key] = formatted_content

            logger.info(f"Successfully resolved {normalized_name} for {language}")
            return formatted_content

        except Exception as e:
            logger.error(f"Error resolving {normalized_name}: {str(e)}")
            self.cache[cache_key] = None
            return None

    def _normalize_data_tree_name(self, data_tree_value: str) -> str:
        """
        Fix known incorrect data-tree names.
        """
        # Handle the specific case
        if data_tree_value == 'QuantConnect.Data.EODHD.MacroIndicators':
            logger.debug(f"Mapping {data_tree_value} -> QuantConnect.DataSource.EODHDMacroIndicator")
            return 'QuantConnect.DataSource.EODHDMacroIndicator'

        # Return original if no mapping needed
        return data_tree_value

    def _call_api(self, data_tree_value: str, language: str) -> Optional[Dict]:
        """Call QuantConnect inspector API"""

        # Convert language format
        lang_param = "python" if language.lower() in ["python", "py"] else "csharp"

        params = {
            'type': f'T:{data_tree_value}',
            'language': lang_param
        }

        try:
            logger.debug(f"Calling API for {data_tree_value} with language {lang_param}")

            response = self.session.get(
                self.api_base_url,
                params=params,
                timeout=10
            )

            response.raise_for_status()

            # Add small delay to be respectful to API
            time.sleep(0.1)

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error calling API for {data_tree_value}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {data_tree_value}: {str(e)}")
            return None

    def _format_api_response(self, json_data: Dict, language: str) -> str:
        """Convert JSON response to formatted HTML content"""

        parts = []

        # Get basic info
        type_name = json_data.get('type-name', '')
        full_type_name = json_data.get('full-type-name', '')
        description = json_data.get('description', '')
        base_type = json_data.get('base-type-full-name', '')

        # Header with type info
        if type_name:
            if 'Enum' in base_type:
                parts.append(f'<h4>{type_name} Enumeration</h4>')
            else:
                parts.append(f'<h4>{type_name}</h4>')

        # Type declaration
        if full_type_name:
            if language.lower() in ["python", "py"]:
                lang_class = "python"
            else:
                lang_class = "csharp"

            object_type = "enum" if 'Enum' in base_type else "class"
            parts.append(
                f'<div class="code-snippet"><span class="object-type">{object_type}</span> <code>{full_type_name}</code></div>')

        # Description
        if description:
            parts.append(f'<p>{description}</p>')

        # Content container
        content_parts = []

        # Process fields (for enums mainly)
        fields = json_data.get('fields', [])
        if fields:
            for field in fields:
                field_name = field.get('field-name', '')
                field_desc = field.get('field-description', '')

                if field_name and field_desc:
                    content_parts.append(
                        f'<div class="code-snippet"><span class="object-type">field</span> <code>{field_name}</code></div>')
                    content_parts.append('<div class="subsection-content">')
                    content_parts.append(f'<p>{field_desc}</p>')
                    content_parts.append('</div>')

        # Process properties
        properties = json_data.get('properties', [])
        if properties:
            for prop in properties:
                prop_name = prop.get('property-name', '')
                prop_desc = prop.get('property-description', '')
                prop_type = prop.get('property-short-type-name', '')

                if prop_name:
                    content_parts.append(
                        f'<div class="code-snippet"><span class="object-type">property</span> <code>{prop_name}</code></div>')
                    content_parts.append('<div class="subsection-content">')
                    if prop_desc:
                        content_parts.append(f'<p>{prop_desc}</p>')
                    if prop_type:
                        content_parts.append('<div class="subsection-header">Type:</div>')
                        content_parts.append(f'<p class="subsection-content">{prop_type}</p>')
                    content_parts.append('</div>')

        # Process methods
        methods = json_data.get('methods', [])
        if methods:
            for method in methods:
                method_name = method.get('method-name', '')
                method_desc = method.get('method-description', '')
                return_type = method.get('method-return-short-type-name', '')

                if method_name:
                    content_parts.append(
                        f'<div class="code-snippet"><span class="object-type">method</span> <code>{method_name}</code></div>')
                    content_parts.append('<div class="subsection-content">')
                    if method_desc:
                        content_parts.append(f'<p>{method_desc}</p>')
                    if return_type:
                        content_parts.append('<div class="subsection-header">Returns:</div>')
                        content_parts.append(f'<p class="subsection-content">{return_type}</p>')
                    content_parts.append('</div>')

        # Wrap content in container if we have any
        if content_parts:
            parts.append('<div class="subsection-content">')
            parts.extend(content_parts)
            parts.append('</div>')

        return '\n'.join(parts)

    def resolve_both_languages(self, data_tree_value: str) -> Dict[str, Optional[str]]:
        """
        Resolve data-tree for both Python and C# languages

        Returns:
            Dict with 'python' and 'csharp' keys
        """
        return {
            'python': self.resolve_data_tree(data_tree_value, 'python'),
            'csharp': self.resolve_data_tree(data_tree_value, 'csharp')
        }

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total = len(self.cache)
        successful = sum(1 for v in self.cache.values() if v is not None)
        failed = total - successful

        return {
            'total_calls': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0
        }


# Test function
if __name__ == "__main__":
    resolver = DataTreeResolver()

    # Test with QuantConnect.Resolution
    print("Testing DataTreeResolver...")

    result = resolver.resolve_data_tree("QuantConnect.Resolution", "python")
    if result:
        print("✅ Successfully resolved QuantConnect.Resolution")
        print("Content preview:")
        print(result[:200] + "..." if len(result) > 200 else result)
    else:
        print("❌ Failed to resolve QuantConnect.Resolution")

    # Test cache
    print(f"\nCache stats: {resolver.get_cache_stats()}")