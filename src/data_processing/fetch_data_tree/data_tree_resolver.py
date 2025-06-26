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
        # ADD: Safe resolution attributes
        self.global_resolution_cache = {}  # type_name -> resolved_content (across all sessions)
        self.active_resolutions = set()  # Currently resolving types (cycle detection)
        self.max_resolution_depth = 3  # Maximum recursion depth
        self.max_total_api_calls = 1000  # Safety limit for total API calls
        self.current_api_call_count = 0  # Track API calls in current session

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
        """Call QuantConnect inspector API - UPDATED to track calls"""

        # Increment call counter
        self.current_api_call_count += 1

        # Convert language format
        lang_param = "python" if language.lower() in ["python", "py"] else "csharp"

        params = {
            'type': f'T:{data_tree_value}',
            'language': lang_param
        }

        try:
            logger.debug(
                f"Calling API for {data_tree_value} with language {lang_param} (call #{self.current_api_call_count})")

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

    # REPLACE method resolve_data_tree_safe trong data_tree_resolver.py

    def resolve_data_tree_safe(self, data_tree_value: str, language: str = "python", depth: int = 0) -> Optional[str]:
        """
        Safe version of resolve_data_tree with cycle detection and dependency resolution
        """
        # Normalize the data tree name
        normalized_name = self._normalize_data_tree_name(data_tree_value)
        cache_key = f"{normalized_name}_{language}"

        # Check global resolution cache first
        if cache_key in self.global_resolution_cache:
            logger.debug(f"Using global resolution cache for {normalized_name}")
            return self.global_resolution_cache[cache_key]

        # Safety check: Total API call limit
        if self.current_api_call_count >= self.max_total_api_calls:
            placeholder = f"[API Limit Reached: {normalized_name}]"
            logger.warning(f"API call limit reached ({self.max_total_api_calls})")
            return placeholder

        # Safety check: Depth limit
        if depth >= self.max_resolution_depth:
            placeholder = f"[Depth Limit: {normalized_name}]"
            logger.debug(f"Depth limit reached for {normalized_name} at depth {depth}")
            self.global_resolution_cache[cache_key] = placeholder
            return placeholder

        # Cycle detection: Check if currently resolving
        if normalized_name in self.active_resolutions:
            placeholder = f"[Circular Reference: {normalized_name}]"
            logger.debug(f"Circular reference detected for {normalized_name}")
            self.global_resolution_cache[cache_key] = placeholder
            return placeholder

        # Mark as currently resolving
        self.active_resolutions.add(normalized_name)

        try:
            # Call original resolution method
            content = self.resolve_data_tree(normalized_name, language)

            if content:
                # NOW IMPLEMENT: Resolve dependencies
                enhanced_content = self._resolve_dependencies_in_content(content, normalized_name, language, depth)

                # Cache the result
                self.global_resolution_cache[cache_key] = enhanced_content
                logger.info(f"Successfully resolved {normalized_name} at depth {depth}")
                return enhanced_content
            else:
                # Failed to resolve
                placeholder = f"[Resolution Failed: {normalized_name}]"
                self.global_resolution_cache[cache_key] = placeholder
                return placeholder

        except Exception as e:
            logger.error(f"Error in safe resolution for {normalized_name}: {str(e)}")
            placeholder = f"[Error: {normalized_name}]"
            self.global_resolution_cache[cache_key] = placeholder
            return placeholder

        finally:
            # Always remove from active resolutions
            self.active_resolutions.discard(normalized_name)

    # REPLACE method _resolve_expandable_links_safe với real implementation

    def _resolve_dependencies_in_content(self, content: str, parent_type: str, language: str, depth: int) -> str:
        """
        Find and resolve dependent types within content
        """
        if depth >= self.max_resolution_depth - 1:  # Leave room for one more level
            logger.debug(f"Skipping dependency resolution at depth {depth}")
            return content

        # Get the original API response to extract dependencies
        api_data = self._get_api_data_for_type(parent_type, language)
        if not api_data:
            return content

        # Extract dependent types
        dependent_types = self._extract_dependent_types_from_api_response(api_data)

        if not dependent_types:
            logger.debug(f"No dependent types found for {parent_type}")
            return content

        logger.debug(f"Found {len(dependent_types)} dependent types for {parent_type}: {dependent_types}")

        # Resolve each dependent type
        enhanced_content = content
        for dep_type in dependent_types:
            try:
                # Recursive resolution with increased depth
                resolved_dep = self.resolve_data_tree_safe(dep_type, language, depth + 1)

                if resolved_dep and not resolved_dep.startswith('['):
                    # Replace type name with expandable section
                    enhanced_content = self._inject_dependent_type_content(
                        enhanced_content, dep_type, resolved_dep, language
                    )
                    logger.debug(f"Successfully injected content for {dep_type}")
                else:
                    logger.debug(f"Failed to resolve dependent type {dep_type}: {resolved_dep}")

            except Exception as e:
                logger.error(f"Error resolving dependent type {dep_type}: {str(e)}")

        return enhanced_content

    def _get_api_data_for_type(self, type_name: str, language: str) -> Optional[Dict]:
        """Get API data for a type (from cache or fresh call)"""
        try:
            return self._call_api(type_name, language)
        except Exception as e:
            logger.error(f"Error getting API data for {type_name}: {str(e)}")
            return None

    def _inject_dependent_type_content(self, content: str, dep_type: str, resolved_content: str, language: str) -> str:
        """
        Inject resolved dependent type content into main content
        """
        # Extract just the type name for matching
        type_name = dep_type.split('.')[-1]

        # Pattern to find where this type is referenced
        # Look for: <p class="subsection-content">TypeName</p>
        pattern = f'<p class="subsection-content">{type_name}</p>'

        if pattern in content:
            # Create expandable section
            expandable_section = f'''<div class="dependent-type-section">
        <p class="subsection-content"><strong>{type_name}</strong> <em>(click to expand)</em></p>
        <div class="dependent-type-content" style="margin-left: 20px; border-left: 2px solid #ccc; padding-left: 10px;">
            {resolved_content}
        </div>
    </div>'''

            # Replace the simple type reference with expandable section
            content = content.replace(pattern, expandable_section)
            logger.debug(f"Injected expandable content for {type_name}")
        else:
            logger.debug(f"Could not find injection point for {type_name} in content")

        return content

    # UPDATE the existing _extract_dependent_types_from_api_response to be more thorough

    def _extract_dependent_types_from_api_response(self, api_data: Dict) -> List[str]:
        """
        Extract dependent type names from API response - UPDATED
        """
        dependent_types = set()
        current_type = api_data.get('full-type-name', '')

        # Check fields
        for field in api_data.get('fields', []):
            field_type = field.get('field-full-type-name', '')
            if field_type and field_type != current_type:
                dependent_types.add(field_type)

        # Check properties
        for prop in api_data.get('properties', []):
            # Try different property type fields
            prop_type = (prop.get('property-full-type-name', '') or
                         prop.get('property-type-name', '') or
                         prop.get('property-short-type-name', ''))

            # If it's a short type name, try to construct full name
            if prop_type and not '.' in prop_type and current_type:
                # Assume same namespace as current type
                namespace = '.'.join(current_type.split('.')[:-1])
                if namespace:
                    prop_type = f"{namespace}.{prop_type}"

            if prop_type and prop_type != current_type:
                dependent_types.add(prop_type)

        # Check methods (return types and parameter types)
        for method in api_data.get('methods', []):
            return_type = (method.get('method-return-full-type-name', '') or
                           method.get('method-return-type-name', '') or
                           method.get('method-return-short-type-name', ''))

            if return_type and return_type != current_type:
                dependent_types.add(return_type)

        # Filter out types we shouldn't resolve
        filtered_types = []
        for dep_type in dependent_types:
            if self._should_resolve_dependent_type(dep_type):
                filtered_types.append(dep_type)

        return filtered_types

    def _resolve_expandable_links_safe(self, content: str, language: str, depth: int) -> str:
        """
        Find and resolve expandable links within content safely

        Args:
            content: HTML content that may contain expandable links
            language: Target language
            depth: Current depth for nested resolution

        Returns:
            Enhanced content with resolved expandable links
        """
        if depth >= self.max_resolution_depth:
            logger.debug(f"Skipping expandable link resolution at depth {depth}")
            return content
        return content

    def _should_resolve_dependent_type(self, type_name: str) -> bool:
        """
        Determine if a dependent type should be resolved

        Args:
            type_name: Type name to check

        Returns:
            True if should resolve, False otherwise
        """
        # Skip system types
        system_prefixes = [
            'System.',
            'Microsoft.',
            'Newtonsoft.',
            'System.Collections.',
            'System.ComponentModel.'
        ]

        for prefix in system_prefixes:
            if type_name.startswith(prefix):
                return False

        # Skip very common QuantConnect types that tend to be circular
        circular_prone_types = [
            'QuantConnect.Algorithm',
            'QuantConnect.QCAlgorithm',
            'QuantConnect.Lean.Engine',
            'QuantConnect.Interfaces'
        ]

        if type_name in circular_prone_types:
            return False

        # Only resolve QuantConnect types
        if not type_name.startswith('QuantConnect.'):
            return False

        return True

    def get_safe_resolution_stats(self) -> Dict:
        """Get statistics about safe resolution"""
        return {
            'global_cache_size': len(self.global_resolution_cache),
            'active_resolutions': len(self.active_resolutions),
            'current_api_calls': self.current_api_call_count,
            'max_api_calls': self.max_total_api_calls,
            'max_depth': self.max_resolution_depth,
            'currently_resolving': list(self.active_resolutions) if self.active_resolutions else []
        }

    def reset_safe_resolution_state(self):
        """Reset safe resolution state (useful for testing)"""
        self.active_resolutions.clear()
        self.current_api_call_count = 0
        logger.info("Safe resolution state reset")


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