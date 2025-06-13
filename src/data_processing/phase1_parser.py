#!/usr/bin/env python3
"""
Phase 1: Core Parser Infrastructure - FIXED with Breadcrumb-based Strategy
Fixed BeautifulSoup syntax và implemented reliable content extraction
"""

import json
import logging
import hashlib
import os
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup, Tag

# Import existing components
from html_preprocessor import HTMLPreprocessor
from cache_manager import CacheManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SectionInfo:
    """Represents a document section with internal links tracking"""
    id: str
    title: str
    level: int
    order: int
    parent_id: Optional[str]
    children_ids: List[str]
    toc_element_id: Optional[str]
    internal_links: Dict[str, str] = None

    def __post_init__(self):
        if self.internal_links is None:
            self.internal_links = {}


@dataclass
class ContentItem:
    """Represents a single content item"""
    id: str
    content_type: str
    text: str
    metadata: Dict
    order: int
    confidence: float


class StructureExtractor:
    """Extract ToC structure and document hierarchy with internal links tracking"""

    def __init__(self):
        self.toc_classes = ['toc-h1', 'toc-h2', 'toc-h3', 'toc-h4', 'toc-h5', 'toc-h6']

        # Skip sections
        self.skip_sections = {
            "24",  # Migrations + children
            "2.1.3.2",  # Rendering Data with CSharp
            "5.8.1"  # Third-Party Libraries
        }

    def extract_toc_hierarchy(self, soup: BeautifulSoup) -> Tuple[List[SectionInfo], Dict[str, str]]:
        """
        Extract ToC hierarchy with FIXED section IDs and internal links tracking
        """
        logger.info("🗂️ Extracting ToC structure with breadcrumb validation...")

        sections = []
        section_mapping = {}
        section_counter = 1
        skipped_sections = []

        # Stack để track parent-child relationships
        parent_stack = []

        # NEW: Track internal links
        internal_links_map = self._extract_internal_links(soup)
        logger.info(f"🔗 Found {len(internal_links_map)} internal links")

        # VALIDATION: Count breadcrumbs và sections
        breadcrumbs = soup.find_all('p', class_='page-breadcrumb')
        all_sections_tags = soup.find_all('section', id=True)
        logger.info(f"📊 Validation: {len(breadcrumbs)} breadcrumbs, {len(all_sections_tags)} section tags")

        try:
            # Find all ToC elements
            toc_elements = []
            for toc_class in self.toc_classes:
                elements = soup.find_all(class_=toc_class)
                for elem in elements:
                    level = int(toc_class.split('-h')[1])
                    toc_elements.append((elem, level))

            # Sort by document order
            toc_elements.sort(key=lambda x: self._get_element_position(x[0]))
            logger.info(f"📋 Found {len(toc_elements)} ToC entries")

            for elem, level in toc_elements:
                # Extract section number from href
                section_number = self._extract_section_number(elem)
                title = self._extract_title_text(elem)

                # Check if should skip this section
                if self._should_skip_section(section_number, title, parent_stack):
                    skipped_sections.append(f"{section_number} - {title}")
                    continue

                # Generate section ID to match HTML id format
                section_id = f"section_{section_number}" if section_number else f"section_{section_counter}"

                # Handle hierarchy
                parent_id = self._find_parent_id(level, parent_stack)

                # Create section
                section = SectionInfo(
                    id=section_id,
                    title=title,
                    level=level,
                    order=section_counter,
                    parent_id=parent_id,
                    children_ids=[],
                    toc_element_id=section_number
                )

                sections.append(section)

                # Update parent's children list
                if parent_id:
                    for s in sections:
                        if s.id == parent_id:
                            s.children_ids.append(section_id)
                            break

                # Update parent stack
                self._update_parent_stack(parent_stack, section, level, section_number, title)

                # Create section mapping with better filename
                filename = f"{section_number.replace('.', '_')}_{self._slugify(title)}.json" if section_number else f"{section_counter:03d}_{self._slugify(title)}.json"
                section_mapping[section_id] = filename

                section_counter += 1

        except Exception as e:
            logger.error(f"❌ Error extracting ToC: {e}")
            # Fallback: create single section
            sections = [SectionInfo("section_1", "Document", 1, 1, None, [], "1")]
            section_mapping = {"section_1": "001_document.json"}

        logger.info(f"✅ Extracted {len(sections)} sections")
        if skipped_sections:
            logger.info(f"🚫 Skipped {len(skipped_sections)} sections:")
            for skipped in skipped_sections:
                logger.info(f"   - {skipped}")

        # Add internal links to sections
        self._add_internal_links_to_sections(sections, internal_links_map)

        return sections, section_mapping

    def _extract_internal_links(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract all internal links and map to target sections"""
        internal_links = {}

        # Find all internal links
        internal_link_elements = soup.find_all('a', href=lambda x: x and x.startswith('#'))

        # Find all sections to create reverse mapping
        all_sections = soup.find_all('section', id=True)
        section_ids = [s.get('id') for s in all_sections]

        for link in internal_link_elements:
            href = link.get('href')
            if not href:
                continue

            target_id = href[1:]  # Remove '#' prefix

            # Try to find which section this link points to
            target_section_id = self._find_target_section_for_link(target_id, section_ids)

            if target_section_id:
                mapped_section_id = f"section_{target_section_id}"
                internal_links[href] = mapped_section_id

        return internal_links

    def _find_target_section_for_link(self, target_id: str, section_ids: List[str]) -> Optional[str]:
        """Find which section a link target belongs to"""
        # Direct match with section ID
        if target_id in section_ids:
            return target_id

        # For other targets, try to find containing section
        for section_id in sorted(section_ids, key=lambda x: len(x.split('.')), reverse=True):
            if target_id.startswith(section_id.replace('.', '-')):
                return section_id

        return None

    def _add_internal_links_to_sections(self, sections: List[SectionInfo], internal_links_map: Dict[str, str]):
        """Add internal_links to SectionInfo objects"""
        section_links = defaultdict(list)

        for link_href, target_section_id in internal_links_map.items():
            section_links[target_section_id].append(link_href)

        for section in sections:
            section.internal_links = {}
            if section.id in section_links:
                for link_href in section_links[section.id]:
                    section.internal_links[link_href] = section.id

    # Keep existing helper methods
    def _get_element_position(self, elem) -> int:
        try:
            position = 0
            for prev in elem.previous_elements:
                if hasattr(prev, 'name'):
                    position += 1
            return position
        except:
            return 0

    def _extract_title_text(self, elem) -> str:
        try:
            title = elem.get_text(strip=True)
            title = re.sub(r'^[\d\.]+\s*', '', title)
            return title[:100] if title else "Untitled"
        except:
            return "Untitled"

    def _find_parent_id(self, current_level: int, parent_stack: List) -> Optional[str]:
        while parent_stack and parent_stack[-1]['level'] >= current_level:
            parent_stack.pop()
        return parent_stack[-1]['section_id'] if parent_stack else None

    def _extract_section_number(self, elem) -> str:
        try:
            href = elem.get('href', '')
            if href.startswith('#'):
                return href[1:]

            text = elem.get_text(strip=True)
            match = re.match(r'^([\d\.]+)', text)
            if match:
                return match.group(1)
            return ""
        except:
            return ""

    def _should_skip_section(self, section_number: str, title: str, parent_stack: List) -> bool:
        if not section_number:
            return False

        if title.strip() == "Market Hours":
            return True

        if section_number in self.skip_sections:
            return True

        for parent_info in parent_stack:
            parent_section = parent_info.get('section_number', '')
            if parent_section in self.skip_sections:
                return True
            parent_title = parent_info.get('title', '')
            if parent_title.strip() == "Market Hours":
                return True

        return False

    def _update_parent_stack(self, parent_stack: List, section: SectionInfo, level: int, section_number: str,
                             title: str):
        while parent_stack and parent_stack[-1]['level'] >= level:
            parent_stack.pop()

        parent_stack.append({
            'section_id': section.id,
            'level': level,
            'title': title,
            'section_number': section_number
        })

    def _slugify(self, text: str) -> str:
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        slug = re.sub(r'[-\s]+', '_', slug)
        return slug[:50]


class ContentClassifier:
    """Classify HTML content using existing rules"""

    def __init__(self, rules_file: str = "pattern_rules.yaml"):
        # FIXED: Handle relative path from src/data_processing/
        if not os.path.exists(rules_file):
            # Try project root
            project_root_path = os.path.join("..", "..", rules_file)
            if os.path.exists(project_root_path):
                rules_file = project_root_path
            else:
                logger.warning(f"⚠️ Rules file not found: {rules_file}")

        self.rules = self._load_rules(rules_file)
        # ADD DEBUG:
        logger.info(f"🔍 DEBUG Rules: {type(self.rules)}")
        logger.info(f"🔍 DEBUG Rules keys: {list(self.rules.keys()) if self.rules else 'None'}")
        logger.info(
            f"🔍 DEBUG Skip rules: {self.rules.get('skip_content', 'Missing') if self.rules else 'Rules is None'}")

    def _load_rules(self, rules_file: str) -> Dict:
        try:
            import yaml
            logger.info(f"📋 Loading rules from: {rules_file}")
            with open(rules_file, 'r', encoding='utf-8') as f:
                rules = yaml.safe_load(f) or {}
            logger.info(f"✅ Rules loaded successfully: {list(rules.keys())}")
            return rules
        except Exception as e:
            logger.error(f"❌ Could not load rules from {rules_file}: {e}")
            return {}

    def classify_element(self, elem: Tag) -> Tuple[str, float]:
        """Classify single HTML element with fixed integration"""
        try:
            if elem is None or not hasattr(elem, 'name'):
                return 'documentation_text', 0.3

            if not self.rules:
                logger.warning("⚠️ Rules not loaded, using fallback classification")
                return self._fallback_classification(elem)

            # Fix classes format for compatibility
            original_classes = elem.get('class', [])

            if isinstance(original_classes, list):
                elem.attrs['class'] = ' '.join(original_classes)
            elif original_classes is None:
                elem.attrs['class'] = ''

            # Try using existing classify_content_type function
            try:
                from production_ready_discovery import classify_content_type
                # Right before calling classify_content_type:
                logger.info(f"🔍 DEBUG: Calling classify_content_type with rules type: {type(self.rules)}")
                logger.info(f"🔍 DEBUG: Rules is None: {self.rules is None}")

                if elem.name == 'img':
                    logger.debug(f"🔍 DEBUG: About to classify IMG element")
                    logger.debug(f"   Classes: {elem.get('class')}")
                    logger.debug(f"   Attributes: {list(elem.attrs.keys())}")

                # Try using existing classify_content_type function
                content_type = classify_content_type(elem, self.rules)

                # ADD DEBUG AFTER:
                if elem.name == 'img':
                    logger.debug(f"🔍 DEBUG: IMG classified as: {content_type}")
            except Exception as classify_error:
                logger.warning(f"⚠️ classify_content_type failed: {classify_error}")
                if isinstance(original_classes, list):
                    elem.attrs['class'] = original_classes
                return self._fallback_classification(elem)

            # Restore original classes
            if isinstance(original_classes, list):
                elem.attrs['class'] = original_classes

            # Calculate confidence
            confidence = self._calculate_confidence(elem, content_type)

            # Skip content should be filtered out
            if content_type == 'SKIP':
                return None, 0.0

            return content_type, confidence

        except Exception as e:
            logger.warning(f"⚠️ Classification error: {e}")
            return self._fallback_classification(elem)

    def _fallback_classification(self, elem: Tag) -> Tuple[str, float]:
        try:
            classes = elem.get('class', [])
            tag = elem.name.lower()

            # QUICK FIX: Skip media elements and other unwanted tags
            skip_tags = ['img', 'video', 'audio', 'iframe', 'embed', 'object', 'canvas',
                        'script', 'style', 'link', 'meta']
            if tag in skip_tags:
                logger.debug(f"🚫 FALLBACK SKIP: {tag} element")
                return None, 0.0

            # Skip elements with media-related classes
            if isinstance(classes, list):
                classes_str = ' '.join(classes).lower()
            else:
                classes_str = str(classes).lower()

            skip_class_indicators = ['image', 'img', 'video', 'audio', 'media', 'gif', 'animation',
                                   'player', 'embed', 'cover-icon', 'cover-image', 'chart', 'graph', 'diagram']
            if any(indicator in classes_str for indicator in skip_class_indicators):
                logger.debug(f"🚫 FALLBACK SKIP: element with classes {classes}")
                return None, 0.0

            # Code content
            if any(indicator in classes_str for indicator in ['code', 'highlight', 'python', 'csharp', 'example']):
                return 'code_content', 0.8

            # API reference
            if 'data-tree' in elem.attrs or any(indicator in classes_str for indicator in ['api', 'reference', 'tree']):
                return 'api_reference', 0.8

            # Table content
            if tag == 'table' or 'table' in classes_str:
                return 'table_content', 0.8

            # Navigation
            if any(indicator in classes_str for indicator in ['toc', 'nav', 'breadcrumb']):
                return 'navigation_content', 0.8

            return 'documentation_text', 0.7

        except Exception as e:
            logger.warning(f"⚠️ Fallback classification failed: {e}")
            return 'documentation_text', 0.5

    def _calculate_confidence(self, elem: Tag, content_type: str) -> float:
        confidence = 0.8

        classes = elem.get('class', [])

        if content_type == 'code_content':
            code_indicators = ['code', 'highlight', 'python', 'csharp', 'example']
            if any(indicator in ' '.join(classes).lower() for indicator in code_indicators):
                confidence = 0.95

        elif content_type == 'api_reference':
            if 'data-tree' in elem.attrs or any('tree' in cls for cls in classes):
                confidence = 0.95

        elif content_type == 'table_content':
            if elem.name == 'table' or any('table' in cls for cls in classes):
                confidence = 0.9

        elif content_type == 'navigation_content':
            nav_indicators = ['toc', 'breadcrumb', 'nav']
            if any(indicator in ' '.join(classes).lower() for indicator in nav_indicators):
                confidence = 0.9

        return confidence


class ContentProcessor:
    """FIXED: Process content using breadcrumb-based boundary detection"""

    def __init__(self, classifier: ContentClassifier):
        self.classifier = classifier
        self.content_counter = 1
        self.all_breadcrumb_elements = []  # NEW: Store breadcrumb elements for dynamic matching

    def process_all_sections_content(self, soup: BeautifulSoup, sections: List[SectionInfo]) -> Dict[
        str, List[ContentItem]]:
        """
        FIXED v3: Dynamic breadcrumb matching content extraction

        Strategy:
        1. Get all breadcrumb positions in document
        2. Dynamically match sections to breadcrumbs (NO HARD-CODE!)
        3. Extract content between consecutive breadcrumb positions
        """
        logger.info("🔄 Starting dynamic breadcrumb matching extraction...")

        # Step 1: Get all breadcrumb positions
        breadcrumb_positions = self._get_all_breadcrumb_positions(soup)
        logger.info(f"📊 Found {len(breadcrumb_positions)} breadcrumb positions")

        # Step 2: Extract content for each section
        all_sections_content = {}

        for section in sections:
            section_id = section.id
            logger.info(f"📝 Processing section '{section.title}' (id: {section_id})")

            try:
                # Find content for this section using position-based approach
                content_elements = self._extract_section_content_by_position(
                    soup, section, breadcrumb_positions
                )

                content_items = []
                for elem in content_elements:
                    try:
                        content_type, confidence = self.classifier.classify_element(elem)

                        if content_type is None or content_type == 'SKIP':
                            continue

                        text_content = self._extract_text_content(elem, content_type)

                        if not text_content.strip():
                            continue

                        content_item = ContentItem(
                            id=f"content_{self.content_counter:06d}",
                            content_type=content_type,
                            text=text_content,
                            metadata={
                                'order': len(content_items) + 1,
                                'html_selector': self._generate_selector(elem),
                                'confidence': confidence,
                                'token_count': self._estimate_token_count(text_content),
                                'char_count': len(text_content),
                                'hash_id': self._generate_hash(text_content),
                                'language': self._detect_language(elem, content_type),
                                'section_id': section_id
                            },
                            order=len(content_items) + 1,
                            confidence=confidence
                        )

                        content_items.append(content_item)
                        self.content_counter += 1

                    except Exception as e:
                        logger.warning(f"⚠️ Error processing element in {section.title}: {e}")
                        continue

                all_sections_content[section_id] = content_items

                if content_items:
                    logger.info(f"✅ Section '{section.title}': {len(content_items)} content items")
                else:
                    logger.warning(f"🚫 Section '{section.title}': No content items")

            except Exception as e:
                logger.error(f"❌ Error processing section '{section.title}': {e}")
                all_sections_content[section_id] = []

        return all_sections_content

    def _get_all_breadcrumb_positions(self, soup: BeautifulSoup) -> Dict[str, Dict]:
        """
        Get positions of all breadcrumbs in document order

        Returns:
            {breadcrumb_text: {element: Tag, position: int, next_position: int}}
        """
        logger.info("🗺️ Getting all breadcrumb positions...")

        breadcrumb_positions = {}
        all_breadcrumbs = soup.find_all('p', class_='page-breadcrumb')

        # Store breadcrumb elements for dynamic matching
        self.all_breadcrumb_elements = all_breadcrumbs  # NEW: Store for later use

        # Get document position for each breadcrumb
        all_elements = soup.find_all(True)  # All tags in document order

        for breadcrumb in all_breadcrumbs:
            try:
                # Find position in document
                position = all_elements.index(breadcrumb)
                breadcrumb_text = breadcrumb.get_text(strip=True)

                breadcrumb_positions[breadcrumb_text] = {
                    'element': breadcrumb,
                    'position': position,
                    'next_position': None  # Will be filled later
                }

                logger.debug(f"   Breadcrumb '{breadcrumb_text}' at position {position}")

            except ValueError:
                logger.warning(f"⚠️ Could not find position for breadcrumb: {breadcrumb.get_text(strip=True)}")

        # Sort by position và set next_position
        sorted_breadcrumbs = sorted(breadcrumb_positions.items(), key=lambda x: x[1]['position'])

        for i, (breadcrumb_text, data) in enumerate(sorted_breadcrumbs):
            if i + 1 < len(sorted_breadcrumbs):
                next_breadcrumb_text, next_data = sorted_breadcrumbs[i + 1]
                data['next_position'] = next_data['position']
            else:
                data['next_position'] = len(all_elements)  # End of document

        logger.info(f"📊 Mapped {len(breadcrumb_positions)} breadcrumb positions")
        return breadcrumb_positions

    def _extract_section_content_by_position(self, soup: BeautifulSoup, section: SectionInfo,
                                             breadcrumb_positions: Dict) -> List[Tag]:
        """
        FIXED v3: Extract content using dynamic breadcrumb matching

        Logic:
        1. Dynamically find breadcrumb text for this section (NO HARD-CODE!)
        2. Get start/end positions from breadcrumb_positions
        3. Extract elements between positions
        """
        content_elements = []

        try:
            # Map section title to breadcrumb text
            section_breadcrumb_text = self._map_section_to_breadcrumb_text(section)

            if not section_breadcrumb_text:
                logger.warning(f"⚠️ Could not map section '{section.title}' to breadcrumb")
                return content_elements

            if section_breadcrumb_text not in breadcrumb_positions:
                logger.warning(f"⚠️ Breadcrumb '{section_breadcrumb_text}' not found in positions")
                return content_elements

            # Get position boundaries
            start_position = breadcrumb_positions[section_breadcrumb_text]['position']
            end_position = breadcrumb_positions[section_breadcrumb_text]['next_position']

            logger.debug(f"🔍 Section '{section.title}': positions {start_position} → {end_position}")

            # Extract content between positions
            all_elements = soup.find_all(True)
            section_elements = all_elements[start_position:end_position]

            logger.debug(f"🔍 DEBUG: Section elements count: {len(section_elements)}")
            for i, elem in enumerate(section_elements[:10]):  # First 10 elements
                if elem.name == 'img':
                    logger.debug(f"   IMG#{i}: classes={elem.get('class')}, text_len={len(elem.get_text())}")
                    logger.debug(f"   IMG#{i}: next_sibling_type={type(elem.next_sibling)}")
                    logger.debug(f"   IMG#{i}: src_length={len(elem.get('src', ''))}")

            # Filter for content elements (skip breadcrumb, page-heading)
            for elem in section_elements:
                if self._is_content_element(elem):
                    content_elements.append(elem)

            logger.debug(f"📊 Extracted {len(content_elements)} content elements for {section.title}")

            # VALIDATION: Log if too many elements
            if len(content_elements) > 150:
                logger.warning(f"⚠️ Section {section.title} has {len(content_elements)} elements (>150)")

        except Exception as e:
            logger.error(f"❌ Error extracting content for section {section.title}: {e}")

        return content_elements

    def _map_section_to_breadcrumb_text(self, section: SectionInfo) -> str:
        """
        FIXED: Dynamic breadcrumb matching - NO HARD-CODE!

        Find breadcrumb that contains section title
        """
        try:
            section_title = section.title.strip()

            # Search through all breadcrumbs for matching text
            for breadcrumb in self.all_breadcrumb_elements:
                breadcrumb_text = breadcrumb.get_text(strip=True)

                # Direct match: section title appears in breadcrumb
                if section_title in breadcrumb_text:
                    logger.debug(f"✅ Matched '{section_title}' → '{breadcrumb_text}'")
                    return breadcrumb_text

                # Partial match: breadcrumb ends with section title
                if breadcrumb_text.endswith(section_title):
                    logger.debug(f"✅ End-matched '{section_title}' → '{breadcrumb_text}'")
                    return breadcrumb_text

                # Split match: last part of breadcrumb matches section
                breadcrumb_parts = breadcrumb_text.split(' > ')
                if breadcrumb_parts[-1].strip() == section_title:
                    logger.debug(f"✅ Part-matched '{section_title}' → '{breadcrumb_text}'")
                    return breadcrumb_text

            # No match found
            logger.warning(f"⚠️ No breadcrumb match found for section: '{section_title}'")
            return ""

        except Exception as e:
            logger.error(f"❌ Error mapping section {section.title}: {e}")
            return ""

    def _is_content_element(self, elem: Tag) -> bool:
        if elem.name == 'img':
            text_len = len(elem.get_text(strip=True))
            logger.debug(f"🔍 DEBUG IMG: name={elem.name}, classes={elem.get('class')}, text_len={text_len}")
            if text_len > 1000:
                logger.warning(f"⚠️ IMG has {text_len} chars text - SUSPICIOUS!")

        """Check if element contains meaningful content - UPDATED for position-based"""
        if not hasattr(elem, 'name') or not elem.name:
            return False

        # Skip breadcrumb elements
        if elem.name == 'p' and elem.get('class'):
            classes = elem.get('class')
            if 'page-breadcrumb' in classes:
                return False

        # Skip page-heading divs
        if elem.name == 'div' and elem.get('class'):
            classes = elem.get('class')
            if 'page-heading' in classes:
                return False

        # Skip page-breaks
        if elem.name == 'p' and elem.get('style') and 'page-break-after' in elem.get('style'):
            return False

        # Skip section tags themselves
        if elem.name == 'section':
            return False

        # Check for meaningful text
        try:
            text = elem.get_text(strip=True)
            if not text or len(text) < 3:  # Reduced minimum length
                return False
        except:
            return False

        return True

    # Keep existing helper methods
    def _extract_text_content(self, elem: Tag, content_type: str) -> str:
        if elem.name == 'img':
            raw_text = elem.get_text()
            logger.warning(f"🚨 DEBUG: Extracting text from IMG element!")
            logger.warning(f"   Text length: {len(raw_text)}")
            logger.warning(f"   Text preview: {raw_text[:200]}...")
            logger.warning(f"   Element structure: {elem}")

        if content_type == 'code_content':
            return elem.get_text()
        else:
            text = elem.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            return text

    def _generate_selector(self, elem: Tag) -> str:
        try:
            selector_parts = []
            selector_parts.append(elem.name)

            classes = elem.get('class', [])
            if classes:
                selector_parts.append(f".{classes[0]}")

            elem_id = elem.get('id')
            if elem_id:
                selector_parts.append(f"#{elem_id}")

            return ''.join(selector_parts)
        except:
            return elem.name or 'unknown'

    def _estimate_token_count(self, text: str) -> int:
        return len(text) // 4

    def _generate_hash(self, text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()[:12]

    def _detect_language(self, elem: Tag, content_type: str) -> Optional[str]:
        if content_type != 'code_content':
            return None

        classes = elem.get('class', [])

        for cls in classes:
            cls_lower = cls.lower()
            if 'python' in cls_lower:
                return 'python'
            elif 'csharp' in cls_lower or 'c#' in cls_lower:
                return 'csharp'
            elif 'javascript' in cls_lower or 'js' in cls_lower:
                return 'javascript'

        text = elem.get_text()
        if 'def ' in text or 'import ' in text:
            return 'python'
        elif 'public class' in text or 'using ' in text:
            return 'csharp'

        return 'unknown'


class FileManager:
    """Manage output files and directory structure"""

    def __init__(self, base_output_dir: str = "data/parsed_content"):
        self.base_output_dir = Path(base_output_dir)

    def setup_output_directory(self, document_name: str) -> Path:
        output_dir = self.base_output_dir / document_name
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sections").mkdir(exist_ok=True)
        logger.info(f"📁 Output directory: {output_dir}")
        return output_dir

    def save_metadata(self, output_dir: Path, metadata: Dict):
        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def save_structure(self, output_dir: Path, structure: Dict):
        with open(output_dir / "structure.json", 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)

    def save_section(self, output_dir: Path, filename: str, section_data: Dict):
        section_path = output_dir / "sections" / filename
        with open(section_path, 'w', encoding='utf-8') as f:
            json.dump(section_data, f, indent=2, ensure_ascii=False)


class HTMLContentParser:
    """Main orchestrator for HTML to JSON conversion"""

    def __init__(self, rules_file: str = "pattern_rules.yaml"):
        self.preprocessor = HTMLPreprocessor()
        self.structure_extractor = StructureExtractor()
        self.classifier = ContentClassifier(rules_file)
        self.content_processor = ContentProcessor(self.classifier)
        self.file_manager = FileManager()

    def parse_document(self, html_file_path: str, document_name: str = None) -> Dict:
        """
        FIXED: Main parsing function with breadcrumb-based extraction
        """
        logger.info(f"🚀 Starting Phase 1 parsing with dynamic breadcrumb matching: {html_file_path}")

        if document_name is None:
            document_name = Path(html_file_path).stem

        try:
            # Step 1: Preprocess HTML
            logger.info("📋 Step 1: HTML preprocessing...")
            processed_file_path = self.preprocessor.preprocess_file(Path(html_file_path).name)

            # Step 2: Load and parse HTML
            logger.info("📋 Step 2: Loading HTML into memory...")
            with open(processed_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'html.parser')
            logger.info(f"✅ Loaded {len(html_content):,} characters into memory")

            # Step 3: Setup output directory
            logger.info("📋 Step 3: Setting up output directory...")
            output_dir = self.file_manager.setup_output_directory(document_name)

            # Step 4: Extract structure with internal links
            logger.info("📋 Step 4: Extracting document structure...")
            sections, section_mapping = self.structure_extractor.extract_toc_hierarchy(soup)

            # FIXED Step 5: Dynamic breadcrumb matching content extraction
            logger.info("📋 Step 5: Dynamic breadcrumb matching content extraction...")
            all_sections_content = self.content_processor.process_all_sections_content(soup, sections)

            # Step 6: Build content items and distribution
            logger.info("📋 Step 6: Building content distribution...")
            all_content_items = []
            content_distribution = defaultdict(int)
            processed_sections = []

            for section in sections:
                section_content_items = all_sections_content.get(section.id, [])
                all_content_items.extend(section_content_items)

                # Count content types
                for item in section_content_items:
                    content_distribution[item.content_type] += 1

                # Prepare section data with internal links
                section_data = {
                    'section_info': {
                        'id': section.id,
                        'title': section.title,
                        'hierarchy': str(section.level),
                        'parent_section': section.parent_id,
                        'child_sections': section.children_ids,
                        'order': section.order,
                        'internal_links': getattr(section, 'internal_links', {})
                    },
                    'content': [asdict(item) for item in section_content_items]
                }

                # Save section
                filename = section_mapping[section.id]
                self.file_manager.save_section(output_dir, filename, section_data)

                processed_sections.append({
                    'id': section.id,
                    'title': section.title,
                    'filename': filename,
                    'content_count': len(section_content_items)
                })

                if section_content_items:
                    logger.info(f"✅ Section '{section.title}': {len(section_content_items)} content items")
                else:
                    logger.warning(f"🚫 Section '{section.title}': No content items")

            # Step 7: Generate metadata
            logger.info("📋 Step 7: Generating metadata...")

            file_checksum = hashlib.md5(html_content.encode()).hexdigest()

            metadata = {
                'extraction_metadata': {
                    'source_file': Path(html_file_path).name,
                    'html_checksum': file_checksum,
                    'extraction_time_utc': datetime.utcnow().isoformat() + 'Z',
                    'tool_version': 'QC-Discovery Phase1 v3.1.0-DYNAMIC-MATCHING',
                    'rules_version': f'pattern_rules.yaml ({len(self.classifier.rules)} categories)',
                    'total_elements': len(soup.find_all()),
                    'processed_elements': len(all_content_items),
                    'extraction_method': 'position_based_dynamic_breadcrumb_matching'
                },
                'document_info': {
                    'title': document_name.replace('-', ' ').title(),
                    'language': 'mixed',
                    'content_types': list(content_distribution.keys()),
                    'estimated_reading_time_minutes': len(html_content) // 1000
                }
            }

            # Step 8: Generate structure
            structure = {
                'toc_hierarchy': [
                    {
                        'id': section.id,
                        'title': section.title,
                        'level': section.level,
                        'order': section.order,
                        'parent_id': section.parent_id,
                        'children_ids': section.children_ids,
                        'internal_links': getattr(section, 'internal_links', {})
                    }
                    for section in sections
                ],
                'section_mapping': section_mapping,
                'content_distribution': dict(content_distribution),
                'processing_stats': {
                    'total_sections': len(sections),
                    'processed_sections': len(processed_sections),
                    'total_content_items': len(all_content_items),
                    'low_confidence_items': len([item for item in all_content_items if item.confidence < 0.7]),
                    'empty_sections': len([s for s in processed_sections if s['content_count'] == 0]),
                    'breadcrumb_validation': f"Found {len(soup.find_all('p', class_='page-breadcrumb'))} breadcrumbs"
                }
            }

            # Step 9: Save metadata and structure
            logger.info("📋 Step 9: Saving metadata and structure...")
            self.file_manager.save_metadata(output_dir, metadata)
            self.file_manager.save_structure(output_dir, structure)

            # Step 10: Generate summary
            logger.info("📋 Step 10: Generating summary...")

            total_tokens = sum(item.metadata.get('token_count', 0) for item in all_content_items)
            avg_confidence = sum(item.confidence for item in all_content_items) / len(
                all_content_items) if all_content_items else 0

            results = {
                'success': True,
                'output_directory': str(output_dir),
                'processing_stats': {
                    'total_sections': len(sections),
                    'processed_sections': len(processed_sections),
                    'total_content_items': len(all_content_items),
                    'total_estimated_tokens': total_tokens,
                    'average_confidence': avg_confidence,
                    'content_distribution': dict(content_distribution),
                    'low_confidence_count': len([item for item in all_content_items if item.confidence < 0.7]),
                    'empty_sections_count': len([s for s in processed_sections if s['content_count'] == 0])
                },
                'file_paths': {
                    'metadata': str(output_dir / "metadata.json"),
                    'structure': str(output_dir / "structure.json"),
                    'sections_dir': str(output_dir / "sections")
                }
            }

            logger.info("🎉 Phase 1 parsing completed successfully!")
            self._log_summary(results)

            return results

        except Exception as e:
            logger.error(f"❌ Phase 1 parsing failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            return {
                'success': False,
                'error': str(e),
                'output_directory': None
            }

    def _log_summary(self, results: Dict):
        """Enhanced logging summary"""
        stats = results['processing_stats']

        logger.info("=" * 60)
        logger.info("🎯 PHASE 1 PARSING SUMMARY - DYNAMIC BREADCRUMB MATCHING")
        logger.info("=" * 60)
        logger.info(f"📁 Output Directory: {results['output_directory']}")
        logger.info(f"📊 Total Sections: {stats['total_sections']}")
        logger.info(f"✅ Sections Processed: {stats['processed_sections']}")
        logger.info(f"📝 Total Content Items: {stats['total_content_items']}")
        logger.info(f"🔤 Estimated Tokens: {stats['total_estimated_tokens']:,}")
        logger.info(f"📈 Average Confidence: {stats['average_confidence']:.2%}")

        if stats['low_confidence_count'] > 0:
            logger.warning(f"⚠️ Low Confidence Items: {stats['low_confidence_count']}")

        if stats['empty_sections_count'] > 0:
            logger.warning(f"🚫 Empty Sections: {stats['empty_sections_count']}")

        logger.info("\n📋 Content Distribution:")
        for content_type, count in stats['content_distribution'].items():
            logger.info(f"   {content_type}: {count}")

        logger.info("\n🗂️ Files Created:")
        for file_type, path in results['file_paths'].items():
            logger.info(f"   {file_type}: {path}")

        # Success rate
        if stats['total_content_items'] > 0:
            success_rate = (stats['processed_sections'] - stats['empty_sections_count']) / stats[
                'processed_sections'] * 100
            logger.info(f"\n🎯 Success Rate: {success_rate:.1f}% sections with content")

        logger.info("=" * 60)


def main():
    """Command line interface for Phase 1 parser"""
    import argparse

    parser = argparse.ArgumentParser(description='Phase 1: DYNAMIC BREADCRUMB MATCHING HTML Parser')
    parser.add_argument('--input', required=True, help='HTML file path to parse')
    parser.add_argument('--output-name', help='Output directory name (default: filename)')
    parser.add_argument('--rules', default='pattern_rules.yaml', help='Pattern rules file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("🐛 Debug logging enabled")

    # Initialize parser
    content_parser = HTMLContentParser(args.rules)

    # Parse document
    results = content_parser.parse_document(args.input, args.output_name)

    if results['success']:
        print("\n🎉 SUCCESS! Phase 1 dynamic breadcrumb matching completed.")
        print(f"📁 Results saved to: {results['output_directory']}")

        stats = results['processing_stats']
        print(f"📊 Processed {stats['processed_sections']}/{stats['total_sections']} sections")
        print(f"📝 Generated {stats['total_content_items']} content items")
        print(f"🔤 Estimated {stats['total_estimated_tokens']:,} tokens")

        if stats['low_confidence_count'] > 0:
            print(f"⚠️ {stats['low_confidence_count']} items with low confidence (<70%)")

    else:
        print(f"\n❌ FAILED: {results['error']}")
        exit(1)


if __name__ == "__main__":
    main()