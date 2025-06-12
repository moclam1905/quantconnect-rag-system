#!/usr/bin/env python3
"""
Phase 1: Core Parser Infrastructure - Hybrid Pipeline
Standalone module for HTML → JSON layers conversion
"""

import json
import logging
import hashlib
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
class ContentItem:
    """Represents a single content item"""
    id: str
    content_type: str
    text: str
    metadata: Dict
    order: int
    confidence: float


@dataclass
class SectionInfo:
    """Represents a document section"""
    id: str
    title: str
    level: int
    order: int
    parent_id: Optional[str]
    children_ids: List[str]
    toc_element_id: Optional[str]


class StructureExtractor:
    """Extract ToC structure and document hierarchy"""

    def __init__(self):
        self.toc_classes = ['toc-h1', 'toc-h2', 'toc-h3', 'toc-h4', 'toc-h5', 'toc-h6']

        # Danh sách sections cần bỏ qua (không bao gồm Market Hours - dùng title match)
        self.skip_sections = {
            "24",  # Migrations + children
            "2.1.3.2",  # Rendering Data with CSharp
            "5.8.1"  # Third-Party Libraries
        }

    def extract_toc_hierarchy(self, soup: BeautifulSoup) -> Tuple[List[SectionInfo], Dict[str, str]]:
        """
        Extract ToC hierarchy from HTML với section filtering

        Returns:
            (sections_list, section_mapping)
        """
        logger.info("🗂️ Đang trích xuất cấu trúc ToC...")

        sections = []
        section_mapping = {}
        section_counter = 1
        skipped_sections = []

        # Stack để track parent-child relationships
        parent_stack = []

        try:
            # Tìm tất cả ToC elements
            toc_elements = []
            for toc_class in self.toc_classes:
                elements = soup.find_all(class_=toc_class)
                for elem in elements:
                    level = int(toc_class.split('-h')[1])  # Extract level từ class
                    toc_elements.append((elem, level))

            # Sắp xếp theo document order
            toc_elements.sort(key=lambda x: self._get_element_position(x[0]))

            logger.info(f"📋 Tìm thấy {len(toc_elements)} ToC entries")

            for elem, level in toc_elements:
                # Extract section number từ href
                section_number = self._extract_section_number(elem)
                title = self._extract_title_text(elem)

                # Kiểm tra xem có nên skip section này không
                if self._should_skip_section(section_number, title, parent_stack):
                    skipped_sections.append(f"{section_number} - {title}")
                    continue

                # Generate section info
                section_id = f"section_{section_counter:03d}"

                # Handle hierarchy
                parent_id = self._find_parent_id(level, parent_stack)

                # Create section với section_number mapping
                section = SectionInfo(
                    id=section_id,
                    title=title,
                    level=level,
                    order=section_counter,
                    parent_id=parent_id,
                    children_ids=[],
                    toc_element_id=section_number  # Store section number thay vì ToC element ID
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

                # Create section mapping
                filename = f"{section_counter:03d}_{self._slugify(title)}.json"
                section_mapping[section_id] = filename

                section_counter += 1

        except Exception as e:
            logger.error(f"❌ Lỗi khi trích xuất ToC: {e}")
            # Fallback: tạo single section
            sections = [SectionInfo("section_001", "Document", 1, 1, None, [], None)]
            section_mapping = {"section_001": "001_document.json"}

        logger.info(f"✅ Đã trích xuất {len(sections)} sections")
        if skipped_sections:
            logger.info(f"🚫 Đã bỏ qua {len(skipped_sections)} sections:")
            for skipped in skipped_sections:
                logger.info(f"   - {skipped}")

        return sections, section_mapping

    def _get_element_position(self, elem) -> int:
        """Get element position in document for sorting"""
        try:
            # Simple approach: count preceding elements
            position = 0
            for prev in elem.previous_elements:
                if hasattr(prev, 'name'):
                    position += 1
            return position
        except:
            return 0

    def _extract_title_text(self, elem) -> str:
        """Extract clean title text from ToC element"""
        try:
            # Get text content, clean up
            title = elem.get_text(strip=True)
            # Remove numbers at start (1.1.1, etc)
            title = re.sub(r'^[\d\.]+\s*', '', title)
            return title[:100] if title else "Untitled"
        except:
            return "Untitled"

    def _find_parent_id(self, current_level: int, parent_stack: List) -> Optional[str]:
        """Find parent section ID based on hierarchy"""
        while parent_stack and parent_stack[-1]['level'] >= current_level:
            parent_stack.pop()

        return parent_stack[-1]['section_id'] if parent_stack else None

    def _extract_section_number(self, elem) -> str:
        """Trích xuất section number từ ToC element"""
        try:
            # Tìm href attribute: href="#1" → "1"
            href = elem.get('href', '')
            if href.startswith('#'):
                return href[1:]  # Remove # prefix: "#2.1.1" → "2.1.1"

            # Fallback: extract từ text content
            text = elem.get_text(strip=True)
            # Tìm pattern số ở đầu: "3.4 Market Hours" → "3.4"
            import re
            match = re.match(r'^([\d\.]+)', text)
            if match:
                return match.group(1)

            return ""
        except:
            return ""

    def _should_skip_section(self, section_number: str, title: str, parent_stack: List) -> bool:
        """
        Kiểm tra xem có nên skip section này không

        Skip logic:
        1. Exact match title "Market Hours" (skip tất cả 12 Market Hours sections)
        2. Specific section numbers (Migrations, CSharp Rendering, Third-Party Libraries)
        3. Parent-child: nếu parent bị skip thì children cũng skip
        """
        if not section_number:
            return False

        # Exact match cho Market Hours (an toàn nhất)
        if title.strip() == "Market Hours":
            return True

        # Kiểm tra specific sections khác
        if section_number in self.skip_sections:
            return True

        # Kiểm tra parent-child: nếu parent bị skip thì skip luôn
        for parent_info in parent_stack:
            parent_section = parent_info.get('section_number', '')
            if parent_section in self.skip_sections:
                return True
            # Cũng check parent title
            parent_title = parent_info.get('title', '')
            if parent_title.strip() == "Market Hours":
                return True

        return False

    def _update_parent_stack(self, parent_stack: List, section: SectionInfo, level: int, section_number: str,
                             title: str):
        """Update parent stack for hierarchy tracking"""
        # Remove items of same or higher level
        while parent_stack and parent_stack[-1]['level'] >= level:
            parent_stack.pop()

        # Add current section
        parent_stack.append({
            'section_id': section.id,
            'level': level,
            'title': title,
            'section_number': section_number
        })

    def _slugify(self, text: str) -> str:
        """Convert title to filename-safe slug"""
        # Remove special characters, replace spaces with underscores
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        slug = re.sub(r'[-\s]+', '_', slug)
        return slug[:50]  # Limit length


class ContentClassifier:
    """Classify HTML content using existing rules"""

    def __init__(self, rules_file: str = "pattern_rules.yaml"):
        self.rules = self._load_rules(rules_file)

    def _load_rules(self, rules_file: str) -> Dict:
        """Load classification rules"""
        try:
            import yaml
            with open(rules_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"⚠️ Could not load rules from {rules_file}: {e}")
            return {}

    def classify_element(self, elem: Tag) -> Tuple[str, float]:
        """
        Classify single HTML element với fixed integration

        Returns:
            (content_type, confidence_score)
        """
        try:
            # Kiểm tra element hợp lệ
            if elem is None or not hasattr(elem, 'name'):
                return 'documentation_text', 0.3

            # Debug: Check if rules loaded
            if not self.rules:
                logger.warning("⚠️ Rules không load được, sử dụng fallback classification")
                return self._fallback_classification(elem)

            # Fix classes format để compatible với existing function
            original_classes = elem.get('class', [])

            # Temporarily modify element để pass correct format
            if isinstance(original_classes, list):
                # Convert list to space-separated string
                elem.attrs['class'] = ' '.join(original_classes)
            elif original_classes is None:
                elem.attrs['class'] = ''

            # Try using existing classify_content_type function
            try:
                from production_ready_discovery import classify_content_type
                content_type = classify_content_type(elem, self.rules)
            except Exception as classify_error:
                logger.warning(f"⚠️ Existing classify_content_type failed: {classify_error}")
                # Restore original classes before fallback
                if isinstance(original_classes, list):
                    elem.attrs['class'] = original_classes
                return self._fallback_classification(elem)

            # Restore original classes
            if isinstance(original_classes, list):
                elem.attrs['class'] = original_classes

            # Calculate confidence based on classification certainty
            confidence = self._calculate_confidence(elem, content_type)

            # Skip content should be filtered out
            if content_type == 'SKIP':
                return None, 0.0

            return content_type, confidence

        except Exception as e:
            logger.warning(f"⚠️ Lỗi classification: {e}")
            return self._fallback_classification(elem)

    def _fallback_classification(self, elem: Tag) -> Tuple[str, float]:
        """Simple fallback classification không dùng external function"""
        try:
            classes = elem.get('class', [])
            tag = elem.name.lower()

            # Simple heuristics
            if isinstance(classes, list):
                classes_str = ' '.join(classes).lower()
            else:
                classes_str = str(classes).lower()

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

            # Default
            return 'documentation_text', 0.7

        except Exception as e:
            logger.warning(f"⚠️ Fallback classification failed: {e}")
            return 'documentation_text', 0.5

    def _calculate_confidence(self, elem: Tag, content_type: str) -> float:
        """Calculate confidence score for classification"""
        confidence = 0.8  # Base confidence

        # Boost confidence for strong indicators
        classes = elem.get('class', [])

        # Code content indicators
        if content_type == 'code_content':
            code_indicators = ['code', 'highlight', 'python', 'csharp', 'example']
            if any(indicator in ' '.join(classes).lower() for indicator in code_indicators):
                confidence = 0.95

        # API reference indicators
        elif content_type == 'api_reference':
            if 'data-tree' in elem.attrs or any('tree' in cls for cls in classes):
                confidence = 0.95

        # Table content
        elif content_type == 'table_content':
            if elem.name == 'table' or any('table' in cls for cls in classes):
                confidence = 0.9

        # Navigation content
        elif content_type == 'navigation_content':
            nav_indicators = ['toc', 'breadcrumb', 'nav']
            if any(indicator in ' '.join(classes).lower() for indicator in nav_indicators):
                confidence = 0.9

        return confidence


class ContentProcessor:
    """Process and extract content from HTML sections"""

    def __init__(self, classifier: ContentClassifier):
        self.classifier = classifier
        self.content_counter = 1

    def process_section_content(self, soup: BeautifulSoup, section: SectionInfo) -> List[ContentItem]:
        """
        Process content cho một section cụ thể

        Args:
            soup: BeautifulSoup object của entire document
            section: Section information

        Returns:
            List of ContentItem objects
        """
        logger.info(f"📝 Đang xử lý nội dung cho section: {section.title}")

        content_items = []

        try:
            # Tìm content elements cho section này
            section_elements = self._find_section_elements(soup, section)

            logger.info(f"🔍 Tìm thấy {len(section_elements)} elements trong section")

            for elem in section_elements:
                try:
                    content_type, confidence = self.classifier.classify_element(elem)

                    # Skip nếu classified as skip hoặc low confidence
                    if content_type is None:
                        continue

                    # Log warning cho low confidence nhưng continue
                    if confidence < 0.7:
                        logger.warning(f"⚠️ Confidence thấp ({confidence:.2f}) cho element: {elem.name}")

                    # Extract text content
                    text_content = self._extract_text_content(elem, content_type)

                    if not text_content.strip():
                        continue

                    # Create content item
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
                            'section_id': section.id
                        },
                        order=len(content_items) + 1,
                        confidence=confidence
                    )

                    content_items.append(content_item)
                    self.content_counter += 1

                except Exception as e:
                    logger.warning(f"⚠️ Lỗi khi xử lý element: {e} - tiếp tục...")
                    continue

        except Exception as e:
            logger.error(f"❌ Lỗi khi xử lý section {section.title}: {e}")

        logger.info(f"✅ Đã xử lý {len(content_items)} content items cho section: {section.title}")
        return content_items

    def _find_section_elements(self, soup: BeautifulSoup, section: SectionInfo) -> List[Tag]:
        """Tìm HTML elements thuộc về section này - FIXED VERSION với debug logging"""

        logger.debug(f"🔍 === STARTING SECTION PROCESSING: '{section.title}' ===")

        # Extract section number từ section title hoặc mapping
        section_number = self._get_section_number_from_info(section, soup)

        logger.debug(f"📋 Section info: title='{section.title}', level={section.level}, order={section.order}")
        logger.debug(f"🔢 Extracted section number: '{section_number}'")

        if not section_number:
            logger.warning(f"⚠️ Không tìm thấy section number cho '{section.title}', sử dụng fallback")
            return self._get_fallback_elements_for_section(soup, section)

        try:
            # Tìm content section với id matching section number
            content_section = soup.find('section', id=section_number)

            if not content_section:
                logger.warning(f"⚠️ Không tìm thấy content section id='{section_number}' cho '{section.title}'")

                # Debug: Show all sections trong HTML
                all_sections = soup.find_all('section')
                logger.debug(f"🔍 Available sections trong HTML: {[s.get('id') for s in all_sections]}")

                return self._get_fallback_elements_for_section(soup, section)

            logger.debug(f"🎯 Tìm thấy content section: <section id='{content_section.get('id')}'>")

            # Extract elements từ section start đến page-break hoặc next section
            elements = self._extract_elements_from_section_boundary(soup, content_section)

            logger.debug(f"📊 SECTION PROCESSING COMPLETE:")
            logger.debug(f"   Section: '{section.title}'")
            logger.debug(f"   Elements found: {len(elements)}")
            logger.debug(f"   Section number: {section_number}")
            logger.debug(f"🔍 === END SECTION PROCESSING: '{section.title}' ===\n")

            return elements

        except Exception as e:
            logger.error(f"❌ Lỗi khi tìm section elements cho '{section.title}': {e}")
            logger.debug(f"🔍 === ERROR IN SECTION PROCESSING: '{section.title}' ===\n")
            return self._get_fallback_elements_for_section(soup, section)

    def _get_section_number_from_info(self, section: SectionInfo, soup: BeautifulSoup) -> str:
        """Extract section number từ section info hoặc re-parse ToC"""
        # Try to extract từ toc_element_id nếu có
        if hasattr(section, 'toc_element_id') and section.toc_element_id:
            return section.toc_element_id

        # Fallback: tìm trong ToC dựa trên title
        try:
            toc_links = soup.find_all('a', class_=lambda x: x and any(
                cls.startswith('toc-h') for cls in x if isinstance(x, list)))

            for link in toc_links:
                link_text = link.get_text(strip=True)
                # Remove section number để compare title
                clean_title = re.sub(r'^[\d\.]+\s*', '', link_text)

                if clean_title.strip() == section.title.strip():
                    href = link.get('href', '')
                    if href.startswith('#'):
                        return href[1:]  # "#2.1" → "2.1"

        except Exception as e:
            logger.debug(f"Lỗi khi tìm section number từ ToC: {e}")

        return ""

    def _extract_elements_from_section_boundary(self, soup: BeautifulSoup, content_section: Tag) -> List[Tag]:
        """Extract elements từ section start đến boundary (page-break hoặc next section)"""
        elements = []
        processed_count = 0

        logger.debug(f"🔍 Starting boundary extraction từ section: {content_section.name}#{content_section.get('id')}")

        # DEBUG: Show section parent và surrounding structure
        logger.debug(f"🏗️ Section parent: {content_section.parent.name if content_section.parent else 'None'}")
        logger.debug(
            f"🏗️ Section parent class: {content_section.parent.get('class') if content_section.parent else 'None'}")

        # DEBUG: Show all siblings của section
        all_siblings = list(content_section.next_siblings)
        logger.debug(f"🏗️ Section có {len(all_siblings)} next siblings:")
        for i, sibling in enumerate(all_siblings[:10]):  # Show first 10
            if hasattr(sibling, 'name') and sibling.name:
                logger.debug(
                    f"   Sibling {i + 1}: <{sibling.name}> id={sibling.get('id')} class={sibling.get('class')}")
            else:
                logger.debug(f"   Sibling {i + 1}: {type(sibling)} - {repr(str(sibling)[:50])}")

        # DEBUG: Check if content nằm trong section thay vì as siblings
        section_children = list(content_section.children)
        logger.debug(f"🏗️ Section có {len(section_children)} children:")
        for i, child in enumerate(section_children[:10]):  # Show first 10
            if hasattr(child, 'name') and child.name:
                logger.debug(f"   Child {i + 1}: <{child.name}> id={child.get('id')} class={child.get('class')}")
            else:
                logger.debug(f"   Child {i + 1}: {type(child)} - {repr(str(child)[:50])}")

        # Start từ content_section
        current = content_section

        # Add section element itself nếu có content
        if self._is_significant_element(current):
            elements.append(current)
            logger.debug(f"✅ Added section element: {current.name}#{current.get('id')}")
        else:
            logger.debug(f"🚫 Section element not significant: {current.name}#{current.get('id')}")

        # Traverse siblings cho đến khi gặp boundary
        while current:
            current = current.next_sibling
            processed_count += 1

            if not current:
                logger.debug("🔚 Reached end of siblings")
                break

            # Debug: Show current element being processed
            if hasattr(current, 'name') and current.name:
                logger.debug(
                    f"🔄 Processing element #{processed_count}: {current.name} class={current.get('class')} id={current.get('id')}")

                # Show text content preview
                try:
                    text_preview = current.get_text(strip=True)[:100] if hasattr(current, 'get_text') else "No text"
                    logger.debug(f"   📝 Text preview: '{text_preview}'")
                except:
                    logger.debug(f"   📝 Could not get text content")
            else:
                logger.debug(f"🔄 Processing non-element #{processed_count}: {type(current)}")
                if hasattr(current, '__str__'):
                    content_preview = str(current)[:100].strip()
                    logger.debug(f"   📝 Content: '{content_preview}'")

            # Kiểm tra page-break boundary
            if self._is_page_break(current):
                logger.debug("📄 Gặp page-break, dừng extraction")
                break

            # Kiểm tra next section boundary
            if hasattr(current, 'name') and current.name == 'section':
                logger.debug(f"📄 Gặp section tiếp theo: {current.get('id')}, dừng extraction")
                break

            # Add significant elements và children
            if hasattr(current, 'name') and current.name:
                if self._is_significant_element(current):
                    elements.append(current)
                    logger.debug(f"✅ Added significant element: {current.name} (total: {len(elements)})")
                else:
                    logger.debug(f"🚫 Element not significant: {current.name}")

                # Add significant children
                try:
                    children_added = 0
                    for child in current.find_all():
                        if self._is_significant_element(child):
                            elements.append(child)
                            children_added += 1

                    if children_added > 0:
                        logger.debug(f"   👶 Added {children_added} significant children")

                except Exception as e:
                    logger.debug(f"   ⚠️ Error finding children: {e}")

            # Safety limit để tránh infinite loop
            if len(elements) > 5000:
                logger.warning(f"⚠️ Section có >5000 elements, dừng extraction")
                break

            # Debug progress every 10 elements
            if processed_count % 10 == 0:
                logger.debug(f"📊 Progress: processed {processed_count} siblings, collected {len(elements)} elements")

        logger.debug(
            f"🏁 Boundary extraction complete: processed {processed_count} siblings, collected {len(elements)} significant elements")

        # DEBUG: Nếu chỉ có 1 element, kiểm tra alternative approaches
        if len(elements) <= 1:
            logger.debug("🔍 VERY FEW ELEMENTS FOUND - INVESTIGATING ALTERNATIVES:")
            self._debug_alternative_content_locations(soup, content_section)

        return elements

    def _debug_alternative_content_locations(self, soup: BeautifulSoup, content_section: Tag):
        """Debug alternative locations where content might be"""
        section_id = content_section.get('id')
        logger.debug(f"🔍 Looking for alternative content locations for section #{section_id}")

        # Method 1: Look for content between this section và next section trong entire document
        logger.debug("🔍 Method 1: Looking for content between sections trong document...")
        all_sections = soup.find_all('section')
        current_section_index = None

        for i, section in enumerate(all_sections):
            if section.get('id') == section_id:
                current_section_index = i
                break

        if current_section_index is not None:
            next_section = all_sections[current_section_index + 1] if current_section_index + 1 < len(
                all_sections) else None
            logger.debug(f"   Current section index: {current_section_index}")
            logger.debug(f"   Next section: {next_section.get('id') if next_section else 'None'}")

            # Find elements between current_section và next_section trong document order
            found_between = self._find_elements_between_sections_in_document(soup, content_section, next_section)
            logger.debug(f"   Found {len(found_between)} elements between sections trong document")

        # Method 2: Look for content inside section's parent container
        logger.debug("🔍 Method 2: Looking trong section's parent container...")
        if content_section.parent:
            parent_children = [child for child in content_section.parent.children if hasattr(child, 'name')]
            logger.debug(f"   Parent has {len(parent_children)} element children")
            for i, child in enumerate(parent_children[:5]):
                logger.debug(f"   Parent child {i + 1}: <{child.name}> id={child.get('id')}")

    def _find_elements_between_sections_in_document(self, soup: BeautifulSoup, current_section: Tag,
                                                    next_section: Tag) -> List[Tag]:
        """Find elements between two sections trong document order"""
        elements = []

        # Get all elements trong document
        all_elements = soup.find_all(True)  # Find all tags

        # Find indices của current và next section
        current_index = None
        next_index = None

        for i, elem in enumerate(all_elements):
            if elem == current_section:
                current_index = i
            elif next_section and elem == next_section:
                next_index = i
                break

        if current_index is not None:
            # Extract elements between sections
            end_index = next_index if next_index else len(all_elements)
            between_elements = all_elements[current_index + 1:end_index]

            # Filter for significant elements
            for elem in between_elements:
                if self._is_significant_element(elem):
                    elements.append(elem)
                    if len(elements) >= 10:  # Limit for debug
                        break

        return elements

    def _is_page_break(self, elem) -> bool:
        """Kiểm tra xem element có phải page-break không với debug logging"""
        try:
            if not hasattr(elem, 'name') or not elem.name:
                return False

            # Check for page-break-after style
            if elem.name == 'p' and elem.get('style'):
                style = elem.get('style', '')
                if 'page-break-after' in style:
                    logger.debug(f"📄 Page-break detected: <{elem.name} style='{style[:50]}...'>")
                    return True

            return False
        except Exception as e:
            logger.debug(f"⚠️ Error checking page-break: {e}")
            return False

    def _get_fallback_elements_for_section(self, soup: BeautifulSoup, section: SectionInfo) -> List[Tag]:
        """Fallback: return limited unique elements cho section này với debug logging"""
        # Thay vì return same elements cho all sections, return empty
        # Để tránh duplicate content
        logger.debug(f"🚫 FALLBACK TRIGGERED cho section '{section.title}'")
        logger.debug(f"   Reason: Không thể extract content với section boundary method")
        logger.debug(f"   Action: Return empty list để tránh duplicate content")
        logger.warning(f"🚫 Không thể extract content cho section '{section.title}' - return empty")
        return []

    def _get_significant_elements(self, soup: BeautifulSoup) -> List[Tag]:
        """Get significant elements từ entire document làm fallback"""
        elements = []

        # Target important tags và classes
        for tag in ['p', 'div', 'code', 'pre', 'table', 'ul', 'ol', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            elements.extend(soup.find_all(tag))

        # Filter và reasonable limit (tăng từ 1000 lên 5000)
        significant = [elem for elem in elements if self._is_significant_element(elem)]
        return significant[:5000]  # Tăng reasonable limit

    def _is_significant_element(self, elem: Tag) -> bool:
        """Check if element is significant for content extraction với debug logging"""
        if not elem.name:
            logger.debug(f"🚫 Not significant: No element name")
            return False

        # Debug element info
        elem_info = f"{elem.name}"
        if elem.get('class'):
            elem_info += f".{elem.get('class')}"
        if elem.get('id'):
            elem_info += f"#{elem.get('id')}"

        # Skip empty elements
        try:
            text = elem.get_text(strip=True)
            if not text or len(text) < 10:
                logger.debug(f"🚫 Not significant ({elem_info}): Empty or too short text (len={len(text)})")
                return False
        except:
            logger.debug(f"🚫 Not significant ({elem_info}): Cannot get text")
            return False

        # Skip pure navigation/UI elements
        classes = elem.get('class', [])
        if isinstance(classes, str):
            classes = classes.split()

        skip_patterns = ['nav', 'menu', 'header', 'footer', 'sidebar', 'breadcrumb']
        classes_str = ' '.join(classes).lower()

        for pattern in skip_patterns:
            if pattern in classes_str:
                logger.debug(f"🚫 Not significant ({elem_info}): Contains skip pattern '{pattern}'")
                return False

        # Significant element found
        logger.debug(f"✅ Significant element: {elem_info} (text_len={len(text)})")
        return True

    def _extract_text_content(self, elem: Tag, content_type: str) -> str:
        """Extract clean text content from element"""
        if content_type == 'code_content':
            # Preserve formatting for code
            return elem.get_text()
        else:
            # Clean text for other content types
            text = elem.get_text(separator=' ', strip=True)
            # Clean up extra whitespace
            text = re.sub(r'\s+', ' ', text)
            return text

    def _generate_selector(self, elem: Tag) -> str:
        """Generate CSS selector for element"""
        try:
            selector_parts = []

            # Add tag
            selector_parts.append(elem.name)

            # Add class if present
            classes = elem.get('class', [])
            if classes:
                selector_parts.append(f".{classes[0]}")  # Use first class

            # Add ID if present
            elem_id = elem.get('id')
            if elem_id:
                selector_parts.append(f"#{elem_id}")

            return ''.join(selector_parts)

        except:
            return elem.name or 'unknown'

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Simple approximation: ~4 characters per token
        return len(text) // 4

    def _generate_hash(self, text: str) -> str:
        """Generate hash ID for content deduplication"""
        return hashlib.sha1(text.encode()).hexdigest()[:12]

    def _detect_language(self, elem: Tag, content_type: str) -> Optional[str]:
        """Detect programming language for code content"""
        if content_type != 'code_content':
            return None

        classes = elem.get('class', [])

        # Check for language indicators in classes
        for cls in classes:
            cls_lower = cls.lower()
            if 'python' in cls_lower:
                return 'python'
            elif 'csharp' in cls_lower or 'c#' in cls_lower:
                return 'csharp'
            elif 'javascript' in cls_lower or 'js' in cls_lower:
                return 'javascript'

        # Check content for language hints
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
        """Setup output directory structure"""
        output_dir = self.base_output_dir / document_name

        # Create directories
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sections").mkdir(exist_ok=True)

        logger.info(f"📁 Output directory: {output_dir}")
        return output_dir

    def save_metadata(self, output_dir: Path, metadata: Dict):
        """Save document metadata"""
        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def save_structure(self, output_dir: Path, structure: Dict):
        """Save document structure"""
        with open(output_dir / "structure.json", 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)

    def save_section(self, output_dir: Path, filename: str, section_data: Dict):
        """Save section content"""
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
        Main parsing function - convert HTML to JSON layers

        Args:
            html_file_path: Path to original HTML file
            document_name: Name for output directory

        Returns:
            Dict with parsing results and output paths
        """
        logger.info(f"🚀 Bắt đầu Phase 1 parsing: {html_file_path}")

        if document_name is None:
            document_name = Path(html_file_path).stem

        try:
            # Step 1: Preprocess HTML
            logger.info("📋 Bước 1: Tiền xử lý HTML...")
            processed_file_path = self.preprocessor.preprocess_file(Path(html_file_path).name)

            # Step 2: Load and parse HTML
            logger.info("📋 Bước 2: Load HTML vào memory...")
            with open(processed_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'html.parser')
            logger.info(f"✅ Đã load {len(html_content):,} ký tự vào memory")

            # Step 3: Setup output directory
            logger.info("📋 Bước 3: Thiết lập thư mục output...")
            output_dir = self.file_manager.setup_output_directory(document_name)

            # Step 4: Extract structure
            logger.info("📋 Bước 4: Trích xuất cấu trúc document...")
            sections, section_mapping = self.structure_extractor.extract_toc_hierarchy(soup)

            # Step 5: Process content by sections
            logger.info("📋 Bước 5: Xử lý nội dung theo sections...")
            all_content_items = []
            content_distribution = defaultdict(int)

            processed_sections = []

            for section in sections:
                try:
                    # Process section content
                    content_items = self.content_processor.process_section_content(soup, section)
                    all_content_items.extend(content_items)

                    # Count content types
                    for item in content_items:
                        content_distribution[item.content_type] += 1

                    # Prepare section data
                    section_data = {
                        'section_info': {
                            'id': section.id,
                            'title': section.title,
                            'hierarchy': str(section.level),
                            'parent_section': section.parent_id,
                            'child_sections': section.children_ids,
                            'order': section.order
                        },
                        'content': [asdict(item) for item in content_items]
                    }

                    # Save section
                    filename = section_mapping[section.id]
                    self.file_manager.save_section(output_dir, filename, section_data)

                    processed_sections.append({
                        'id': section.id,
                        'title': section.title,
                        'filename': filename,
                        'content_count': len(content_items)
                    })

                    logger.info(f"✅ Section '{section.title}': {len(content_items)} content items")

                except Exception as e:
                    logger.error(f"❌ Không thể xử lý section '{section.title}': {e}")
                    logger.warning("⚠️ Tiếp tục với section tiếp theo...")
                    continue

            # Step 6: Generate metadata
            logger.info("📋 Bước 6: Tạo metadata...")

            file_checksum = hashlib.md5(html_content.encode()).hexdigest()

            metadata = {
                'extraction_metadata': {
                    'source_file': Path(html_file_path).name,
                    'html_checksum': file_checksum,
                    'extraction_time_utc': datetime.utcnow().isoformat() + 'Z',
                    'tool_version': 'QC-Discovery Phase1 v1.0.1',
                    'rules_version': 'pattern_rules.yaml',
                    'total_elements': len(soup.find_all()),
                    'processed_elements': len(all_content_items)
                },
                'document_info': {
                    'title': document_name.replace('-', ' ').title(),
                    'language': 'mixed',
                    'content_types': list(content_distribution.keys()),
                    'estimated_reading_time_minutes': len(html_content) // 1000  # Rough estimate
                }
            }

            # Step 7: Generate structure
            structure = {
                'toc_hierarchy': [
                    {
                        'id': section.id,
                        'title': section.title,
                        'level': section.level,
                        'order': section.order,
                        'parent_id': section.parent_id,
                        'children_ids': section.children_ids
                    }
                    for section in sections
                ],
                'section_mapping': section_mapping,
                'content_distribution': dict(content_distribution),
                'processing_stats': {
                    'total_sections': len(sections),
                    'processed_sections': len(processed_sections),
                    'total_content_items': len(all_content_items),
                    'low_confidence_items': len([item for item in all_content_items if item.confidence < 0.7])
                }
            }

            # Step 8: Save metadata and structure
            logger.info("📋 Bước 8: Lưu metadata và structure...")
            self.file_manager.save_metadata(output_dir, metadata)
            self.file_manager.save_structure(output_dir, structure)

            # Step 9: Generate summary
            logger.info("📋 Bước 9: Tạo tổng kết...")

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
                    'low_confidence_count': len([item for item in all_content_items if item.confidence < 0.7])
                },
                'file_paths': {
                    'metadata': str(output_dir / "metadata.json"),
                    'structure': str(output_dir / "structure.json"),
                    'sections_dir': str(output_dir / "sections")
                }
            }

            logger.info("🎉 Phase 1 parsing hoàn thành thành công!")
            self._log_summary(results)

            return results

        except Exception as e:
            logger.error(f"❌ Phase 1 parsing thất bại: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            return {
                'success': False,
                'error': str(e),
                'output_directory': None
            }

    def _log_summary(self, results: Dict):
        """Log parsing summary"""
        stats = results['processing_stats']

        logger.info("=" * 60)
        logger.info("🎯 TỔNG KẾT PHASE 1 PARSING")
        logger.info("=" * 60)
        logger.info(f"📁 Thư mục Output: {results['output_directory']}")
        logger.info(f"📊 Tổng Sections: {stats['total_sections']}")
        logger.info(f"✅ Sections đã xử lý: {stats['processed_sections']}")
        logger.info(f"📝 Tổng Content Items: {stats['total_content_items']}")
        logger.info(f"🔤 Ước tính Tokens: {stats['total_estimated_tokens']:,}")
        logger.info(f"📈 Confidence trung bình: {stats['average_confidence']:.2%}")

        if stats['low_confidence_count'] > 0:
            logger.warning(f"⚠️ Items có Confidence thấp: {stats['low_confidence_count']}")

        logger.info("\n📋 Phân bố Content:")
        for content_type, count in stats['content_distribution'].items():
            logger.info(f"   {content_type}: {count}")

        logger.info("\n🗂️ Files đã tạo:")
        for file_type, path in results['file_paths'].items():
            logger.info(f"   {file_type}: {path}")

        logger.info("=" * 60)


def main():
    """Command line interface for Phase 1 parser với debug support"""
    import argparse

    parser = argparse.ArgumentParser(description='Phase 1: HTML Content Parser')
    parser.add_argument('--input', required=True, help='HTML file path to parse')
    parser.add_argument('--output-name', help='Output directory name (default: filename)')
    parser.add_argument('--rules', default='pattern_rules.yaml', help='Pattern rules file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("🐛 Debug logging enabled")

    # Initialize parser
    content_parser = HTMLContentParser(args.rules)

    # Parse document
    results = content_parser.parse_document(args.input, args.output_name)

    if results['success']:
        print("\n🎉 THÀNH CÔNG! Phase 1 parsing hoàn thành.")
        print(f"📁 Kết quả đã lưu tại: {results['output_directory']}")

        # Show quick stats
        stats = results['processing_stats']
        print(f"📊 Đã xử lý {stats['processed_sections']}/{stats['total_sections']} sections")
        print(f"📝 Tạo ra {stats['total_content_items']} content items")
        print(f"🔤 Ước tính {stats['total_estimated_tokens']:,} tokens")

        if stats['low_confidence_count'] > 0:
            print(f"⚠️ {stats['low_confidence_count']} items có confidence thấp (<70%)")

    else:
        print(f"\n❌ THẤT BẠI: {results['error']}")
        exit(1)


if __name__ == "__main__":
    main()