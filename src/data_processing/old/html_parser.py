import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup, Tag, NavigableString
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import logger
from src.data_processing.old import parser_utils


@dataclass
class ContentElement:
    """
    Đại diện cho một element trong mixed content stream.
    Giúp track đúng thứ tự của text và code trong document.
    """
    type: str  # 'text', 'code', 'table'
    content: str
    order: int
    language: Optional[str] = None  # For code elements
    title: Optional[str] = None     # Descriptive title for code/table
    context: Optional[str] = None   # Context description


@dataclass
class CodeBlock:
    """Đại diện cho một khối code trong documentation"""
    language: str  # 'python', 'csharp', 'cli', etc.
    content: str
    section_id: str  # ID của section chứa code block
    line_number: Optional[int] = None  # Dòng bắt đầu trong HTML gốc
    # Loại bỏ is_inline field

    def __post_init__(self):
        # Clean up code content
        self.content = self.content.strip()
        # Normalize language name
        if self.language.lower() in ['c#', 'cs', 'csharp']:
            self.language = 'csharp'
        elif self.language.lower() in ['py', 'python']:
            self.language = 'python'
        elif self.language.lower() in ['cli', 'bash', 'shell', 'cmd']:
            self.language = 'cli'


@dataclass
class TableData:
    """Đại diện cho một bảng trong documentation"""
    headers: List[str]
    rows: List[List[str]]
    section_id: str
    caption: Optional[str] = None


@dataclass
class Section:
    """
    Đại diện cho một section trong documentation.
    Mỗi section có thể chứa text, code blocks, tables, và subsections.
    """
    id: str  # ID unique từ HTML (dùng cho navigation) - e.g., "1.2.3"
    title: str
    level: int  # Được xác định từ số dấu chấm trong ID

    # NEW: Mixed content tracking (preserves exact order)
    mixed_content: List[ContentElement] = field(default_factory=list)

    # Traditional fields (computed properties for backward compatibility)
    _content: Optional[str] = None
    _code_blocks: Optional[List[CodeBlock]] = None

    tables: List[TableData] = field(default_factory=list)
    subsections: List['Section'] = field(default_factory=list)
    parent_id: Optional[str] = None
    section_number: Optional[str] = None  # e.g., "1.2.3"
    breadcrumb: Optional[str] = None  # Full breadcrumb path

    @property
    def content(self) -> str:
        """
        Backward compatible content property.
        Generates from mixed_content if available, otherwise returns stored content.
        """
        if self._content is not None:
            return self._content

        # Generate from mixed_content
        if self.mixed_content:
            text_parts = []
            for element in self.mixed_content:
                if element.type == "text":
                    text_parts.append(element.content)
                elif element.type == "code":
                    # Add placeholder for code in text flow
                    title = element.title or f"{element.language} code"
                    text_parts.append(f"[{title}]")
            return " ".join(text_parts)

        return ""

    @content.setter
    def content(self, value: str):
        """Allow setting content for backward compatibility"""
        self._content = value

    @property
    def code_blocks(self) -> List[CodeBlock]:
        """
        Backward compatible code_blocks property.
        Generates from mixed_content if available, otherwise returns stored code_blocks.
        """
        if self._code_blocks is not None:
            return self._code_blocks

        # Generate from mixed_content
        if self.mixed_content:
            codes = []
            for element in self.mixed_content:
                if element.type == "code":
                    codes.append(CodeBlock(
                        language=element.language or 'text',
                        content=element.content,
                        section_id=self.id
                    ))
            return codes

        return []

    @code_blocks.setter
    def code_blocks(self, value: List[CodeBlock]):
        """Allow setting code_blocks for backward compatibility"""
        self._code_blocks = value

    def get_full_path(self) -> str:
        """Trả về đường dẫn đầy đủ của section (e.g., "1.2.3 Section Title")"""
        if self.section_number:
            return f"{self.section_number} {self.title}"
        return self.title

    def to_dict(self) -> Dict:
        """Convert section thành dictionary để dễ serialize"""
        return {
            'id': self.id,
            'title': self.title,
            'level': self.level,
            'section_number': self.section_number,
            'breadcrumb': self.breadcrumb,
            'mixed_content': [
                {
                    'type': elem.type,
                    'content': elem.content,
                    'order': elem.order,
                    'language': elem.language,
                    'title': elem.title,
                    'context': elem.context
                } for elem in self.mixed_content
            ],
            'tables': [
                {
                    'headers': t.headers,
                    'rows': t.rows,
                    'caption': t.caption
                } for t in self.tables
            ],
            'subsections': [s.to_dict() for s in self.subsections],
            'parent_id': self.parent_id
        }


class QuantConnectHTMLParser:
    """
    FIXED Parser chính cho QuantConnect HTML documentation.
    Sửa lại để xử lý đúng cấu trúc code blocks thực tế.
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_name = file_path.name
        self.soup: Optional[BeautifulSoup] = None
        self.toc_structure: Dict = {}  # Lưu structure từ Table of Contents
        self.sections: List[Section] = []  # Danh sách các sections đã parse
        self.section_map: Dict[str, Section] = {}  # Map section ID to Section object

        logger.info(f"Initializing ENHANCED parser for: {self.file_name}")

    def parse(self, target_document_index=0) -> List[Section]:
        """
        Main parsing method - orchestrates toàn bộ quá trình parsing.
        """
        try:
            logger.info(f"Starting to parse {self.file_name}")

            # Step 1: Load HTML file
            self._load_html(target_document_index=target_document_index)

            # Step 2: Parse Table of Contents
            self._parse_table_of_contents()

            # Step 3: Remove unnecessary elements (but preserve code structures)
            self._clean_html_preserve_code()

            # Step 4: Parse content sections with mixed content tracking
            self._parse_content_sections()

            # Step 5: Build hierarchy
            self._build_section_hierarchy()

            # Step 6: Post-process và validate
            self._post_process()

            logger.info(f"Successfully parsed {len(self.sections)} sections from {self.file_name}")
            logger.info(f"Found {sum(len(s.code_blocks) for s in self.sections)} total code blocks")
            logger.info(f"Mixed content elements: {sum(len(s.mixed_content) for s in self.sections)}")

            return self.sections

        except Exception as e:
            logger.error(f"Error parsing {self.file_name}: {str(e)}")
            raise

    def _load_html(self, target_document_index=0):
        """Load HTML file vào BeautifulSoup"""
        logger.info(f"Loading document at index {target_document_index} from {self.file_path}")

        if not self.file_path.exists():
            raise FileNotFoundError(f"HTML file not found: {self.file_path}")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            full_file_content = f.read()

        # Find DOCTYPE declarations
        doctype_pattern = r'<!DOCTYPE\s+html[^>]*>'
        starts = [m.start() for m in re.finditer(doctype_pattern, full_file_content, flags=re.IGNORECASE)]

        if not starts:
            if target_document_index == 0:
                selected_document_content = full_file_content
            else:
                raise IndexError(f"No <!DOCTYPE html> found. Cannot get document at index {target_document_index}.")
        elif target_document_index < len(starts):
            start_pos = starts[target_document_index]
            end_pos = starts[target_document_index + 1] if (target_document_index + 1) < len(starts) else len(
                full_file_content)
            selected_document_content = full_file_content[start_pos:end_pos].strip()
        else:
            raise IndexError(
                f"Target document index {target_document_index} out of range. File contains {len(starts)} documents.")

        if not selected_document_content:
            raise ValueError(f"Extracted document content for index {target_document_index} is empty.")

        # Remove DOCTYPE và parse
        html_content_for_soup = re.sub(r'<!DOCTYPE[^>]*>', '', selected_document_content, count=1, flags=re.IGNORECASE)
        self.soup = BeautifulSoup(html_content_for_soup, 'html.parser')

        logger.info(f"Loaded HTML for document index {target_document_index}")

    def _clean_html_preserve_code(self):
        """
        Loại bỏ các elements không cần thiết nhưng GIỮ LẠI code structures.
        """
        logger.info("Cleaning HTML while preserving code structures...")

        # Remove scripts, styles, etc. but NOT pre, code, or div.section-example-container
        elements_to_remove = [
            'script',
            'style', 
            'link',
            'meta',
            'img',
            'video',
            'audio',
            'iframe',
            'embed',
            'object'
        ]

        for selector in elements_to_remove:
            for element in self.soup.select(selector):
                element.decompose()

        # KEEP nav elements with ToC
        for nav in self.soup.find_all('nav'):
            has_toc_links = nav.find('a', class_=re.compile(r'toc-h\d+'))
            if not has_toc_links:
                nav.decompose()

        logger.info("HTML cleaned while preserving code structures")

    def _parse_table_of_contents(self):
        """Parse Table of Contents"""
        logger.info("Parsing Table of Contents...")

        toc_heading = None
        for heading in self.soup.find_all(['h3', 'h2', 'h4']):
            if 'table of content' in heading.get_text().lower():
                toc_heading = heading
                break

        if not toc_heading:
            logger.warning("Table of Contents heading not found")
            return

        # Find nav element
        toc_container = None
        current = toc_heading.next_sibling
        while current and not toc_container:
            if isinstance(current, Tag) and current.name == 'nav':
                toc_container = current
                break
            current = current.next_sibling

        if not toc_container:
            logger.warning("Table of Contents nav element not found")
            return

        # Parse ToC entries
        toc_entries = toc_container.find_all('a', href=re.compile('^#'))

        for entry in toc_entries:
            href = entry.get('href', '').lstrip('#')
            full_text = entry.get_text(strip=True)

            match = re.match(r'^([\d\.]+)\s+(.+)$', full_text)
            if match:
                section_id = match.group(1)
                title = match.group(2)
            else:
                section_id = href
                title = full_text

            # Get level from class
            level = 1
            entry_classes = entry.get('class', [])
            for class_name in entry_classes:
                match = re.search(r'toc-h(\d+)', str(class_name))
                if match:
                    level = int(match.group(1))
                    break

            calculated_level = len(section_id.split('.'))
            if calculated_level != level:
                level = calculated_level

            self.toc_structure[section_id] = {
                'id': section_id,
                'title': title,
                'level': level,
                'href': href,
                'full_text': full_text
            }

        logger.info(f"Found {len(self.toc_structure)} entries in Table of Contents")

    def _parse_content_sections(self):
        """ENHANCED: Parse content sections với mixed content tracking"""
        logger.info("Parsing content sections with mixed content tracking...")
        
        # DEBUG: Check for all data-tree elements in entire document
        all_data_trees = self.soup.find_all('div', attrs={'data-tree': True})
        logger.info(f"Total data-tree elements found in document: {len(all_data_trees)}")
        for dt in all_data_trees:
            logger.info(f"  - data-tree: {dt.get('data-tree')}")
            section_elements = self.soup.find_all('section', id=True)

        if not section_elements:
            logger.warning("No <section> elements found, falling back to heading-based parsing")
            self._parse_sections_from_headings()
            return

        for section_elem in tqdm(section_elements, desc="Parsing sections"):
            section_id = section_elem.get('id', '')

            if not section_id:
                continue

            # Get ToC info
            toc_info = self.toc_structure.get(section_id, {})

            # Extract title và breadcrumb
            title = ""
            breadcrumb = ""

            breadcrumb_elem = section_elem.find_previous('p', class_='page-breadcrumb')
            if breadcrumb_elem:
                breadcrumb = breadcrumb_elem.get_text(strip=True)

            h1 = section_elem.find('h1')
            h2 = section_elem.find('h2')

            if h2:
                title = h2.get_text(strip=True)
            elif h1:
                title = h1.get_text(strip=True)
            else:
                title = toc_info.get('title', f'Section {section_id}')

            level = toc_info.get('level', len(section_id.split('.')))

            # Create Section object
            section = Section(
                id=section_id,
                title=title,
                level=level,
                section_number=section_id,
                breadcrumb=breadcrumb
            )

            # ENHANCED: Parse content với mixed content tracking
            self._parse_section_content_with_mixed_tracking(section_elem, section)

            self.sections.append(section)
            self.section_map[section_id] = section

        logger.info(f"Parsed {len(self.sections)} content sections")

    def _parse_section_content_with_mixed_tracking(self, element: Tag, section: Section):
        """
        ENHANCED: Parse nội dung section và track mixed_content order.
        Giữ nguyên 100% logic xử lý text cũ để ensure backward compatibility.
        """
        # Traditional content parts (for backward compatibility)
        content_parts = []

        # NEW: Mixed content tracking
        mixed_elements = []
        order = 1

        # Find the parent div.page-heading
        page_heading_div = element.parent
        if not page_heading_div or page_heading_div.name != 'div' or 'page-heading' not in page_heading_div.get('class',
                                                                                                                []):
            page_heading_div = element

        # DEBUG: Kiểm tra data-tree trong phạm vi section này
        logger.info(f"=== PARSING SECTION {section.id}: {section.title} ===")

        # Find the parent div.page-heading
        page_heading_div = element.parent
        if not page_heading_div or page_heading_div.name != 'div' or 'page-heading' not in page_heading_div.get('class',
                                                                                                                []):
            page_heading_div = element

        # DEBUG: Check data-tree trong toàn bộ phạm vi section
        current_check = page_heading_div.next_sibling
        section_data_trees = []
        sibling_count = 0

        while current_check:
            sibling_count += 1
            if isinstance(current_check, Tag):
                logger.debug(f"  Sibling {sibling_count}: {current_check.name}, classes: {current_check.get('class')}")

                # Check direct data-tree
                if current_check.get('data-tree'):
                    section_data_trees.append(f"Direct: {current_check.get('data-tree')}")
                    logger.info(f"  ✓ FOUND DIRECT DATA-TREE: {current_check.get('data-tree')}")

                # Check nested data-tree
                nested = current_check.find_all('div', attrs={'data-tree': True})
                for n in nested:
                    section_data_trees.append(f"Nested: {n.get('data-tree')}")
                    logger.info(f"  ✓ FOUND NESTED DATA-TREE: {n.get('data-tree')}")

                # Stop conditions
                if current_check.name == 'p' and 'page-breadcrumb' in current_check.get('class', []):
                    logger.debug(f"  Stopping at breadcrumb")
                    break
                if current_check.name == 'div' and 'page-heading' in current_check.get('class', []):
                    logger.debug(f"  Stopping at next page-heading")
                    break

            current_check = current_check.next_sibling

            # Safety break
            if sibling_count > 50:
                logger.warning(f"  Breaking after {sibling_count} siblings")
                break

        logger.info(f"Section {section.id} total siblings checked: {sibling_count}")
        logger.info(f"Section {section.id} data-trees found: {section_data_trees}")

        # Start looking for content after the page-heading div
        current = page_heading_div.next_sibling

        # Process siblings until we hit the next section or page break
        while current:
            if isinstance(current, Tag):
                # Check if this is our target div
                if current.get_text() and "Backtest Handled Error" in current.get_text():
                    logger.info(f"🎯 FOUND TARGET DIV: {current.name}, classes: {current.get('class', [])}")

                logger.debug(f"Processing: {current.name}, classes: {current.get('class', [])}")

                # Stop conditions
                logger.debug(f"Processing element: {current.name}, classes: {current.get('class')}, data-tree: {current.get('data-tree')}")
            
                if current.name == 'p' and 'page-breadcrumb' in current.get('class', []):
                    break

                if current.name == 'div' and 'page-heading' in current.get('class', []):
                    break

                # Skip page breaks
                if current.name == 'p' and current.get('style') and 'page-break' in current.get('style'):
                    current = current.next_sibling
                    continue

                # Process data-tree
                data_tree_processed = False

                # Check direct data-tree
                if current.get('data-tree'):
                    logger.info(f"🎯 PROCESSING DIRECT DATA-TREE: {current.get('data-tree')}")
                    new_order = self._process_patched_data_tree_element(current, section, mixed_elements, order)
                    if isinstance(new_order, int):
                        order = new_order
                    data_tree_processed = True

                # Check nested data-tree
                nested_trees = current.find_all('div', attrs={'data-tree': True})
                for tree_elem in nested_trees:
                    logger.info(f"🎯 PROCESSING NESTED DATA-TREE: {tree_elem.get('data-tree')}")
                    new_order = self._process_patched_data_tree_element(tree_elem, section, mixed_elements, order)
                    if isinstance(new_order, int):
                        order = new_order

                # For direct data-tree, skip other processing of this element
                if data_tree_processed and current.get('data-tree'):
                    current = current.next_sibling
                    continue

                # Process different content types
                if current.name == 'h3':
                    heading_text = current.get_text(strip=True)
                    if heading_text:
                        content_parts.append(f"### {heading_text}")
                        # Track in mixed content
                        mixed_elements.append(ContentElement(
                            type="text",
                            content=f"### {heading_text}",
                            order=order
                        ))
                        order += 1

                elif current.name == 'html':
                    # Main content block
                    body = current.find('body')
                    if body:
                        self._extract_content_from_body_with_mixed_tracking(
                            body, section, content_parts, mixed_elements, order
                        )
                        order += len([e for e in mixed_elements if e.order >= order])

                elif current.name == 'pre':
                    # Direct code block
                    code_info = self._extract_code_block_improved_with_mixed_tracking(current, section)
                    if code_info:
                        mixed_elements.append(ContentElement(
                            type="code",
                            content=code_info['content'],
                            language=code_info['language'],
                            title=code_info.get('title', f"{code_info['language']} code"),
                            order=order
                        ))
                        order += 1

                elif current.name == 'table':
                    self._extract_table(current, section)
                    # Track table in mixed content
                    mixed_elements.append(ContentElement(
                        type="table",
                        content="[Table Data]",
                        title="Data Table",
                        order=order
                    ))
                    order += 1
                else:
                    # GIỮ NGUYÊN logic xử lý text với inline code
                    text_with_inline_code = self._process_text_with_inline_code_and_extract_containers_with_mixed_tracking(
                        current, section, mixed_elements, order
                    )
                    if text_with_inline_code and text_with_inline_code not in content_parts:
                        content_parts.append(text_with_inline_code)

            current = current.next_sibling

        # Set traditional content (backward compatibility)
        section.content = '\n\n'.join(content_parts)

        # Set mixed content (new functionality)
        section.mixed_content = mixed_elements

    def _extract_content_from_body_with_mixed_tracking(
            self,
            body: Tag,
            section: Section,
            content_parts: List[str],
            mixed_elements: List[ContentElement],
            start_order: int
    ):
        """
        ENHANCED: Extract content from <body> với mixed content tracking.
        Giữ nguyên logic xử lý text cũ.
        """
        current_order = start_order

        for element in body.children:
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text:
                    content_parts.append(text)
                    # Track in mixed content
                    mixed_elements.append(ContentElement(
                        type="text",
                        content=text,
                        order=current_order
                    ))
                    current_order += 1

            elif isinstance(element, Tag):
                # ✅ NEW: Special div handling
                if element.name == 'div':
                    div_classes = element.get('class', [])
                    div_classes_str = ' '.join(str(c) for c in div_classes)

                    # Handle error messages
                    if 'error-messages' in div_classes_str:
                        error_content = self._extract_error_message(element)
                        if error_content:
                            content_parts.append(error_content)
                            mixed_elements.append(ContentElement(
                                type="text",
                                content=error_content,
                                title="Error Message",
                                order=current_order
                            ))
                            current_order += 1
                            logger.debug(f"✅ Added error message in body: {error_content[:50]}...")
                        continue  # Skip normal processing

                    # Handle tutorial steps
                    elif 'tutorial-step' in div_classes_str:
                        tutorial_content = self._extract_tutorial_step(element)
                        if tutorial_content:
                            content_parts.append(tutorial_content)
                            mixed_elements.append(ContentElement(
                                type="text",
                                content=tutorial_content,
                                title="Tutorial Step",
                                order=current_order
                            ))
                            current_order += 1
                            logger.debug(f"✅ Added tutorial step in body: {tutorial_content[:50]}...")
                        continue  # Skip normal processing

                    # Handle example fieldsets
                    elif 'example-fieldset' in div_classes_str:
                        example_content = self._extract_example_fieldset(element)
                        if example_content:
                            content_parts.append(example_content)
                            mixed_elements.append(ContentElement(
                                type="text",
                                content=example_content,
                                title="Example Algorithms",
                                order=current_order
                            ))
                            current_order += 1
                            logger.debug(f"✅ Added example fieldset in body: {example_content[:50]}...")
                        continue  # Skip normal processing

                if element.name == 'pre':
                    # Code block
                    code_info = self._extract_code_block_improved_with_mixed_tracking(element, section)
                    if code_info:
                        content_parts.append("[Code Block]")
                        # Track in mixed content
                        mixed_elements.append(ContentElement(
                            type="code",
                            content=code_info['content'],
                            language=code_info['language'],
                            title=code_info.get('title', f"{code_info['language']} code"),
                            order=current_order
                        ))
                        current_order += 1

                elif element.name == 'table':
                    self._extract_table(element, section)
                    content_parts.append("[Table]")
                    # Track in mixed content
                    mixed_elements.append(ContentElement(
                        type="table",
                        content="[Table Data]",
                        title="Data Table",
                        order=current_order
                    ))
                    current_order += 1

                else:
                    # GIỮ NGUYÊN logic xử lý text với inline code
                    text_with_inline = self._process_text_with_inline_code_and_extract_containers_with_mixed_tracking(
                        element, section, mixed_elements, current_order
                    )
                    if text_with_inline:
                        # Skip JSON metadata
                        if not (text_with_inline.startswith('{') and text_with_inline.endswith('}')):
                            content_parts.append(text_with_inline)

    def _extract_code_block_improved_with_mixed_tracking(self, pre_element: Tag, section: Section) -> Optional[Dict]:
        """
        ENHANCED: Extract code block và return info for mixed content tracking.
        Giữ nguyên 100% logic detection cũ.
        """
        # Kiểm tra xem pre element này có phải là part của code container không
        parent = pre_element.parent
        if parent and parent.name == 'div':
            parent_classes = parent.get('class', [])
            parent_classes_str = ' '.join(str(c) for c in parent_classes)
            if 'section-example-container' not in parent_classes_str:
                # Đây không phải code container, skip
                return None

        # GIỮ NGUYÊN logic detect language
        language = 'text'  # default
        pre_classes = pre_element.get('class', [])

        for class_name in pre_classes:
            class_str = str(class_name).lower()
            if 'python' in class_str:
                language = 'python'
                break
            elif 'csharp' in class_str or 'c#' in class_str:
                language = 'csharp'
                break
            elif 'cli' in class_str or 'bash' in class_str or 'shell' in class_str:
                language = 'cli'
                break

        # Check parent div classes if pre doesn't have language class
        if language == 'text' and parent:
            parent_classes = parent.get('class', [])
            for class_name in parent_classes:
                class_str = str(class_name).lower()
                if 'python' in class_str:
                    language = 'python'
                    break
                elif 'csharp' in class_str:
                    language = 'csharp'
                    break
                elif 'cli' in class_str:
                    language = 'cli'
                    break

        # GIỮ NGUYÊN logic extract content
        code_elem = pre_element.find('code')
        if code_elem:
            code_content = code_elem.get_text(strip=False)
            # Check code element classes too
            code_classes = code_elem.get('class', [])
            for class_name in code_classes:
                class_str = str(class_name).lower()
                if 'python' in class_str:
                    language = 'python'
                elif 'csharp' in class_str:
                    language = 'csharp'
                elif 'cli' in class_str:
                    language = 'cli'
        else:
            code_content = pre_element.get_text(strip=False)

        # Final language detection từ content nếu vẫn chưa xác định
        if language == 'text':
            language = self._detect_language_from_content(code_content)

        # GIỮ NGUYÊN việc tạo CodeBlock cho backward compatibility
        if code_content.strip():  # Only create if has content
            code_block = CodeBlock(
                language=language,
                content=code_content,
                section_id=section.id
            )
            # Ensure section has _code_blocks list
            if section._code_blocks is None:
                section._code_blocks = []
            section._code_blocks.append(code_block)
            logger.debug(f"Added {language} code block to section {section.id}")

            # Return info for mixed content tracking
            return {
                'content': code_content,
                'language': language,
                'title': self._generate_code_title_from_context(language)
            }

        return None

    def _generate_code_title_from_context(self, language: str) -> str:
        """Generate descriptive title for code blocks"""
        return parser_utils.generate_code_title(language)

    def _process_text_with_inline_code_and_extract_containers_with_mixed_tracking(
            self,
            element: Tag,
            section: Section,
            mixed_elements: List[ContentElement],
            current_order: int
    ) -> str:
        """
        ENHANCED: Process text với mixed content tracking.
        ✅ UPDATED: Thêm special div search cho nested elements.
        """
        special_content_found = []
        order = current_order

        # ✅ NEW: Search for special divs within this element using find_all
        # Error messages
        error_divs = element.find_all('div', class_='error-messages')
        for error_div in error_divs:
            error_content = self._extract_error_message(error_div)
            if error_content:
                mixed_elements.append(ContentElement(
                    type="text",
                    content=error_content,
                    title="Error Message",
                    order=order
                ))
                special_content_found.append(error_content)
                order += 1
                logger.debug(f"✅ Found nested error message: {error_content[:50]}...")

        # Tutorial steps
        tutorial_divs = element.find_all('div', class_='tutorial-step')
        for tutorial_div in tutorial_divs:
            tutorial_content = self._extract_tutorial_step(tutorial_div)
            if tutorial_content:
                mixed_elements.append(ContentElement(
                    type="text",
                    content=tutorial_content,
                    title="Tutorial Step",
                    order=order
                ))
                special_content_found.append(tutorial_content)
                order += 1
                logger.debug(f"✅ Found nested tutorial step: {tutorial_content[:50]}...")

        # Example fieldsets
        example_divs = element.find_all('div', class_='example-fieldset')
        for example_div in example_divs:
            example_content = self._extract_example_fieldset(example_div)
            if example_content:
                mixed_elements.append(ContentElement(
                    type="text",
                    content=example_content,
                    title="Example Algorithms",
                    order=order
                ))
                special_content_found.append(example_content)
                order += 1
                logger.debug(f"✅ Found nested example fieldset: {example_content[:50]}...")

        # If we found special content, return it
        if special_content_found:
            return '\n\n'.join(special_content_found)
        # GIỮ NGUYÊN logic extract code containers
        self._extract_all_code_containers_recursive_with_mixed_tracking(element, section, mixed_elements, current_order)

        # GIỮ NGUYÊN logic format inline code
        text_result = self._get_text_with_inline_code_formatting(element)

        # Track text trong mixed content nếu có substantial content
        if text_result and len(text_result.strip()) > 10:
            mixed_elements.append(ContentElement(
                type="text",
                content=text_result,
                order=current_order
            ))

        return text_result

    def _extract_all_code_containers_recursive_with_mixed_tracking(
            self,
            element: Tag,
            section: Section,
            mixed_elements: List[ContentElement],
            current_order: int
    ):
        """
        ENHANCED: Recursively find and extract code containers với mixed tracking.
        GIỮ NGUYÊN 100% logic detection cũ.
        """
        if not isinstance(element, Tag):
            return

        # GIỮ NGUYÊN logic check code container
        if element.name == 'div':
            div_classes = element.get('class', [])
            div_classes_str = ' '.join(str(c) for c in div_classes)

            if 'section-example-container' in div_classes_str:
                # This is a code container - extract it
                pre_elements = element.find_all('pre')
                for pre_elem in pre_elements:
                    code_info = self._extract_code_block_improved_with_mixed_tracking(pre_elem, section)
                    if code_info:
                        # Track in mixed content
                        mixed_elements.append(ContentElement(
                            type="code",
                            content=code_info['content'],
                            language=code_info['language'],
                            title=code_info.get('title', f"{code_info['language']} code"),
                            order=current_order
                        ))
                return  # Don't process children since we've handled this container

        # GIỮ NGUYÊN recursive logic
        for child in element.children:
            if isinstance(child, Tag):
                self._extract_all_code_containers_recursive_with_mixed_tracking(child, section, mixed_elements,
                                                                                current_order)


    def _extract_code_block_improved(self, pre_element: Tag, section: Section):
        """
        IMPROVED: Extract code block với better language detection.
        """
        # Kiểm tra xem pre element này có phải là part của code container không
        parent = pre_element.parent
        if parent and parent.name == 'div':
            parent_classes = parent.get('class', [])
            parent_classes_str = ' '.join(str(c) for c in parent_classes)
            if 'section-example-container' not in parent_classes_str:
                # Đây không phải code container, skip
                return

        # Determine language từ pre element classes
        language = 'text'  # default
        pre_classes = pre_element.get('class', [])

        for class_name in pre_classes:
            class_str = str(class_name).lower()
            if 'python' in class_str:
                language = 'python'
                break
            elif 'csharp' in class_str or 'c#' in class_str:
                language = 'csharp'
                break
            elif 'cli' in class_str or 'bash' in class_str or 'shell' in class_str:
                language = 'cli'
                break

        # Check parent div classes if pre doesn't have language class
        if language == 'text' and parent:
            parent_classes = parent.get('class', [])
            for class_name in parent_classes:
                class_str = str(class_name).lower()
                if 'python' in class_str:
                    language = 'python'
                    break
                elif 'csharp' in class_str:
                    language = 'csharp'
                    break
                elif 'cli' in class_str:
                    language = 'cli'
                    break

        # Extract code content
        code_elem = pre_element.find('code')
        if code_elem:
            code_content = code_elem.get_text(strip=False)
            # Check code element classes too
            code_classes = code_elem.get('class', [])
            for class_name in code_classes:
                class_str = str(class_name).lower()
                if 'python' in class_str:
                    language = 'python'
                elif 'csharp' in class_str:
                    language = 'csharp'
                elif 'cli' in class_str:
                    language = 'cli'
        else:
            code_content = pre_element.get_text(strip=False)

        # Final language detection từ content nếu vẫn chưa xác định
        if language == 'text':
            language = self._detect_language_from_content(code_content)

        # Create CodeBlock object
        if code_content.strip():  # Only create if has content
            code_block = CodeBlock(
                language=language,
                content=code_content,
                section_id=section.id
            )
            section.code_blocks.append(code_block)
            logger.debug(f"Added {language} code block to section {section.id}")

    def _detect_language_from_content(self, code_content: str) -> str:
        """Detect programming language từ code content"""
        return parser_utils.detect_code_language(code_content)

    def _extract_table(self, table_element: Tag, section: Section):
        """Extract table data"""
        table_data = parser_utils.extract_table_data(table_element, section.id)
        if table_data:
            section.tables.append(table_data)

    def _parse_sections_from_headings(self):
        """GIỮ NGUYÊN: Fallback method khi không có section tags"""
        logger.info("Parsing sections from heading elements...")

        headings = self.soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

        for heading in tqdm(headings, desc="Parsing sections"):
            if 'table of content' in heading.get_text().lower():
                continue

            section_id = heading.get('id', '')
            if not section_id:
                parent_section = heading.find_parent('section', id=True)
                if parent_section:
                    section_id = parent_section.get('id', '')

            if not section_id:
                section_id = self._generate_section_id(heading.get_text())

            section_title = heading.get_text(strip=True)
            section_level = int(heading.name[1])

            section = Section(
                id=section_id,
                title=section_title,
                level=section_level
            )

            # Parse content around this heading
            self._parse_content_around_heading(heading, section)

            self.sections.append(section)
            self.section_map[section_id] = section

    def _parse_content_around_heading(self, heading: Tag, section: Section):
        """GIỮ NGUYÊN: Parse content around a heading element"""
        content_parts = []

        # Look for content after this heading
        current = heading.next_sibling

        while current:
            if isinstance(current, Tag):
                # Stop at next heading of same or higher level
                if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    current_level = int(current.name[1])
                    if current_level <= section.level:
                        break

                # Process content
                if current.name == 'pre':
                    self._extract_code_block_improved_with_mixed_tracking(current, section)
                elif current.name in ['p', 'ul', 'ol']:
                    # GIỮ NGUYÊN logic xử lý inline code
                    text = self._get_text_with_inline_code_formatting(current)
                    if text:
                        content_parts.append(text)

            current = current.next_sibling

        section.content = '\n\n'.join(content_parts)

    def _build_section_hierarchy(self):
        """GIỮ NGUYÊN: Build hierarchy for sections"""
        logger.info("Building section hierarchy...")

        self.sections.sort(key=lambda s: [int(x) for x in s.id.split('.') if x.isdigit()])

        for section in self.sections:
            if '.' in section.id:
                parent_id = '.'.join(section.id.split('.')[:-1])
                if parent_id in self.section_map:
                    parent = self.section_map[parent_id]
                    section.parent_id = parent_id
                    parent.subsections.append(section)

        logger.info("Section hierarchy built successfully")

    def _generate_section_id(self, title: str) -> str:
        """Generate unique section ID from title"""
        return parser_utils.generate_section_id(title)

    def _post_process(self):
        """GIỮ NGUYÊN: Post-process sections"""
        logger.info("Post-processing parsed sections...")

        # Remove empty sections
        self.sections = [
            s for s in self.sections
            if s.content or s.code_blocks or s.tables or s.subsections or s.mixed_content
        ]

        # Update section map
        self.section_map = {s.id: s for s in self.sections}

        # Clean content
        for section in self.sections:
            if section._content:  # Only clean if we have stored content
                section._content = parser_utils.clean_text_content(section._content)

        logger.info(f"Post-processing complete. Final: {len(self.sections)} sections, "
                   f"{sum(len(s.code_blocks) for s in self.sections)} code blocks, "
                   f"{sum(len(s.mixed_content) for s in self.sections)} mixed content elements")

    def save_parsed_data(self, output_dir: Path):
        """Save parsed data to JSON file"""
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = self.file_path.stem
        output_file = output_dir / f"{base_name}_parsed.json"

        # Include enhanced statistics
        total_code_blocks = sum(len(s.code_blocks) for s in self.sections)
        total_mixed_elements = sum(len(s.mixed_content) for s in self.sections)

        languages = set()
        for s in self.sections:
            for cb in s.code_blocks:
                languages.add(cb.language)

        data = {
            'source_file': self.file_name,
            'sections': [s.to_dict() for s in self.sections],
            'statistics': {
                'total_sections': len(self.sections),
                'total_code_blocks': total_code_blocks,
                'total_tables': sum(len(s.tables) for s in self.sections),
                'total_mixed_elements': total_mixed_elements,
                'languages': list(languages),
                'enhanced_parsing': True  # Flag to indicate mixed content support
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved parsed data to: {output_file}")
        logger.info(f"Statistics: {data['statistics']}")

        return output_file

    def _extract_all_code_containers_recursive(self, element: Tag, section: Section):
        """
        UPDATED: Recursively find and extract code containers
        """
        if not isinstance(element, Tag):
            return

        # Check if this element itself is a code container
        if element.name == 'div':
            div_classes = element.get('class', [])
            div_classes_str = ' '.join(str(c) for c in div_classes)

            if 'section-example-container' in div_classes_str:
                # This is a code container - extract it
                pre_elements = element.find_all('pre')
                for pre_elem in pre_elements:
                    self._extract_code_block_improved(pre_elem, section)
                return  # Don't process children since we've handled this container

        # Recursively process children only if current element is not a code container
        for child in element.children:
            if isinstance(child, Tag):
                self._extract_all_code_containers_recursive(child, section)

    def _get_text_with_inline_code_formatting(self, element: Tag) -> str:
        if element.get_text() and "Backtest Handled Error" in element.get_text():
            logger.info(f"🎯 FOUND IN TEXT FORMATTING: {element.name}, classes: {element.get('class', [])}")
        """
        GIỮ NGUYÊN: Get text content và format inline code properly
        """
        result_parts = []

        for child in element.children:
            if isinstance(child, NavigableString):
                result_parts.append(str(child))
            elif isinstance(child, Tag):
                if child.name == 'code':
                    # Đây là inline code - format theo yêu cầu
                    formatted_code = parser_utils.format_inline_code(child)
                    if formatted_code:
                        result_parts.append(formatted_code)

                elif child.name == 'div':
                    div_classes = child.get('class', [])
                    div_classes_str = ' '.join(str(c) for c in div_classes)

                    if 'section-example-container' in div_classes_str:
                        # Code container đã được extract - skip content để tránh duplicate
                        pass  # Không add gì vào result_parts
                    else:
                        # Regular div - process recursively
                        text_content = self._get_text_with_inline_code_formatting(child)
                        if text_content:
                            result_parts.append(text_content)

                elif child.name == 'pre':
                    # Pre element có thể là code block - check if it's in a container
                    parent = child.parent
                    if parent and parent.name == 'div':
                        parent_classes = parent.get('class', [])
                        if any('section-example-container' in str(c) for c in parent_classes):
                            # Đây là code block trong container - skip để tránh duplicate
                            pass
                        else:
                            # Pre element độc lập - get text
                            result_parts.append(child.get_text())
                    else:
                        result_parts.append(child.get_text())

                else:
                    # Other tags - process recursively
                    text_content = self._get_text_with_inline_code_formatting(child)
                    if text_content:
                        result_parts.append(text_content)

        return ''.join(result_parts).strip()

    def _process_patched_data_tree_element(self, element: Tag, section: Section, mixed_elements: List[ContentElement],
                                           order: int) -> int:
        """Process patched data-tree elements"""
        data_tree_value = element.get('data-tree')
        logger.info(f"🔥 ENTERING _process_patched_data_tree_element with: {data_tree_value}")

        if not data_tree_value:
            logger.warning("No data-tree value found")
            return order

        # Check if this is a patched element
        content_container = element.find('div', class_='base-expandable-type')
        logger.info(f"Content container found: {content_container is not None}")

        if content_container:
            logger.info(f"Processing PATCHED data-tree: {data_tree_value}")

            # Extract Python content
            python_div = element.find('div', class_='python')
            logger.info(f"Python div found: {python_div is not None}")

            if python_div:
                python_content = self._extract_api_content_from_div(python_div, 'python')
                logger.info(f"Python content length: {len(python_content) if python_content else 0}")

                if python_content:
                    mixed_elements.append(ContentElement(
                        type="api_content",
                        content=python_content,
                        language="python",
                        title=f"API: {data_tree_value}",
                        context=f"Resolved from data-tree: {data_tree_value}",
                        order=order
                    ))
                    logger.info(f"✅ Added Python API content for {data_tree_value}")
                    order += 1

            # Extract C# content
            csharp_div = element.find('div', class_='csharp')
            logger.info(f"C# div found: {csharp_div is not None}")

            if csharp_div:
                csharp_content = self._extract_api_content_from_div(csharp_div, 'csharp')
                logger.info(f"C# content length: {len(csharp_content) if csharp_content else 0}")

                if csharp_content:
                    mixed_elements.append(ContentElement(
                        type="api_content",
                        content=csharp_content,
                        language="csharp",
                        title=f"API: {data_tree_value} (C#)",
                        context=f"Resolved from data-tree: {data_tree_value}",
                        order=order
                    ))
                    logger.info(f"✅ Added C# API content for {data_tree_value}")
                    order += 1
        else:
            logger.warning(f"UNPATCHED data-tree element: {data_tree_value}")

        logger.info(f"🔥 EXITING _process_patched_data_tree_element, returning order: {order}")
        return order

    def _extract_api_content_from_div(self, api_div: Tag, language: str) -> str:
        """Extract API content from language-specific div"""
        return parser_utils.extract_api_content_from_div(api_div, language)

    def _convert_api_html_to_text(self, html_content: str, data_tree_value: str) -> str:
        """Convert API HTML content to readable text format"""
        return parser_utils.convert_api_html_to_text(html_content, data_tree_value)

    def _extract_error_message(self, element: Tag) -> str:
        """Extract error message content with warning prefix"""
        return parser_utils.extract_error_message(element)

    def _extract_tutorial_step(self, element: Tag) -> str:
        """Extract tutorial step content from all paragraphs"""
        return parser_utils.extract_tutorial_step(element)

    def _extract_example_fieldset(self, element: Tag) -> str:
        """Extract example algorithm links with clean filenames"""
        return parser_utils.extract_example_fieldset(element)

# Test function
def test_fixed_parser():
    """Test the fixed parser"""
    print("Testing FIXED HTML Parser...")
    
if __name__ == "__main__":
    test_fixed_parser()