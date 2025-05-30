"""
FIXED HTML Parser cho QuantConnect Documentation
Sửa lại để xử lý đúng cấu trúc code blocks thực tế
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from bs4 import BeautifulSoup, Tag, NavigableString
import lxml
import re
from tqdm import tqdm
import json
import hashlib

# Import config và logger từ modules đã tạo
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import logger
from config.config import settings


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
    content: str  # Text content của section
    code_blocks: List[CodeBlock] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    subsections: List['Section'] = field(default_factory=list)
    parent_id: Optional[str] = None
    section_number: Optional[str] = None  # e.g., "1.2.3"
    breadcrumb: Optional[str] = None  # Full breadcrumb path

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
            'content': self.content,
            'section_number': self.section_number,
            'breadcrumb': self.breadcrumb,
            'code_blocks': [
                {
                    'language': cb.language,
                    'content': cb.content
                } for cb in self.code_blocks
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

        logger.info(f"Initializing FIXED parser for: {self.file_name}")

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

            # Step 4: Parse content sections
            self._parse_content_sections()

            # Step 5: Build hierarchy
            self._build_section_hierarchy()

            # Step 6: Post-process và validate
            self._post_process()

            logger.info(f"Successfully parsed {len(self.sections)} sections from {self.file_name}")
            logger.info(f"Found {sum(len(s.code_blocks) for s in self.sections)} total code blocks")
            
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
            end_pos = starts[target_document_index + 1] if (target_document_index + 1) < len(starts) else len(full_file_content)
            selected_document_content = full_file_content[start_pos:end_pos].strip()
        else:
            raise IndexError(f"Target document index {target_document_index} out of range. File contains {len(starts)} documents.")

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
        """Parse content sections với improved code detection"""
        logger.info("Parsing content sections with improved code detection...")

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
                content="",
                section_number=section_id,
                breadcrumb=breadcrumb
            )

            # Parse content với improved logic
            self._parse_section_content_improved(section_elem, section)

            self.sections.append(section)
            self.section_map[section_id] = section

        logger.info(f"Parsed {len(self.sections)} content sections")

    def _parse_section_content_improved(self, element: Tag, section: Section):
        """
        UPDATED: Parse nội dung của một section với improved recursive code detection
        """
        content_parts = []

        # Find the parent div.page-heading
        page_heading_div = element.parent
        if not page_heading_div or page_heading_div.name != 'div' or 'page-heading' not in page_heading_div.get('class', []):
            page_heading_div = element

        # Start looking for content after the page-heading div
        current = page_heading_div.next_sibling

        # Process siblings until we hit the next section or page break
        while current:
            if isinstance(current, Tag):
                # Stop conditions
                if current.name == 'p' and 'page-breadcrumb' in current.get('class', []):
                    break

                if current.name == 'div' and 'page-heading' in current.get('class', []):
                    break

                # Skip page breaks
                if current.name == 'p' and current.get('style') and 'page-break' in current.get('style'):
                    current = current.next_sibling
                    continue

                # Process different content types
                if current.name == 'h3':
                    heading_text = current.get_text(strip=True)
                    if heading_text:
                        content_parts.append(f"### {heading_text}")

                elif current.name == 'html':
                    # Main content block
                    body = current.find('body')
                    if body:
                        self._extract_content_from_body(body, section, content_parts)

                elif current.name == 'pre':
                    # Direct code block
                    self._extract_code_block_improved(current, section)

                elif current.name == 'table':
                    self._extract_table(current, section)

                else:
                    # UPDATED: Use new method that handles nested code containers
                    text_with_inline_code = self._process_text_with_inline_code_and_extract_containers(current, section)
                    if text_with_inline_code and text_with_inline_code not in content_parts:
                        content_parts.append(text_with_inline_code)

            current = current.next_sibling

        # Combine all text parts
        section.content = '\n\n'.join(content_parts)

    def _extract_content_from_body(self, body: Tag, section: Section, content_parts: List[str]):
        """
        UPDATED: Extract content from <body> with improved recursive code detection
        """
        for element in body.children:
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text:
                    content_parts.append(text)

            elif isinstance(element, Tag):
                if element.name == 'pre':
                    # Code block
                    self._extract_code_block_improved(element, section)
                    content_parts.append("[Code Block]")

                elif element.name == 'table':
                    self._extract_table(element, section)
                    content_parts.append("[Table]")

                else:
                    # UPDATED: Use new method that handles nested code containers
                    text_with_inline = self._process_text_with_inline_code_and_extract_containers(element, section)
                    if text_with_inline:
                        # Skip JSON metadata
                        if not (text_with_inline.startswith('{') and text_with_inline.endswith('}')):
                            content_parts.append(text_with_inline)
    
    def _handle_div_element(self, div_element: Tag, section: Section, content_parts: List[str]):
        """
        FIXED: Xử lý div elements, đặc biệt các div chứa code.
        """
        div_classes = div_element.get('class', [])
        div_classes_str = ' '.join(div_classes)

        # Check for code container divs
        if 'section-example-container' in div_classes_str:
            # This is a code container
            self._extract_code_from_container(div_element, section)
            content_parts.append("[Code Example]")  # Placeholder in text
            
        elif any(lang in div_classes_str for lang in ['python', 'csharp', 'cli']):
            # Language-specific div
            self._extract_code_from_container(div_element, section)
            content_parts.append("[Code Example]")
            
        else:
            # Regular div - extract text content
            text_content = self._process_text_with_inline_code(div_element, section)
            if text_content:
                content_parts.append(text_content)

    def _extract_code_from_container(self, container: Tag, section: Section):
        """
        FIXED: Extract code từ các container divs.
        """
        # Find all pre elements trong container
        pre_elements = container.find_all('pre')
        
        for pre_elem in pre_elements:
            self._extract_code_block_improved(pre_elem, section)

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
        content_lower = code_content.lower()
        
        # Python indicators
        python_indicators = [
            'import ', 'from ', 'def ', 'class ', 'self.', 'print(', '__init__',
            'import numpy', 'import pandas', 'def ', 'elif', 'True', 'False', 'None'
        ]
        
        # C# indicators  
        csharp_indicators = [
            'using ', 'namespace ', 'public class', 'private ', 'public ', 'void ',
            'string ', 'int ', 'var ', 'new ', '();', 'Console.', 'public override'
        ]
        
        # CLI indicators
        cli_indicators = [
            '$ ', 'conda ', 'pip ', 'dotnet ', 'git ', 'cd ', 'ls ', 'mkdir',
            '--', 'sudo ', 'chmod ', 'export '
        ]
        
        python_score = sum(1 for indicator in python_indicators if indicator in content_lower)
        csharp_score = sum(1 for indicator in csharp_indicators if indicator in content_lower)
        cli_score = sum(1 for indicator in cli_indicators if indicator in content_lower)
        
        if cli_score > 0:
            return 'cli'
        elif python_score > csharp_score:
            return 'python'
        elif csharp_score > 0:
            return 'csharp'
        else:
            return 'text'

    def _process_text_with_inline_code(self, element: Tag, section: Section) -> str:
        """
        FIXED: Process text content và handle inline code properly - KHÔNG tạo CodeBlock cho inline code
        """
        result_parts = []

        for child in element.children:
            if isinstance(child, NavigableString):
                result_parts.append(str(child))
            elif isinstance(child, Tag):
                if child.name == 'code':
                    # Đây là inline code - chỉ format trong text, KHÔNG tạo CodeBlock
                    code_content = child.get_text(strip=True)
                    if code_content:
                        # Determine language from class
                        language = None
                        code_classes = child.get('class', [])
                        for class_name in code_classes:
                            class_str = str(class_name).lower()
                            if 'python' in class_str:
                                language = 'python'
                                break
                            elif 'csharp' in class_str:
                                language = 'csharp'
                                break

                        # Format inline code trong text
                        if language:
                            result_parts.append(f"{code_content}({language})")
                        else:
                            result_parts.append(f"{code_content}(code)")
                else:
                    # Regular tag, get text content
                    result_parts.append(child.get_text())

        return ''.join(result_parts).strip()

    def _extract_table(self, table_element: Tag, section: Section):
        """Extract table data"""
        headers = []
        rows = []

        # Extract headers
        thead = table_element.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        else:
            first_row = table_element.find('tr')
            if first_row and first_row.find('th'):
                headers = [th.get_text(strip=True) for th in first_row.find_all('th')]

        # Extract rows
        tbody = table_element.find('tbody') or table_element
        for tr in tbody.find_all('tr'):
            if tr.find('th') and not rows:
                continue
            row_data = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if row_data:
                rows.append(row_data)

        # Extract caption
        caption_element = table_element.find('caption')
        caption = caption_element.get_text(strip=True) if caption_element else None

        if headers or rows:
            table_data = TableData(
                headers=headers,
                rows=rows,
                section_id=section.id,
                caption=caption
            )
            section.tables.append(table_data)

    def _parse_sections_from_headings(self):
        """Fallback method khi không có section tags"""
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
                level=section_level,
                content=""
            )

            # Parse content around this heading
            self._parse_content_around_heading(heading, section)

            self.sections.append(section)
            self.section_map[section_id] = section

    def _parse_content_around_heading(self, heading: Tag, section: Section):
        """Parse content around a heading element"""
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
                    self._extract_code_block_improved(current, section)
                elif current.name == 'div':
                    self._handle_div_element(current, section, content_parts)
                elif current.name in ['p', 'ul', 'ol']:
                    text = self._process_text_with_inline_code(current, section)
                    if text:
                        content_parts.append(text)
            
            current = current.next_sibling
        
        section.content = '\n\n'.join(content_parts)

    def _build_section_hierarchy(self):
        """Build hierarchy for sections"""
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
        section_id = title.lower().replace(' ', '-')
        section_id = re.sub(r'[^a-z0-9\-]', '', section_id)
        section_id = re.sub(r'-+', '-', section_id)
        hash_suffix = hashlib.md5(title.encode()).hexdigest()[:6]
        return f"{section_id}-{hash_suffix}"

    def _post_process(self):
        """Post-process sections"""
        logger.info("Post-processing parsed sections...")

        # Remove empty sections
        self.sections = [
            s for s in self.sections
            if s.content or s.code_blocks or s.tables or s.subsections
        ]

        # Update section map
        self.section_map = {s.id: s for s in self.sections}

        # Clean content
        for section in self.sections:
            if section.content:
                # Basic cleaning
                section.content = re.sub(r'\s+', ' ', section.content).strip()
                section.content = section.content.replace('\xa0', ' ')

        logger.info(f"Post-processing complete. Final: {len(self.sections)} sections, "
                   f"{sum(len(s.code_blocks) for s in self.sections)} code blocks")

    def save_parsed_data(self, output_dir: Path):
        """Save parsed data to JSON file"""
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = self.file_path.stem
        output_file = output_dir / f"{base_name}_parsed.json"

        # Include code block statistics
        total_code_blocks = sum(len(s.code_blocks) for s in self.sections)
        inline_code_blocks = sum(len([cb for cb in s.code_blocks if cb.is_inline]) for s in self.sections)
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
                'inline_code_blocks': inline_code_blocks,
                'block_code_blocks': total_code_blocks - inline_code_blocks,
                'total_tables': sum(len(s.tables) for s in self.sections),
                'languages': list(languages)
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

    def _process_text_with_inline_code_and_extract_containers(self, element: Tag, section: Section) -> str:
        """
        FIXED: Process text content, handle inline code properly, và extract code containers
        """
        # First, extract any nested code containers recursively
        self._extract_all_code_containers_recursive(element, section)

        # Then process the text content with inline code handling
        return self._get_text_with_inline_code_formatting(element)

    def _get_text_with_inline_code_formatting(self, element: Tag) -> str:
        """
        NEW METHOD: Get text content và format inline code properly
        """
        result_parts = []

        for child in element.children:
            if isinstance(child, NavigableString):
                result_parts.append(str(child))
            elif isinstance(child, Tag):
                if child.name == 'code':
                    # Đây là inline code - format theo yêu cầu
                    code_content = child.get_text(strip=True)
                    if code_content:
                        # Determine language from class
                        language = None
                        code_classes = child.get('class', [])
                        for class_name in code_classes:
                            class_str = str(class_name).lower()
                            if 'python' in class_str:
                                language = 'python'
                                break
                            elif 'csharp' in class_str:
                                language = 'csharp'
                                break

                        # Format inline code
                        if language:
                            result_parts.append(f"{code_content}({language})")
                        else:
                            result_parts.append(f"{code_content}(code)")

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

    # def _get_text_without_code_containers(self, element: Tag) -> str:
    #     """
    #     NEW METHOD: Get text content from element but exclude text from code containers
    #     since those are handled separately
    #     """
    #     result_parts = []
    #
    #     for child in element.children:
    #         if isinstance(child, NavigableString):
    #             result_parts.append(str(child))
    #         elif isinstance(child, Tag):
    #             if child.name == 'div':
    #                 div_classes = child.get('class', [])
    #                 div_classes_str = ' '.join(str(c) for c in div_classes)
    #
    #                 if 'section-example-container' in div_classes_str:
    #                     # Skip code container text - it's handled separately
    #                     result_parts.append("[Code Example]")
    #                 else:
    #                     # Regular div - get its text recursively
    #                     result_parts.append(self._get_text_without_code_containers(child))
    #             elif child.name == 'pre':
    #                 # Skip pre elements - they're handled as code blocks
    #                 result_parts.append("[Code Block]")
    #             else:
    #                 # Recursive call for other elements
    #                 result_parts.append(self._get_text_without_code_containers(child))
    #
    #     return ''.join(result_parts).strip()

# Test function
def test_fixed_parser():
    """Test the fixed parser"""
    print("Testing FIXED HTML Parser...")
    
    # Test với file HTML thực tế
    test_files = [
        "Quantconnect-Lean-Engine.html",
        "Quantconnect-Writing-Algorithms.html", 
        "Quantconnect-Lean-Cli.html"
    ]
    
    for file_name in test_files:
        test_file = Path(settings.raw_html_path) / file_name
        
        if not test_file.exists():
            print(f"❌ Test file not found: {test_file}")
            continue
            
        print(f"\n🔍 Testing with: {file_name}")
        
        try:
            parser = QuantConnectHTMLParser(test_file)
            sections = parser.parse(target_document_index=1)
            
            # Statistics
            total_sections = len(sections)
            total_code_blocks = sum(len(s.code_blocks) for s in sections)
            inline_code = sum(len([cb for cb in s.code_blocks if cb.is_inline]) for s in sections)
            block_code = total_code_blocks - inline_code
            languages = set()
            for s in sections:
                for cb in s.code_blocks:
                    languages.add(cb.language)
            
            print(f"✅ Parsed {total_sections} sections")
            print(f"📝 Found {total_code_blocks} total code blocks:")
            print(f"   - {block_code} code blocks")  
            print(f"   - {inline_code} inline code")
            print(f"🗣️ Languages: {', '.join(languages)}")
            
            # Show sample with code
            sections_with_code = [s for s in sections if s.code_blocks][:3]
            if sections_with_code:
                print(f"\n📋 Sample sections with code:")
                for s in sections_with_code:
                    print(f"   - {s.id}: {s.title} ({len(s.code_blocks)} code blocks)")
                    for i, cb in enumerate(s.code_blocks[:2]):
                        cb_type = "inline" if cb.is_inline else "block"
                        preview = cb.content[:50].replace('\n', ' ') + "..." if len(cb.content) > 50 else cb.content
                        print(f"     {i+1}. {cb.language} ({cb_type}): {preview}")
            
            # Save results
            output_file = parser.save_parsed_data(settings.processed_data_path)
            print(f"💾 Saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error parsing {file_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_fixed_parser()