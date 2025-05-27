"""
HTML Parser cho QuantConnect Documentation
Nhiệm vụ: Parse các file HTML single-page lớn và trích xuất nội dung có cấu trúc

Cách tiếp cận:
1. Đọc file HTML theo chunks để xử lý file lớn
2. Parse structure (Table of Contents) trước
3. Extract content theo sections
4. Phân loại và tag các loại content (text, code, table)
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
    language: str  # 'python' hoặc 'csharp'
    content: str
    section_id: str  # ID của section chứa code block
    line_number: Optional[int] = None  # Dòng bắt đầu trong HTML gốc

    def __post_init__(self):
        # Clean up code content
        self.content = self.content.strip()
        # Normalize language name
        if self.language.lower() in ['c#', 'cs', 'csharp']:
            self.language = 'csharp'
        elif self.language.lower() in ['py', 'python']:
            self.language = 'python'


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
    Parser chính cho QuantConnect HTML documentation.

    Được thiết kế dựa trên cấu trúc HTML thực tế được tạo bởi SinglePageDocGenerator.py
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_name = file_path.name
        self.soup: Optional[BeautifulSoup] = None
        self.toc_structure: Dict = {}  # Lưu structure từ Table of Contents
        self.sections: List[Section] = []  # Danh sách các sections đã parse
        self.section_map: Dict[str, Section] = {}  # Map section ID to Section object

        logger.info(f"Initializing parser for: {self.file_name}")

    def parse(self, target_document_index=0) -> List[Section]:
        """
        Main parsing method - orchestrates toàn bộ quá trình parsing.

        Returns:
            List[Section]: Danh sách các sections đã được parse và structure
        """
        try:
            logger.info(f"Starting to parse {self.file_name}")

            # Step 1: Load HTML file
            self._load_html(target_document_index=target_document_index)

            # Step 2: Parse Table of Contents
            self._parse_table_of_contents()

            # Step 3: Remove unnecessary elements
            self._clean_html()

            # Step 4: Parse content sections
            self._parse_content_sections()

            # Step 5: Build hierarchy
            self._build_section_hierarchy()

            # Step 6: Post-process và validate
            self._post_process()

            logger.info(f"Successfully parsed {len(self.sections)} sections from {self.file_name}")
            return self.sections

        except Exception as e:
            logger.error(f"Error parsing {self.file_name}: {str(e)}")
            raise

    def _load_html(self, target_document_index=0): # Mặc định là tài liệu đầu tiên (index 0)
        """
        Load HTML file vào BeautifulSoup.
        Xử lý trường hợp file có multiple DOCTYPE declarations và nested HTML tags.
        Sẽ parse tài liệu ở vị trí target_document_index (0-based).
        """
        logger.info(f"Initializing parser for: {self.file_name}")
        logger.info(f"Attempting to load document at index {target_document_index} from {self.file_path}")

        # Kiểm tra file tồn tại
        if not self.file_path.exists():
            raise FileNotFoundError(f"HTML file not found: {self.file_path}")

        # Đọc toàn bộ nội dung file gốc
        with open(self.file_path, 'r', encoding='utf-8') as f:
            full_file_content = f.read()

        # Tìm tất cả các vị trí bắt đầu của <!DOCTYPE html... (không phân biệt chữ hoa thường cho DOCTYPE HTML)
        # Pattern này tìm đúng <!DOCTYPE html ... >
        doctype_marker_pattern = r'<!DOCTYPE\s+html[^>]*>'
        starts = [m.start() for m in re.finditer(doctype_marker_pattern, full_file_content, flags=re.IGNORECASE)]
        
        selected_document_content = ""
        if not starts:
            # Nếu không tìm thấy DOCTYPE nào, coi toàn bộ file là một tài liệu
            if target_document_index == 0:
                selected_document_content = full_file_content
            else:
                raise IndexError(f"No <!DOCTYPE html> found in file. Cannot get document at index {target_document_index}.")
        elif target_document_index < len(starts):
            start_pos = starts[target_document_index]
            # Điểm kết thúc là điểm bắt đầu của DOCTYPE tiếp theo, hoặc cuối file
            end_pos = starts[target_document_index + 1] if (target_document_index + 1) < len(starts) else len(full_file_content)
            selected_document_content = full_file_content[start_pos:end_pos].strip()
        else:
            raise IndexError(f"Target document index {target_document_index} is out of range. File contains {len(starts)} documents (0-indexed).")

        if not selected_document_content:
            # Trường hợp này không nên xảy ra nếu logic trên đúng và file có nội dung
            raise ValueError(f"Extracted document content for index {target_document_index} is empty. This might indicate an issue with the splitting logic or the source file structure.")

        # selected_document_content bây giờ chứa chuỗi HTML của tài liệu bạn muốn parse,
        # bắt đầu bằng <!DOCTYPE html...> của chính nó.

        # Loại bỏ khai báo DOCTYPE khỏi phần nội dung đã chọn này.
        # Mã gốc loại bỏ TẤT CẢ các DOCTYPE. Vì selected_document_content chỉ nên có một ở đầu,
        # việc này vẫn ổn.
        html_content_for_soup = re.sub(r'<!DOCTYPE[^>]*>', '', selected_document_content, count=1, flags=re.IGNORECASE) # [html_parser.py]
        # count=1 để đảm bảo chỉ loại bỏ cái đầu tiên (nếu có nhiều, dù không nên)

        # Parse với BeautifulSoup
        self.soup = BeautifulSoup(html_content_for_soup, 'html.parser') 

        logger.info(f"Loaded HTML for document index {target_document_index}, effective content size after DOCTYPE removal: {len(html_content_for_soup) / 1024 / 1024:.2f} MB")
        if not self.soup.contents or (self.soup.html and not self.soup.html.body.contents and not self.soup.html.head.contents):
             logger.warning(f"Parsed soup for document index {target_document_index} appears to be empty or minimal. Original chunk started with: '{selected_document_content[:200]}...'")

    def _parse_table_of_contents(self):
        """
        Parse Table of Contents theo cấu trúc được tạo bởi SinglePageDocGenerator:
        <h3>Table of Content</h3>
        <nav>
            <ul>
                <li><a href="#{id}" class="toc-h{level}">{id} {title}</a></li>
                ...
            </ul>
        </nav>
        """
        logger.info("Parsing Table of Contents...")

        # Tìm heading "Table of Content"
        toc_heading = None
        for heading in self.soup.find_all(['h3', 'h2', 'h4']):
            if 'table of content' in heading.get_text().lower():
                toc_heading = heading
                break

        if not toc_heading:
            logger.warning("Table of Contents heading not found, will parse sections directly")
            return

        # Tìm nav element sau heading này
        toc_container = None
        current = toc_heading.next_sibling
        while current and not toc_container:
            if isinstance(current, Tag):
                if current.name == 'nav':
                    toc_container = current
                    break
            current = current.next_sibling

        if not toc_container:
            logger.warning("Table of Contents nav element not found")
            return

        # Parse các entries trong ToC
        toc_entries = toc_container.find_all('a', href=re.compile('^#'))

        for entry in toc_entries:
            # Extract thông tin từ mỗi ToC entry
            href = entry.get('href', '').lstrip('#')  # Remove # prefix
            full_text = entry.get_text(strip=True)

            # Parse text format: "{id} {title}" (e.g., "1.2.3 Section Title")
            match = re.match(r'^([\d\.]+)\s+(.+)$', full_text)
            if match:
                section_id = match.group(1)
                title = match.group(2)
            else:
                # Fallback nếu format không match
                section_id = href
                title = full_text

            # Xác định level từ class attribute
            level = 1  # default
            entry_classes = entry.get('class', [])
            for class_name in entry_classes:
                match = re.search(r'toc-h(\d+)', str(class_name))
                if match:
                    level = int(match.group(1))
                    break

            # Verify level bằng cách đếm số dấu chấm
            # Level = số dấu chấm + 1 (e.g., "1.2.3" = level 3)
            calculated_level = len(section_id.split('.'))
            if calculated_level != level:
                logger.debug(f"Level mismatch for {section_id}: class says {level}, dots say {calculated_level}")
                level = calculated_level  # Trust the dots

            # Lưu vào toc_structure
            self.toc_structure[section_id] = {
                'id': section_id,
                'title': title,
                'level': level,
                'href': href,
                'full_text': full_text
            }

        logger.info(f"Found {len(self.toc_structure)} entries in Table of Contents")

        # Debug: print first few entries
        if self.toc_structure:
            logger.debug("Sample ToC entries:")
            for i, (section_id, info) in enumerate(list(self.toc_structure.items())[:5]):
                logger.debug(f"  {section_id}: Level {info['level']} - {info['title']}")

    def _clean_html(self):
        """
        Loại bỏ các elements không cần thiết khỏi HTML.
        Cẩn thận với nested HTML structure của QuantConnect.
        """
        logger.info("Cleaning HTML content...")

        # Danh sách các elements cần remove
        elements_to_remove = [
            # Scripts và styles
            'script',
            'style',
            'link',
            'meta',

            # Media elements (theo yêu cầu)
            'img',
            'video',
            'audio',
            'iframe',
            'embed',
            'object',

            # KHÔNG remove 'html' và 'body' vì chúng chứa content

            # Ads hoặc promotional content
            'div.advertisement',
            'div.promo',
            'div.banner'
        ]

        # Remove các elements
        for selector in elements_to_remove:
            for element in self.soup.select(selector):
                element.decompose()

        # Remove empty nested html/body tags but keep ones with content
        for html_tag in self.soup.find_all('html'):
            # Check if this html tag has actual content
            body = html_tag.find('body')
            if body:
                # Check if body has content
                text_content = body.get_text(strip=True)
                if not text_content or text_content == '\xa0':  # empty or just &nbsp;
                    html_tag.decompose()
            else:
                # No body tag, check direct content
                text_content = html_tag.get_text(strip=True)
                if not text_content:
                    html_tag.decompose()

        # Remove nav elements EXCEPT the ToC nav
        for nav in self.soup.find_all('nav'):
            # Check if this nav contains ToC links
            has_toc_links = nav.find('a', class_=re.compile(r'toc-h\d+'))
            if not has_toc_links:
                nav.decompose()

        # DON'T remove page break paragraphs yet - we need them to identify section boundaries

        # Remove comments
        comments = self.soup.find_all(string=lambda text: isinstance(text, NavigableString) and '<!--' in str(text))
        for comment in comments:
            comment.extract()

        logger.info("HTML cleaned successfully")

    def _parse_content_sections(self):
        """
        Parse content sections dựa trên cấu trúc được tạo bởi SinglePageDocGenerator.
        Sections được wrap trong <section id="{number}"> tags.
        """
        logger.info("Parsing content sections...")

        # Tìm tất cả <section> elements với id
        section_elements = self.soup.find_all('section', id=True)

        if not section_elements:
            logger.warning("No <section> elements found, falling back to heading-based parsing")
            self._parse_sections_from_headings()
            return

        for section_elem in tqdm(section_elements, desc="Parsing sections"):
            section_id = section_elem.get('id', '')

            # Skip empty IDs
            if not section_id:
                continue

            # Extract section info từ ToC nếu có
            toc_info = self.toc_structure.get(section_id, {})

            # Extract title từ headings trong section
            title = ""
            breadcrumb = ""

            # Tìm breadcrumb
            breadcrumb_elem = section_elem.find_previous('p', class_='page-breadcrumb')
            if breadcrumb_elem:
                breadcrumb = breadcrumb_elem.get_text(strip=True)

            # Tìm headings trong section
            h1 = section_elem.find('h1')
            h2 = section_elem.find('h2')

            if h2:  # Nếu có h2, đó là title chính
                title = h2.get_text(strip=True)
            elif h1:  # Nếu chỉ có h1
                title = h1.get_text(strip=True)
            else:
                # Fallback to ToC title
                title = toc_info.get('title', f'Section {section_id}')

            # Determine level
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

            # Parse content của section
            self._parse_section_content(section_elem, section)

            # Add to collections
            self.sections.append(section)
            self.section_map[section_id] = section

        logger.info(f"Parsed {len(self.sections)} content sections")

    def _parse_sections_from_headings(self):
        """
        Fallback method: Parse sections từ headings nếu không có <section> tags.
        """
        logger.info("Parsing sections from heading elements...")

        # Tìm tất cả heading elements (h1-h6)
        headings = self.soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

        for heading in tqdm(headings, desc="Parsing sections"):
            # Skip ToC heading
            if 'table of content' in heading.get_text().lower():
                continue

            # Extract section info
            section_id = heading.get('id', '')
            if not section_id:
                # Try parent section
                parent_section = heading.find_parent('section', id=True)
                if parent_section:
                    section_id = parent_section.get('id', '')

            if not section_id:
                # Generate ID
                section_id = self._generate_section_id(heading.get_text())

            section_title = heading.get_text(strip=True)
            section_level = int(heading.name[1])  # h1 -> 1, h2 -> 2, etc.

            # Create Section object
            section = Section(
                id=section_id,
                title=section_title,
                level=section_level,
                content=""
            )

            # Parse content
            self._parse_section_content(heading, section)

            # Add to collections
            self.sections.append(section)
            self.section_map[section_id] = section

    def _parse_section_content(self, element: Tag, section: Section):
        """
        Parse nội dung của một section theo cấu trúc thực tế của QuantConnect HTML.

        Cấu trúc:
        1. <p class='page-breadcrumb'>
        2. <div class='page-heading'><section id="X"><h1>Title</h1></section></div>
        3. <h3>Subsection</h3> (optional)
        4. <html><body>actual content</body></html>
        5. <p style="page-break-after: always;">

        Args:
            element: BeautifulSoup Tag - <section> element
            section: Section object để populate
        """
        content_parts = []

        # Find the parent div.page-heading
        page_heading_div = element.parent
        if not page_heading_div or page_heading_div.name != 'div' or 'page-heading' not in page_heading_div.get('class', []):
            # Fallback: try to find content after section
            page_heading_div = element

        # Start looking for content after the page-heading div
        current = page_heading_div.next_sibling

        # Process siblings until we hit the next section or page break
        while current:
            if isinstance(current, Tag):
                # Stop conditions
                if current.name == 'p' and 'page-breadcrumb' in current.get('class', []):
                    # Next section starting
                    break

                if current.name == 'div' and 'page-heading' in current.get('class', []):
                    # Another section
                    break

                # Skip page breaks
                if current.name == 'p' and current.get('style') and 'page-break' in current.get('style'):
                    # This usually marks the end of this section's content
                    current = current.next_sibling
                    continue

                # Process different content types
                if current.name == 'h3':
                    # Subsection heading
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
                    self._extract_code_block(current, section)

                elif current.name == 'table':
                    # Direct table
                    self._extract_table(current, section)

                elif current.name in ['p', 'div', 'ul', 'ol', 'h4', 'h5', 'h6']:
                    # Other content
                    text = current.get_text(strip=True)
                    if text and text not in content_parts:
                        content_parts.append(text)

            current = current.next_sibling

        # Combine all text parts
        section.content = '\n\n'.join(content_parts)

    def _extract_content_from_body(self, body: Tag, section: Section, content_parts: List[str]):
        """
        Extract content from <body> tag trong nested HTML structure.
        """
        for element in body.children:
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text:
                    content_parts.append(text)

            elif isinstance(element, Tag):
                if element.name == 'pre':
                    # Code block
                    self._extract_code_block(element, section)
                    # Also add to content for context
                    content_parts.append(f"[Code Block - {section.code_blocks[-1].language}]")

                elif element.name == 'table':
                    # Table
                    self._extract_table(element, section)
                    content_parts.append(f"[Table - {len(section.tables[-1].rows)} rows]")

                elif element.name in ['p', 'div', 'ul', 'ol', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    text = element.get_text(strip=True)
                    if text:
                        # Check for special content like JSON
                        if text.startswith('{') and text.endswith('}'):
                            try:
                                # Try to parse as JSON for better formatting
                                import json
                                data = json.loads(text)
                                content_parts.append(f"[Metadata: {data.get('type', 'unknown')} - {data.get('heading', '')}]")
                            except:
                                content_parts.append(text)
                        else:
                            content_parts.append(text)

    def _extract_code_block(self, pre_element: Tag, section: Section):
        """
        Extract code block từ <pre> element.
        Xác định ngôn ngữ từ class attribute hoặc content.
        """
        # Xác định ngôn ngữ từ class
        language = 'text'  # default
        classes = pre_element.get('class', [])

        for class_name in classes:
            class_str = str(class_name).lower()
            if 'python' in class_str or 'py' in class_str:
                language = 'python'
                break
            elif 'csharp' in class_str or 'c#' in class_str or 'cs' in class_str:
                language = 'csharp'
                break

        # Extract code content
        # Check for nested <code> tag first
        code_elem = pre_element.find('code')
        if code_elem:
            code_content = code_elem.get_text(strip=False)
            # Also check code element classes
            code_classes = code_elem.get('class', [])
            for class_name in code_classes:
                class_str = str(class_name).lower()
                if 'python' in class_str:
                    language = 'python'
                elif 'csharp' in class_str:
                    language = 'csharp'
        else:
            code_content = pre_element.get_text(strip=False)

        # If language still not determined, analyze content
        if language == 'text':
            # Import từ parser_utils nếu cần
            try:
                from src.data_processing.parser_utils import CodeLanguageDetector
                language = CodeLanguageDetector.detect_language(code_content, classes)
            except:
                # Fallback to simple detection
                if 'import ' in code_content or 'def ' in code_content or 'self.' in code_content:
                    language = 'python'
                elif 'using ' in code_content or 'namespace ' in code_content or 'public class' in code_content:
                    language = 'csharp'

        # Create CodeBlock object
        code_block = CodeBlock(
            language=language,
            content=code_content,
            section_id=section.id
        )

        section.code_blocks.append(code_block)

    def _extract_table(self, table_element: Tag, section: Section):
        """
        Extract table data từ <table> element.
        """
        headers = []
        rows = []

        # Extract headers
        thead = table_element.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        else:
            # Try first row as headers
            first_row = table_element.find('tr')
            if first_row and first_row.find('th'):
                headers = [th.get_text(strip=True) for th in first_row.find_all('th')]

        # Extract rows
        tbody = table_element.find('tbody') or table_element
        for tr in tbody.find_all('tr'):
            # Skip header row if already processed
            if tr.find('th') and not rows:
                continue

            row_data = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if row_data:
                rows.append(row_data)

        # Extract caption if exists
        caption_element = table_element.find('caption')
        caption = caption_element.get_text(strip=True) if caption_element else None

        # Create TableData object
        if headers or rows:  # Only create if table has content
            table_data = TableData(
                headers=headers,
                rows=rows,
                section_id=section.id,
                caption=caption
            )
            section.tables.append(table_data)

    def _build_section_hierarchy(self):
        """
        Xây dựng hierarchy cho sections dựa trên section IDs.
        E.g., "1.2.3" là con của "1.2", "1.2" là con của "1"
        """
        logger.info("Building section hierarchy...")

        # Sort sections by ID để process theo thứ tự
        self.sections.sort(key=lambda s: [int(x) for x in s.id.split('.') if x.isdigit()])

        # Build parent-child relationships
        for section in self.sections:
            if '.' in section.id:
                # Find parent ID
                parent_id = '.'.join(section.id.split('.')[:-1])
                if parent_id in self.section_map:
                    parent = self.section_map[parent_id]
                    section.parent_id = parent_id
                    parent.subsections.append(section)

        logger.info("Section hierarchy built successfully")

    def _generate_section_id(self, title: str) -> str:
        """
        Generate a unique section ID từ title.
        Convert to lowercase, replace spaces with hyphens, remove special chars.
        """
        # Convert to lowercase và replace spaces
        section_id = title.lower().replace(' ', '-')

        # Remove special characters
        section_id = re.sub(r'[^a-z0-9\-]', '', section_id)

        # Remove multiple hyphens
        section_id = re.sub(r'-+', '-', section_id)

        # Add hash suffix để ensure unique
        hash_suffix = hashlib.md5(title.encode()).hexdigest()[:6]

        return f"{section_id}-{hash_suffix}"

    def _post_process(self):
        """
        Post-process các sections đã parse.
        - Validate data
        - Clean up empty sections
        - Fix any issues
        """
        logger.info("Post-processing parsed sections...")

        # Remove empty sections (but keep if they have subsections)
        self.sections = [
            s for s in self.sections
            if s.content or s.code_blocks or s.tables or s.subsections
        ]

        # Update section map
        self.section_map = {s.id: s for s in self.sections}

        # Clean content using utilities if available
        try:
            from src.data_processing.parser_utils import ContentCleaner
            cleaner = ContentCleaner()
            for section in self.sections:
                section.content = cleaner.clean_text(section.content)
                for code_block in section.code_blocks:
                    code_block.content = cleaner.clean_code(code_block.content, code_block.language)
        except ImportError:
            logger.warning("ContentCleaner not available, skipping content cleaning")

        logger.info(f"Post-processing complete. Final section count: {len(self.sections)}")

    def save_parsed_data(self, output_dir: Path):
        """
        Save parsed data ra file để có thể reuse sau này.
        Lưu dưới dạng JSON để dễ load lại.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create output filename
        base_name = self.file_path.stem
        output_file = output_dir / f"{base_name}_parsed.json"

        # Convert sections to JSON-serializable format
        data = {
            'source_file': self.file_name,
            'sections': [s.to_dict() for s in self.sections],
            'statistics': {
                'total_sections': len(self.sections),
                'total_code_blocks': sum(len(s.code_blocks) for s in self.sections),
                'total_tables': sum(len(s.tables) for s in self.sections),
                'languages': list(set(cb.language for s in self.sections for cb in s.code_blocks))
            }
        }

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved parsed data to: {output_file}")

        return output_file


def test_parser():
    """
    Test function để kiểm tra parser với một file HTML cụ thể
    """
    # Path to test file
    test_file = Path(settings.raw_html_path) / "Quantconnect-Lean-Engine.html"

    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return

    # Create parser và parse file
    parser = QuantConnectHTMLParser(test_file)
    sections = parser.parse(target_document_index=1)

    # Print some statistics
    print(f"\n=== Parsing Results ===")
    print(f"Total sections: {len(sections)}")
    print(f"Total code blocks: {sum(len(s.code_blocks) for s in sections)}")
    print(f"Total tables: {sum(len(s.tables) for s in sections)}")

    # Print first few sections as examples
    print(f"\n=== First 5 Sections ===")
    for i, section in enumerate(sections[:5]):
        print(f"\n{i+1}. Section {section.id}: {section.title}")
        print(f"   Level: {section.level}")
        print(f"   Breadcrumb: {section.breadcrumb}")
        print(f"   Content preview: {section.content[:100]}..." if section.content else "   No content")
        print(f"   Code blocks: {len(section.code_blocks)}")
        print(f"   Tables: {len(section.tables)}")
        print(f"   Subsections: {len(section.subsections)}")

    # Save parsed data
    output_dir = Path(settings.processed_data_path)
    output_file = parser.save_parsed_data(output_dir)
    print(f"\nSaved parsed data to: {output_file}")


def test_with_sample_html():
    """
    Test với sample HTML được cung cấp
    """
    sample_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
</body>
</html><p style="page-break-after: always;">&nbsp;</p>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    <div class="cover-content" style="padding-left:200px;padding-top: 300px;">
        <h2 class= "cover-section-name" style="color: #484948;font-size:30px;font-weight: 400;margin-top:1rem;">LEAN ENGINE</h2>
        <h1 class="cover-section-tagline" style="color:#1a1919;font-size:54px;font-weight: 400;margin:1rem 0">Radically open-source <br>algorithmic trading <br>engine</h1>
        <h2 class="cover-section-description" style="color: #484948;font-size:32px;font-weight: 400;line-height: 1.4;">Multi-asset with full portfolio modeling, <br>LEAN is data agnostic, empowering you <br>to explore faster than ever before.</h2>
    </div>
</body>
</html>
<p style="page-break-after: always;">&nbsp;</p>
<h3>Table of Content</h3>
<nav>
<ul>
<li><a href="#1" class="toc-h1" target="_parent">1 Getting Started</a></li>
<li><a href="#2" class="toc-h1" target="_parent">2 Contributions</a></li>
</ul>
</nav>
<p style="page-break-after: always;">&nbsp;</p>
<p class='page-breadcrumb'>Getting Started</p>
<div class='page-heading'>
    <section id="1">
        <h1>Getting Started</h1>
        
    </section>
</div>
<h3>Introduction</h3>
<html>
 <body>
  <p>
   Lean Engine is an open-source algorithmic trading engine built for easy strategy research, backtesting and live trading. We integrate with common data providers and brokerages so you can quickly deploy algorithmic trading strategies.
  </p>
  <p>
   The core of the LEAN Engine is written in C#; but it operates seamlessly on Linux, Mac and Windows operating systems. It supports algorithms written in Python 3.11 or C#. Lean drives the web-based algorithmic trading platform
   <a href="https://www.quantconnect.com/">
    QuantConnect
   </a>
   .
  </p>
 </body>
</html>
<p style="page-break-after: always;">&nbsp;</p>
<p class='page-breadcrumb'>Class Reference</p>
<div class='page-heading'>
    <section id="5">
        <h1>Class Reference</h1>
        
    </section>
</div>
<html>
 <body>
  <p>
   {
   "type": "link",
   "heading": "Class Reference",
   "subHeading": "",
   "content": "",
   "alsoLinks": [],
   "href": "https://www.lean.io/docs/v2/lean-engine/class-reference/"
}
  </p>
 </body>
</html>
<p style="page-break-after: always;">&nbsp;</p>'''

    # Save sample to temp file
    temp_file = Path(settings.raw_html_path) / "test_sample.html"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(sample_html)

    # Parse
    parser = QuantConnectHTMLParser(temp_file)
    # Để parse tài liệu HTML thứ hai (index 1)
    sections = parser.parse(target_document_index=1)

    print(f"\n=== Sample HTML Parsing Results ===")
    print(f"Total sections found: {len(sections)}")

    for section in sections:
        print(f"\n--- Section {section.id}: {section.title} ---")
        print(f"Level: {section.level}")
        print(f"Breadcrumb: {section.breadcrumb}")
        print(f"Content: {section.content[:200]}..." if len(section.content) > 200 else f"Content: {section.content}")
        print(f"Code blocks: {len(section.code_blocks)}")
        print(f"Tables: {len(section.tables)}")

    # Clean up
    temp_file.unlink()


if __name__ == "__main__":
    # Test với sample HTML trước
    # print("Testing with sample HTML...")
    # test_with_sample_html()

    # print("\n" + "="*50 + "\n")

    # Sau đó test với file thực nếu có
    test_parser()