"""
Utilities và helper functions cho QuantConnect HTML Parser
Xử lý các edge cases và cung cấp các chức năng bổ sung
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent.parent))

import hashlib
from bs4 import BeautifulSoup, Tag

from src.data_processing.old.html_parser import TableData
from src.utils.logger import logger

class SectionHierarchyBuilder:
    """
    Xây dựng hierarchy (cấu trúc phân cấp) cho các sections.
    Giúp hiểu được mối quan hệ parent-child giữa các sections.
    """
    
    @staticmethod
    def build_hierarchy(sections: List['Section']) -> List['Section']:
        """
        Tổ chức các sections thành cấu trúc cây dựa trên level.
        
        Args:
            sections: Danh sách sections phẳng
            
        Returns:
            Danh sách sections đã được tổ chức thành hierarchy
        """
        if not sections:
            return []
        
        # Stack để track current parent at each level
        parent_stack = []
        root_sections = []
        
        for section in sections:
            # Pop stack until we find appropriate parent level
            while parent_stack and parent_stack[-1].level >= section.level:
                parent_stack.pop()
            
            if parent_stack:
                # This section is a child of the last item in stack
                parent = parent_stack[-1]
                section.parent_id = parent.id
                parent.subsections.append(section)
            else:
                # This is a root section
                root_sections.append(section)
            
            # Add this section to stack as potential parent
            parent_stack.append(section)
        
        return root_sections
    
    @staticmethod
    def get_section_path(section: 'Section', all_sections: Dict[str, 'Section']) -> str:
        """
        Lấy full path của section (e.g., "1. Getting Started > 1.2 Installation > 1.2.3 Python Setup")
        
        Args:
            section: Section cần lấy path
            all_sections: Dictionary mapping section IDs to sections
            
        Returns:
            Full path string
        """
        path_parts = [section.title]
        current = section
        
        while current.parent_id and current.parent_id in all_sections:
            current = all_sections[current.parent_id]
            path_parts.insert(0, current.title)
        
        return " > ".join(path_parts)


class TableProcessor:
    """
    Xử lý và tối ưu hóa tables từ HTML.
    Convert tables thành format phù hợp cho LLM processing.
    """
    
    @staticmethod
    def table_to_markdown(table_data: 'TableData') -> str:
        """
        Convert table thành markdown format.
        """
        if not table_data.headers and not table_data.rows:
            return ""
        
        lines = []
        
        # Add caption if exists
        if table_data.caption:
            lines.append(f"**{table_data.caption}**\n")
        
        # Headers
        if table_data.headers:
            lines.append("| " + " | ".join(table_data.headers) + " |")
            lines.append("|" + "|".join([" --- " for _ in table_data.headers]) + "|")
        
        # Rows
        for row in table_data.rows:
            # Ensure row has same number of columns as headers
            if table_data.headers:
                while len(row) < len(table_data.headers):
                    row.append("")
                row = row[:len(table_data.headers)]
            
            lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)


def export_sections_for_rag(sections: List['Section'], output_file: Path, metadata: Dict = None):
    """
    Export sections trong format tối ưu cho RAG processing.
    Mỗi section sẽ được format để dễ dàng chunk và embed.
    
    Args:
        sections: List of Section objects
        output_file: Path to save the output
        metadata: Additional metadata to include (e.g., document_index, source_file)
    """
    rag_documents = []
    
    # Build hierarchy first
    hierarchy_builder = SectionHierarchyBuilder()
    root_sections = hierarchy_builder.build_hierarchy(sections.copy())
    
    # Create ID mapping
    all_sections = {s.id: s for s in sections}
    
    # Process each section
    for section in sections:
        # Create document for this section
        doc = {
            'id': section.id,
            'title': section.title,
            'level': section.level,
            'path': hierarchy_builder.get_section_path(section, all_sections),
            'content': section.content,
            'metadata': {
                'has_code': len(section.code_blocks) > 0,
                'has_tables': len(section.tables) > 0,
                'code_languages': list(set(cb.language for cb in section.code_blocks)),
                'parent_id': section.parent_id,
                'breadcrumb': section.breadcrumb,
                'section_number': section.section_number
            }
        }
        
        # Add additional metadata if provided
        if metadata:
            doc['metadata'].update(metadata)
        
        # Add code blocks as separate sub-documents
        for i, code_block in enumerate(section.code_blocks):
            doc[f'code_block_{i}'] = {
                'language': code_block.language,
                'content': code_block.content
            }
        
        # Add tables as markdown
        table_processor = TableProcessor()
        for i, table in enumerate(section.tables):
            doc[f'table_{i}'] = table_processor.table_to_markdown(table)
        
        rag_documents.append(doc)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rag_documents, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Exported {len(rag_documents)} documents for RAG to {output_file}")


def count_documents_in_html(file_path: Path) -> int:
    """
    Đếm số lượng documents trong một file HTML dựa trên số lượng DOCTYPE declarations.
    
    Args:
        file_path: Path to HTML file
        
    Returns:
        Số lượng documents found
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern để tìm DOCTYPE declarations
        doctype_pattern = r'<!DOCTYPE\s+html[^>]*>'
        matches = re.findall(doctype_pattern, content, flags=re.IGNORECASE)
        
        # Nếu không tìm thấy DOCTYPE nào, coi như có 1 document
        return len(matches) if matches else 1
        
    except Exception as e:
        logger.error(f"Error counting documents in {file_path}: {str(e)}")
        return 0


def split_html_documents(file_path: Path) -> List[Tuple[int, str]]:
    """
    Split một file HTML thành các documents riêng biệt.
    
    Args:
        file_path: Path to HTML file
        
    Returns:
        List of tuples (document_index, html_content)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern để tìm DOCTYPE declarations
        doctype_pattern = r'<!DOCTYPE\s+html[^>]*>'
        
        # Tìm tất cả vị trí bắt đầu của DOCTYPE
        starts = [m.start() for m in re.finditer(doctype_pattern, content, flags=re.IGNORECASE)]
        
        if not starts:
            # Không có DOCTYPE, return toàn bộ content
            return [(0, content)]
        
        documents = []
        for i, start in enumerate(starts):
            # End position là start của document tiếp theo hoặc end of file
            end = starts[i + 1] if i + 1 < len(starts) else len(content)
            doc_content = content[start:end].strip()
            documents.append((i, doc_content))
        
        return documents
        
    except Exception as e:
        logger.error(f"Error splitting documents in {file_path}: {str(e)}")
        return []


# =============================================================================
# Language Detection Utilities
# =============================================================================

def detect_code_language(code_content: str) -> str:
    """
    Detect programming language từ code content.

    Args:
        code_content: Source code content

    Returns:
        Detected language: 'python', 'csharp', 'cli', or 'text'
    """
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


def generate_code_title(language: str) -> str:
    """
    Generate descriptive title for code blocks.

    Args:
        language: Programming language

    Returns:
        Descriptive title string
    """
    language_map = {
        'python': 'Python Code Example',
        'csharp': 'C# Code Example',
        'cli': 'Command Line Example',
        'text': 'Code Example'
    }
    return language_map.get(language, f"{language} Code Example")


# =============================================================================
# Content Extraction Utilities
# =============================================================================

def extract_error_message(element: Tag) -> str:
    """
    Extract error message content with warning prefix.

    Args:
        element: HTML element containing error message

    Returns:
        Formatted error message
    """
    error_text = element.get_text(strip=True)
    return f"⚠️ Error: {error_text}"


def extract_tutorial_step(element: Tag) -> str:
    """
    Extract tutorial step content from all paragraphs.

    Args:
        element: HTML element containing tutorial step

    Returns:
        Formatted tutorial step content
    """
    content_parts = []

    # Process all paragraphs in tutorial step
    for p in element.find_all('p'):
        text = p.get_text(strip=True)
        if text:
            content_parts.append(text)

    # If no paragraphs found, get direct text
    if not content_parts:
        direct_text = element.get_text(strip=True)
        if direct_text:
            content_parts.append(direct_text)

    return '\n\n'.join(content_parts)


def extract_example_fieldset(element: Tag) -> str:
    """
    Extract example algorithm links with clean filenames.

    Args:
        element: HTML element containing example fieldset

    Returns:
        Formatted example algorithms list
    """
    content_parts = []

    # Get legend/title
    legend = element.find('div', class_='example-legend')
    if legend:
        legend_text = legend.get_text(strip=True)
        content_parts.append(f"📚 {legend_text}:")

    # Extract algorithm links
    links = element.find_all('a', class_='example-algorithm-link')
    if links:
        for link in links:
            # Extract filename from href or text
            href = link.get('href', '')
            if href:
                filename = href.split('/')[-1]
            else:
                filename = link.get_text(strip=True)

            # Get language badge
            badge = link.find('span', class_=re.compile(r'badge.*'))
            if badge:
                lang = badge.get_text(strip=True)
                content_parts.append(f"  - {filename} ({lang})")
            else:
                content_parts.append(f"  - {filename}")

    return '\n'.join(content_parts) if content_parts else ""


def extract_api_content_from_div(api_div: Tag, language: str) -> str:
    """
    Extract API content from language-specific div.

    Args:
        api_div: HTML div containing API content
        language: Programming language ('python' or 'csharp')

    Returns:
        Extracted API content as text
    """
    content_parts = []

    # Find the content container
    container = api_div.find('div', class_='inner-tree-container')
    if not container:
        return ""

    # Extract all content recursively
    for element in container.descendants:
        if hasattr(element, 'name') and element.name:
            if element.name == 'h4':
                content_parts.append(f"## {element.get_text(strip=True)}")
            elif element.name == 'p':
                text = element.get_text(strip=True)
                if text:
                    content_parts.append(text)
            elif element.name == 'div' and 'code-snippet' in element.get('class', []):
                code_text = element.get_text(strip=True)
                if code_text:
                    content_parts.append(f"`{code_text}`")

    return '\n'.join(content_parts)


# =============================================================================
# Table Processing Utilities
# =============================================================================

def extract_table_data(table_element: Tag, section_id: str) -> Optional[TableData]:
    """
    Extract table data from HTML table element.

    Args:
        table_element: HTML table element
        section_id: ID of the section containing the table

    Returns:
        TableData object or None if no valid table data
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
        return TableData(
            headers=headers,
            rows=rows,
            section_id=section_id,
            caption=caption
        )

    return None


# =============================================================================
# Text Processing Utilities
# =============================================================================

def format_inline_code(code_element: Tag) -> str:
    """
    Format inline code element with language detection.

    Args:
        code_element: HTML code element

    Returns:
        Formatted inline code string
    """
    code_content = code_element.get_text(strip=True)
    if not code_content:
        return ""

    # Determine language from class
    language = None
    code_classes = code_element.get('class', [])
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
        return f"{code_content}({language})"
    else:
        return f"{code_content}(code)"


def clean_text_content(text: str) -> str:
    """
    Clean and normalize text content.

    Args:
        text: Raw text content

    Returns:
        Cleaned text content
    """
    if not text:
        return ""

    # Basic cleaning
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('\xa0', ' ')
    return text


# =============================================================================
# ID Generation Utilities
# =============================================================================

def generate_section_id(title: str) -> str:
    """
    Generate unique section ID from title.

    Args:
        title: Section title

    Returns:
        Unique section ID
    """
    section_id = title.lower().replace(' ', '-')
    section_id = re.sub(r'[^a-z0-9\-]', '', section_id)
    section_id = re.sub(r'-+', '-', section_id)
    hash_suffix = hashlib.md5(title.encode()).hexdigest()[:6]
    return f"{section_id}-{hash_suffix}"


# =============================================================================
# HTML Document Utilities
# =============================================================================

def count_documents_in_html(file_path: Path) -> int:
    """
    Count số lượng documents trong HTML file dựa trên DOCTYPE declarations.

    Args:
        file_path: Path to HTML file

    Returns:
        Number of documents found
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find DOCTYPE declarations
        doctype_pattern = r'<!DOCTYPE\s+html[^>]*>'
        matches = re.findall(doctype_pattern, content, flags=re.IGNORECASE)

        count = len(matches)
        logger.debug(f"Found {count} document(s) in {file_path.name}")
        return count

    except Exception as e:
        logger.error(f"Error counting documents in {file_path}: {str(e)}")
        return 0


def convert_api_html_to_text(html_content: str, data_tree_value: str) -> str:
    """
    Convert API HTML content to readable text format.

    Args:
        html_content: HTML content from API
        data_tree_value: Original data-tree value

    Returns:
        Text representation
    """
    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    text_parts = [f"[API Reference: {data_tree_value}]"]

    # Extract main content
    for element in soup.find_all(['h4', 'p', 'div']):
        if element.name == 'h4':
            text_parts.append(f"\n{element.get_text(strip=True)}")
        elif element.name == 'p':
            text_parts.append(element.get_text(strip=True))
        elif 'code-snippet' in element.get('class', []):
            code_text = element.get_text(strip=True)
            if code_text:
                text_parts.append(f"- {code_text}")

    return '\n'.join(text_parts)


# =============================================================================
# Export Functions for RAG
# =============================================================================

def export_sections_for_rag(sections: List, output_file: Path, metadata: Dict = None):
    """
    Export parsed sections in RAG-friendly format.

    Args:
        sections: List of Section objects
        output_file: Output JSON file path
        metadata: Additional metadata to include
    """
    import json
    from datetime import datetime

    # Prepare data for RAG
    rag_data = {
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {},
        'sections': []
    }

    for section in sections:
        section_data = {
            'id': section.id,
            'title': section.title,
            'level': section.level,
            'content': section.content,
            'breadcrumb': section.breadcrumb,
            'code_blocks': [
                {
                    'language': cb.language,
                    'content': cb.content
                } for cb in section.code_blocks
            ],
            'tables': [
                {
                    'headers': t.headers,
                    'rows': t.rows[:10],  # Limit rows for RAG
                    'caption': t.caption
                } for t in section.tables
            ]
        }
        rag_data['sections'].append(section_data)

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rag_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported {len(sections)} sections for RAG to {output_file}")


# Example usage function
def demonstrate_utilities():
    """
    Demo các utilities đã tạo
    """
    # Test language detection
    print('Test')


if __name__ == "__main__":
    demonstrate_utilities()