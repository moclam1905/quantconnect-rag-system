"""
Utilities và helper functions cho QuantConnect HTML Parser
Xử lý các edge cases và cung cấp các chức năng bổ sung
"""

import re
from typing import List, Dict, Optional, Tuple
from bs4 import Tag, NavigableString
import html
from pathlib import Path
import json
from dataclasses import asdict

# Import logger
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import logger


class CodeLanguageDetector:
    """
    Phát hiện ngôn ngữ lập trình của code blocks một cách thông minh hơn.
    Sử dụng nhiều signals: class names, content patterns, syntax patterns.
    """
    
    # Patterns để nhận diện Python code
    PYTHON_PATTERNS = [
        r'^\s*import\s+\w+',           # import statements
        r'^\s*from\s+\w+\s+import',     # from X import Y
        r'^\s*def\s+\w+\s*\(',         # function definitions
        r'^\s*class\s+\w+',             # class definitions
        r'^\s*if\s+__name__\s*==',     # if __name__ == "__main__"
        r'^\s*#\s*\w+',                 # Python comments
        r'self\.',                      # self reference
        r'print\s*\(',                  # print function
        r'\.append\(',                  # list append
        r'\.items\(\)',                 # dict items
        r'range\(',                     # range function
        r'len\(',                       # len function
    ]
    
    # Patterns để nhận diện C# code
    CSHARP_PATTERNS = [
        r'^\s*using\s+\w+;',            # using statements
        r'^\s*namespace\s+\w+',         # namespace declarations
        r'^\s*public\s+class',          # class declarations
        r'^\s*private\s+\w+',           # private members
        r'^\s*protected\s+\w+',         # protected members
        r'^\s*static\s+void\s+Main',   # Main method
        r'^\s*//\s*\w+',                # C# comments
        r'var\s+\w+\s*=',               # var declarations
        r'new\s+\w+\(',                 # object instantiation
        r'\.ToString\(\)',              # ToString method
        r'Console\.WriteLine',          # Console output
        r'public\s+\w+\s+\w+\s*{',     # properties
    ]
    
    # QuantConnect-specific patterns
    QUANTCONNECT_PYTHON_PATTERNS = [
        r'self\.Initialize',            # QC Initialize method
        r'self\.SetStartDate',          # Setting dates
        r'self\.SetCash',               # Setting cash
        r'self\.AddEquity',             # Adding securities
        r'self\.Debug',                 # Debug output
        r'self\.Log',                   # Logging
        r'Algorithm',                   # Base class
        r'QCAlgorithm',                 # Full class name
        r'OnData\s*\(',                 # OnData method
        r'self\.Schedule',              # Scheduling
    ]
    
    QUANTCONNECT_CSHARP_PATTERNS = [
        r'Initialize\s*\(\)',           # Initialize method
        r'SetStartDate',                # Setting dates
        r'SetCash',                     # Setting cash
        r'AddEquity',                   # Adding securities
        r'Debug\s*\(',                  # Debug output
        r'Log\s*\(',                    # Logging
        r'QCAlgorithm',                 # Base class
        r'OnData\s*\(',                 # OnData method
        r'Schedule\.',                  # Scheduling
        r'Securities\[',                # Securities access
    ]
    
    @classmethod
    def detect_language(cls, code_content: str, element_classes: List[str] = None) -> str:
        """
        Phát hiện ngôn ngữ của code block.
        
        Args:
            code_content: Nội dung code
            element_classes: CSS classes của element chứa code
            
        Returns:
            'python', 'csharp', hoặc 'text'
        """
        # Kiểm tra classes trước (most reliable)
        if element_classes:
            classes_str = ' '.join(str(c).lower() for c in element_classes)
            if any(py in classes_str for py in ['python', 'py', 'python3']):
                return 'python'
            elif any(cs in classes_str for cs in ['csharp', 'c#', 'cs', 'c-sharp']):
                return 'csharp'
        
        # Nếu không có class rõ ràng, analyze content
        python_score = 0
        csharp_score = 0
        
        # Check generic patterns
        for pattern in cls.PYTHON_PATTERNS:
            if re.search(pattern, code_content, re.MULTILINE):
                python_score += 1
        
        for pattern in cls.CSHARP_PATTERNS:
            if re.search(pattern, code_content, re.MULTILINE):
                csharp_score += 1
        
        # Check QuantConnect-specific patterns (higher weight)
        for pattern in cls.QUANTCONNECT_PYTHON_PATTERNS:
            if re.search(pattern, code_content, re.MULTILINE | re.IGNORECASE):
                python_score += 2
        
        for pattern in cls.QUANTCONNECT_CSHARP_PATTERNS:
            if re.search(pattern, code_content, re.MULTILINE | re.IGNORECASE):
                csharp_score += 2
        
        # Make decision
        if python_score > csharp_score:
            return 'python'
        elif csharp_score > python_score:
            return 'csharp'
        else:
            # Default to text if can't determine
            return 'text'


class ContentCleaner:
    """
    Làm sạch và normalize content từ HTML.
    Xử lý các vấn đề về encoding, whitespace, special characters.
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Làm sạch text content.
        """
        if not text:
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Replace non-breaking spaces với regular spaces
        text = text.replace('\u00A0', ' ')
        text = text.replace('&nbsp;', ' ')
        
        # Remove zero-width characters
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\u200c', '')  # Zero-width non-joiner
        text = text.replace('\u200d', '')  # Zero-width joiner
        text = text.replace('\ufeff', '')  # Zero-width no-break space
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        
        # Remove excessive whitespace while preserving paragraph structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip whitespace from each line
            cleaned_line = line.strip()
            if cleaned_line:  # Keep non-empty lines
                cleaned_lines.append(cleaned_line)
            elif cleaned_lines and cleaned_lines[-1]:  # Keep one empty line between paragraphs
                cleaned_lines.append('')
        
        # Join back and remove multiple consecutive empty lines
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    @staticmethod
    def clean_code(code: str, language: str = 'text') -> str:
        """
        Làm sạch code content while preserving formatting.
        """
        if not code:
            return ""
        
        # Decode HTML entities
        code = html.unescape(code)
        
        # Replace non-breaking spaces in code (but preserve indentation)
        code = code.replace('\u00A0', ' ')
        code = code.replace('&nbsp;', ' ')
        
        # Normalize line endings
        code = code.replace('\r\n', '\n')
        code = code.replace('\r', '\n')
        
        # Remove trailing whitespace from each line but preserve indentation
        lines = code.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        
        # Remove leading and trailing empty lines
        while cleaned_lines and not cleaned_lines[0]:
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)


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
    
    @staticmethod
    def table_to_text(table_data: 'TableData') -> str:
        """
        Convert table thành plain text format cho embedding.
        """
        lines = []
        
        if table_data.caption:
            lines.append(f"Table: {table_data.caption}")
        
        if table_data.headers:
            lines.append("Columns: " + ", ".join(table_data.headers))
        
        if table_data.rows:
            lines.append(f"Data ({len(table_data.rows)} rows):")
            for i, row in enumerate(table_data.rows[:5]):  # Limit to first 5 rows for embedding
                row_text = []
                for j, cell in enumerate(row):
                    if table_data.headers and j < len(table_data.headers):
                        row_text.append(f"{table_data.headers[j]}: {cell}")
                    else:
                        row_text.append(cell)
                lines.append(f"  Row {i+1}: " + ", ".join(row_text))
            
            if len(table_data.rows) > 5:
                lines.append(f"  ... and {len(table_data.rows) - 5} more rows")
        
        return "\n".join(lines)


class ParsedDataValidator:
    """
    Validate và sửa các vấn đề trong parsed data.
    """
    
    @staticmethod
    def validate_sections(sections: List['Section']) -> Tuple[bool, List[str]]:
        """
        Validate danh sách sections và trả về các issues found.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        if not sections:
            issues.append("No sections found in document")
            return False, issues
        
        # Check for duplicate IDs
        section_ids = [s.id for s in sections]
        if len(section_ids) != len(set(section_ids)):
            issues.append("Duplicate section IDs found")
        
        # Check for empty sections
        empty_sections = []
        for section in sections:
            if not section.content and not section.code_blocks and not section.tables:
                empty_sections.append(section.title)
        
        if empty_sections:
            issues.append(f"Found {len(empty_sections)} empty sections")
        
        # Check for sections without titles
        untitled = [s for s in sections if not s.title or s.title.strip() == ""]
        if untitled:
            issues.append(f"Found {len(untitled)} sections without titles")
        
        # Check code blocks
        for section in sections:
            for code_block in section.code_blocks:
                if not code_block.content:
                    issues.append(f"Empty code block in section: {section.title}")
                if code_block.language not in ['python', 'csharp', 'text']:
                    issues.append(f"Unknown language '{code_block.language}' in section: {section.title}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @staticmethod
    def fix_common_issues(sections: List['Section']) -> List['Section']:
        """
        Tự động sửa các issues phổ biến trong parsed data.
        """
        # Remove empty sections
        sections = [s for s in sections if s.content or s.code_blocks or s.tables]
        
        # Fix duplicate IDs
        seen_ids = set()
        for section in sections:
            if section.id in seen_ids:
                # Generate new ID
                base_id = section.id
                counter = 1
                while f"{base_id}-{counter}" in seen_ids:
                    counter += 1
                section.id = f"{base_id}-{counter}"
            seen_ids.add(section.id)
        
        # Clean content
        cleaner = ContentCleaner()
        for section in sections:
            section.content = cleaner.clean_text(section.content)
            
            for code_block in section.code_blocks:
                code_block.content = cleaner.clean_code(code_block.content, code_block.language)
        
        return sections


def create_section_summary(section: 'Section', max_length: int = 200) -> str:
    """
    Tạo summary ngắn gọn cho một section.
    Useful cho indexing và preview.
    """
    summary_parts = []
    
    # Add title
    summary_parts.append(f"[{section.title}]")
    
    # Add content preview
    if section.content:
        content_preview = section.content[:max_length]
        if len(section.content) > max_length:
            content_preview += "..."
        summary_parts.append(content_preview)
    
    # Add code info
    if section.code_blocks:
        languages = set(cb.language for cb in section.code_blocks)
        summary_parts.append(f"Contains {len(section.code_blocks)} code examples ({', '.join(languages)})")
    
    # Add table info
    if section.tables:
        summary_parts.append(f"Contains {len(section.tables)} tables")
    
    return " | ".join(summary_parts)


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


# Example usage function
def demonstrate_utilities():
    """
    Demo các utilities đã tạo
    """
    # Test language detection
    python_code = """
import pandas as pd
from QuantConnect import *

class MyAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)
    """
    
    csharp_code = """
using System;
using QuantConnect;

namespace QuantConnect.Algorithm
{
    public class MyAlgorithm : QCAlgorithm
    {
        public override void Initialize()
        {
            SetStartDate(2020, 1, 1);
            SetCash(100000);
        }
    }
}
    """
    
    detector = CodeLanguageDetector()
    print(f"Python code detected as: {detector.detect_language(python_code)}")
    print(f"C# code detected as: {detector.detect_language(csharp_code)}")
    
    # Test content cleaning
    dirty_text = "This is   some\u00A0text with&nbsp;weird   spacing\n\n\n\nAnd multiple breaks"
    cleaner = ContentCleaner()
    print(f"Cleaned text: {cleaner.clean_text(dirty_text)}")
    
    # Test document counting (if you have a test file)
    # test_file = Path("test.html")
    # if test_file.exists():
    #     doc_count = count_documents_in_html(test_file)
    #     print(f"Found {doc_count} documents in {test_file}")


if __name__ == "__main__":
    demonstrate_utilities()