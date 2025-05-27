"""
Simplified parser functions cho QuantConnect files.
Tự động xử lý việc skip document 0 (phần dư thừa).
"""

from pathlib import Path
from typing import List, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing.html_parser import QuantConnectHTMLParser, Section
from src.data_processing.parser_utils import count_documents_in_html
from src.utils.logger import logger
from config.config import settings


def parse_quantconnect_file(file_path: Path) -> List[Section]:
    """
    Parse một file QuantConnect HTML, tự động skip document 0 (phần dư thừa).
    
    Args:
        file_path: Path to QuantConnect HTML file
        
    Returns:
        List of Section objects từ main content (document 1)
        
    Raises:
        Exception if file has less than 2 documents
    """
    # Check file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Count documents
    doc_count = count_documents_in_html(file_path)
    
    if doc_count < 2:
        logger.warning(f"File {file_path.name} only has {doc_count} document(s). Expected 2.")
        if doc_count == 1:
            logger.warning("Parsing the only available document (index 0)")
            target_index = 0
        else:
            raise ValueError(f"No documents found in {file_path.name}")
    else:
        # Normal case: parse document 1 (main content)
        target_index = 1
    
    # Parse
    parser = QuantConnectHTMLParser(file_path)
    sections = parser.parse(target_document_index=target_index)
    
    logger.info(f"Successfully parsed {len(sections)} sections from {file_path.name} (document {target_index})")
    
    return sections


def parse_all_quantconnect_files(output_dir: Optional[Path] = None) -> dict:
    """
    Parse tất cả file QuantConnect mặc định, skip document 0.
    
    Args:
        output_dir: Optional output directory. If None, use default processed path.
        
    Returns:
        Dictionary với results cho mỗi file
    """
    default_files = [
        "Quantconnect-Lean-Cli.html",
        "Quantconnect-Lean-Engine.html",
        "Quantconnect-Research-Environment.html",
        "Quantconnect-Writing-Algorithms.html"
    ]
    
    if output_dir is None:
        output_dir = settings.processed_data_path
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for file_name in default_files:
        file_path = settings.raw_html_path / file_name
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            results[file_name] = {
                'status': 'not_found',
                'error': 'File not found'
            }
            continue
        
        try:
            # Parse file
            sections = parse_quantconnect_file(file_path)
            
            # Save parsed data
            output_file = output_dir / f"{file_path.stem}_parsed.json"
            
            # Save
            parser = QuantConnectHTMLParser(file_path)
            parser.sections = sections
            saved_file = parser.save_parsed_data(output_dir)
            
            results[file_name] = {
                'status': 'success',
                'sections': len(sections),
                'code_blocks': sum(len(s.code_blocks) for s in sections),
                'tables': sum(len(s.tables) for s in sections),
                'output_file': str(saved_file)
            }
            
        except Exception as e:
            logger.error(f"Error parsing {file_name}: {str(e)}")
            results[file_name] = {
                'status': 'error',
                'error': str(e)
            }
    
    return results


def quick_parse(file_name: str) -> List[Section]:
    """
    Quick parse function - chỉ cần tên file (không cần full path).
    
    Args:
        file_name: Tên file (e.g., "Quantconnect-Writing-Algorithms.html")
        
    Returns:
        List of sections
    """
    file_path = settings.raw_html_path / file_name
    return parse_quantconnect_file(file_path)


def get_section_by_id(sections: List[Section], section_id: str) -> Optional[Section]:
    """
    Tìm section theo ID.
    
    Args:
        sections: List of sections
        section_id: ID to find (e.g., "1.2.3")
        
    Returns:
        Section object or None
    """
    for section in sections:
        if section.id == section_id:
            return section
    return None


def get_sections_by_level(sections: List[Section], level: int) -> List[Section]:
    """
    Lấy tất cả sections ở một level cụ thể.
    
    Args:
        sections: List of sections
        level: Level to filter (1, 2, 3, etc.)
        
    Returns:
        List of sections at that level
    """
    return [s for s in sections if s.level == level]


def get_sections_with_code(sections: List[Section], language: Optional[str] = None) -> List[Section]:
    """
    Lấy tất cả sections có code blocks.
    
    Args:
        sections: List of sections
        language: Optional - filter by language ('python' or 'csharp')
        
    Returns:
        List of sections containing code blocks
    """
    result = []
    for section in sections:
        if section.code_blocks:
            if language is None:
                result.append(section)
            else:
                # Check if section has code in specified language
                for cb in section.code_blocks:
                    if cb.language == language:
                        result.append(section)
                        break
    return result


def print_section_tree(sections: List[Section], max_depth: int = 3):
    """
    In cây sections để dễ visualize structure.
    
    Args:
        sections: List of sections
        max_depth: Maximum depth to display
    """
    # Build hierarchy first
    root_sections = [s for s in sections if s.parent_id is None]
    
    def print_section(section: Section, indent: int = 0):
        if indent // 2 >= max_depth:
            return
            
        prefix = " " * indent + ("└─ " if indent > 0 else "")
        print(f"{prefix}{section.id} {section.title}")
        
        # Print subsections
        for subsection in section.subsections:
            print_section(subsection, indent + 2)
    
    for section in root_sections:
        print_section(section)


# Example usage
if __name__ == "__main__":
    # Test với một file
    print("Testing QuantConnect parser helper...")
    
    try:
        # Parse một file
        sections = quick_parse("Quantconnect-Writing-Algorithms.html")
        print(f"\nParsed {len(sections)} sections")
        
        # Print section tree
        print("\nSection Structure:")
        print_section_tree(sections, max_depth=2)
        
        # Find sections with Python code
        python_sections = get_sections_with_code(sections, 'python')
        print(f"\n{len(python_sections)} sections contain Python code")
        
        # Get all level 1 sections
        level1_sections = get_sections_by_level(sections, 1)
        print(f"\n{len(level1_sections)} top-level sections:")
        for s in level1_sections:
            print(f"  - {s.id} {s.title}")
            
    except Exception as e:
        print(f"Error: {e}")