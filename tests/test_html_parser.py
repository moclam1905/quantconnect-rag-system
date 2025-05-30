"""
Test script chi tiết cho QuantConnect HTML Parser
Giúp kiểm tra parser với các file HTML thực tế và debug các vấn đề
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing.html_parser import QuantConnectHTMLParser, Section, CodeBlock
from src.utils.logger import logger
from config.config import settings
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from rich import print as rprint
import json
from typing import List, Dict
import time


console = Console()


def analyze_html_file(file_path: Path, target_document_index: int = 1) -> Dict:
    """
    Phân tích chi tiết một file HTML và trả về statistics
    
    Args:
        file_path: Path to HTML file
        target_document_index: Index của document cần parse (default=1 cho QuantConnect files)
    """
    logger.info(f"Analyzing {file_path.name} - Document index: {target_document_index}...")
    start_time = time.time()
    
    try:
        # Parse file
        parser = QuantConnectHTMLParser(file_path)
        sections = parser.parse(target_document_index=target_document_index)
        
        # Collect statistics
        stats = {
            'file_name': file_path.name,
            'document_index': target_document_index,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'parse_time_seconds': time.time() - start_time,
            'total_sections': len(sections),
            'sections_by_level': {},
            'total_code_blocks': 0,
            'code_blocks_by_language': {},
            'total_tables': 0,
            'sections_with_content': 0,
            'sections_empty': 0,
            'avg_content_length': 0,
            'largest_section': None,
            'sections_with_code': 0,
            'sections_with_tables': 0
        }
        
        # Analyze sections
        total_content_length = 0
        largest_content_length = 0
        
        for section in sections:
            # Count by level
            level = f"h{section.level}"
            stats['sections_by_level'][level] = stats['sections_by_level'].get(level, 0) + 1
            
            # Content analysis
            content_length = len(section.content)
            total_content_length += content_length
            
            if content_length > 0:
                stats['sections_with_content'] += 1
            else:
                stats['sections_empty'] += 1
            
            if content_length > largest_content_length:
                largest_content_length = content_length
                stats['largest_section'] = {
                    'title': section.title,
                    'length': content_length
                }
            
            # Code blocks analysis
            if section.code_blocks:
                stats['sections_with_code'] += 1
                stats['total_code_blocks'] += len(section.code_blocks)
                
                for code_block in section.code_blocks:
                    lang = code_block.language
                    stats['code_blocks_by_language'][lang] = stats['code_blocks_by_language'].get(lang, 0) + 1
            
            # Tables analysis
            if section.tables:
                stats['sections_with_tables'] += 1
                stats['total_tables'] += len(section.tables)
        
        # Calculate averages
        if stats['sections_with_content'] > 0:
            stats['avg_content_length'] = total_content_length / stats['sections_with_content']
        
        # Save parsed data
        output_file = parser.save_parsed_data(settings.processed_data_path)
        stats['output_file'] = str(output_file)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error analyzing {file_path.name}: {str(e)}")
        return {
            'file_name': file_path.name,
            'document_index': target_document_index,
            'error': str(e)
        }


def display_section_details(sections: List[Section], max_sections: int = 10):
    """
    Hiển thị chi tiết một số sections đầu tiên
    """
    console.print(f"\n[cyan]Displaying first {max_sections} sections in detail:[/cyan]")
    
    for i, section in enumerate(sections[:max_sections]):
        # Section header
        console.print(f"\n[bold green]Section {i+1}: {section.title}[/bold green]")
        console.print(f"Level: H{section.level} | ID: {section.id}")
        
        # Content preview
        if section.content:
            content_preview = section.content[:200] + "..." if len(section.content) > 200 else section.content
            console.print(Panel(content_preview, title="Content Preview", border_style="blue"))
        
        # Code blocks
        if section.code_blocks:
            console.print(f"\n[yellow]Code Blocks ({len(section.code_blocks)}):[/yellow]")
            for j, code_block in enumerate(section.code_blocks[:2]):  # Show max 2 code blocks
                code_preview = code_block.content[:150] + "..." if len(code_block.content) > 150 else code_block.content
                syntax = Syntax(code_preview, code_block.language, theme="monokai", line_numbers=True)
                console.print(Panel(syntax, title=f"Code Block {j+1} ({code_block.language})", border_style="green"))
        
        # Tables
        if section.tables:
            console.print(f"\n[yellow]Tables ({len(section.tables)}):[/yellow]")
            for j, table_data in enumerate(section.tables[:1]):  # Show max 1 table
                table = Table(title=f"Table {j+1}")
                
                # Add headers
                for header in table_data.headers[:5]:  # Show max 5 columns
                    table.add_column(header)
                
                # Add rows (max 3 rows)
                for row in table_data.rows[:3]:
                    table.add_row(*row[:5])
                
                if len(table_data.rows) > 3:
                    table.add_row(*["..." for _ in range(min(5, len(table_data.headers)))])
                
                console.print(table)


def test_single_file(file_name: str, document_index: int = 1):
    """
    Test parser với một file cụ thể
    
    Args:
        file_name: Tên file HTML
        document_index: Index của document cần parse (default=1 cho QuantConnect files)
    """
    file_path = settings.raw_html_path / file_name
    
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        return
    
    console.print(Panel.fit(
        f"[bold cyan]Testing HTML Parser with {file_name}[/bold cyan]\n"
        f"Document Index: {document_index}", 
        border_style="cyan"
    ))
    
    # Analyze file
    stats = analyze_html_file(file_path, document_index)
    
    if 'error' in stats:
        console.print(f"[red]Error: {stats['error']}[/red]")
        return
    
    # Display statistics
    table = Table(title="Parsing Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("File Size", f"{stats['file_size_mb']:.2f} MB")
    table.add_row("Document Index", str(stats['document_index']))
    table.add_row("Parse Time", f"{stats['parse_time_seconds']:.2f} seconds")
    table.add_row("Total Sections", str(stats['total_sections']))
    table.add_row("Sections with Content", str(stats['sections_with_content']))
    table.add_row("Empty Sections", str(stats['sections_empty']))
    table.add_row("Avg Content Length", f"{stats['avg_content_length']:.0f} chars")
    
    if stats['largest_section']:
        table.add_row("Largest Section", 
                     f"{stats['largest_section']['title'][:50]}... ({stats['largest_section']['length']} chars)")
    
    table.add_row("Total Code Blocks", str(stats['total_code_blocks']))
    table.add_row("Sections with Code", str(stats['sections_with_code']))
    table.add_row("Total Tables", str(stats['total_tables']))
    table.add_row("Sections with Tables", str(stats['sections_with_tables']))
    
    console.print(table)
    
    # Display section levels breakdown
    if stats['sections_by_level']:
        table = Table(title="Sections by Level", show_header=True)
        table.add_column("Level", style="cyan")
        table.add_column("Count", style="green")
        
        for level in sorted(stats['sections_by_level'].keys()):
            table.add_row(level.upper(), str(stats['sections_by_level'][level]))
        
        console.print(table)
    
    # Display code languages breakdown
    if stats['code_blocks_by_language']:
        table = Table(title="Code Blocks by Language", show_header=True)
        table.add_column("Language", style="cyan")
        table.add_column("Count", style="green")
        
        for lang, count in stats['code_blocks_by_language'].items():
            table.add_row(lang, str(count))
        
        console.print(table)
    
    console.print(f"\n[green]✓ Output saved to: {stats['output_file']}[/green]")
    
    # Ask if user wants to see section details
    console.print("\n[yellow]Would you like to see detailed section examples? (y/n)[/yellow]")
    if input().lower() == 'y':
        # Load and display sections
        parser = QuantConnectHTMLParser(file_path)
        sections = parser.parse(target_document_index=document_index)
        display_section_details(sections)


def test_all_files():
    """
    Test parser với tất cả các file HTML trong thư mục
    """
    html_files = [
        "Quantconnect-Lean-Cli.html",
        "Quantconnect-Lean-Engine.html",
        "Quantconnect-Research-Environment.html",
        "Quantconnect-Writing-Algorithms.html"
    ]
    
    console.print(Panel.fit(
        "[bold cyan]Testing HTML Parser with All Files[/bold cyan]",
        border_style="cyan"
    ))
    
    all_stats = []
    
    for file_name in html_files:
        file_path = settings.raw_html_path / file_name
        if file_path.exists():
            # Always use document index 1 for QuantConnect files
            stats = analyze_html_file(file_path, target_document_index=1)
            all_stats.append(stats)
        else:
            console.print(f"[yellow]Skipping {file_name} (not found)[/yellow]")
    
    # Display summary
    table = Table(title="Summary of All Files", show_header=True)
    table.add_column("File", style="cyan")
    table.add_column("Size (MB)", style="green")
    table.add_column("Sections", style="green")
    table.add_column("Code Blocks", style="green")
    table.add_column("Tables", style="green")
    table.add_column("Parse Time (s)", style="green")
    
    for stats in all_stats:
        if 'error' not in stats:
            table.add_row(
                stats['file_name'][:30] + "..." if len(stats['file_name']) > 30 else stats['file_name'],
                f"{stats['file_size_mb']:.1f}",
                str(stats['total_sections']),
                str(stats['total_code_blocks']),
                str(stats['total_tables']),
                f"{stats['parse_time_seconds']:.2f}"
            )
    
    console.print(table)


def main():
    """
    Main function với menu để chọn test options
    """
    console.print(Panel.fit(
        "[bold cyan]QuantConnect HTML Parser Test Suite[/bold cyan]\n"
        "Test và validate HTML parser với các file thực tế",
        border_style="cyan"
    ))
    
    while True:
        console.print("\n[cyan]Choose an option:[/cyan]")
        console.print("1. Test single file (Quantconnect-Lean-Engine.html) - Document 0")
        console.print("2. Test all files")
        console.print("3. Test specific file (enter name)")
        console.print("4. Test specific document in file")
        console.print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ")
        
        if choice == '1':
            test_single_file("Quantconnect-Lean-Engine.html", 1)
        elif choice == '2':
            test_all_files()
        elif choice == '3':
            file_name = input("Enter file name: ")
            test_single_file(file_name, 1)  # Default to document 1
        elif choice == '4':
            file_name = input("Enter file name: ")
            try:
                doc_index = int(input("Enter document index (default=1 for QuantConnect files): ") or "1")
                test_single_file(file_name, doc_index)
            except ValueError:
                console.print("[red]Invalid document index![/red]")
        elif choice == '5':
            break
        else:
            console.print("[red]Invalid choice![/red]")


if __name__ == "__main__":
    main()