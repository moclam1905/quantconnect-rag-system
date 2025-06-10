"""
Batch processing script để parse tất cả documents trong các file HTML của QuantConnect.
Xử lý trường hợp mỗi file có thể chứa nhiều documents.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing.old.html_parser import QuantConnectHTMLParser
from src.data_processing.old.parser_utils import count_documents_in_html, export_sections_for_rag
from src.utils.logger import logger
from config.config import settings
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
import json
from datetime import datetime
from typing import List, Dict

console = Console()


def process_all_documents(html_files: List[str] = None, skip_first_document: bool = True) -> Dict:
    """
    Process documents trong các file HTML.
    
    Args:
        html_files: List of HTML files to process. If None, process all default files.
        skip_first_document: If True, skip document 0 (dư thừa) và chỉ process từ document 1
        
    Returns:
        Dictionary với statistics về quá trình processing
    """
    if html_files is None:
        html_files = [
            "Quantconnect-Lean-Cli.html",
            "Quantconnect-Lean-Engine.html",
            "Quantconnect-Research-Environment.html",
            "Quantconnect-Writing-Algorithms.html"
        ]
    
    results = {
        'processed_files': 0,
        'total_documents': 0,
        'successful_documents': 0,
        'failed_documents': 0,
        'skipped_documents': 0,
        'total_sections': 0,
        'total_code_blocks': 0,
        'total_tables': 0,
        'details': []
    }
    
    start_time = datetime.now()
    
    console.print(Panel.fit(
        "[bold cyan]QuantConnect Document Batch Processor[/bold cyan]\n"
        f"Processing {len(html_files)} HTML files\n"
        f"Skip first document: {skip_first_document}",
        border_style="cyan"
    ))
    
    # Create output directory for batch results
    batch_output_dir = settings.processed_data_path / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        # Process each file
        for file_name in html_files:
            file_path = settings.raw_html_path / file_name
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                results['details'].append({
                    'file': file_name,
                    'status': 'not_found',
                    'error': 'File not found'
                })
                continue
            
            results['processed_files'] += 1
            
            # Count documents in file
            doc_count = count_documents_in_html(file_path)
            
            # Determine which documents to process
            start_index = 1 if skip_first_document and doc_count > 1 else 0
            docs_to_process = doc_count - start_index
            
            results['total_documents'] += docs_to_process
            if skip_first_document and doc_count > 1:
                results['skipped_documents'] += 1
            
            file_task = progress.add_task(f"[cyan]Processing {file_name}", total=docs_to_process)
            
            file_results = {
                'file': file_name,
                'document_count': doc_count,
                'documents_processed': docs_to_process,
                'documents': []
            }
            
            # Process each document in the file (starting from start_index)
            for doc_index in range(start_index, doc_count):
                try:
                    # Parse document
                    parser = QuantConnectHTMLParser(file_path)
                    sections = parser.parse(target_document_index=doc_index)
                    
                    # Calculate statistics
                    doc_stats = {
                        'document_index': doc_index,
                        'sections': len(sections),
                        'code_blocks': sum(len(s.code_blocks) for s in sections),
                        'tables': sum(len(s.tables) for s in sections),
                        'status': 'success'
                    }
                    
                    # Save parsed data
                    output_filename = f"{file_path.stem}_parsed.json"
                    output_path = batch_output_dir / output_filename
                    
                    # Save with additional metadata
                    data = {
                        'source_file': file_name,
                        'document_index': doc_index,
                        'total_documents_in_file': doc_count,
                        'parse_timestamp': datetime.now().isoformat(),
                        'sections': [s.to_dict() for s in sections],
                        'statistics': doc_stats
                    }
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    # Export for RAG
                    rag_output = batch_output_dir / f"{file_path.stem}_rag.json"
                    export_sections_for_rag(
                        sections, 
                        rag_output,
                        metadata={
                            'source_file': file_name,
                            'document_index': doc_index
                        }
                    )
                    
                    # Update results
                    results['successful_documents'] += 1
                    results['total_sections'] += doc_stats['sections']
                    results['total_code_blocks'] += doc_stats['code_blocks']
                    results['total_tables'] += doc_stats['tables']
                    
                    file_results['documents'].append(doc_stats)
                    
                    progress.update(file_task, advance=1)
                    
                except Exception as e:
                    logger.error(f"Error processing {file_name} document {doc_index}: {str(e)}")
                    results['failed_documents'] += 1
                    
                    file_results['documents'].append({
                        'document_index': doc_index,
                        'status': 'failed',
                        'error': str(e)
                    })
                    
                    progress.update(file_task, advance=1)
            
            results['details'].append(file_results)
            progress.remove_task(file_task)
    
    # Calculate processing time
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    results['processing_time_seconds'] = processing_time
    
    # Save batch summary
    summary_path = batch_output_dir / "batch_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Display results
    display_batch_results(results, batch_output_dir)
    
    return results


def display_batch_results(results: Dict, output_dir: Path):
    """
    Display batch processing results in a nice format.
    """
    console.print("\n" + "="*50 + "\n")
    
    # Summary table
    table = Table(title="Batch Processing Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Files Processed", str(results['processed_files']))
    table.add_row("Total Documents", str(results['total_documents']))
    table.add_row("Skipped Documents", str(results.get('skipped_documents', 0)))
    table.add_row("Successful Documents", str(results['successful_documents']))
    table.add_row("Failed Documents", str(results['failed_documents']))
    table.add_row("Total Sections", str(results['total_sections']))
    table.add_row("Total Code Blocks", str(results['total_code_blocks']))
    table.add_row("Total Tables", str(results['total_tables']))
    table.add_row("Processing Time", f"{results['processing_time_seconds']:.2f} seconds")
    
    console.print(table)
    
    # Details by file
    console.print("\n[bold cyan]Details by File:[/bold cyan]")
    
    for file_detail in results['details']:
        if file_detail.get('status') == 'not_found':
            console.print(f"\n[red]✗ {file_detail['file']}: {file_detail['error']}[/red]")
            continue
        
        console.print(f"\n[green]✓ {file_detail['file']}[/green] "
                     f"({file_detail['document_count']} total documents, "
                     f"{file_detail.get('documents_processed', file_detail['document_count'])} processed)")
        
        for doc in file_detail['documents']:
            if doc['status'] == 'success':
                console.print(f"  Document {doc['document_index']}: "
                             f"{doc['sections']} sections, "
                             f"{doc['code_blocks']} code blocks, "
                             f"{doc['tables']} tables")
            else:
                console.print(f"  [red]Document {doc['document_index']}: Failed - {doc.get('error', 'Unknown error')}[/red]")
    
    console.print(f"\n[green]✓ All results saved to: {output_dir}[/green]")


def process_single_file_all_documents(file_name: str, skip_first: bool = True) -> Dict:
    """
    Process documents trong một file HTML cụ thể.
    
    Args:
        file_name: Tên file HTML
        skip_first: If True, skip document 0 (mặc định True cho QuantConnect files)
    """
    return process_all_documents([file_name], skip_first_document=skip_first)


def main():
    """
    Main function với options để batch process documents.
    """
    console.print(Panel.fit(
        "[bold cyan]QuantConnect Batch Document Processor[/bold cyan]\n"
        "Process multiple documents across HTML files",
        border_style="cyan"
    ))
    
    while True:
        console.print("\n[cyan]Choose an option:[/cyan]")
        console.print("1. Process all QuantConnect files (skip document 0)")
        console.print("2. Process specific file (skip document 0)")
        console.print("3. Process ALL documents in file (including document 0)")
        console.print("4. List documents in file")
        console.print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ")
        
        if choice == '1':
            process_all_documents(skip_first_document=True)
            
        elif choice == '2':
            file_name = input("Enter file name: ")
            process_single_file_all_documents(file_name, skip_first=True)
            
        elif choice == '3':
            file_name = input("Enter file name: ")
            process_single_file_all_documents(file_name, skip_first=False)
            
        elif choice == '4':
            file_name = input("Enter file name: ")
            file_path = settings.raw_html_path / file_name
            
            if file_path.exists():
                doc_count = count_documents_in_html(file_path)
                console.print(f"\n[green]{file_name} contains {doc_count} document(s)[/green]")
                console.print("[yellow]Note: Document 0 is usually redundant in QuantConnect files[/yellow]")
                
                # Try to get more info by parsing ToC of document 1
                if doc_count > 1:
                    try:
                        parser = QuantConnectHTMLParser(file_path)
                        parser._load_html(1)
                        parser._parse_table_of_contents()
                        if parser.toc_structure:
                            console.print("\nTable of Contents (Document 1 - Main content):")
                            for i, (section_id, info) in enumerate(list(parser.toc_structure.items())[:10]):
                                console.print(f"  {section_id}: {info['title']}")
                            if len(parser.toc_structure) > 10:
                                console.print(f"  ... and {len(parser.toc_structure) - 10} more sections")
                    except:
                        pass
            else:
                console.print(f"[red]File not found: {file_path}[/red]")
                
        elif choice == '5':
            break
        else:
            console.print("[red]Invalid choice![/red]")


if __name__ == "__main__":
    main()