"""
Test text chunker với real QuantConnect data từ parsed JSON files.
"""

import json
from pathlib import Path
from typing import List, Dict
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing.text_chunker import TextChunker, AdvancedTextChunker
from src.data_processing.chunk_models import ChunkingConfig, ChunkingPresets
from src.data_processing.html_parser import Section
from src.data_processing.chunking_config import get_chunking_config_for_file
from src.utils.logger import logger
from config.config import settings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
import numpy as np

console = Console()


def load_parsed_sections(json_file: Path) -> List[Section]:
    """Load parsed sections từ JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sections = []
    for section_dict in data.get('sections', []):
        # Recreate Section object từ dict
        section = Section(
            id=section_dict['id'],
            title=section_dict['title'],
            level=section_dict['level'],
            content=section_dict.get('content', ''),
            section_number=section_dict.get('section_number'),
            breadcrumb=section_dict.get('breadcrumb'),
            parent_id=section_dict.get('parent_id')
        )
        
        # Add code blocks và tables nếu cần (simplified for now)
        section.code_blocks = []
        section.tables = []
        
        sections.append(section)
    
    return sections


def analyze_chunks(chunks: List) -> Dict:
    """Analyze chunk statistics"""
    if not chunks:
        return {}
    
    sizes = [chunk.char_count for chunk in chunks]
    
    return {
        'total_chunks': len(chunks),
        'total_chars': sum(sizes),
        'avg_size': np.mean(sizes),
        'min_size': min(sizes),
        'max_size': max(sizes),
        'std_size': np.std(sizes),
        'size_distribution': {
            '<500': sum(1 for s in sizes if s < 500),
            '500-1000': sum(1 for s in sizes if 500 <= s < 1000),
            '1000-1500': sum(1 for s in sizes if 1000 <= s < 1500),
            '>1500': sum(1 for s in sizes if s >= 1500)
        }
    }


def test_chunker_on_section(
    section: Section,
    chunker: TextChunker,
    source_file: str,
    display_examples: bool = True
) -> Dict:
    """Test chunker on a single section"""
    # Skip if no content
    if not section.content:
        return {'skipped': True, 'reason': 'No content'}
    
    # Chunk the section
    chunks = chunker.chunk_text(
        section.content,
        source_file,
        section,
        doc_index=1
    )
    
    # Analyze results
    stats = analyze_chunks(chunks)
    stats['section_id'] = section.id
    stats['section_title'] = section.title
    stats['original_size'] = len(section.content)
    
    # Display examples if requested
    if display_examples and chunks:
        console.print(f"\n[cyan]Section {section.id}: {section.title}[/cyan]")
        console.print(f"Original size: {stats['original_size']} chars")
        console.print(f"Chunks created: {len(chunks)}")
        
        # Show first 2 chunks as examples
        for i, chunk in enumerate(chunks[:2]):
            console.print(f"\n[green]Chunk {i+1}/{len(chunks)}:[/green]")
            console.print(f"Size: {chunk.char_count} chars")
            
            # Show content preview
            preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            console.print(Panel(preview, title=f"Content Preview", border_style="blue"))
            
            # Show metadata
            console.print(f"Metadata:")
            console.print(f"  - Chunk ID: {chunk.chunk_id[:8]}...")
            console.print(f"  - Position: {chunk.metadata.chunk_index + 1}/{chunk.metadata.total_chunks_in_section}")
            console.print(f"  - Overlap prev: {chunk.metadata.overlap_with_previous}")
            console.print(f"  - Overlap next: {chunk.metadata.overlap_with_next}")
    
    return stats


def test_chunker_on_file(
    parsed_json_file: Path,
    chunker_type: str = "basic",
    max_sections: int = 10,
    display_examples: bool = True
) -> Dict:
    """Test chunker on entire parsed file"""
    # Load sections
    sections = load_parsed_sections(parsed_json_file)
    source_file = parsed_json_file.stem.replace('_parsed', '.html')
    
    console.print(Panel.fit(
        f"[bold cyan]Testing {chunker_type} chunker on {source_file}[/bold cyan]\n"
        f"Total sections: {len(sections)}",
        border_style="cyan"
    ))
    
    # Get appropriate config
    config = get_chunking_config_for_file(source_file)
    
    # Create chunker
    if chunker_type == "basic":
        chunker = TextChunker(config)
    else:
        chunker = AdvancedTextChunker(config)
    
    # Test on sections
    all_stats = []
    sections_to_test = sections[:max_sections] if max_sections else sections
    
    for section in sections_to_test:
        stats = test_chunker_on_section(
            section, 
            chunker, 
            source_file,
            display_examples=display_examples
        )
        all_stats.append(stats)
    
    # Aggregate statistics
    valid_stats = [s for s in all_stats if not s.get('skipped')]
    
    if valid_stats:
        total_chunks = sum(s['total_chunks'] for s in valid_stats)
        total_chars = sum(s['total_chars'] for s in valid_stats)
        all_avg_sizes = [s['avg_size'] for s in valid_stats]
        
        summary = {
            'file': source_file,
            'chunker_type': chunker_type,
            'sections_processed': len(valid_stats),
            'sections_skipped': len(all_stats) - len(valid_stats),
            'total_chunks': total_chunks,
            'total_chars': total_chars,
            'avg_chunks_per_section': total_chunks / len(valid_stats),
            'avg_chunk_size': np.mean(all_avg_sizes),
            'config': {
                'max_chunk_size': config.max_chunk_size,
                'chunk_overlap': config.chunk_overlap,
                'strategy': config.default_strategy.value
            }
        }
        
        # Display summary
        table = Table(title="Chunking Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Sections Processed", str(summary['sections_processed']))
        table.add_row("Total Chunks Created", str(summary['total_chunks']))
        table.add_row("Avg Chunks per Section", f"{summary['avg_chunks_per_section']:.1f}")
        table.add_row("Avg Chunk Size", f"{summary['avg_chunk_size']:.0f} chars")
        table.add_row("Max Chunk Size Config", str(config.max_chunk_size))
        table.add_row("Overlap Config", str(config.chunk_overlap))
        
        console.print(table)
        
        return summary
    
    return {}


def compare_chunkers(parsed_json_file: Path):
    """Compare basic vs advanced chunker"""
    sections = load_parsed_sections(parsed_json_file)
    source_file = parsed_json_file.stem.replace('_parsed', '.html')
    
    # Get a sample section with reasonable content
    sample_section = None
    for section in sections:
        if section.content and len(section.content) > 500:
            sample_section = section
            break
    
    if not sample_section:
        console.print("[red]No suitable section found for comparison[/red]")
        return
    
    console.print(Panel.fit(
        f"[bold cyan]Comparing Chunkers on Section {sample_section.id}[/bold cyan]\n"
        f"Section: {sample_section.title}\n"
        f"Original size: {len(sample_section.content)} chars",
        border_style="cyan"
    ))
    
    # Get config
    config = get_chunking_config_for_file(source_file)
    config.max_chunk_size = 500  # Smaller for better comparison
    config.chunk_overlap = 100
    
    # Test both chunkers
    basic_chunker = TextChunker(config)
    advanced_chunker = AdvancedTextChunker(config)
    
    basic_chunks = basic_chunker.chunk_text(
        sample_section.content,
        source_file,
        sample_section
    )
    
    advanced_chunks = advanced_chunker.chunk_text(
        sample_section.content,
        source_file,
        sample_section
    )
    
    # Compare results
    table = Table(title="Chunker Comparison", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Basic Chunker", style="green")
    table.add_column("Advanced Chunker", style="blue")
    
    table.add_row("Total Chunks", str(len(basic_chunks)), str(len(advanced_chunks)))
    table.add_row(
        "Avg Chunk Size",
        f"{np.mean([c.char_count for c in basic_chunks]):.0f}",
        f"{np.mean([c.char_count for c in advanced_chunks]):.0f}"
    )
    table.add_row(
        "Size Std Dev",
        f"{np.std([c.char_count for c in basic_chunks]):.0f}",
        f"{np.std([c.char_count for c in advanced_chunks]):.0f}"
    )
    
    console.print(table)
    
    # Show example chunks
    console.print("\n[yellow]Basic Chunker - First Chunk:[/yellow]")
    if basic_chunks:
        console.print(Panel(
            basic_chunks[0].content[:300] + "...",
            border_style="green"
        ))
    
    console.print("\n[yellow]Advanced Chunker - First Chunk:[/yellow]")
    if advanced_chunks:
        console.print(Panel(
            advanced_chunks[0].content[:300] + "...",
            border_style="blue"
        ))


def main():
    """Main test function"""
    console.print(Panel.fit(
        "[bold cyan]Text Chunker Test Suite[/bold cyan]\n"
        "Test text chunking với QuantConnect parsed data",
        border_style="cyan"
    ))
    
    # Find available parsed JSON files
    processed_dir = settings.processed_data_path
    json_files = list(processed_dir.rglob("*_parsed.json"))
    
    if not json_files:
        console.print("[red]No parsed JSON files found![/red]")
        console.print("Please run batch_process_documents.py first.")
        return
    
    while True:
        console.print("\n[cyan]Available parsed files:[/cyan]")
        for i, file in enumerate(json_files):
            console.print(f"{i+1}. {file.name}")
        
        console.print("\n[cyan]Choose an option:[/cyan]")
        console.print("1. Test basic chunker on file")
        console.print("2. Test advanced chunker on file")
        console.print("3. Compare chunkers")
        console.print("4. Test with custom config")
        console.print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ")
        
        if choice in ['1', '2', '3', '4']:
            file_idx = input(f"Select file (1-{len(json_files)}): ")
            try:
                file_idx = int(file_idx) - 1
                if 0 <= file_idx < len(json_files):
                    selected_file = json_files[file_idx]
                else:
                    console.print("[red]Invalid file index![/red]")
                    continue
            except:
                console.print("[red]Invalid input![/red]")
                continue
            
            if choice == '1':
                test_chunker_on_file(selected_file, "basic", max_sections=5)
            elif choice == '2':
                test_chunker_on_file(selected_file, "advanced", max_sections=5)
            elif choice == '3':
                compare_chunkers(selected_file)
            elif choice == '4':
                # Custom config test
                max_size = input("Max chunk size (default 1000): ") or "1000"
                overlap = input("Chunk overlap (default 100): ") or "100"
                
                config = ChunkingConfig(
                    max_chunk_size=int(max_size),
                    chunk_overlap=int(overlap),
                    ensure_complete_sentences=True
                )
                
                chunker = TextChunker(config)
                sections = load_parsed_sections(selected_file)[:3]
                
                for section in sections:
                    if section.content:
                        chunks = chunker.chunk_text(
                            section.content,
                            selected_file.stem,
                            section
                        )
                        console.print(f"\nSection {section.id}: {len(chunks)} chunks")
                        
        elif choice == '5':
            break
        else:
            console.print("[red]Invalid choice![/red]")


if __name__ == "__main__":
    main()