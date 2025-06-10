"""
Integration pipeline để chunk parsed QuantConnect documents.
Đọc từ parsed JSON và output chunks ready cho embedding.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing.old.text_chunker import TextChunker, AdvancedTextChunker
from src.data_processing.old.chunk_models import Chunk, ChunkingConfig
from src.data_processing.old.html_parser import Section
from src.data_processing.old.chunking_config import (
    get_chunking_config_for_file,
    get_chunking_config_for_section
)
from src.utils.logger import logger
from config.config import settings
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn


class ChunkingPipeline:
    """
    Pipeline để chunk parsed documents.
    Handles loading, chunking, và saving.
    """
    
    def __init__(
        self,
        chunker_type: str = "advanced",
        config: Optional[ChunkingConfig] = None
    ):
        self.chunker_type = chunker_type
        self.default_config = config
        
    def process_parsed_file(
        self,
        parsed_json_path: Path,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Process một parsed JSON file và tạo chunks.
        
        Args:
            parsed_json_path: Path to parsed JSON file
            output_dir: Output directory (default: same as input)
            
        Returns:
            Dictionary với processing statistics
        """
        logger.info(f"Processing {parsed_json_path.name}")
        
        # Load parsed data
        with open(parsed_json_path, 'r', encoding='utf-8') as f:
            parsed_data = json.load(f)
        
        source_file = parsed_data.get('source_file', parsed_json_path.stem)
        doc_index = parsed_data.get('document_index', 1)
        
        # Convert section dicts back to Section objects
        sections = self._load_sections(parsed_data['sections'])
        
        # Get appropriate config if not provided
        if not self.default_config:
            config = get_chunking_config_for_file(source_file)
        else:
            config = self.default_config
        
        # Process each section
        all_chunks = []
        stats = {
            'total_sections': len(sections),
            'sections_processed': 0,
            'sections_skipped': 0,
            'total_chunks': 0,
            'total_chars': 0
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            
            task = progress.add_task(
                f"[cyan]Chunking sections...", 
                total=len(sections)
            )
            
            for section in sections:
                chunks = self._process_section(
                    section, 
                    source_file, 
                    doc_index,
                    config
                )
                
                if chunks:
                    all_chunks.extend(chunks)
                    stats['sections_processed'] += 1
                    stats['total_chunks'] += len(chunks)
                    stats['total_chars'] += sum(c.char_count for c in chunks)
                else:
                    stats['sections_skipped'] += 1
                
                progress.update(task, advance=1)
        
        # Save chunks
        output_path = self._save_chunks(
            all_chunks,
            parsed_json_path,
            output_dir,
            stats
        )
        
        stats['output_file'] = str(output_path)
        
        logger.info(f"Chunking complete: {stats['total_chunks']} chunks created")
        
        return stats
    
    def process_batch(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        file_pattern: str = "*_parsed.json"
    ) -> List[Dict]:
        """
        Process multiple parsed files in batch.
        
        Args:
            input_dir: Directory containing parsed JSON files
            output_dir: Output directory
            file_pattern: Pattern to match parsed files
            
        Returns:
            List of processing statistics
        """
        # Find all parsed files
        parsed_files = list(input_dir.glob(file_pattern))
        
        if not parsed_files:
            logger.warning(f"No files matching {file_pattern} found in {input_dir}")
            return []
        
        logger.info(f"Found {len(parsed_files)} files to process")
        
        # Process each file
        results = []
        for parsed_file in parsed_files:
            try:
                stats = self.process_parsed_file(parsed_file, output_dir)
                results.append(stats)
            except Exception as e:
                logger.error(f"Error processing {parsed_file.name}: {str(e)}")
                results.append({
                    'file': str(parsed_file),
                    'error': str(e)
                })
        
        return results
    
    def _load_sections(self, section_dicts: List[Dict]) -> List[Section]:
        """Convert section dictionaries back to Section objects"""
        sections = []
        
        for s_dict in section_dicts:
            section = Section(
                id=s_dict['id'],
                title=s_dict['title'],
                level=s_dict['level'],
                content=s_dict.get('content', ''),
                section_number=s_dict.get('section_number'),
                breadcrumb=s_dict.get('breadcrumb'),
                parent_id=s_dict.get('parent_id')
            )
            
            # Simplified - not loading code blocks and tables for now
            # In full implementation, would reconstruct these as well
            section.code_blocks = []
            section.tables = []
            
            sections.append(section)
        
        return sections
    
    def _process_section(
        self,
        section: Section,
        source_file: str,
        doc_index: int,
        config: ChunkingConfig
    ) -> List[Chunk]:
        """Process a single section into chunks"""
        # Skip empty sections
        if not section.content or not section.content.strip():
            return []
        
        # Get section-specific config if needed
        if section.code_blocks:
            # If section has code, might want different config
            languages = set(cb.language for cb in section.code_blocks)
            primary_language = list(languages)[0] if languages else None
            
            section_config = get_chunking_config_for_section(
                section.title,
                has_code=True,
                code_language=primary_language
            )
        else:
            section_config = config
        
        # Create appropriate chunker
        if self.chunker_type == "basic":
            chunker = TextChunker(section_config)
        else:
            chunker = AdvancedTextChunker(section_config)
        
        # Chunk the section
        try:
            chunks = chunker.chunk_text(
                section.content,
                source_file,
                section,
                doc_index
            )
            return chunks
        except Exception as e:
            logger.error(f"Error chunking section {section.id}: {str(e)}")
            return []
    
    def _save_chunks(
        self,
        chunks: List[Chunk],
        source_path: Path,
        output_dir: Optional[Path],
        stats: Dict
    ) -> Path:
        """Save chunks to JSON file"""
        # Determine output path
        if output_dir is None:
            output_dir = source_path.parent
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output filename
        base_name = source_path.stem.replace('_parsed', '')
        output_file = output_dir / f"{base_name}_chunks.json"
        
        # Prepare data for saving
        output_data = {
            'source_file': source_path.name,
            'chunking_config': {
                'chunker_type': self.chunker_type,
                'timestamp': datetime.now().isoformat()
            },
            'statistics': stats,
            'chunks': [chunk.to_dict() for chunk in chunks]
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_file}")
        
        return output_file


def chunk_single_file(
    parsed_json_path: Path,
    chunker_type: str = "advanced",
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Convenience function to chunk a single parsed file.
    
    Args:
        parsed_json_path: Path to parsed JSON
        chunker_type: "basic" or "advanced"
        output_dir: Optional output directory
        
    Returns:
        Processing statistics
    """
    pipeline = ChunkingPipeline(chunker_type=chunker_type)
    return pipeline.process_parsed_file(parsed_json_path, output_dir)


def chunk_all_parsed_files(
    chunker_type: str = "advanced",
    batch_dir: Optional[Path] = None
) -> List[Dict]:
    """
    Chunk all parsed files in the latest batch directory.
    
    Args:
        chunker_type: "basic" or "advanced"
        batch_dir: Specific batch directory, or auto-find latest
        
    Returns:
        List of processing statistics
    """
    if batch_dir is None:
        # Find latest batch directory
        batch_dirs = list(settings.processed_data_path.glob("batch_*"))
        if not batch_dirs:
            logger.error("No batch directories found")
            return []
        
        # Get most recent
        batch_dir = max(batch_dirs, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using latest batch directory: {batch_dir}")
    
    pipeline = ChunkingPipeline(chunker_type=chunker_type)
    return pipeline.process_batch(batch_dir)


# Example usage
if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    console.print("[bold cyan]Chunking Pipeline Test[/bold cyan]")
    
    # Find a parsed file to test
    parsed_files = list(settings.processed_data_path.rglob("*_parsed.json"))
    
    if parsed_files:
        test_file = parsed_files[0]
        console.print(f"\nTesting with: {test_file.name}")
        
        # Process file
        stats = chunk_single_file(test_file, chunker_type="advanced")
        
        # Display results
        table = Table(title="Chunking Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in stats.items():
            if key != 'output_file':
                table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)
        console.print(f"\n[green]Output saved to: {stats.get('output_file')}[/green]")
    else:
        console.print("[red]No parsed files found. Run batch_process_documents.py first.[/red]")