"""
Fixed version of chunking_pipeline.py
Properly loads code blocks and tables from parsed JSON
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing.text_chunker import TextChunker, AdvancedTextChunker
from src.data_processing.chunk_models import Chunk, ChunkingConfig
from src.data_processing.html_parser import Section, CodeBlock, TableData
from src.data_processing.chunking_config import (
    get_chunking_config_for_file,
    get_chunking_config_for_section
)
from src.utils.logger import logger
from config.config import settings
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn


class FixedChunkingPipeline:
    """
    Fixed Pipeline để chunk parsed documents.
    Properly handles code blocks and tables.
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
        
        # ✅ FIX: Convert section dicts back to Section objects WITH code blocks
        sections = self._load_sections_with_code(parsed_data['sections'])
        
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
            'total_chars': 0,
            'code_blocks_found': 0,
            'tables_found': 0
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
                    stats['code_blocks_found'] += len(section.code_blocks)
                    stats['tables_found'] += len(section.tables)
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
        logger.info(f"Code blocks found: {stats['code_blocks_found']}")
        logger.info(f"Tables found: {stats['tables_found']}")
        
        return stats
    
    def _load_sections_with_code(self, section_dicts: List[Dict]) -> List[Section]:
        """✅ FIXED: Convert section dictionaries back to Section objects WITH code blocks"""
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
            
            # ✅ FIX: Properly load code blocks
            section.code_blocks = []
            for cb_dict in s_dict.get('code_blocks', []):
                code_block = CodeBlock(
                    language=cb_dict.get('language', 'text'),
                    content=cb_dict.get('content', ''),
                    section_id=section.id,
                    line_number=cb_dict.get('line_number')
                )
                section.code_blocks.append(code_block)
            
            # ✅ FIX: Properly load tables
            section.tables = []
            for table_dict in s_dict.get('tables', []):
                table = TableData(
                    headers=table_dict.get('headers', []),
                    rows=table_dict.get('rows', []),
                    section_id=section.id,
                    caption=table_dict.get('caption')
                )
                section.tables.append(table)
            
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
        if not section.content and not section.code_blocks and not section.tables:
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
        
        # ✅ ENHANCED: Try to use code-aware chunker if available and section has code
        if section.code_blocks and self.chunker_type == "auto":
            try:
                from src.data_processing.code_aware_chunker import CodeAwareChunker
                chunker = CodeAwareChunker(section_config)
                logger.info(f"Using CodeAwareChunker for section {section.id} with {len(section.code_blocks)} code blocks")
            except ImportError:
                logger.warning("CodeAwareChunker not available, using AdvancedTextChunker")
                chunker = AdvancedTextChunker(section_config)
        
        # Chunk the section
        try:
            chunks = chunker.chunk_text(
                section.content,
                source_file,
                section,
                doc_index
            )
            
            # ✅ ADD: Create separate chunks for large code blocks
            code_chunks = self._create_code_chunks(section, source_file, doc_index, section_config)
            chunks.extend(code_chunks)
            
            # ✅ ADD: Create chunks for tables
            table_chunks = self._create_table_chunks(section, source_file, doc_index, section_config)
            chunks.extend(table_chunks)
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking section {section.id}: {str(e)}")
            return []
    
    def _create_code_chunks(
        self,
        section: Section,
        source_file: str,
        doc_index: int,
        config: ChunkingConfig
    ) -> List[Chunk]:
        """Create separate chunks for code blocks"""
        code_chunks = []
        
        for i, code_block in enumerate(section.code_blocks):
            # Only create separate chunk if code is substantial
            if len(code_block.content) > 200:  # Threshold for separate chunk
                
                # Create content with context
                content_parts = [
                    f"[Code Example from Section {section.id}: {section.title}]",
                    f"Language: {code_block.language}",
                    f"```{code_block.language}",
                    code_block.content,
                    "```"
                ]
                
                content = '\n'.join(content_parts)
                
                # Create metadata
                from src.data_processing.chunk_models import ChunkMetadata, ChunkType
                metadata = ChunkMetadata(
                    source_file=source_file,
                    document_index=doc_index,
                    section_id=section.id,
                    section_title=section.title,
                    section_path=section.breadcrumb or section.get_full_path(),
                    chunk_index=i,
                    total_chunks_in_section=len(section.code_blocks),
                    start_char=0,
                    end_char=len(content),
                    chunk_type=ChunkType.CODE,
                    language=code_block.language,
                    has_code=True,
                    parent_section_id=section.parent_id,
                    level=section.level
                )
                
                # Create chunk
                chunk = Chunk(
                    chunk_id="",
                    content=content,
                    metadata=metadata
                )
                
                code_chunks.append(chunk)
        
        return code_chunks
    
    def _create_table_chunks(
        self,
        section: Section,
        source_file: str,
        doc_index: int,
        config: ChunkingConfig
    ) -> List[Chunk]:
        """Create chunks for tables"""
        table_chunks = []
        
        for i, table in enumerate(section.tables):
            # Create table content in markdown format
            content_parts = [f"[Table from Section {section.id}: {section.title}]"]
            
            if table.caption:
                content_parts.append(f"Caption: {table.caption}")
            
            # Add table in markdown format
            if table.headers:
                content_parts.append("| " + " | ".join(table.headers) + " |")
                content_parts.append("|" + "|".join([" --- " for _ in table.headers]) + "|")
            
            for row in table.rows[:20]:  # Limit to first 20 rows
                if table.headers:
                    # Ensure row has same number of columns
                    padded_row = row + [""] * (len(table.headers) - len(row))
                    padded_row = padded_row[:len(table.headers)]
                else:
                    padded_row = row
                
                content_parts.append("| " + " | ".join(str(cell) for cell in padded_row) + " |")
            
            if len(table.rows) > 20:
                content_parts.append(f"... and {len(table.rows) - 20} more rows")
            
            content = '\n'.join(content_parts)
            
            # Create metadata
            from src.data_processing.chunk_models import ChunkMetadata, ChunkType
            metadata = ChunkMetadata(
                source_file=source_file,
                document_index=doc_index,
                section_id=section.id,
                section_title=section.title,
                section_path=section.breadcrumb or section.get_full_path(),
                chunk_index=i,
                total_chunks_in_section=len(section.tables),
                start_char=0,
                end_char=len(content),
                chunk_type=ChunkType.TABLE,
                has_table=True,
                parent_section_id=section.parent_id,
                level=section.level
            )
            
            # Create chunk
            chunk = Chunk(
                chunk_id="",
                content=content,
                metadata=metadata
            )
            
            table_chunks.append(chunk)
        
        return table_chunks
    
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
        output_file = output_dir / f"{base_name}_chunks_fixed.json"
        
        # Prepare data for saving
        output_data = {
            'source_file': source_path.name,
            'chunking_config': {
                'chunker_type': self.chunker_type,
                'timestamp': datetime.now().isoformat(),
                'fixed_version': True,
                'includes_code_blocks': True,
                'includes_tables': True
            },
            'statistics': stats,
            'chunks': [chunk.to_dict() for chunk in chunks]
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_file}")
        
        return output_file


def run_fixed_chunking():
    """Run the fixed chunking pipeline"""
    from rich.console import Console
    
    console = Console()
    console.print("[bold cyan]🔧 Running Fixed Chunking Pipeline[/bold cyan]")
    
    # Find latest parsed file
    processed_dirs = list(settings.processed_data_path.glob("batch_*"))
    if not processed_dirs:
        console.print("[red]❌ No processed directories found![/red]")
        return
    
    latest_dir = max(processed_dirs, key=lambda p: p.stat().st_mtime)
    parsed_files = list(latest_dir.glob("*_parsed.json"))
    
    if not parsed_files:
        console.print(f"[red]❌ No parsed files found in {latest_dir}![/red]")
        return
    
    console.print(f"[green]📁 Found {len(parsed_files)} parsed files[/green]")
    
    # Process each file
    pipeline = FixedChunkingPipeline(chunker_type="advanced")
    
    for parsed_file in parsed_files:
        console.print(f"\n[yellow]🔄 Processing {parsed_file.name}...[/yellow]")
        
        try:
            stats = pipeline.process_parsed_file(parsed_file)
            
            console.print(f"[green]✅ Success![/green]")
            console.print(f"  📊 Chunks created: {stats['total_chunks']}")
            console.print(f"  💻 Code blocks: {stats['code_blocks_found']}")
            console.print(f"  📋 Tables: {stats['tables_found']}")
            console.print(f"  💾 Output: {Path(stats['output_file']).name}")
            
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")


if __name__ == "__main__":
    run_fixed_chunking()