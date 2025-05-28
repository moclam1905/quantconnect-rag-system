"""
Định nghĩa các chunking strategies cho QuantConnect documentation.
Mỗi strategy có approach riêng để chia content thành chunks.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import re

from src.data_processing.chunk_models import (
    Chunk, ChunkMetadata, ChunkType, ChunkingConfig, ChunkingStrategy
)
from src.data_processing.html_parser import Section, CodeBlock, TableData


class BaseChunkingStrategy(ABC):
    """
    Base class cho tất cả chunking strategies.
    Định nghĩa interface chung.
    """
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.strategy_type = ChunkingStrategy.FIXED_SIZE  # Override in subclasses
    
    @abstractmethod
    def chunk_section(self, section: Section, source_file: str, doc_index: int) -> List[Chunk]:
        """
        Chunk một section thành list of chunks.
        
        Args:
            section: Section object cần chunk
            source_file: Tên file gốc
            doc_index: Document index
            
        Returns:
            List of Chunk objects
        """
        pass
    
    def create_chunk_metadata(
        self, 
        section: Section, 
        source_file: str,
        doc_index: int,
        chunk_index: int,
        total_chunks: int,
        start_char: int,
        end_char: int,
        chunk_type: ChunkType,
        **kwargs
    ) -> ChunkMetadata:
        """Helper method để tạo chunk metadata"""
        return ChunkMetadata(
            source_file=source_file,
            document_index=doc_index,
            section_id=section.id,
            section_title=section.title,
            section_path=section.breadcrumb or section.get_full_path(),
            chunk_index=chunk_index,
            total_chunks_in_section=total_chunks,
            start_char=start_char,
            end_char=end_char,
            chunk_type=chunk_type,
            parent_section_id=section.parent_id,
            level=section.level,
            chunking_strategy=self.strategy_type,
            **kwargs
        )
    
    def add_section_header(self, content: str, section: Section) -> str:
        """Add section header to chunk content if configured"""
        if self.config.include_section_header_in_chunks:
            header = f"[Section {section.id}: {section.title}]\n\n"
            return header + content
        return content


class FixedSizeChunker(BaseChunkingStrategy):
    """
    Chunk theo kích thước cố định.
    Simple nhưng effective cho general text.
    """
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.strategy_type = ChunkingStrategy.FIXED_SIZE
    
    def chunk_section(self, section: Section, source_file: str, doc_index: int) -> List[Chunk]:
        chunks = []
        
        # Get full content
        full_content = section.content
        if not full_content:
            return chunks
        
        # Calculate chunk positions
        positions = self._calculate_chunk_positions(full_content)
        
        # Create chunks
        for i, (start, end) in enumerate(positions):
            chunk_content = full_content[start:end]
            
            # Add section header to first chunk
            if i == 0:
                chunk_content = self.add_section_header(chunk_content, section)
            
            # Create metadata
            metadata = self.create_chunk_metadata(
                section=section,
                source_file=source_file,
                doc_index=doc_index,
                chunk_index=i,
                total_chunks=len(positions),
                start_char=start,
                end_char=end,
                chunk_type=ChunkType.TEXT,
                has_code=bool(section.code_blocks),
                has_table=bool(section.tables)
            )
            
            # Create chunk
            chunk = Chunk(
                chunk_id="",  # Auto-generated
                content=chunk_content.strip(),
                metadata=metadata
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _calculate_chunk_positions(self, text: str) -> List[Tuple[int, int]]:
        """Calculate start and end positions for chunks"""
        positions = []
        text_length = len(text)
        
        start = 0
        while start < text_length:
            # Calculate end position
            end = min(start + self.config.max_chunk_size, text_length)
            
            # Try to find a good break point (sentence end)
            if end < text_length and self.config.ensure_complete_sentences:
                # Look for sentence end
                search_start = max(start + self.config.min_chunk_size, end - 100)
                sentence_end = self._find_sentence_end(text, search_start, end)
                if sentence_end != -1:
                    end = sentence_end
            
            positions.append((start, end))
            
            # Calculate next start with overlap
            start = end - self.config.chunk_overlap
            
            # Ensure we make progress
            if start <= positions[-1][0]:
                start = end
        
        return positions
    
    def _find_sentence_end(self, text: str, start: int, end: int) -> int:
        """Find the best sentence end position"""
        # Look for sentence endings
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        
        best_pos = -1
        for ending in sentence_endings:
            pos = text.rfind(ending, start, end)
            if pos != -1:
                # Position after the ending
                pos += len(ending) - 1
                if pos > best_pos:
                    best_pos = pos
        
        return best_pos


class SentenceBasedChunker(BaseChunkingStrategy):
    """
    Chunk theo câu.
    Đảm bảo mỗi chunk chứa complete sentences.
    """
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.strategy_type = ChunkingStrategy.SENTENCE_BASED
        self.sentence_pattern = re.compile(self.config.sentence_splitter_regex)
    
    def chunk_section(self, section: Section, source_file: str, doc_index: int) -> List[Chunk]:
        chunks = []
        
        if not section.content:
            return chunks
        
        # Split into sentences
        sentences = self._split_into_sentences(section.content)
        if not sentences:
            return chunks
        
        # Group sentences into chunks
        chunk_groups = self._group_sentences_into_chunks(sentences)
        
        # Create chunks
        current_pos = 0
        for i, sentence_group in enumerate(chunk_groups):
            chunk_content = ' '.join(sentence_group)
            
            # Add section header to first chunk
            if i == 0:
                chunk_content = self.add_section_header(chunk_content, section)
            
            # Calculate positions
            start_char = current_pos
            end_char = current_pos + len(chunk_content)
            
            # Create metadata
            metadata = self.create_chunk_metadata(
                section=section,
                source_file=source_file,
                doc_index=doc_index,
                chunk_index=i,
                total_chunks=len(chunk_groups),
                start_char=start_char,
                end_char=end_char,
                chunk_type=ChunkType.TEXT
            )
            
            # Create chunk
            chunk = Chunk(
                chunk_id="",
                content=chunk_content.strip(),
                metadata=metadata
            )
            
            chunks.append(chunk)
            current_pos = end_char
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Split by sentence endings
        parts = self.sentence_pattern.split(text)
        
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                # Reconstruct sentence with its ending
                sentence = parts[i] + parts[i + 1]
                sentences.append(sentence.strip())
        
        # Don't forget the last part if it exists
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())
        
        return sentences
    
    def _group_sentences_into_chunks(self, sentences: List[str]) -> List[List[str]]:
        """Group sentences into chunks of appropriate size"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Check if adding this sentence would exceed max size
            if current_size + sentence_size > self.config.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk)
                
                # Start new chunk with overlap
                if self.config.chunk_overlap > 0:
                    # Include last few sentences as overlap
                    overlap_size = 0
                    overlap_sentences = []
                    
                    for s in reversed(current_chunk):
                        overlap_size += len(s)
                        overlap_sentences.insert(0, s)
                        if overlap_size >= self.config.chunk_overlap:
                            break
                    
                    current_chunk = overlap_sentences
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


class ParagraphBasedChunker(BaseChunkingStrategy):
    """
    Chunk theo đoạn văn.
    Giữ nguyên cấu trúc paragraph.
    """
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.strategy_type = ChunkingStrategy.PARAGRAPH_BASED
    
    def chunk_section(self, section: Section, source_file: str, doc_index: int) -> List[Chunk]:
        chunks = []
        
        if not section.content:
            return chunks
        
        # Split into paragraphs
        paragraphs = section.content.split(self.config.paragraph_splitter)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return chunks
        
        # Group paragraphs into chunks
        chunk_groups = self._group_paragraphs_into_chunks(paragraphs)
        
        # Create chunks
        current_pos = 0
        for i, para_group in enumerate(chunk_groups):
            chunk_content = '\n\n'.join(para_group)
            
            # Add section header to first chunk
            if i == 0:
                chunk_content = self.add_section_header(chunk_content, section)
            
            # Calculate positions
            start_char = current_pos
            end_char = current_pos + len(chunk_content)
            
            # Create metadata
            metadata = self.create_chunk_metadata(
                section=section,
                source_file=source_file,
                doc_index=doc_index,
                chunk_index=i,
                total_chunks=len(chunk_groups),
                start_char=start_char,
                end_char=end_char,
                chunk_type=ChunkType.TEXT
            )
            
            # Create chunk
            chunk = Chunk(
                chunk_id="",
                content=chunk_content,
                metadata=metadata
            )
            
            chunks.append(chunk)
            current_pos = end_char
        
        return chunks
    
    def _group_paragraphs_into_chunks(self, paragraphs: List[str]) -> List[List[str]]:
        """Group paragraphs into chunks"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # Check if adding this paragraph would exceed max size
            if current_size + para_size > self.config.max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            
            # Add paragraph to current chunk
            current_chunk.append(para)
            current_size += para_size + len(self.config.paragraph_splitter)
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


class SectionBasedChunker(BaseChunkingStrategy):
    """
    Chunk dựa trên section structure.
    Respect section boundaries và hierarchy.
    """
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.strategy_type = ChunkingStrategy.SECTION_BASED
    
    def chunk_section(self, section: Section, source_file: str, doc_index: int) -> List[Chunk]:
        chunks = []
        
        # Check if section is small enough to be a single chunk
        total_size = len(section.content or "")
        
        # Add size for code blocks
        for code_block in section.code_blocks:
            total_size += len(code_block.content) + 100  # Extra for formatting
        
        # Add size for tables
        for table in section.tables:
            total_size += len(str(table.headers)) + sum(len(str(row)) for row in table.rows)
        
        # If small enough, create single chunk
        if total_size <= self.config.max_chunk_size:
            return self._create_single_chunk(section, source_file, doc_index)
        
        # Otherwise, create multiple chunks
        return self._create_multiple_chunks(section, source_file, doc_index)
    
    def _create_single_chunk(self, section: Section, source_file: str, doc_index: int) -> List[Chunk]:
        """Create a single chunk for the entire section"""
        # Combine all content
        content_parts = []
        
        # Add section header
        content_parts.append(f"[Section {section.id}: {section.title}]")
        
        # Add text content
        if section.content:
            content_parts.append(section.content)
        
        # Add code blocks
        for i, code_block in enumerate(section.code_blocks):
            content_parts.append(f"\n```{code_block.language}\n{code_block.content}\n```")
        
        # Add tables (simplified)
        for i, table in enumerate(section.tables):
            if table.caption:
                content_parts.append(f"\nTable: {table.caption}")
            content_parts.append(f"[Table with {len(table.rows)} rows]")
        
        # Create metadata
        metadata = self.create_chunk_metadata(
            section=section,
            source_file=source_file,
            doc_index=doc_index,
            chunk_index=0,
            total_chunks=1,
            start_char=0,
            end_char=len('\n\n'.join(content_parts)),
            chunk_type=ChunkType.MIXED if section.code_blocks or section.tables else ChunkType.TEXT,
            has_code=bool(section.code_blocks),
            has_table=bool(section.tables),
            language=section.code_blocks[0].language if section.code_blocks else None
        )
        
        # Create chunk
        chunk = Chunk(
            chunk_id="",
            content='\n\n'.join(content_parts),
            metadata=metadata
        )
        
        return [chunk]
    
    def _create_multiple_chunks(self, section: Section, source_file: str, doc_index: int) -> List[Chunk]:
        """Create multiple chunks for large section"""
        chunks = []
        
        # First, create a summary chunk if configured
        if self.config.create_section_summary_chunk:
            summary_chunk = self._create_summary_chunk(section, source_file, doc_index)
            chunks.append(summary_chunk)
        
        # Then chunk the content
        if section.content:
            # Use sentence-based chunking for the text content
            sentence_chunker = SentenceBasedChunker(self.config)
            text_chunks = sentence_chunker.chunk_section(section, source_file, doc_index)
            chunks.extend(text_chunks)
        
        # Add code blocks as separate chunks
        for i, code_block in enumerate(section.code_blocks):
            code_chunk = self._create_code_chunk(
                code_block, section, source_file, doc_index, 
                chunk_index=len(chunks)
            )
            chunks.append(code_chunk)
        
        # Add tables as separate chunks
        for i, table in enumerate(section.tables):
            table_chunk = self._create_table_chunk(
                table, section, source_file, doc_index,
                chunk_index=len(chunks)
            )
            chunks.append(table_chunk)
        
        # Update total chunks count
        for chunk in chunks:
            chunk.metadata.total_chunks_in_section = len(chunks)
        
        return chunks
    
    def _create_summary_chunk(self, section: Section, source_file: str, doc_index: int) -> Chunk:
        """Create a summary chunk for the section"""
        summary_parts = [
            f"[Section Summary: {section.id} - {section.title}]",
            f"Level: {section.level}",
            f"Path: {section.breadcrumb or section.get_full_path()}"
        ]
        
        if section.content:
            # Add first paragraph or first 200 chars as preview
            preview = section.content[:200] + "..." if len(section.content) > 200 else section.content
            summary_parts.append(f"\nPreview: {preview}")
        
        if section.code_blocks:
            summary_parts.append(f"\nContains {len(section.code_blocks)} code example(s)")
            languages = set(cb.language for cb in section.code_blocks)
            summary_parts.append(f"Languages: {', '.join(languages)}")
        
        if section.tables:
            summary_parts.append(f"\nContains {len(section.tables)} table(s)")
        
        # Create metadata
        metadata = self.create_chunk_metadata(
            section=section,
            source_file=source_file,
            doc_index=doc_index,
            chunk_index=0,
            total_chunks=1,  # Will be updated later
            start_char=0,
            end_char=len('\n'.join(summary_parts)),
            chunk_type=ChunkType.SECTION_HEADER
        )
        
        return Chunk(
            chunk_id="",
            content='\n'.join(summary_parts),
            metadata=metadata
        )
    
    def _create_code_chunk(
        self, 
        code_block: CodeBlock, 
        section: Section,
        source_file: str,
        doc_index: int,
        chunk_index: int
    ) -> Chunk:
        """Create chunk for code block"""
        content = f"[Code Example from Section {section.id}: {section.title}]\n\n"
        content += f"```{code_block.language}\n{code_block.content}\n```"
        
        metadata = self.create_chunk_metadata(
            section=section,
            source_file=source_file,
            doc_index=doc_index,
            chunk_index=chunk_index,
            total_chunks=1,  # Will be updated
            start_char=0,
            end_char=len(content),
            chunk_type=ChunkType.CODE,
            language=code_block.language,
            has_code=True
        )
        
        return Chunk(
            chunk_id="",
            content=content,
            metadata=metadata
        )
    
    def _create_table_chunk(
        self,
        table: TableData,
        section: Section,
        source_file: str,
        doc_index: int,
        chunk_index: int
    ) -> Chunk:
        """Create chunk for table"""
        content_parts = [f"[Table from Section {section.id}: {section.title}]"]
        
        if table.caption:
            content_parts.append(f"Caption: {table.caption}")
        
        # Simple table representation
        if table.headers:
            content_parts.append(f"Headers: {' | '.join(table.headers)}")
        
        content_parts.append(f"Data: {len(table.rows)} rows")
        
        # Include first few rows as sample
        for i, row in enumerate(table.rows[:5]):
            content_parts.append(f"Row {i+1}: {' | '.join(str(cell) for cell in row)}")
        
        if len(table.rows) > 5:
            content_parts.append(f"... and {len(table.rows) - 5} more rows")
        
        metadata = self.create_chunk_metadata(
            section=section,
            source_file=source_file,
            doc_index=doc_index,
            chunk_index=chunk_index,
            total_chunks=1,  # Will be updated
            start_char=0,
            end_char=len('\n'.join(content_parts)),
            chunk_type=ChunkType.TABLE,
            has_table=True
        )
        
        return Chunk(
            chunk_id="",
            content='\n'.join(content_parts),
            metadata=metadata
        )


# Strategy factory
def get_chunking_strategy(strategy: ChunkingStrategy, config: ChunkingConfig) -> BaseChunkingStrategy:
    """Factory function to get appropriate chunking strategy"""
    strategy_map = {
        ChunkingStrategy.FIXED_SIZE: FixedSizeChunker,
        ChunkingStrategy.SENTENCE_BASED: SentenceBasedChunker,
        ChunkingStrategy.PARAGRAPH_BASED: ParagraphBasedChunker,
        ChunkingStrategy.SECTION_BASED: SectionBasedChunker,
    }
    
    strategy_class = strategy_map.get(strategy)
    if not strategy_class:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    return strategy_class(config)


# Example usage
if __name__ == "__main__":
    from src.data_processing.chunk_models import ChunkingPresets
    
    # Create sample section
    section = Section(
        id="1.2.3",
        title="Sample Section",
        level=3,
        content="This is a sample section with some content. It has multiple sentences. Each sentence should be preserved.",
        section_number="1.2.3",
        breadcrumb="Getting Started > Basics > Sample Section"
    )
    
    # Test different strategies
    config = ChunkingPresets.for_qa()
    
    # Test fixed size chunker
    fixed_chunker = FixedSizeChunker(config)
    chunks = fixed_chunker.chunk_section(section, "test.html", 1)
    print(f"Fixed size chunker produced {len(chunks)} chunks")
    
    # Test sentence based chunker
    sentence_chunker = SentenceBasedChunker(config)
    chunks = sentence_chunker.chunk_section(section, "test.html", 1)
    print(f"Sentence based chunker produced {len(chunks)} chunks")