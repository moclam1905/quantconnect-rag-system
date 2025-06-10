"""
Data models và structures cho chunking system.
Định nghĩa các loại chunks và metadata cần thiết cho RAG.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from datetime import datetime
import hashlib
from enum import Enum


class ChunkType(Enum):
    """Các loại chunk khác nhau"""
    TEXT = "text"              # Plain text content
    CODE = "code"              # Code block
    TABLE = "table"            # Table data
    MIXED = "mixed"            # Mixed content (text + code/table)
    SECTION_HEADER = "header"  # Section header/title


class ChunkingStrategy(Enum):
    """Các chiến lược chunking khác nhau"""
    FIXED_SIZE = "fixed_size"          # Chunk by character/token count
    SENTENCE_BASED = "sentence_based"  # Chunk by sentences
    PARAGRAPH_BASED = "paragraph_based" # Chunk by paragraphs
    SECTION_BASED = "section_based"    # Chunk by document sections
    SEMANTIC = "semantic"              # Chunk by semantic similarity
    CODE_AWARE = "code_aware"          # Special handling for code
    HYBRID = "hybrid"                  # Combination of strategies


@dataclass
class ChunkMetadata:
    """
    Metadata cho mỗi chunk.
    Chứa thông tin context và traceability.
    """
    # Source information
    source_file: str                    # Original HTML file
    document_index: int                 # Document index trong file (thường là 1)
    section_id: str                     # Section ID (e.g., "1.2.3")
    section_title: str                  # Section title
    section_path: str                   # Full section path/breadcrumb
    
    # Position information
    chunk_index: int                    # Index của chunk trong section
    total_chunks_in_section: int        # Tổng số chunks trong section
    start_char: int                     # Starting character position in section
    end_char: int                       # Ending character position in section
    
    # Content information
    chunk_type: ChunkType              # Loại chunk
    language: Optional[str] = None     # Programming language (for code chunks)
    has_code: bool = False             # Chunk có chứa code không
    has_table: bool = False            # Chunk có chứa table không
    
    # Hierarchy information
    parent_section_id: Optional[str] = None  # Parent section ID
    level: int = 1                           # Section level (1, 2, 3...)
    
    # Chunking information
    chunking_strategy: Optional[ChunkingStrategy] = None
    overlap_with_previous: int = 0     # Number of overlapping tokens/chars
    overlap_with_next: int = 0         # Number of overlapping tokens/chars
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'source_file': self.source_file,
            'document_index': self.document_index,
            'section_id': self.section_id,
            'section_title': self.section_title,
            'section_path': self.section_path,
            'chunk_index': self.chunk_index,
            'total_chunks_in_section': self.total_chunks_in_section,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'chunk_type': self.chunk_type.value,
            'language': self.language,
            'has_code': self.has_code,
            'has_table': self.has_table,
            'parent_section_id': self.parent_section_id,
            'level': self.level,
            'chunking_strategy': self.chunking_strategy.value if self.chunking_strategy else None,
            'overlap_with_previous': self.overlap_with_previous,
            'overlap_with_next': self.overlap_with_next,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class Chunk:
    """
    Đại diện cho một chunk của document.
    Đây là unit cơ bản sẽ được embed và lưu vào vector database.
    """
    # Unique identifier
    chunk_id: str                      # Unique ID cho chunk
    
    # Content
    content: str                       # Actual text content của chunk
    
    # Metadata
    metadata: ChunkMetadata
    
    # Size information
    char_count: int = field(init=False)
    word_count: int = field(init=False)
    
    # Embedding placeholder (sẽ được fill sau)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Calculate size metrics"""
        self.char_count = len(self.content)
        self.word_count = len(self.content.split())
        
        # Generate chunk_id if not provided
        if not self.chunk_id:
            self.chunk_id = self._generate_chunk_id()
    
    def _generate_chunk_id(self) -> str:
        """Generate unique chunk ID based on content and metadata"""
        # Combine key information for uniqueness
        unique_string = f"{self.metadata.source_file}_{self.metadata.section_id}_{self.metadata.chunk_index}_{self.content[:50]}"
        
        # Create hash
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'metadata': self.metadata.to_dict(),
            'char_count': self.char_count,
            'word_count': self.word_count,
            'embedding': self.embedding
        }
    
    def get_context_string(self) -> str:
        """
        Generate context string for this chunk.
        Used for adding context when querying.
        """
        context_parts = [
            f"Source: {self.metadata.source_file}",
            f"Section: {self.metadata.section_path}",
        ]
        
        if self.metadata.language:
            context_parts.append(f"Language: {self.metadata.language}")
        
        return " | ".join(context_parts)
    
    def __repr__(self) -> str:
        return (f"Chunk(id={self.chunk_id[:8]}..., "
                f"type={self.metadata.chunk_type.value}, "
                f"size={self.char_count} chars)")


@dataclass
class ChunkingConfig:
    """
    Configuration cho chunking process.
    Có thể customize cho từng use case.
    """
    # Size limits (in characters)
    max_chunk_size: int = 1000         # Maximum size của một chunk
    min_chunk_size: int = 100          # Minimum size của một chunk
    chunk_overlap: int = 200           # Overlap between consecutive chunks
    
    # Size limits (in tokens) - optional, for token-based chunking
    max_chunk_tokens: Optional[int] = None
    chunk_overlap_tokens: Optional[int] = None
    
    # Strategy settings
    default_strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    
    # Code handling
    preserve_small_code_blocks: bool = True      # Keep small code blocks intact
    max_code_block_size: int = 1500              # Max size for intact code blocks
    code_chunk_by_function: bool = True          # Try to chunk code by functions/classes
    
    # Section handling  
    respect_section_boundaries: bool = True       # Don't mix content from different sections
    include_section_header_in_chunks: bool = True # Include section title in each chunk
    create_section_summary_chunk: bool = True     # Create a summary chunk for large sections
    
    # Table handling
    preserve_small_tables: bool = True            # Keep small tables intact
    max_table_rows_intact: int = 20              # Max rows to keep table intact
    
    # Text processing
    sentence_splitter_regex: str = r'[.!?]\s+'   # Regex for sentence splitting
    paragraph_splitter: str = '\n\n'             # Paragraph delimiter
    
    # Quality settings
    ensure_complete_sentences: bool = True        # Don't cut in middle of sentences
    ensure_complete_code_blocks: bool = True      # Don't cut in middle of code blocks
    
    # Metadata settings
    include_chunk_position_meta: bool = True      # Include position info in metadata
    include_hierarchy_meta: bool = True           # Include section hierarchy in metadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'max_chunk_size': self.max_chunk_size,
            'min_chunk_size': self.min_chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_chunk_tokens': self.max_chunk_tokens,
            'chunk_overlap_tokens': self.chunk_overlap_tokens,
            'default_strategy': self.default_strategy.value,
            'preserve_small_code_blocks': self.preserve_small_code_blocks,
            'max_code_block_size': self.max_code_block_size,
            'code_chunk_by_function': self.code_chunk_by_function,
            'respect_section_boundaries': self.respect_section_boundaries,
            'include_section_header_in_chunks': self.include_section_header_in_chunks,
            'create_section_summary_chunk': self.create_section_summary_chunk,
            'preserve_small_tables': self.preserve_small_tables,
            'max_table_rows_intact': self.max_table_rows_intact,
            'sentence_splitter_regex': self.sentence_splitter_regex,
            'paragraph_splitter': self.paragraph_splitter,
            'ensure_complete_sentences': self.ensure_complete_sentences,
            'ensure_complete_code_blocks': self.ensure_complete_code_blocks,
            'include_chunk_position_meta': self.include_chunk_position_meta,
            'include_hierarchy_meta': self.include_hierarchy_meta
        }


# Preset configurations cho different use cases
class ChunkingPresets:
    """Các preset configuration cho chunking"""
    
    @staticmethod
    def for_embedding() -> ChunkingConfig:
        """Config tối ưu cho embedding models (smaller chunks)"""
        return ChunkingConfig(
            max_chunk_size=500,
            min_chunk_size=50,
            chunk_overlap=50,
            max_chunk_tokens=256,  # Typical embedding model limit
            default_strategy=ChunkingStrategy.SENTENCE_BASED
        )
    
    @staticmethod
    def for_qa() -> ChunkingConfig:
        """Config tối ưu cho Q&A (medium chunks với good context)"""
        return ChunkingConfig(
            max_chunk_size=1000,
            min_chunk_size=100,
            chunk_overlap=200,
            default_strategy=ChunkingStrategy.HYBRID,
            include_section_header_in_chunks=True
        )
    
    @staticmethod
    def for_code_understanding() -> ChunkingConfig:
        """Config tối ưu cho code understanding"""
        return ChunkingConfig(
            max_chunk_size=1500,
            min_chunk_size=200,
            chunk_overlap=100,
            preserve_small_code_blocks=True,
            max_code_block_size=2000,
            code_chunk_by_function=True,
            default_strategy=ChunkingStrategy.CODE_AWARE
        )
    
    @staticmethod
    def for_documentation() -> ChunkingConfig:
        """Config cho general documentation (QuantConnect use case)"""
        return ChunkingConfig(
            max_chunk_size=1200,
            min_chunk_size=100,
            chunk_overlap=150,
            default_strategy=ChunkingStrategy.HYBRID,
            preserve_small_code_blocks=True,
            max_code_block_size=1500,
            respect_section_boundaries=True,
            include_section_header_in_chunks=True,
            create_section_summary_chunk=True
        )


# Example usage
if __name__ == "__main__":
    # Create sample metadata
    metadata = ChunkMetadata(
        source_file="Quantconnect-Writing-Algorithms.html",
        document_index=1,
        section_id="1.2.3",
        section_title="Creating Your First Algorithm",
        section_path="Getting Started > Writing Algorithms > Creating Your First Algorithm",
        chunk_index=0,
        total_chunks_in_section=5,
        start_char=0,
        end_char=500,
        chunk_type=ChunkType.MIXED,
        has_code=True,
        language="python"
    )
    
    # Create sample chunk
    chunk = Chunk(
        chunk_id="",  # Will be auto-generated
        content="This is a sample chunk content with some Python code examples...",
        metadata=metadata
    )
    
    print(f"Created chunk: {chunk}")
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Context: {chunk.get_context_string()}")
    
    # Test configurations
    qa_config = ChunkingPresets.for_qa()
    print(f"\nQ&A Config: max_size={qa_config.max_chunk_size}, overlap={qa_config.chunk_overlap}")