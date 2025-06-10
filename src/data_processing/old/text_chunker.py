"""
Basic text chunker implementation.
Chia text thành chunks với size và overlap có thể config.
"""

import re
from typing import List, Tuple, Dict
import nltk
from pathlib import Path

# Import models và config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data_processing.old.chunk_models import Chunk, ChunkMetadata, ChunkType, ChunkingConfig
from src.data_processing.old.html_parser import Section
from src.utils.logger import logger

# Try to download punkt tokenizer for sentence splitting
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')


class TextChunker:
    """
    Basic text chunker với các features:
    - Chia theo size với overlap
    - Respect word boundaries
    - Respect sentence boundaries (optional)
    - Handle edge cases
    """
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        
        # Compile regex patterns
        self.word_boundary_pattern = re.compile(r'\b')
        self.sentence_end_pattern = re.compile(self.config.sentence_splitter_regex)
        
    def chunk_text(
        self,
        text: str,
        source_file: str,
        section: Section,
        doc_index: int = 1
    ) -> List[Chunk]:
        """
        Chia text thành chunks.
        
        Args:
            text: Text content cần chunk
            source_file: Source file name
            section: Section object chứa text
            doc_index: Document index
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip() or len(text.strip()) < 50:
            logger.warning(f"Section {section.id} has insufficient content ({len(text or '')} chars)")
            return []  # Return empty list instead of trying to chunk
        
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Calculate chunk boundaries
        boundaries = self._calculate_chunk_boundaries(text)
        
        # Create chunks
        chunks = []
        for i, (start, end) in enumerate(boundaries):
            chunk_text = text[start:end].strip()
            
            if not chunk_text:
                continue
            
            # Add section header to first chunk if configured
            if i == 0 and self.config.include_section_header_in_chunks:
                chunk_text = f"[Section {section.id}: {section.title}]\n\n{chunk_text}"
            
            # Create metadata
            metadata = ChunkMetadata(
                source_file=source_file,
                document_index=doc_index,
                section_id=section.id,
                section_title=section.title,
                section_path=section.breadcrumb or section.get_full_path(),
                chunk_index=i,
                total_chunks_in_section=len(boundaries),
                start_char=start,
                end_char=end,
                chunk_type=ChunkType.TEXT,
                parent_section_id=section.parent_id,
                level=section.level,
                overlap_with_previous=self._calculate_overlap(i, boundaries, 'previous'),
                overlap_with_next=self._calculate_overlap(i, boundaries, 'next')
            )
            
            # Create chunk
            chunk = Chunk(
                chunk_id="",  # Auto-generated
                content=chunk_text,
                metadata=metadata
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text trước khi chunking"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure text ends with proper punctuation for sentence detection
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text.strip()
    
    def _calculate_chunk_boundaries(self, text: str) -> List[Tuple[int, int]]:
        """
        Calculate start và end positions cho mỗi chunk.
        Respect word và sentence boundaries.
        """
        boundaries = []
        text_length = len(text)
        
        start = 0
        while start < text_length:
            # Calculate ideal end position
            ideal_end = min(start + self.config.max_chunk_size, text_length)
            
            # Find actual end position respecting boundaries
            if ideal_end < text_length:
                end = self._find_best_break_point(text, start, ideal_end)
            else:
                end = text_length
            
            # Ensure minimum chunk size (except for last chunk)
            if end - start < self.config.min_chunk_size and end < text_length:
                # Try to extend to minimum size
                extended_end = min(start + self.config.min_chunk_size, text_length)
                end = self._find_best_break_point(text, start, extended_end)
            
            boundaries.append((start, end))
            
            # Calculate next start with overlap
            if self.config.chunk_overlap > 0 and end < text_length:
                # Find overlap start point
                overlap_start = max(end - self.config.chunk_overlap, start)
                
                # Adjust to sentence boundary if possible
                if self.config.ensure_complete_sentences:
                    overlap_start = self._find_sentence_start(text, overlap_start, end)
                else:
                    # At least adjust to word boundary
                    overlap_start = self._find_word_start(text, overlap_start, end)
                
                start = overlap_start
            else:
                start = end
        
        return boundaries
    
    def _find_best_break_point(self, text: str, start: int, ideal_end: int) -> int:
        """
        Find the best position to break text.
        Ưu tiên: sentence end > paragraph end > word boundary
        """
        # Try to find sentence end first
        if self.config.ensure_complete_sentences:
            sentence_end = self._find_sentence_end(text, start, ideal_end)
            if sentence_end != -1:
                return sentence_end
        
        # Try to find paragraph end
        paragraph_end = text.rfind('\n\n', start, ideal_end)
        if paragraph_end != -1:
            return paragraph_end + 2  # After the double newline
        
        # Try to find single newline
        newline_pos = text.rfind('\n', start, ideal_end)
        if newline_pos != -1:
            return newline_pos + 1
        
        # Find word boundary
        return self._find_word_boundary(text, start, ideal_end)
    
    def _find_sentence_end(self, text: str, start: int, end: int) -> int:
        """Find the position after a sentence end"""
        # Use NLTK for better sentence detection
        try:
            sentences = nltk.sent_tokenize(text[start:end])
            if len(sentences) > 1:
                # Find position of last complete sentence
                last_complete = ''.join(sentences[:-1])
                return start + len(last_complete)
        except:
            pass
        
        # Fallback to regex
        search_text = text[start:end]
        matches = list(self.sentence_end_pattern.finditer(search_text))
        
        if matches:
            # Get the last match
            last_match = matches[-1]
            return start + last_match.end()
        
        return -1
    
    def _find_word_boundary(self, text: str, start: int, end: int) -> int:
        """Find the nearest word boundary before end"""
        # Don't break in middle of word
        if end >= len(text):
            return len(text)
        
        # If we're already at a word boundary, use it
        if text[end-1].isspace() or not text[end-1].isalnum():
            return end
        
        # Find the last space before end
        space_pos = text.rfind(' ', start, end)
        if space_pos != -1:
            return space_pos + 1
        
        # No good break point found, just use the position
        return end
    
    def _find_sentence_start(self, text: str, start: int, end: int) -> int:
        """Find the start of a sentence for overlap"""
        # Use NLTK to find sentence boundaries
        try:
            sentences = nltk.sent_tokenize(text[start:end])
            if sentences:
                # Start from the beginning of the first sentence
                return start
        except:
            pass
        
        # Fallback: look for sentence start patterns
        sentence_start_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        match = sentence_start_pattern.search(text, start, end)
        
        if match:
            return match.end()
        
        return start
    
    def _find_word_start(self, text: str, start: int, end: int) -> int:
        """Find the start of a word for overlap"""
        # If we're at a space, move forward to next word
        while start < end and text[start].isspace():
            start += 1
        
        return start
    
    def _calculate_overlap(
        self, 
        chunk_index: int, 
        boundaries: List[Tuple[int, int]], 
        direction: str
    ) -> int:
        """Calculate overlap size with previous or next chunk"""
        if direction == 'previous' and chunk_index > 0:
            prev_end = boundaries[chunk_index - 1][1]
            curr_start = boundaries[chunk_index][0]
            return max(0, prev_end - curr_start)
        
        elif direction == 'next' and chunk_index < len(boundaries) - 1:
            curr_end = boundaries[chunk_index][1]
            next_start = boundaries[chunk_index + 1][0]
            return max(0, curr_end - next_start)
        
        return 0


class AdvancedTextChunker(TextChunker):
    """
    Advanced text chunker với thêm features:
    - Smart paragraph grouping
    - Heading detection và preservation
    - Better overlap calculation
    """
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        
        # Pattern để detect headings
        self.heading_pattern = re.compile(r'^#+\s+.+$', re.MULTILINE)
        self.numbered_heading_pattern = re.compile(r'^\d+\.[\d\.]*\s+.+$', re.MULTILINE)
    
    def chunk_text(
        self,
        text: str,
        source_file: str,
        section: Section,
        doc_index: int = 1
    ) -> List[Chunk]:
        """
        Advanced chunking với smart paragraph grouping.
        """
        if not text or not text.strip():
            return []
        
        # Split into paragraphs first
        paragraphs = self._split_into_paragraphs(text)
        
        # Group paragraphs into chunks
        paragraph_groups = self._group_paragraphs(paragraphs)
        
        # Create chunks from paragraph groups
        chunks = []
        char_offset = 0
        
        for i, para_group in enumerate(paragraph_groups):
            chunk_text = '\n\n'.join(para_group['paragraphs'])
            
            # Add section header to first chunk
            if i == 0 and self.config.include_section_header_in_chunks:
                chunk_text = f"[Section {section.id}: {section.title}]\n\n{chunk_text}"
            
            # Calculate positions
            start_char = char_offset
            end_char = char_offset + len(chunk_text)
            
            # Create metadata
            metadata = ChunkMetadata(
                source_file=source_file,
                document_index=doc_index,
                section_id=section.id,
                section_title=section.title,
                section_path=section.breadcrumb or section.get_full_path(),
                chunk_index=i,
                total_chunks_in_section=len(paragraph_groups),
                start_char=start_char,
                end_char=end_char,
                chunk_type=ChunkType.TEXT,
                parent_section_id=section.parent_id,
                level=section.level
            )
            
            # Create chunk
            chunk = Chunk(
                chunk_id="",
                content=chunk_text,
                metadata=metadata
            )
            
            chunks.append(chunk)
            char_offset = end_char + 2  # Account for paragraph separator
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[Dict]:
        """
        Split text into paragraphs với metadata.
        """
        # Split by double newline
        raw_paragraphs = text.split('\n\n')
        
        paragraphs = []
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if this is a heading
            is_heading = bool(
                self.heading_pattern.match(para) or 
                self.numbered_heading_pattern.match(para)
            )
            
            paragraphs.append({
                'text': para,
                'length': len(para),
                'is_heading': is_heading,
                'sentences': len(nltk.sent_tokenize(para)) if not is_heading else 1
            })
        
        return paragraphs
    
    def _group_paragraphs(self, paragraphs: List[Dict]) -> List[Dict]:
        """
        Group paragraphs into chunks một cách thông minh.
        """
        groups = []
        current_group = {
            'paragraphs': [],
            'total_length': 0,
            'has_heading': False
        }
        
        for para in paragraphs:
            # Check if adding this paragraph would exceed max size
            if (current_group['total_length'] + para['length'] > self.config.max_chunk_size 
                and current_group['paragraphs']):
                
                # Save current group
                groups.append(current_group)
                
                # Start new group
                current_group = {
                    'paragraphs': [],
                    'total_length': 0,
                    'has_heading': False
                }
                
                # Add overlap if configured
                if self.config.chunk_overlap > 0 and groups:
                    # Add last paragraph from previous group as overlap
                    overlap_para = groups[-1]['paragraphs'][-1]
                    current_group['paragraphs'].append(overlap_para)
                    current_group['total_length'] += len(overlap_para)
            
            # Add paragraph to current group
            current_group['paragraphs'].append(para['text'])
            current_group['total_length'] += para['length'] + 2  # +2 for \n\n
            if para['is_heading']:
                current_group['has_heading'] = True
        
        # Don't forget the last group
        if current_group['paragraphs']:
            groups.append(current_group)
        
        return groups


# Example usage and testing
if __name__ == "__main__":
    from src.data_processing.old.chunk_models import ChunkingPresets
    
    # Create sample section
    sample_text = """
    This is the introduction to our algorithm. It explains the basic concepts and provides an overview of what we'll be building.
    
    In this tutorial, we'll create a simple moving average crossover strategy. The strategy will buy when the fast moving average crosses above the slow moving average.
    
    Here are the key components:
    - Data handling and preprocessing
    - Signal generation based on moving averages
    - Order execution and position management
    - Performance tracking and analysis
    
    Let's start by importing the necessary libraries and setting up our algorithm class. This will form the foundation of our trading strategy.
    
    The next section will cover the implementation details step by step.
    """
    
    section = Section(
        id="1.2",
        title="Introduction to Algorithm Development",
        level=2,
        content=sample_text,
        section_number="1.2",
        breadcrumb="Getting Started > Introduction"
    )
    
    # Test basic chunker
    config = ChunkingPresets.for_qa()
    config.max_chunk_size = 200  # Small size for testing
    config.chunk_overlap = 50
    
    basic_chunker = TextChunker(config)
    chunks = basic_chunker.chunk_text(
        sample_text,
        "test.html",
        section,
        doc_index=1
    )
    
    print(f"Basic Chunker produced {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Size: {chunk.char_count} chars")
        print(f"  Content: {chunk.content[:100]}...")
        print(f"  Overlap with next: {chunk.metadata.overlap_with_next}")
    
    # Test advanced chunker
    print("\n" + "="*50 + "\n")
    
    advanced_chunker = AdvancedTextChunker(config)
    chunks = advanced_chunker.chunk_text(
        sample_text,
        "test.html",
        section,
        doc_index=1
    )
    
    print(f"Advanced Chunker produced {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Size: {chunk.char_count} chars")
        print(f"  Content preview: {chunk.content[:100]}...")