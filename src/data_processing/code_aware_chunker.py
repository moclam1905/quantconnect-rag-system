"""
Code-Aware Chunker cho QuantConnect documentation.
Xử lý mixed content (text + code) một cách thông minh.
Giữ nguyên code blocks nhỏ, split code blocks lớn theo logical boundaries.
"""

import re
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing.chunk_models import (
    Chunk, ChunkMetadata, ChunkType, ChunkingConfig, ChunkingStrategy
)
from src.data_processing.html_parser import Section, CodeBlock
from src.data_processing.chunking_strategies import BaseChunkingStrategy
from src.utils.logger import logger


@dataclass
class CodeSegment:
    """Đại diện cho một đoạn code được identify từ text"""
    content: str
    language: str
    start_pos: int
    end_pos: int
    is_inline: bool = False  # True nếu là inline code, False nếu là code block
    logical_units: List[str] = None  # Functions, classes, etc. trong code này


class CodeAnalyzer(ABC):
    """Base class cho các code analyzers"""
    
    @abstractmethod
    def detect_logical_units(self, code: str) -> List[Tuple[str, int, int]]:
        """
        Detect logical units trong code (functions, classes, etc.)
        Returns: List of (unit_name, start_pos, end_pos)
        """
        pass
    
    @abstractmethod
    def find_good_split_points(self, code: str, max_size: int) -> List[int]:
        """
        Find good positions to split code nếu quá lớn
        Returns: List of character positions
        """
        pass


class PythonCodeAnalyzer(CodeAnalyzer):
    """Analyzer cho Python code"""
    
    def __init__(self):
        # Patterns để detect Python constructs
        self.function_pattern = re.compile(r'^(\s*)def\s+(\w+)\s*\(', re.MULTILINE)
        self.class_pattern = re.compile(r'^(\s*)class\s+(\w+)\s*[:\(]', re.MULTILINE)
        self.method_pattern = re.compile(r'^(\s+)def\s+(\w+)\s*\(', re.MULTILINE)
        self.import_pattern = re.compile(r'^(from\s+\S+\s+)?import\s+.*$', re.MULTILINE)
        self.comment_pattern = re.compile(r'^\s*#.*$', re.MULTILINE)
        self.decorator_pattern = re.compile(r'^\s*@\w+.*$', re.MULTILINE)
    
    def detect_logical_units(self, code: str) -> List[Tuple[str, int, int]]:
        """Detect functions và classes trong Python code"""
        units = []
        lines = code.split('\n')
        
        # Find classes
        for match in self.class_pattern.finditer(code):
            class_name = match.group(2)
            start_line = code[:match.start()].count('\n')
            indent = len(match.group(1))
            
            # Find end of class by looking for next item at same or lower indent
            end_line = len(lines)
            for i in range(start_line + 1, len(lines)):
                line = lines[i]
                if line.strip() and not line.startswith(' ' * (indent + 1)):
                    if not line.startswith(' ' * indent) or line.startswith(' ' * indent + 'class') or line.startswith(' ' * indent + 'def'):
                        end_line = i
                        break
            
            start_pos = match.start()
            end_pos = len('\n'.join(lines[:end_line]))
            units.append((f"class {class_name}", start_pos, end_pos))
        
        # Find standalone functions (not inside classes)
        for match in self.function_pattern.finditer(code):
            func_name = match.group(2)
            start_line = code[:match.start()].count('\n')
            indent = len(match.group(1))
            
            # Skip if this is a method inside a class (indented)
            if indent > 0:
                continue
            
            # Find end of function
            end_line = len(lines)
            for i in range(start_line + 1, len(lines)):
                line = lines[i]
                if line.strip() and not line.startswith(' '):
                    end_line = i
                    break
            
            start_pos = match.start()
            end_pos = len('\n'.join(lines[:end_line]))
            units.append((f"def {func_name}", start_pos, end_pos))
        
        return units
    
    def find_good_split_points(self, code: str, max_size: int) -> List[int]:
        """Find good split points trong Python code"""
        split_points = []
        lines = code.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            current_pos += len(line) + 1  # +1 for newline
            
            # If we're approaching max size, look for a good break
            if current_pos >= max_size:
                # Prefer to break at:
                # 1. End of function/class
                # 2. Import statements
                # 3. Comments
                # 4. Empty lines
                
                if (line.strip() == '' or 
                    self.comment_pattern.match(line) or
                    self.import_pattern.match(line)):
                    split_points.append(current_pos)
                elif i < len(lines) - 1:
                    next_line = lines[i + 1]
                    if (self.function_pattern.match(next_line) or 
                        self.class_pattern.match(next_line)):
                        split_points.append(current_pos)
        
        return split_points


class CSharpCodeAnalyzer(CodeAnalyzer):
    """Analyzer cho C# code"""
    
    def __init__(self):
        self.class_pattern = re.compile(r'^(\s*)(?:public|private|internal|protected)?\s*(?:static|abstract|sealed)?\s*class\s+(\w+)', re.MULTILINE)
        self.method_pattern = re.compile(r'^(\s*)(?:public|private|internal|protected)?\s*(?:static|virtual|override|abstract)?\s*\w+\s+(\w+)\s*\(', re.MULTILINE)
        self.property_pattern = re.compile(r'^(\s*)(?:public|private|internal|protected)?\s*\w+\s+(\w+)\s*\{', re.MULTILINE)
        self.namespace_pattern = re.compile(r'^namespace\s+(\w+(?:\.\w+)*)', re.MULTILINE)
        self.using_pattern = re.compile(r'^using\s+.*$', re.MULTILINE)
    
    def detect_logical_units(self, code: str) -> List[Tuple[str, int, int]]:
        """Detect classes, methods trong C# code"""
        units = []
        
        # Find namespaces
        for match in self.namespace_pattern.finditer(code):
            namespace_name = match.group(1)
            start_pos = match.start()
            # Find matching closing brace (simplified)
            brace_count = 0
            pos = match.end()
            while pos < len(code):
                if code[pos] == '{':
                    brace_count += 1
                elif code[pos] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        break
                pos += 1
            
            end_pos = pos + 1 if pos < len(code) else len(code)
            units.append((f"namespace {namespace_name}", start_pos, end_pos))
        
        # Find classes
        for match in self.class_pattern.finditer(code):
            class_name = match.group(2)
            start_pos = match.start()
            # Simple approach: find matching braces
            brace_count = 0
            pos = code.find('{', match.end())
            if pos == -1:
                continue
            
            brace_count = 1
            pos += 1
            while pos < len(code) and brace_count > 0:
                if code[pos] == '{':
                    brace_count += 1
                elif code[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            end_pos = pos
            units.append((f"class {class_name}", start_pos, end_pos))
        
        return units
    
    def find_good_split_points(self, code: str, max_size: int) -> List[int]:
        """Find good split points trong C# code"""
        split_points = []
        lines = code.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            current_pos += len(line) + 1
            
            if current_pos >= max_size:
                # Good places to split in C#:
                if (line.strip() == '' or
                    line.strip() == '}' or
                    self.using_pattern.match(line) or
                    line.strip().startswith('//')):
                    split_points.append(current_pos)
        
        return split_points


class CodeAwareChunker(BaseChunkingStrategy):
    """
    Chunker chuyên biệt cho content có code.
    
    Strategies:
    1. Keep small code blocks intact
    2. Split large code blocks intelligently
    3. Maintain context between text and code
    4. Create separate chunks for complex code examples
    """
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.strategy_type = ChunkingStrategy.CODE_AWARE
        
        # Initialize code analyzers
        self.analyzers = {
            'python': PythonCodeAnalyzer(),
            'csharp': CSharpCodeAnalyzer()
        }
        
        # Patterns để detect code trong text
        self.inline_code_pattern = re.compile(r'`([^`]+)`')
        self.code_block_pattern = re.compile(r'```(\w+)?\n(.*?)\n```', re.DOTALL)
    
    def chunk_section(self, section: Section, source_file: str, doc_index: int) -> List[Chunk]:
        """
        Chunk section với code-aware strategy.
        """
        chunks = []
        
        # Analyze section content
        content_segments = self._analyze_mixed_content(section)
        
        if not content_segments:
            return chunks
        
        # Group segments into chunks
        chunk_groups = self._group_segments_into_chunks(content_segments)
        
        # Create chunks
        for i, group in enumerate(chunk_groups):
            chunk_content = self._build_chunk_content(group, section)
            
            # Determine chunk type
            chunk_type = self._determine_chunk_type(group)
            
            # Get primary language if this is a code-heavy chunk
            primary_language = self._get_primary_language(group)
            
            # Create metadata
            metadata = self.create_chunk_metadata(
                section=section,
                source_file=source_file,
                doc_index=doc_index,
                chunk_index=i,
                total_chunks=len(chunk_groups),
                start_char=group['start_pos'],
                end_char=group['end_pos'],
                chunk_type=chunk_type,
                language=primary_language,
                has_code=group['has_code'],
                has_table=bool(section.tables)  # Simplified
            )
            
            # Create chunk
            chunk = Chunk(
                chunk_id="",
                content=chunk_content,
                metadata=metadata
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _analyze_mixed_content(self, section: Section) -> List[Dict]:
        """
        Analyze section content và identify text vs code segments.
        """
        segments = []
        
        # Start with text content
        if section.content:
            text_segments = self._analyze_text_with_inline_code(section.content)
            segments.extend(text_segments)
        
        # Add dedicated code blocks
        for code_block in section.code_blocks:
            segments.append({
                'type': 'code_block',
                'content': code_block.content,
                'language': code_block.language,
                'size': len(code_block.content),
                'is_large': len(code_block.content) > self.config.max_code_block_size,
                'logical_units': self._analyze_code_structure(code_block.content, code_block.language)
            })
        
        return segments
    
    def _analyze_text_with_inline_code(self, text: str) -> List[Dict]:
        """
        Analyze text content và extract inline code.
        """
        segments = []
        current_pos = 0
        
        # Find code blocks first
        for match in self.code_block_pattern.finditer(text):
            # Add text before code block
            if match.start() > current_pos:
                text_content = text[current_pos:match.start()]
                if text_content.strip():
                    segments.append({
                        'type': 'text',
                        'content': text_content,
                        'size': len(text_content),
                        'has_inline_code': bool(self.inline_code_pattern.search(text_content))
                    })
            
            # Add code block
            language = match.group(1) or 'text'
            code_content = match.group(2)
            segments.append({
                'type': 'code_block',
                'content': code_content,
                'language': language,
                'size': len(code_content),
                'is_large': len(code_content) > self.config.max_code_block_size
            })
            
            current_pos = match.end()
        
        # Add remaining text
        if current_pos < len(text):
            remaining_text = text[current_pos:]
            if remaining_text.strip():
                segments.append({
                    'type': 'text',
                    'content': remaining_text,
                    'size': len(remaining_text),
                    'has_inline_code': bool(self.inline_code_pattern.search(remaining_text))
                })
        
        return segments
    
    def _analyze_code_structure(self, code: str, language: str) -> List[Dict]:
        """
        Analyze code structure để hiểu logical units.
        """
        if language not in self.analyzers:
            return []
        
        analyzer = self.analyzers[language]
        logical_units = analyzer.detect_logical_units(code)
        
        return [
            {
                'name': name,
                'start_pos': start,
                'end_pos': end,
                'size': end - start
            }
            for name, start, end in logical_units
        ]
    
    def _group_segments_into_chunks(self, segments: List[Dict]) -> List[Dict]:
        """
        Group segments into chunks theo code-aware rules.
        """
        chunks = []
        current_chunk = {
            'segments': [],
            'total_size': 0,
            'has_code': False,
            'start_pos': 0,
            'end_pos': 0
        }
        
        for segment in segments:
            segment_size = segment['size']
            
            # Special handling cho large code blocks
            if segment['type'] == 'code_block' and segment.get('is_large', False):
                # Save current chunk if it has content
                if current_chunk['segments']:
                    chunks.append(current_chunk)
                    current_chunk = self._new_chunk_group()
                
                # Split large code block
                code_chunks = self._split_large_code_block(segment)
                chunks.extend(code_chunks)
                continue
            
            # Check if adding this segment would exceed max size
            if (current_chunk['total_size'] + segment_size > self.config.max_chunk_size 
                and current_chunk['segments']):
                
                # Save current chunk
                chunks.append(current_chunk)
                current_chunk = self._new_chunk_group()
            
            # Add segment to current chunk
            current_chunk['segments'].append(segment)
            current_chunk['total_size'] += segment_size
            if segment['type'] == 'code_block':
                current_chunk['has_code'] = True
        
        # Don't forget the last chunk
        if current_chunk['segments']:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_large_code_block(self, code_segment: Dict) -> List[Dict]:
        """
        Split large code block thành smaller chunks.
        """
        code = code_segment['content']
        language = code_segment['language']
        
        # If we have logical units, try to split by them
        if 'logical_units' in code_segment and code_segment['logical_units']:
            return self._split_by_logical_units(code_segment)
        
        # Fallback: split by good break points
        if language in self.analyzers:
            analyzer = self.analyzers[language]
            split_points = analyzer.find_good_split_points(code, self.config.max_code_block_size)
            
            if split_points:
                return self._split_by_positions(code_segment, split_points)
        
        # Last resort: split by lines
        return self._split_by_lines(code_segment)
    
    def _split_by_logical_units(self, code_segment: Dict) -> List[Dict]:
        """Split code theo logical units (functions, classes)."""
        chunks = []
        code = code_segment['content']
        language = code_segment['language']
        logical_units = code_segment['logical_units']
        
        # Group small units together, large units get their own chunk
        current_group = []
        current_size = 0
        
        for unit in logical_units:
            unit_size = unit['size']
            
            if unit_size > self.config.max_code_block_size:
                # Save current group
                if current_group:
                    chunks.append(self._create_code_chunk_from_units(current_group, code, language))
                    current_group = []
                    current_size = 0
                
                # Large unit gets its own chunk
                chunks.append(self._create_code_chunk_from_units([unit], code, language))
                
            elif current_size + unit_size > self.config.max_code_block_size:
                # Save current group
                if current_group:
                    chunks.append(self._create_code_chunk_from_units(current_group, code, language))
                
                # Start new group
                current_group = [unit]
                current_size = unit_size
                
            else:
                current_group.append(unit)
                current_size += unit_size
        
        # Don't forget the last group
        if current_group:
            chunks.append(self._create_code_chunk_from_units(current_group, code, language))
        
        return chunks
    
    def _create_code_chunk_from_units(self, units: List[Dict], full_code: str, language: str) -> Dict:
        """Create chunk từ logical units."""
        if not units:
            return self._new_chunk_group()
        
        # Extract code for these units
        start_pos = min(unit['start_pos'] for unit in units)
        end_pos = max(unit['end_pos'] for unit in units)
        chunk_code = full_code[start_pos:end_pos]
        
        return {
            'segments': [{
                'type': 'code_block',
                'content': chunk_code,
                'language': language,
                'size': len(chunk_code),
                'logical_units': units
            }],
            'total_size': len(chunk_code),
            'has_code': True,
            'start_pos': start_pos,
            'end_pos': end_pos
        }
    
    def _split_by_positions(self, code_segment: Dict, split_points: List[int]) -> List[Dict]:
        """Split code tại các positions được specify."""
        chunks = []
        code = code_segment['content']
        language = code_segment['language']
        
        start = 0
        for split_point in split_points:
            if split_point > start:
                chunk_code = code[start:split_point]
                chunks.append({
                    'segments': [{
                        'type': 'code_block',
                        'content': chunk_code,
                        'language': language,
                        'size': len(chunk_code)
                    }],
                    'total_size': len(chunk_code),
                    'has_code': True,
                    'start_pos': start,
                    'end_pos': split_point
                })
                start = split_point
        
        # Add remaining code
        if start < len(code):
            remaining_code = code[start:]
            chunks.append({
                'segments': [{
                    'type': 'code_block',
                    'content': remaining_code,
                    'language': language,
                    'size': len(remaining_code)
                }],
                'total_size': len(remaining_code),
                'has_code': True,
                'start_pos': start,
                'end_pos': len(code)
            })
        
        return chunks
    
    def _split_by_lines(self, code_segment: Dict) -> List[Dict]:
        """Fallback: split code by lines."""
        chunks = []
        code = code_segment['content']
        language = code_segment['language']
        lines = code.split('\n')
        
        current_lines = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > self.config.max_code_block_size and current_lines:
                # Save current chunk
                chunk_code = '\n'.join(current_lines)
                chunks.append({
                    'segments': [{
                        'type': 'code_block',
                        'content': chunk_code,
                        'language': language,
                        'size': len(chunk_code)
                    }],
                    'total_size': len(chunk_code),
                    'has_code': True,
                    'start_pos': 0,  # Simplified
                    'end_pos': len(chunk_code)
                })
                
                current_lines = []
                current_size = 0
            
            current_lines.append(line)
            current_size += line_size
        
        # Add remaining lines
        if current_lines:
            chunk_code = '\n'.join(current_lines)
            chunks.append({
                'segments': [{
                    'type': 'code_block',
                    'content': chunk_code,
                    'language': language,
                    'size': len(chunk_code)
                }],
                'total_size': len(chunk_code),
                'has_code': True,
                'start_pos': 0,
                'end_pos': len(chunk_code)
            })
        
        return chunks
    
    def _new_chunk_group(self) -> Dict:
        """Create new empty chunk group."""
        return {
            'segments': [],
            'total_size': 0,
            'has_code': False,
            'start_pos': 0,
            'end_pos': 0
        }
    
    def _build_chunk_content(self, chunk_group: Dict, section: Section) -> str:
        """Build final content string từ chunk group."""
        content_parts = []
        
        # Add section header to first chunk nếu configured
        if self.config.include_section_header_in_chunks:
            content_parts.append(f"[Section {section.id}: {section.title}]")
        
        # Add each segment
        for segment in chunk_group['segments']:
            if segment['type'] == 'text':
                content_parts.append(segment['content'])
            elif segment['type'] == 'code_block':
                # Format code block
                language = segment.get('language', 'text')
                code_content = segment['content']
                
                # Add code block markers
                content_parts.append(f"```{language}")
                content_parts.append(code_content)
                content_parts.append("```")
        
        return '\n\n'.join(content_parts)
    
    def _determine_chunk_type(self, chunk_group: Dict) -> ChunkType:
        """Determine chunk type dựa trên content."""
        has_text = any(s['type'] == 'text' for s in chunk_group['segments'])
        has_code = chunk_group['has_code']
        
        if has_code and has_text:
            return ChunkType.MIXED
        elif has_code:
            return ChunkType.CODE
        else:
            return ChunkType.TEXT
    
    def _get_primary_language(self, chunk_group: Dict) -> Optional[str]:
        """Get primary programming language của chunk."""
        languages = [
            s.get('language') 
            for s in chunk_group['segments'] 
            if s['type'] == 'code_block' and s.get('language')
        ]
        
        if languages:
            # Return most common language, or first one if tie
            from collections import Counter
            lang_counts = Counter(languages)
            return lang_counts.most_common(1)[0][0]
        
        return None


# Integration với existing chunking system
def get_code_aware_chunker(config: ChunkingConfig) -> CodeAwareChunker:
    """Factory function để create code-aware chunker."""
    return CodeAwareChunker(config)


# Example usage
if __name__ == "__main__":
    from src.data_processing.chunk_models import ChunkingPresets
    from src.data_processing.html_parser import Section, CodeBlock
    
    # Create sample section với mixed content
    sample_content = """
    This section explains how to create a simple trading algorithm in QuantConnect.
    
    First, let's import the necessary libraries:
    
    ```python
    from QuantConnect import *
    from QuantConnect.Algorithm import *
    from QuantConnect.Data.Market import TradeBar
    
    class MyAlgorithm(QCAlgorithm):
        def Initialize(self):
            self.SetStartDate(2020, 1, 1)
            self.SetEndDate(2023, 1, 1)
            self.SetCash(100000)
            
            # Add equity data
            self.spy = self.AddEquity("SPY", Resolution.Daily)
            
            # Create indicators
            self.sma_fast = self.SMA("SPY", 10, Resolution.Daily)
            self.sma_slow = self.SMA("SPY", 30, Resolution.Daily)
    ```
    
    The Initialize method sets up our algorithm parameters and adds the data we need.
    
    Next, we implement the trading logic:
    
    ```python
    def OnData(self, data):
        if not self.sma_fast.IsReady or not self.sma_slow.IsReady:
            return
            
        if self.sma_fast.Current.Value > self.sma_slow.Current.Value:
            if not self.Portfolio["SPY"].IsLong:
                self.SetHoldings("SPY", 1.0)
                self.Debug("Bought SPY")
        else:
            if self.Portfolio["SPY"].IsLong:
                self.Liquidate("SPY")
                self.Debug("Sold SPY")
    ```
    
    This creates a simple moving average crossover strategy.
    """
    
    # Create section với code blocks
    section = Section(
        id="2.1.3",
        title="Creating a Moving Average Strategy",
        level=3,
        content=sample_content,
        section_number="2.1.3",
        breadcrumb="Algorithms > Examples > Moving Average Strategy"
    )
    
    # Add some code blocks
    section.code_blocks = [
        CodeBlock(
            language="python",
            content="""
class AdvancedAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(1000000)
        
        # Add multiple securities
        symbols = ["SPY", "QQQ", "IWM", "TLT"]
        self.securities = {}
        
        for symbol in symbols:
            equity = self.AddEquity(symbol, Resolution.Daily)
            self.securities[symbol] = {
                'equity': equity,
                'sma': self.SMA(symbol, 20, Resolution.Daily),
                'rsi': self.RSI(symbol, 14, Resolution.Daily)
            }
    
    def OnData(self, data):
        for symbol, security_data in self.securities.items():
            if not security_data['sma'].IsReady or not security_data['rsi'].IsReady:
                continue
                
            current_price = data[symbol].Close
            sma_value = security_data['sma'].Current.Value
            rsi_value = security_data['rsi'].Current.Value
            
            # Buy signal: price above SMA and RSI oversold
            if current_price > sma_value and rsi_value < 30:
                if not self.Portfolio[symbol].IsLong:
                    self.SetHoldings(symbol, 0.25)  # 25% allocation
                    self.Debug(f"Bought {symbol} at {current_price}")
            
            # Sell signal: price below SMA or RSI overbought
            elif current_price < sma_value or rsi_value > 70:
                if self.Portfolio[symbol].IsLong:
                    self.Liquidate(symbol)
                    self.Debug(f"Sold {symbol} at {current_price}")
            """,
            section_id=section.id
        )
    ]
    
    # Test code-aware chunker
    config = ChunkingPresets.for_code_understanding()
    config.max_chunk_size = 800  # Smaller for testing
    config.max_code_block_size = 1000
    
    chunker = CodeAwareChunker(config)
    chunks = chunker.chunk_section(section, "test.html", 1)
    
    print(f"Code-Aware Chunker produced {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Type: {chunk.metadata.chunk_type.value}")
        print(f"  Size: {chunk.char_count} chars")
        print(f"  Has code: {chunk.metadata.has_code}")
        print(f"  Language: {chunk.metadata.language}")
        print(f"  Content preview: {chunk.content[:150]}...")