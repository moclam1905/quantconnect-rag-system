"""
Hybrid Chunker cho QuantConnect documentation.
Kết hợp tất cả chunking strategies với smart routing và metadata enrichment.
Tự động chọn strategy tốt nhất cho từng loại content.
"""

from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import Counter
import math

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing.chunk_models import (
    Chunk, ChunkMetadata, ChunkType, ChunkingConfig, ChunkingStrategy
)
from src.data_processing.html_parser import Section, CodeBlock, TableData
from src.data_processing.chunking_strategies import (
    BaseChunkingStrategy, FixedSizeChunker, SentenceBasedChunker, 
    ParagraphBasedChunker, SectionBasedChunker
)
from src.data_processing.code_aware_chunker import CodeAwareChunker
from src.data_processing.section_based_chunker import (
    SectionBasedChunker, SectionHierarchyAnalyzer, SectionMetricsCalculator
)
from src.utils.logger import logger


class ContentType(Enum):
    """Types of content for smart routing"""
    PURE_TEXT = "pure_text"
    CODE_HEAVY = "code_heavy"
    MIXED_CONTENT = "mixed_content"
    STRUCTURED_DOC = "structured_doc"
    API_REFERENCE = "api_reference"
    TUTORIAL = "tutorial"
    EXAMPLES = "examples"
    TABLE_HEAVY = "table_heavy"


@dataclass
class ContentAnalysis:
    """Comprehensive analysis of content characteristics"""
    content_type: ContentType
    text_ratio: float
    code_ratio: float
    table_ratio: float
    structure_complexity: float
    readability_score: float
    technical_density: float
    code_languages: List[str]
    section_depth: int
    has_lists: bool
    has_formulas: bool
    has_references: bool
    recommended_strategy: ChunkingStrategy
    confidence_score: float  # 0-1, how confident we are in the analysis
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for metadata"""
        return {
            'content_type': self.content_type.value,
            'text_ratio': self.text_ratio,
            'code_ratio': self.code_ratio,
            'table_ratio': self.table_ratio,
            'structure_complexity': self.structure_complexity,
            'readability_score': self.readability_score,
            'technical_density': self.technical_density,
            'code_languages': self.code_languages,
            'section_depth': self.section_depth,
            'has_lists': self.has_lists,
            'has_formulas': self.has_formulas,
            'has_references': self.has_references,
            'recommended_strategy': self.recommended_strategy.value,
            'confidence_score': self.confidence_score
        }


class ContentAnalyzer:
    """Analyzes content to determine optimal chunking strategy"""
    
    def __init__(self):
        # Patterns for content analysis
        self.code_patterns = [
            r'```\w*\n.*?\n```',  # Code blocks
            r'`[^`]+`',  # Inline code
            r'^\s*(?:def|class|import|from|if|for|while)\s+',  # Python keywords
            r'^\s*(?:public|private|class|namespace|using)\s+',  # C# keywords
        ]
        
        self.technical_patterns = [
            r'\b(?:algorithm|function|method|class|variable|parameter|return|exception)\b',
            r'\b(?:API|SDK|JSON|XML|HTTP|REST|URL|URI)\b',
            r'\b(?:database|query|table|index|schema)\b',
            r'\b(?:server|client|request|response|authentication)\b',
        ]
        
        self.list_patterns = [
            r'^\s*[-*+]\s+',  # Bullet points
            r'^\s*\d+\.\s+',  # Numbered lists
        ]
        
        self.formula_patterns = [
            r'\$[^$]+\$',  # LaTeX math
            r'\\\([^)]+\\\)',  # LaTeX inline math
            r'\b(?:sum|integral|derivative|equation|formula)\b',
        ]
        
        self.reference_patterns = [
            r'\[[^\]]+\]',  # References like [1], [API]
            r'(?:see|refer to|check|visit)\s+\w+',  # Reference phrases
        ]
    
    def analyze_section(self, section: Section, hierarchy_analyzer: Optional[SectionHierarchyAnalyzer] = None) -> ContentAnalysis:
        """Comprehensive analysis of a section"""
        
        # Get all text content
        full_text = self._extract_all_text(section)
        total_chars = len(full_text)
        
        if total_chars == 0:
            return self._create_empty_analysis()
        
        # Calculate ratios
        text_chars = len(section.content) if section.content else 0
        code_chars = sum(len(cb.content) for cb in section.code_blocks)
        table_chars = self._estimate_table_chars(section.tables)
        
        text_ratio = text_chars / total_chars
        code_ratio = code_chars / total_chars
        table_ratio = table_chars / total_chars
        
        # Analyze content characteristics
        structure_complexity = self._calculate_structure_complexity(section, hierarchy_analyzer)
        readability_score = self._calculate_readability_score(section.content or "")
        technical_density = self._calculate_technical_density(full_text)
        
        # Extract code languages
        code_languages = list(set(cb.language for cb in section.code_blocks))
        
        # Section depth
        section_depth = section.level
        
        # Content flags
        has_lists = bool(re.search('|'.join(self.list_patterns), full_text, re.MULTILINE))
        has_formulas = bool(re.search('|'.join(self.formula_patterns), full_text, re.IGNORECASE))
        has_references = bool(re.search('|'.join(self.reference_patterns), full_text, re.IGNORECASE))
        
        # Determine content type
        content_type = self._determine_content_type(
            text_ratio, code_ratio, table_ratio, section, technical_density
        )
        
        # Recommend strategy
        recommended_strategy, confidence = self._recommend_strategy(
            content_type, structure_complexity, code_ratio, table_ratio, section
        )
        
        return ContentAnalysis(
            content_type=content_type,
            text_ratio=text_ratio,
            code_ratio=code_ratio,
            table_ratio=table_ratio,
            structure_complexity=structure_complexity,
            readability_score=readability_score,
            technical_density=technical_density,
            code_languages=code_languages,
            section_depth=section_depth,
            has_lists=has_lists,
            has_formulas=has_formulas,
            has_references=has_references,
            recommended_strategy=recommended_strategy,
            confidence_score=confidence
        )
    
    def _extract_all_text(self, section: Section) -> str:
        """Extract all text content from section"""
        parts = []
        
        if section.content:
            parts.append(section.content)
        
        for code_block in section.code_blocks:
            parts.append(code_block.content)
        
        for table in section.tables:
            # Add table content
            if table.headers:
                parts.append(' '.join(table.headers))
            for row in table.rows:
                parts.append(' '.join(str(cell) for cell in row))
        
        return '\n'.join(parts)
    
    def _estimate_table_chars(self, tables: List[TableData]) -> int:
        """Estimate character count for tables"""
        total = 0
        for table in tables:
            if table.headers:
                total += sum(len(h) for h in table.headers)
            for row in table.rows:
                total += sum(len(str(cell)) for cell in row)
            if table.caption:
                total += len(table.caption)
        return total
    
    def _calculate_structure_complexity(self, section: Section, hierarchy_analyzer: Optional[SectionHierarchyAnalyzer]) -> float:
        """Calculate structural complexity (0-1)"""
        complexity_factors = []
        
        # Section depth
        depth_score = min(section.level / 5, 1.0)
        complexity_factors.append(depth_score)
        
        # Number of code blocks
        code_score = min(len(section.code_blocks) / 5, 1.0)
        complexity_factors.append(code_score)
        
        # Number of tables
        table_score = min(len(section.tables) / 3, 1.0)
        complexity_factors.append(table_score)
        
        # Hierarchy complexity
        if hierarchy_analyzer:
            children = hierarchy_analyzer.get_children(section.id)
            hierarchy_score = min(len(children) / 10, 1.0)
            complexity_factors.append(hierarchy_score)
        
        # Content structure (lists, headings, etc.)
        content = section.content or ""
        headings = len(re.findall(r'^#+\s+', content, re.MULTILINE))
        lists = len(re.findall(r'^\s*[-*+]\s+', content, re.MULTILINE))
        structure_score = min((headings + lists) / 20, 1.0)
        complexity_factors.append(structure_score)
        
        return sum(complexity_factors) / len(complexity_factors)
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate readability score (0-1, higher = more readable)"""
        if not text:
            return 0.5
        
        # Simple readability metrics
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        
        if sentences == 0 or words == 0:
            return 0.5
        
        # Average sentence length (shorter = more readable)
        avg_sentence_length = words / sentences
        sentence_score = max(0, 1 - (avg_sentence_length - 15) / 20)  # Optimal around 15 words
        
        # Syllable complexity (simplified)
        avg_word_length = len(text.replace(' ', '')) / words
        word_score = max(0, 1 - (avg_word_length - 5) / 5)  # Optimal around 5 chars
        
        # Technical term density
        technical_matches = len(re.findall('|'.join(self.technical_patterns), text, re.IGNORECASE))
        tech_density = technical_matches / words
        tech_score = max(0, 1 - tech_density)
        
        return (sentence_score + word_score + tech_score) / 3
    
    def _calculate_technical_density(self, text: str) -> float:
        """Calculate technical density (0-1)"""
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # Count technical terms
        technical_matches = 0
        for pattern in self.technical_patterns:
            technical_matches += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Count code-like patterns
        code_matches = 0
        for pattern in self.code_patterns:
            code_matches += len(re.findall(pattern, text, re.DOTALL))
        
        # Normalize by word count
        density = (technical_matches + code_matches * 2) / len(words)  # Code weighted higher
        return min(density, 1.0)
    
    def _determine_content_type(self, text_ratio: float, code_ratio: float, table_ratio: float, section: Section, technical_density: float) -> ContentType:
        """Determine the primary content type"""
        
        # Code-heavy content
        if code_ratio > 0.4 or (code_ratio > 0.2 and technical_density > 0.3):
            return ContentType.CODE_HEAVY
        
        # Table-heavy content
        if table_ratio > 0.3:
            return ContentType.TABLE_HEAVY
        
        # Mixed content
        if code_ratio > 0.1 and text_ratio > 0.5:
            return ContentType.MIXED_CONTENT
        
        # API Reference (high technical density, structured)
        if technical_density > 0.4 and section.level > 2:
            return ContentType.API_REFERENCE
        
        # Tutorial (moderate technical density, structured)
        if 0.1 < technical_density < 0.4 and section.level <= 3:
            return ContentType.TUTORIAL
        
        # Examples (has code but primarily explanatory)
        if code_ratio > 0.05 and text_ratio > 0.6:
            return ContentType.EXAMPLES
        
        # Structured document (high text ratio, good structure)
        if text_ratio > 0.8 and section.level > 1:
            return ContentType.STRUCTURED_DOC
        
        # Default to pure text
        return ContentType.PURE_TEXT
    
    def _recommend_strategy(self, content_type: ContentType, structure_complexity: float, code_ratio: float, table_ratio: float, section: Section) -> Tuple[ChunkingStrategy, float]:
        """Recommend chunking strategy with confidence score"""
        
        recommendations = {
            ContentType.PURE_TEXT: (ChunkingStrategy.SENTENCE_BASED, 0.8),
            ContentType.CODE_HEAVY: (ChunkingStrategy.CODE_AWARE, 0.9),
            ContentType.MIXED_CONTENT: (ChunkingStrategy.HYBRID, 0.8),
            ContentType.STRUCTURED_DOC: (ChunkingStrategy.SECTION_BASED, 0.9),
            ContentType.API_REFERENCE: (ChunkingStrategy.SECTION_BASED, 0.8),
            ContentType.TUTORIAL: (ChunkingStrategy.PARAGRAPH_BASED, 0.7),
            ContentType.EXAMPLES: (ChunkingStrategy.CODE_AWARE, 0.8),
            ContentType.TABLE_HEAVY: (ChunkingStrategy.SECTION_BASED, 0.8),
        }
        
        base_strategy, base_confidence = recommendations.get(content_type, (ChunkingStrategy.HYBRID, 0.6))
        
        # Adjust confidence based on additional factors
        confidence_adjustments = []
        
        # Structure complexity
        if structure_complexity > 0.6:
            confidence_adjustments.append(-0.1)  # More complex = less confident
        
        # Content balance
        if 0.3 < code_ratio < 0.7:  # Balanced content is harder to chunk
            confidence_adjustments.append(-0.1)
        
        # Section size
        total_size = len(section.content or "") + sum(len(cb.content) for cb in section.code_blocks)
        if total_size > 3000:  # Large sections are harder
            confidence_adjustments.append(-0.1)
        
        final_confidence = max(0.3, base_confidence + sum(confidence_adjustments))
        
        return base_strategy, final_confidence
    
    def _create_empty_analysis(self) -> ContentAnalysis:
        """Create analysis for empty content"""
        return ContentAnalysis(
            content_type=ContentType.PURE_TEXT,
            text_ratio=0.0,
            code_ratio=0.0,
            table_ratio=0.0,
            structure_complexity=0.0,
            readability_score=0.5,
            technical_density=0.0,
            code_languages=[],
            section_depth=1,
            has_lists=False,
            has_formulas=False,
            has_references=False,
            recommended_strategy=ChunkingStrategy.FIXED_SIZE,
            confidence_score=0.5
        )


class ChunkQualityScorer:
    """Scores chunk quality for optimization"""
    
    def score_chunk(self, chunk: Chunk, content_analysis: ContentAnalysis) -> float:
        """Score chunk quality (0-1, higher = better)"""
        
        scores = []
        
        # Size score (optimal size gets highest score)
        size_score = self._score_size(chunk.char_count)
        scores.append(size_score)
        
        # Completeness score (complete sentences, code blocks)
        completeness_score = self._score_completeness(chunk.content)
        scores.append(completeness_score)
        
        # Coherence score (topic consistency)
        coherence_score = self._score_coherence(chunk.content, content_analysis)
        scores.append(coherence_score)
        
        # Context score (has proper context)
        context_score = self._score_context(chunk)
        scores.append(context_score)
        
        return sum(scores) / len(scores)
    
    def _score_size(self, size: int) -> float:
        """Score based on chunk size"""
        # Optimal range is 500-1200 characters
        if 500 <= size <= 1200:
            return 1.0
        elif 200 <= size < 500:
            return 0.8
        elif 1200 < size <= 2000:
            return 0.7
        elif size < 200:
            return 0.5
        else:
            return 0.3
    
    def _score_completeness(self, content: str) -> float:
        """Score based on content completeness"""
        scores = []
        
        # Check for complete sentences
        sentences = re.findall(r'[.!?]+', content)
        if sentences:
            # Check if content ends with proper punctuation
            if content.rstrip().endswith(('.', '!', '?', '```')):
                scores.append(1.0)
            else:
                scores.append(0.7)
        else:
            scores.append(0.5)
        
        # Check for complete code blocks
        code_blocks = re.findall(r'```.*?```', content, re.DOTALL)
        incomplete_code = re.search(r'```[^`]*$', content)
        if code_blocks and not incomplete_code:
            scores.append(1.0)
        elif incomplete_code:
            scores.append(0.3)
        else:
            scores.append(0.8)  # No code blocks is fine
        
        return sum(scores) / len(scores)
    
    def _score_coherence(self, content: str, content_analysis: ContentAnalysis) -> float:
        """Score based on topic coherence"""
        # Simplified coherence scoring
        # Check for topic consistency by looking at keyword density
        
        words = content.lower().split()
        if len(words) < 10:
            return 0.6  # Too short to evaluate
        
        # Count repetition of key terms
        word_counts = Counter(words)
        most_common = word_counts.most_common(5)
        
        # Good coherence if key terms appear multiple times
        repetition_score = 0
        for word, count in most_common:
            if len(word) > 3 and count > 1:  # Ignore short words
                repetition_score += min(count / len(words), 0.2)
        
        return min(repetition_score * 5, 1.0)  # Scale to 0-1
    
    def _score_context(self, chunk: Chunk) -> float:
        """Score based on contextual information"""
        scores = []
        
        # Check if chunk has section information
        if chunk.metadata.section_title:
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # Check if chunk has hierarchy information
        if chunk.metadata.section_path:
            scores.append(1.0)
        else:
            scores.append(0.7)
        
        # Check metadata completeness
        if chunk.metadata.chunk_type != ChunkType.TEXT:  # Has specific type
            scores.append(1.0)
        else:
            scores.append(0.8)
        
        return sum(scores) / len(scores)


class HybridChunker(BaseChunkingStrategy):
    """
    Hybrid chunker that combines all strategies with smart routing.
    
    Features:
    - Automatic strategy selection based on content analysis
    - Metadata enrichment for better retrieval
    - Quality scoring and optimization
    - Adaptive chunking parameters
    - Fallback strategies for edge cases
    """
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.strategy_type = ChunkingStrategy.HYBRID
        
        # Initialize all chunkers
        self.chunkers = self._initialize_chunkers(config)
        
        # Initialize analyzers
        self.content_analyzer = ContentAnalyzer()
        self.quality_scorer = ChunkQualityScorer()
        
        # Adaptive parameters
        self.min_quality_threshold = 0.6
        self.max_retry_attempts = 3
    
    def _initialize_chunkers(self, config: ChunkingConfig) -> Dict[ChunkingStrategy, BaseChunkingStrategy]:
        """Initialize all available chunkers"""
        chunkers = {}
        
        try:
            chunkers[ChunkingStrategy.FIXED_SIZE] = FixedSizeChunker(config)
            chunkers[ChunkingStrategy.SENTENCE_BASED] = SentenceBasedChunker(config)
            chunkers[ChunkingStrategy.PARAGRAPH_BASED] = ParagraphBasedChunker(config)
            chunkers[ChunkingStrategy.SECTION_BASED] = SectionBasedChunker(config)
            chunkers[ChunkingStrategy.CODE_AWARE] = CodeAwareChunker(config)
        except Exception as e:
            logger.warning(f"Failed to initialize some chunkers: {e}")
            # Fallback to basic chunker
            chunkers[ChunkingStrategy.FIXED_SIZE] = FixedSizeChunker(config)
        
        return chunkers
    
    def chunk_section(self, section: Section, source_file: str, doc_index: int) -> List[Chunk]:
        """
        Chunk section using hybrid approach with smart routing.
        """
        # Analyze content
        content_analysis = self.content_analyzer.analyze_section(section)
        
        # Select primary strategy
        primary_strategy = content_analysis.recommended_strategy
        
        # Generate chunks with primary strategy
        chunks = self._chunk_with_strategy(
            section, source_file, doc_index, primary_strategy, content_analysis
        )
        
        # Quality check and potential retry
        if chunks:
            avg_quality = self._evaluate_chunks_quality(chunks, content_analysis)
            
            if avg_quality < self.min_quality_threshold:
                # Try fallback strategies
                chunks = self._try_fallback_strategies(
                    section, source_file, doc_index, primary_strategy, content_analysis
                )
        
        # Enrich metadata
        chunks = self._enrich_chunks_metadata(chunks, content_analysis)
        
        return chunks
    
    def chunk_sections(self, sections: List[Section], source_file: str, doc_index: int) -> List[Chunk]:
        """
        Chunk multiple sections with global optimization.
        """
        if not sections:
            return []
        
        # Build hierarchy analyzer
        hierarchy_analyzer = SectionHierarchyAnalyzer(sections)
        
        # Analyze each section
        section_analyses = {}
        for section in sections:
            section_analyses[section.id] = self.content_analyzer.analyze_section(
                section, hierarchy_analyzer
            )
        
        # Determine global strategy
        global_strategy = self._determine_global_strategy(section_analyses)
        
        # Use section-based chunker for global approach
        if global_strategy == ChunkingStrategy.SECTION_BASED:
            section_chunker = self.chunkers[ChunkingStrategy.SECTION_BASED]
            if hasattr(section_chunker, 'chunk_sections'):
                return section_chunker.chunk_sections(sections, source_file, doc_index)
        
        # Fallback to individual chunking
        all_chunks = []
        for section in sections:
            chunks = self.chunk_section(section, source_file, doc_index)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _chunk_with_strategy(
        self, 
        section: Section, 
        source_file: str, 
        doc_index: int, 
        strategy: ChunkingStrategy,
        content_analysis: ContentAnalysis
    ) -> List[Chunk]:
        """Chunk section with specific strategy"""
        
        if strategy not in self.chunkers:
            logger.warning(f"Strategy {strategy} not available, using fixed size")
            strategy = ChunkingStrategy.FIXED_SIZE
        
        chunker = self.chunkers[strategy]
        
        try:
            # Adapt config based on content analysis
            adapted_config = self._adapt_config_for_content(content_analysis)
            if adapted_config:
                chunker.config = adapted_config
            
            return chunker.chunk_section(section, source_file, doc_index)
        
        except Exception as e:
            logger.error(f"Error chunking with {strategy}: {e}")
            # Fallback to fixed size
            return self.chunkers[ChunkingStrategy.FIXED_SIZE].chunk_section(
                section, source_file, doc_index
            )
    
    def _try_fallback_strategies(
        self,
        section: Section,
        source_file: str,
        doc_index: int,
        failed_strategy: ChunkingStrategy,
        content_analysis: ContentAnalysis
    ) -> List[Chunk]:
        """Try fallback strategies if primary fails"""
        
        # Define fallback order based on content type
        fallback_orders = {
            ContentType.CODE_HEAVY: [ChunkingStrategy.SECTION_BASED, ChunkingStrategy.PARAGRAPH_BASED, ChunkingStrategy.FIXED_SIZE],
            ContentType.MIXED_CONTENT: [ChunkingStrategy.PARAGRAPH_BASED, ChunkingStrategy.SENTENCE_BASED, ChunkingStrategy.FIXED_SIZE],
            ContentType.STRUCTURED_DOC: [ChunkingStrategy.PARAGRAPH_BASED, ChunkingStrategy.SENTENCE_BASED, ChunkingStrategy.FIXED_SIZE],
            ContentType.PURE_TEXT: [ChunkingStrategy.PARAGRAPH_BASED, ChunkingStrategy.FIXED_SIZE],
        }
        
        fallback_list = fallback_orders.get(
            content_analysis.content_type, 
            [ChunkingStrategy.PARAGRAPH_BASED, ChunkingStrategy.FIXED_SIZE]
        )
        
        # Remove failed strategy from fallback list
        fallback_list = [s for s in fallback_list if s != failed_strategy]
        
        for strategy in fallback_list:
            try:
                chunks = self._chunk_with_strategy(
                    section, source_file, doc_index, strategy, content_analysis
                )
                
                if chunks:
                    avg_quality = self._evaluate_chunks_quality(chunks, content_analysis)
                    if avg_quality >= self.min_quality_threshold:
                        logger.info(f"Fallback to {strategy} successful")
                        return chunks
            
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy} failed: {e}")
                continue
        
        # Last resort: simple fixed size
        logger.warning("All strategies failed, using simple fixed size")
        return self.chunkers[ChunkingStrategy.FIXED_SIZE].chunk_section(
            section, source_file, doc_index
        )
    
    def _evaluate_chunks_quality(self, chunks: List[Chunk], content_analysis: ContentAnalysis) -> float:
        """Evaluate average quality of chunks"""
        if not chunks:
            return 0.0
        
        total_score = 0.0
        for chunk in chunks:
            score = self.quality_scorer.score_chunk(chunk, content_analysis)
            total_score += score
        
        return total_score / len(chunks)
    
    def _enrich_chunks_metadata(self, chunks: List[Chunk], content_analysis: ContentAnalysis) -> List[Chunk]:
        """Enrich chunk metadata with analysis results"""
        
        for chunk in chunks:
            # Add content analysis to metadata
            chunk.metadata.chunking_strategy = self.strategy_type
            
            # Add custom metadata
            if hasattr(chunk.metadata, 'custom_metadata'):
                chunk.metadata.custom_metadata = content_analysis.to_dict()
            
            # Enhance existing metadata
            if not chunk.metadata.language and content_analysis.code_languages:
                chunk.metadata.language = content_analysis.code_languages[0]
            
            # Add quality score
            quality_score = self.quality_scorer.score_chunk(chunk, content_analysis)
            # Store quality score in a custom way since we can't modify the dataclass
            # In a real implementation, you might extend ChunkMetadata
        
        return chunks
    
    def _determine_global_strategy(self, section_analyses: Dict[str, ContentAnalysis]) -> ChunkingStrategy:
        """Determine best global strategy for multiple sections"""
        
        # Count strategy recommendations
        strategy_votes = Counter()
        confidence_weights = {}
        
        for section_id, analysis in section_analyses.items():
            strategy = analysis.recommended_strategy
            confidence = analysis.confidence_score
            
            strategy_votes[strategy] += 1
            if strategy not in confidence_weights:
                confidence_weights[strategy] = []
            confidence_weights[strategy].append(confidence)
        
        # Weight votes by confidence
        weighted_scores = {}
        for strategy, votes in strategy_votes.items():
            avg_confidence = sum(confidence_weights[strategy]) / len(confidence_weights[strategy])
            weighted_scores[strategy] = votes * avg_confidence
        
        # Return strategy with highest weighted score
        if weighted_scores:
            best_strategy = max(weighted_scores.items(), key=lambda x: x[1])[0]
            return best_strategy
        
        return ChunkingStrategy.HYBRID
    
    def _adapt_config_for_content(self, content_analysis: ContentAnalysis) -> Optional[ChunkingConfig]:
        """Adapt chunking config based on content analysis"""
        
        # Create adaptive config
        adapted_config = ChunkingConfig(
            max_chunk_size=self.config.max_chunk_size,
            min_chunk_size=self.config.min_chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # Adjust based on content type
        if content_analysis.content_type == ContentType.CODE_HEAVY:
            adapted_config.max_chunk_size = int(self.config.max_chunk_size * 1.2)  # Larger for code
            adapted_config.preserve_small_code_blocks = True
        
        elif content_analysis.content_type == ContentType.API_REFERENCE:
            adapted_config.max_chunk_size = int(self.config.max_chunk_size * 0.8)  # Smaller for precise retrieval
            adapted_config.respect_section_boundaries = True
        
        elif content_analysis.content_type == ContentType.TUTORIAL:
            adapted_config.chunk_overlap = int(self.config.chunk_overlap * 1.3)  # More overlap for continuity
            adapted_config.ensure_complete_sentences = True
        
        elif content_analysis.readability_score < 0.5:  # Complex content
            adapted_config.max_chunk_size = int(self.config.max_chunk_size * 0.9)  # Smaller chunks
            adapted_config.chunk_overlap = int(self.config.chunk_overlap * 1.2)  # More overlap
        
        return adapted_config


# Factory function
def get_hybrid_chunker(config: ChunkingConfig) -> HybridChunker:
    """Factory function to create hybrid chunker"""
    return HybridChunker(config)


# Example usage
if __name__ == "__main__":
    from src.data_processing.chunk_models import ChunkingPresets
    from src.data_processing.html_parser import Section, CodeBlock
    
    # Create complex test section
    complex_section = Section(
        id="3.2.1",
        title="Advanced Trading Strategies",
        level=3,
        content="""
        This section covers advanced algorithmic trading strategies in QuantConnect.
        
        We'll explore momentum strategies, mean reversion, and statistical arbitrage.
        Each strategy has different risk profiles and market conditions where they perform best.
        
        ## Risk Management
        
        Before implementing any strategy, consider these risk factors:
        - Market volatility
        - Position sizing
        - Drawdown limits
        - Correlation between assets
        
        ## Implementation Guidelines
        
        Follow these steps when implementing a strategy:
        1. Backtest thoroughly
        2. Validate on out-of-sample data
        3. Monitor performance metrics
        4. Adjust parameters as needed
        """,
        section_number="3.2.1",
        breadcrumb="Algorithms > Advanced > Trading Strategies"
    )
    
    # Add code blocks
    complex_section.code_blocks = [
        CodeBlock(
            language="python",
            content="""
class MomentumStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(1000000)
        
        # Universe selection
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        
        # Strategy parameters
        self.lookback_period = 252  # 1 year
        self.top_percentile = 0.1   # Top 10%
        self.rebalance_frequency = 30  # Monthly
        
        # Tracking variables
        self.securities = {}
        self.last_rebalance = datetime.min
    
    def CoarseSelectionFunction(self, coarse):
        # Filter by price and volume
        filtered = [x for x in coarse if x.Price > 10 and x.DollarVolume > 1000000]
        
        # Sort by momentum (12-month return)
        sorted_by_momentum = sorted(filtered, 
                                  key=lambda x: x.Price / x.AdjustedPrice, 
                                  reverse=True)
        
        # Return top performers
        return [x.Symbol for x in sorted_by_momentum[:50]]
    
    def OnData(self, data):
        # Rebalance monthly
        if (self.Time - self.last_rebalance).days < self.rebalance_frequency:
            return
        
        # Calculate momentum for each security
        momentum_scores = {}
        for symbol in self.Securities.Keys:
            if symbol in data and data[symbol] is not None:
                history = self.History(symbol, self.lookback_period, Resolution.Daily)
                if len(history) > 0:
                    momentum_scores[symbol] = history['close'][-1] / history['close'][0] - 1
        
        # Select top performers
        sorted_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        selected_symbols = [x[0] for x in sorted_symbols[:int(len(sorted_symbols) * self.top_percentile)]]
        
        # Rebalance portfolio
        for symbol in self.Securities.Keys:
            if symbol in selected_symbols:
                self.SetHoldings(symbol, 1.0 / len(selected_symbols))
            else:
                self.Liquidate(symbol)
        
        self.last_rebalance = self.Time
            """,
            section_id="3.2.1"
        ),
        
        CodeBlock(
            language="python",
            content="""
def CalculateRiskMetrics(self, returns):
    \"\"\"Calculate comprehensive risk metrics for the strategy\"\"\"
    
    # Convert to numpy array
    returns_array = np.array(returns)
    
    # Basic statistics
    mean_return = np.mean(returns_array)
    volatility = np.std(returns_array)
    sharpe_ratio = mean_return / volatility if volatility > 0 else 0
    
    # Downside risk metrics
    negative_returns = returns_array[returns_array < 0]
    downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
    sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + returns_array)
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = np.min(drawdowns)
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(returns_array, 5)
    
    # Expected Shortfall (Conditional VaR)
    expected_shortfall = np.mean(returns_array[returns_array <= var_95])
    
    return {
        'mean_return': mean_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'expected_shortfall': expected_shortfall
    }
            """,
            section_id="3.2.1"
        )
    ]
    
    # Test hybrid chunker
    config = ChunkingPresets.for_documentation()
    config.max_chunk_size = 1000
    
    chunker = HybridChunker(config)
    chunks = chunker.chunk_section(complex_section, "advanced_strategies.html", 1)
    
    print(f"Hybrid Chunker produced {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Type: {chunk.metadata.chunk_type.value}")
        print(f"  Size: {chunk.char_count} chars")
        print(f"  Strategy: {chunk.metadata.chunking_strategy.value}")
        print(f"  Section: {chunk.metadata.section_id} - {chunk.metadata.section_title}")
        print(f"  Has code: {chunk.metadata.has_code}")
        print(f"  Language: {chunk.metadata.language}")
        print(f"  Content preview: {chunk.content[:150]}...")  