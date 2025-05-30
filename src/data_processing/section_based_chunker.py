"""
Section-Based Chunker cho QuantConnect documentation.
Chunk dựa trên cấu trúc sections, respect boundaries và hierarchy.
Đặc biệt phù hợp cho structured documentation với clear section organization.
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import re

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing.chunk_models import (
    Chunk, ChunkMetadata, ChunkType, ChunkingConfig, ChunkingStrategy
)
from src.data_processing.html_parser import Section, CodeBlock, TableData
from src.data_processing.chunking_strategies import BaseChunkingStrategy
from src.utils.logger import logger


class SectionChunkingStrategy(Enum):
    """Different strategies for section-based chunking"""
    PRESERVE_SMALL = "preserve_small"           # Keep small sections intact
    SPLIT_LARGE = "split_large"                 # Split large sections
    HIERARCHICAL = "hierarchical"               # Group by hierarchy
    SUMMARY_FIRST = "summary_first"             # Create summary + detail chunks
    BALANCED = "balanced"                       # Balance between strategies


@dataclass
class SectionMetrics:
    """Metrics for analyzing a section"""
    id: str
    title: str
    level: int
    text_size: int
    code_size: int
    table_size: int
    total_size: int
    code_blocks_count: int
    tables_count: int
    subsections_count: int
    complexity_score: float
    is_leaf: bool  # No subsections
    
    @property
    def is_small(self) -> bool:
        """Check if section is small enough to keep intact"""
        return self.total_size <= 800  # Configurable threshold
    
    @property
    def is_large(self) -> bool:
        """Check if section is too large and needs splitting"""
        return self.total_size > 2000  # Configurable threshold
    
    @property
    def is_code_heavy(self) -> bool:
        """Check if section is primarily code"""
        return self.code_size > (self.text_size * 1.5)
    
    @property
    def is_content_heavy(self) -> bool:
        """Check if section has substantial content"""
        return self.text_size > 500 or self.code_blocks_count > 2


class SectionHierarchyAnalyzer:
    """Analyze section hierarchy and relationships"""
    
    def __init__(self, sections: List[Section]):
        self.sections = sections
        self.section_map = {s.id: s for s in sections}
        self._build_hierarchy_cache()
    
    def _build_hierarchy_cache(self):
        """Build caches for efficient hierarchy queries"""
        self.children_map = {}  # section_id -> list of child sections
        self.parent_map = {}    # section_id -> parent section
        self.depth_map = {}     # section_id -> depth in hierarchy
        
        # Build parent-child relationships
        for section in self.sections:
            if section.parent_id:
                if section.parent_id not in self.children_map:
                    self.children_map[section.parent_id] = []
                self.children_map[section.parent_id].append(section)
                self.parent_map[section.id] = self.section_map.get(section.parent_id)
        
        # Calculate depths
        for section in self.sections:
            self.depth_map[section.id] = self._calculate_depth(section)
    
    def _calculate_depth(self, section: Section) -> int:
        """Calculate depth of section in hierarchy"""
        if not section.parent_id:
            return 0
        
        parent = self.parent_map.get(section.id)
        if parent:
            return 1 + self._calculate_depth(parent)
        return 0
    
    def get_children(self, section_id: str) -> List[Section]:
        """Get direct children of a section"""
        return self.children_map.get(section_id, [])
    
    def get_all_descendants(self, section_id: str) -> List[Section]:
        """Get all descendants (children, grandchildren, etc.)"""
        descendants = []
        direct_children = self.get_children(section_id)
        
        for child in direct_children:
            descendants.append(child)
            descendants.extend(self.get_all_descendants(child.id))
        
        return descendants
    
    def get_siblings(self, section_id: str) -> List[Section]:
        """Get sibling sections (same parent)"""
        section = self.section_map.get(section_id)
        if not section or not section.parent_id:
            return []
        
        return [s for s in self.get_children(section.parent_id) if s.id != section_id]
    
    def get_section_path(self, section_id: str) -> List[Section]:
        """Get path from root to this section"""
        path = []
        current = self.section_map.get(section_id)
        
        while current:
            path.insert(0, current)
            current = self.parent_map.get(current.id)
        
        return path
    
    def is_leaf_section(self, section_id: str) -> bool:
        """Check if section has no children"""
        return len(self.get_children(section_id)) == 0


class SectionMetricsCalculator:
    """Calculate metrics for sections to aid chunking decisions"""
    
    @staticmethod
    def calculate_metrics(section: Section, hierarchy_analyzer: SectionHierarchyAnalyzer) -> SectionMetrics:
        """Calculate comprehensive metrics for a section"""
        
        # Size calculations
        text_size = len(section.content) if section.content else 0
        code_size = sum(len(cb.content) for cb in section.code_blocks)
        table_size = SectionMetricsCalculator._estimate_table_size(section.tables)
        total_size = text_size + code_size + table_size
        
        # Count calculations
        code_blocks_count = len(section.code_blocks)
        tables_count = len(section.tables)
        subsections_count = len(hierarchy_analyzer.get_children(section.id))
        
        # Complexity score (0-1, higher = more complex)
        complexity_score = SectionMetricsCalculator._calculate_complexity(
            section, text_size, code_blocks_count, tables_count, subsections_count
        )
        
        return SectionMetrics(
            id=section.id,
            title=section.title,
            level=section.level,
            text_size=text_size,
            code_size=code_size,
            table_size=table_size,
            total_size=total_size,
            code_blocks_count=code_blocks_count,
            tables_count=tables_count,
            subsections_count=subsections_count,
            complexity_score=complexity_score,
            is_leaf=hierarchy_analyzer.is_leaf_section(section.id)
        )
    
    @staticmethod
    def _estimate_table_size(tables: List[TableData]) -> int:
        """Estimate size of tables in characters"""
        total_size = 0
        for table in tables:
            # Headers
            if table.headers:
                total_size += sum(len(h) for h in table.headers) + len(table.headers) * 3  # separators
            
            # Rows
            for row in table.rows:
                total_size += sum(len(str(cell)) for cell in row) + len(row) * 3
            
            # Caption
            if table.caption:
                total_size += len(table.caption)
        
        return total_size
    
    @staticmethod
    def _calculate_complexity(
        section: Section, 
        text_size: int, 
        code_blocks: int, 
        tables: int, 
        subsections: int
    ) -> float:
        """Calculate complexity score (0-1) based on various factors"""
        
        # Base complexity from content size
        size_complexity = min(text_size / 2000, 1.0)  # Max at 2000 chars
        
        # Code complexity
        code_complexity = min(code_blocks / 5, 1.0)  # Max at 5 code blocks
        
        # Table complexity
        table_complexity = min(tables / 3, 1.0)  # Max at 3 tables
        
        # Structure complexity
        structure_complexity = min(subsections / 10, 1.0)  # Max at 10 subsections
        
        # Check for complex patterns in text
        text_complexity = 0.0
        if section.content:
            # Look for lists, references, formulas, etc.
            list_matches = len(re.findall(r'^\s*[-*+]\s+', section.content, re.MULTILINE))
            reference_matches = len(re.findall(r'\[[^\]]+\]', section.content))
            formula_matches = len(re.findall(r'\$[^$]+\$', section.content))
            
            text_complexity = min((list_matches + reference_matches + formula_matches) / 20, 1.0)
        
        # Weighted average
        weights = [0.3, 0.25, 0.15, 0.15, 0.15]  # size, code, table, structure, text patterns
        complexities = [size_complexity, code_complexity, table_complexity, structure_complexity, text_complexity]
        
        return sum(w * c for w, c in zip(weights, complexities))


class SectionBasedChunker(BaseChunkingStrategy):
    """
    Section-based chunker that respects document structure.
    
    Strategies:
    1. Keep small sections intact
    2. Split large sections intelligently 
    3. Create section summaries for navigation
    4. Maintain hierarchy context in chunks
    5. Group related subsections when appropriate
    """
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.strategy_type = ChunkingStrategy.SECTION_BASED
        
        # Section-specific configuration
        self.section_strategy = SectionChunkingStrategy.BALANCED
        self.min_section_chunk_size = config.min_chunk_size
        self.max_section_chunk_size = config.max_chunk_size
        self.create_section_summaries = config.create_section_summary_chunk
        self.include_hierarchy_context = config.include_hierarchy_meta
    
    def chunk_section(self, section: Section, source_file: str, doc_index: int) -> List[Chunk]:
        """
        Chunk a single section - this is the main entry point.
        For section-based chunking, we'll often want to process multiple sections together.
        """
        # For now, process single section
        return self._chunk_single_section(section, source_file, doc_index, [section])
    
    def chunk_sections(self, sections: List[Section], source_file: str, doc_index: int) -> List[Chunk]:
        """
        Chunk multiple sections together - preferred method for section-based chunking.
        This allows for better hierarchy analysis and cross-section optimization.
        """
        if not sections:
            return []
        
        # Build hierarchy analyzer
        hierarchy_analyzer = SectionHierarchyAnalyzer(sections)
        
        # Calculate metrics for all sections
        section_metrics = {}
        for section in sections:
            section_metrics[section.id] = SectionMetricsCalculator.calculate_metrics(
                section, hierarchy_analyzer
            )
        
        # Determine chunking strategy for each section
        chunking_plan = self._create_chunking_plan(sections, section_metrics, hierarchy_analyzer)
        
        # Execute chunking plan
        all_chunks = []
        for plan_item in chunking_plan:
            chunks = self._execute_chunking_plan_item(
                plan_item, sections, source_file, doc_index, hierarchy_analyzer
            )
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _chunk_single_section(
        self, 
        section: Section, 
        source_file: str, 
        doc_index: int,
        all_sections: List[Section]
    ) -> List[Chunk]:
        """Chunk a single section when called individually"""
        
        hierarchy_analyzer = SectionHierarchyAnalyzer(all_sections)
        metrics = SectionMetricsCalculator.calculate_metrics(section, hierarchy_analyzer)
        
        # Decide strategy based on section characteristics
        if metrics.is_small:
            return self._create_single_section_chunk(section, source_file, doc_index, hierarchy_analyzer)
        elif metrics.is_large:
            return self._split_large_section(section, source_file, doc_index, hierarchy_analyzer)
        else:
            return self._create_balanced_section_chunks(section, source_file, doc_index, hierarchy_analyzer)
    
    def _create_chunking_plan(
        self, 
        sections: List[Section], 
        metrics: Dict[str, SectionMetrics],
        hierarchy_analyzer: SectionHierarchyAnalyzer
    ) -> List[Dict]:
        """Create a plan for how to chunk all sections"""
        
        plan = []
        processed_sections = set()
        
        # Sort sections by hierarchy level first
        sorted_sections = sorted(sections, key=lambda s: (s.level, s.id))
        
        for section in sorted_sections:
            if section.id in processed_sections:
                continue
            
            section_metrics = metrics[section.id]
            
            # Determine strategy for this section
            if section_metrics.is_small and section_metrics.is_leaf:
                # Small leaf sections might be grouped with siblings
                plan_item = self._plan_small_section_grouping(
                    section, sections, metrics, hierarchy_analyzer, processed_sections
                )
            
            elif section_metrics.is_large:
                # Large sections need splitting
                plan_item = {
                    'type': 'split_large_section',
                    'section': section,
                    'strategy': 'content_aware_split'
                }
                processed_sections.add(section.id)
            
            elif not section_metrics.is_leaf:
                # Parent sections with children
                plan_item = self._plan_hierarchical_section(
                    section, sections, metrics, hierarchy_analyzer, processed_sections
                )
            
            else:
                # Medium-sized leaf sections
                plan_item = {
                    'type': 'single_section',
                    'section': section,
                    'strategy': 'balanced'
                }
                processed_sections.add(section.id)
            
            if plan_item:
                plan.append(plan_item)
        
        return plan
    
    def _plan_small_section_grouping(
        self,
        section: Section,
        all_sections: List[Section],
        metrics: Dict[str, SectionMetrics],
        hierarchy_analyzer: SectionHierarchyAnalyzer,
        processed_sections: Set[str]
    ) -> Optional[Dict]:
        """Plan grouping of small sections"""
        
        # Get siblings
        siblings = hierarchy_analyzer.get_siblings(section.id)
        
        # Find small siblings that can be grouped together
        groupable_siblings = []
        total_size = metrics[section.id].total_size
        
        for sibling in siblings:
            if (sibling.id not in processed_sections and 
                sibling.id in metrics and 
                metrics[sibling.id].is_small):
                
                if total_size + metrics[sibling.id].total_size <= self.max_section_chunk_size:
                    groupable_siblings.append(sibling)
                    total_size += metrics[sibling.id].total_size
                    processed_sections.add(sibling.id)
        
        processed_sections.add(section.id)
        
        if groupable_siblings:
            return {
                'type': 'group_small_sections',
                'sections': [section] + groupable_siblings,
                'strategy': 'sibling_grouping'
            }
        else:
            return {
                'type': 'single_section',
                'section': section,
                'strategy': 'preserve_small'
            }
    
    def _plan_hierarchical_section(
        self,
        section: Section,
        all_sections: List[Section],
        metrics: Dict[str, SectionMetrics],
        hierarchy_analyzer: SectionHierarchyAnalyzer,
        processed_sections: Set[str]
    ) -> Dict:
        """Plan chunking for sections with children"""
        
        children = hierarchy_analyzer.get_children(section.id)
        
        # Mark children as processed since we'll handle them here
        for child in children:
            processed_sections.add(child.id)
        
        processed_sections.add(section.id)
        
        if self.create_section_summaries:
            return {
                'type': 'hierarchical_with_summary',
                'parent_section': section,
                'child_sections': children,
                'strategy': 'summary_then_details'
            }
        else:
            return {
                'type': 'hierarchical_flat',
                'parent_section': section,
                'child_sections': children,
                'strategy': 'include_children'
            }
    
    def _execute_chunking_plan_item(
        self,
        plan_item: Dict,
        all_sections: List[Section],
        source_file: str,
        doc_index: int,
        hierarchy_analyzer: SectionHierarchyAnalyzer
    ) -> List[Chunk]:
        """Execute a single item from the chunking plan"""
        
        plan_type = plan_item['type']
        
        if plan_type == 'single_section':
            section = plan_item['section']
            strategy = plan_item['strategy']
            
            if strategy == 'preserve_small':
                return self._create_single_section_chunk(section, source_file, doc_index, hierarchy_analyzer)
            else:
                return self._create_balanced_section_chunks(section, source_file, doc_index, hierarchy_analyzer)
        
        elif plan_type == 'group_small_sections':
            sections = plan_item['sections']
            return self._create_grouped_sections_chunk(sections, source_file, doc_index, hierarchy_analyzer)
        
        elif plan_type == 'split_large_section':
            section = plan_item['section']
            return self._split_large_section(section, source_file, doc_index, hierarchy_analyzer)
        
        elif plan_type == 'hierarchical_with_summary':
            parent = plan_item['parent_section']
            children = plan_item['child_sections']
            return self._create_hierarchical_chunks_with_summary(
                parent, children, source_file, doc_index, hierarchy_analyzer
            )
        
        elif plan_type == 'hierarchical_flat':
            parent = plan_item['parent_section']
            children = plan_item['child_sections']
            return self._create_hierarchical_chunks_flat(
                parent, children, source_file, doc_index, hierarchy_analyzer
            )
        
        return []
    
    def _create_single_section_chunk(
        self,
        section: Section,
        source_file: str,
        doc_index: int,
        hierarchy_analyzer: SectionHierarchyAnalyzer
    ) -> List[Chunk]:
        """Create a single chunk for an entire section"""
        
        # Build content
        content_parts = []
        
        # Add section header
        section_path = " > ".join(s.title for s in hierarchy_analyzer.get_section_path(section.id))
        content_parts.append(f"[Section {section.id}: {section.title}]")
        if self.include_hierarchy_context:
            content_parts.append(f"Path: {section_path}")
        
        # Add main content
        if section.content:
            content_parts.append(section.content)
        
        # Add code blocks
        for i, code_block in enumerate(section.code_blocks):
            content_parts.append(f"\n```{code_block.language}")
            content_parts.append(code_block.content)
            content_parts.append("```")
        
        # Add tables (simplified representation)
        for i, table in enumerate(section.tables):
            table_repr = self._format_table_for_chunk(table)
            content_parts.append(table_repr)
        
        # Create metadata
        metadata = self.create_chunk_metadata(
            section=section,
            source_file=source_file,
            doc_index=doc_index,
            chunk_index=0,
            total_chunks=1,
            start_char=0,
            end_char=len('\n\n'.join(content_parts)),
            chunk_type=self._determine_section_chunk_type(section),
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
    
    def _create_grouped_sections_chunk(
        self,
        sections: List[Section],
        source_file: str,
        doc_index: int,
        hierarchy_analyzer: SectionHierarchyAnalyzer
    ) -> List[Chunk]:
        """Create a single chunk from multiple small sections"""
        
        content_parts = []
        
        # Add group header
        section_titles = [s.title for s in sections]
        content_parts.append(f"[Grouped Sections: {', '.join(section_titles)}]")
        
        # Add each section
        for section in sections:
            content_parts.append(f"\n## {section.title}")
            
            if section.content:
                content_parts.append(section.content)
            
            # Add code blocks
            for code_block in section.code_blocks:
                content_parts.append(f"\n```{code_block.language}")
                content_parts.append(code_block.content)
                content_parts.append("```")
        
        # Use first section for metadata (representative)
        primary_section = sections[0]
        
        metadata = self.create_chunk_metadata(
            section=primary_section,
            source_file=source_file,
            doc_index=doc_index,
            chunk_index=0,
            total_chunks=1,
            start_char=0,
            end_char=len('\n\n'.join(content_parts)),
            chunk_type=ChunkType.MIXED,
            has_code=any(s.code_blocks for s in sections),
            has_table=any(s.tables for s in sections)
        )
        
        chunk = Chunk(
            chunk_id="",
            content='\n\n'.join(content_parts),
            metadata=metadata
        )
        
        return [chunk]
    
    def _split_large_section(
        self,
        section: Section,
        source_file: str,
        doc_index: int,
        hierarchy_analyzer: SectionHierarchyAnalyzer
    ) -> List[Chunk]:
        """Split a large section into multiple chunks"""
        
        chunks = []
        
        # Create summary chunk if configured
        if self.create_section_summaries:
            summary_chunk = self._create_section_summary_chunk(
                section, source_file, doc_index, hierarchy_analyzer
            )
            chunks.append(summary_chunk)
        
        # Split content into logical parts
        content_parts = self._split_section_content(section)
        
        for i, part in enumerate(content_parts):
            # Adjust chunk index if we have summary
            chunk_index = i + (1 if self.create_section_summaries else 0)
            
            metadata = self.create_chunk_metadata(
                section=section,
                source_file=source_file,
                doc_index=doc_index,
                chunk_index=chunk_index,
                total_chunks=len(content_parts) + (1 if self.create_section_summaries else 0),
                start_char=part['start_pos'],
                end_char=part['end_pos'],
                chunk_type=part['type'],
                has_code=part.get('has_code', False),
                has_table=part.get('has_table', False),
                language=part.get('language')
            )
            
            chunk = Chunk(
                chunk_id="",
                content=part['content'],
                metadata=metadata
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _create_section_summary_chunk(
        self,
        section: Section,
        source_file: str,
        doc_index: int,
        hierarchy_analyzer: SectionHierarchyAnalyzer
    ) -> Chunk:
        """Create a summary chunk for a large section"""
        
        summary_parts = []
        
        # Section info
        section_path = " > ".join(s.title for s in hierarchy_analyzer.get_section_path(section.id))
        summary_parts.append(f"[Section Summary: {section.id} - {section.title}]")
        summary_parts.append(f"Path: {section_path}")
        summary_parts.append(f"Level: {section.level}")
        
        # Content preview
        if section.content:
            preview_length = min(300, len(section.content))
            preview = section.content[:preview_length]
            if len(section.content) > preview_length:
                preview += "..."
            summary_parts.append(f"\nContent Preview:\n{preview}")
        
        # Code blocks info
        if section.code_blocks:
            languages = list(set(cb.language for cb in section.code_blocks))
            summary_parts.append(f"\nContains {len(section.code_blocks)} code example(s)")
            summary_parts.append(f"Languages: {', '.join(languages)}")
        
        # Tables info
        if section.tables:
            summary_parts.append(f"\nContains {len(section.tables)} table(s)")
        
        # Subsections info
        children = hierarchy_analyzer.get_children(section.id)
        if children:
            summary_parts.append(f"\nSubsections ({len(children)}):")
            for child in children[:5]:  # Show max 5
                summary_parts.append(f"  - {child.id} {child.title}")
            if len(children) > 5:
                summary_parts.append(f"  ... and {len(children) - 5} more")
        
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
    
    def _split_section_content(self, section: Section) -> List[Dict]:
        """Split section content into logical parts"""
        parts = []
        
        # Split text content by paragraphs
        if section.content:
            paragraphs = section.content.split('\n\n')
            current_part = []
            current_size = 0
            
            for para in paragraphs:
                para_size = len(para)
                
                if current_size + para_size > self.max_section_chunk_size and current_part:
                    # Save current part
                    parts.append({
                        'content': '\n\n'.join(current_part),
                        'type': ChunkType.TEXT,
                        'start_pos': 0,  # Simplified
                        'end_pos': len('\n\n'.join(current_part))
                    })
                    
                    current_part = []
                    current_size = 0
                
                current_part.append(para)
                current_size += para_size
            
            # Add remaining paragraphs
            if current_part:
                parts.append({
                    'content': '\n\n'.join(current_part),
                    'type': ChunkType.TEXT,
                    'start_pos': 0,
                    'end_pos': len('\n\n'.join(current_part))
                })
        
        # Add code blocks as separate parts
        for code_block in section.code_blocks:
            content = f"```{code_block.language}\n{code_block.content}\n```"
            parts.append({
                'content': content,
                'type': ChunkType.CODE,
                'has_code': True,
                'language': code_block.language,
                'start_pos': 0,
                'end_pos': len(content)
            })
        
        # Add tables as separate parts
        for table in section.tables:
            table_content = self._format_table_for_chunk(table)
            parts.append({
                'content': table_content,
                'type': ChunkType.TABLE,
                'has_table': True,
                'start_pos': 0,
                'end_pos': len(table_content)
            })
        
        return parts
    
    def _create_hierarchical_chunks_with_summary(
        self,
        parent: Section,
        children: List[Section],
        source_file: str,
        doc_index: int,
        hierarchy_analyzer: SectionHierarchyAnalyzer
    ) -> List[Chunk]:
        """Create chunks for hierarchical section with summary approach"""
        
        chunks = []
        
        # Create parent summary
        summary_chunk = self._create_section_summary_chunk(
            parent, source_file, doc_index, hierarchy_analyzer
        )
        chunks.append(summary_chunk)
        
        # Create parent content chunk if it has substantial content
        if parent.content or parent.code_blocks or parent.tables:
            parent_chunks = self._create_single_section_chunk(
                parent, source_file, doc_index, hierarchy_analyzer
            )
            chunks.extend(parent_chunks)
        
        # Create chunks for children
        for child in children:
            child_chunks = self._chunk_single_section(
                child, source_file, doc_index, [parent] + children
            )
            chunks.extend(child_chunks)
        
        return chunks
    
    def _create_hierarchical_chunks_flat(
        self,
        parent: Section,
        children: List[Section],
        source_file: str,
        doc_index: int,
        hierarchy_analyzer: SectionHierarchyAnalyzer
    ) -> List[Chunk]:
        """Create chunks for hierarchical section with flat approach"""
        
        # Include parent content with first child, or create separate chunk
        all_sections = [parent] + children
        
        chunks = []
        for section in all_sections:
            section_chunks = self._chunk_single_section(
                section, source_file, doc_index, all_sections
            )
            chunks.extend(section_chunks)
        
        return chunks
    
    def _determine_section_chunk_type(self, section: Section) -> ChunkType:
        """Determine appropriate chunk type for section"""
        has_text = bool(section.content and section.content.strip())
        has_code = bool(section.code_blocks)
        has_tables = bool(section.tables)
        
        if has_code and has_text:
            return ChunkType.MIXED
        elif has_code:
            return ChunkType.CODE
        elif has_tables:
            return ChunkType.TABLE
        else:
            return ChunkType.TEXT
    
    def _format_table_for_chunk(self, table: TableData) -> str:
        """Format table for inclusion in chunk"""
        lines = []
        
        if table.caption:
            lines.append(f"Table: {table.caption}")
        
        # Simple table representation
        if table.headers:
            lines.append("| " + " | ".join(table.headers) + " |")
            lines.append("|" + "|".join([" --- " for _ in table.headers]) + "|")
        
        # Add rows (limit to first 10 for readability)
        for row in table.rows[:10]:
            if table.headers:
                # Ensure row has same number of columns
                padded_row = row + [""] * (len(table.headers) - len(row))
                padded_row = padded_row[:len(table.headers)]
            else:
                padded_row = row
            
            lines.append("| " + " | ".join(str(cell) for cell in padded_row) + " |")
        
        if len(table.rows) > 10:
            lines.append(f"... and {len(table.rows) - 10} more rows")
        
        return "\n".join(lines)
    
    def _create_balanced_section_chunks(
        self,
        section: Section,
        source_file: str,
        doc_index: int,
        hierarchy_analyzer: SectionHierarchyAnalyzer
    ) -> List[Chunk]:
        """Create balanced chunks for medium-sized sections"""
        
        # For now, use the single section approach
        # In a more advanced implementation, we might split differently
        return self._create_single_section_chunk(section, source_file, doc_index, hierarchy_analyzer)


# Factory function
def get_section_based_chunker(config: ChunkingConfig) -> SectionBasedChunker:
    """Factory function to create section-based chunker"""
    return SectionBasedChunker(config)


# Example usage
if __name__ == "__main__":
    from src.data_processing.chunk_models import ChunkingPresets
    from src.data_processing.html_parser import Section, CodeBlock
    
    # Create sample hierarchical sections
    parent_section = Section(
        id="2",
        title="Writing Algorithms",
        level=1,
        content="This section covers how to write algorithms in QuantConnect. We'll start with basic concepts and move to advanced topics.",
        section_number="2",
        breadcrumb="Documentation > Writing Algorithms"
    )
    
    child1 = Section(
        id="2.1",
        title="Getting Started",
        level=2,
        content="Let's begin by creating your first algorithm. You'll need to understand the basic structure and lifecycle of a QuantConnect algorithm.",
        section_number="2.1",
        breadcrumb="Documentation > Writing Algorithms > Getting Started",
        parent_id="2"
    )
    
    child2 = Section(
        id="2.2",
        title="Algorithm Structure",
        level=2,
        content="""Every QuantConnect algorithm must inherit from QCAlgorithm and implement the Initialize method.

Here's the basic structure:""",
        section_number="2.2",
        breadcrumb="Documentation > Writing Algorithms > Algorithm Structure",
        parent_id="2"
    )
    
    # Add code block to child2
    child2.code_blocks = [
        CodeBlock(
            language="python",
            content="""
class MyAlgorithm(QCAlgorithm):
    def Initialize(self):
        # Set the start and end dates
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 1, 1)
        
        # Set initial cash
        self.SetCash(100000)
        
        # Add equity data
        self.AddEquity("SPY", Resolution.Daily)
    
    def OnData(self, data):
        # Your trading logic goes here
        pass
            """,
            section_id="2.2"
        )
    ]
    
    # Test section-based chunker
    config = ChunkingPresets.for_documentation()
    config.max_chunk_size = 800
    config.create_section_summary_chunk = True
    
    chunker = SectionBasedChunker(config)
    
    # Test with multiple sections
    sections = [parent_section, child1, child2]
    chunks = chunker.chunk_sections(sections, "test.html", 1)
    
    print(f"Section-Based Chunker produced {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Type: {chunk.metadata.chunk_type.value}")
        print(f"  Size: {chunk.char_count} chars")
        print(f"  Section: {chunk.metadata.section_id} - {chunk.metadata.section_title}")
        print(f"  Content preview: {chunk.content[:100]}...")