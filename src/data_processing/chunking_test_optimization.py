"""
Comprehensive Testing và Optimization Suite cho tất cả Chunking Strategies.
Evaluate performance, quality, và optimize parameters cho QuantConnect documentation.
"""

import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing.chunk_models import (
    Chunk, ChunkingConfig, ChunkingStrategy, ChunkingPresets
)
from src.data_processing.html_parser import Section
from src.data_processing.text_chunker import TextChunker, AdvancedTextChunker
from src.data_processing.code_aware_chunker import CodeAwareChunker
from src.data_processing.section_based_chunker import SectionBasedChunker
from src.data_processing.hybrid_chunker import HybridChunker, ContentAnalyzer
from src.data_processing.chunking_config import get_chunking_config_for_file
from src.utils.logger import logger
from config.config import settings
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich import print as rprint


@dataclass
class ChunkingMetrics:
    """Comprehensive metrics for evaluating chunking quality"""
    
    # Basic statistics
    total_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    std_chunk_size: float
    
    # Size distribution
    small_chunks_count: int  # < 300 chars
    medium_chunks_count: int  # 300-1000 chars
    large_chunks_count: int  # > 1000 chars
    
    # Content preservation
    total_original_chars: int
    total_chunked_chars: int
    content_preservation_ratio: float
    
    # Overlap analysis
    avg_overlap: float
    overlap_efficiency: float  # Meaningful overlap vs redundant
    
    # Quality scores
    completeness_score: float  # Complete sentences, code blocks
    coherence_score: float     # Topic consistency within chunks
    context_score: float       # Proper context information
    overall_quality_score: float
    
    # Performance
    processing_time_seconds: float
    chunks_per_second: float
    
    # Strategy-specific
    strategy_used: str
    config_parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class ComparisonResult:
    """Result of comparing multiple chunkers"""
    chunker_metrics: Dict[str, ChunkingMetrics]
    best_chunker: str
    best_score: float
    recommendations: List[str]
    detailed_analysis: Dict[str, Any]


class ChunkQualityEvaluator:
    """Evaluates chunk quality across multiple dimensions"""
    
    def __init__(self):
        self.content_analyzer = ContentAnalyzer()
    
    def evaluate_chunks(self, chunks: List[Chunk], original_sections: List[Section]) -> Dict[str, float]:
        """Comprehensive evaluation of chunk quality"""
        
        if not chunks:
            return self._empty_evaluation()
        
        # Calculate individual scores
        completeness_score = self._evaluate_completeness(chunks)
        coherence_score = self._evaluate_coherence(chunks)
        context_score = self._evaluate_context(chunks)
        size_distribution_score = self._evaluate_size_distribution(chunks)
        content_preservation_score = self._evaluate_content_preservation(chunks, original_sections)
        overlap_score = self._evaluate_overlap_quality(chunks)
        
        # Calculate overall score (weighted average)
        weights = {
            'completeness': 0.25,
            'coherence': 0.20,
            'context': 0.15,
            'size_distribution': 0.15,
            'content_preservation': 0.15,
            'overlap': 0.10
        }
        
        overall_score = (
            weights['completeness'] * completeness_score +
            weights['coherence'] * coherence_score +
            weights['context'] * context_score +
            weights['size_distribution'] * size_distribution_score +
            weights['content_preservation'] * content_preservation_score +
            weights['overlap'] * overlap_score
        )
        
        return {
            'completeness_score': completeness_score,
            'coherence_score': coherence_score,
            'context_score': context_score,
            'size_distribution_score': size_distribution_score,
            'content_preservation_score': content_preservation_score,
            'overlap_score': overlap_score,
            'overall_quality_score': overall_score
        }
    
    def _evaluate_completeness(self, chunks: List[Chunk]) -> float:
        """Evaluate if chunks contain complete thoughts/code blocks"""
        scores = []
        
        for chunk in chunks:
            content = chunk.content
            
            # Check sentence completeness
            if content.strip():
                # Check if ends with proper punctuation or code block
                if (content.rstrip().endswith(('.', '!', '?', '```', ':', ';')) or
                    chunk.metadata.chunk_type.value == 'code'):
                    scores.append(1.0)
                elif content.rstrip().endswith(','):
                    scores.append(0.3)  # Incomplete sentence
                else:
                    scores.append(0.7)  # Partial completeness
            else:
                scores.append(0.0)
        
        return statistics.mean(scores) if scores else 0.0
    
    def _evaluate_coherence(self, chunks: List[Chunk]) -> float:
        """Evaluate topic coherence within chunks"""
        scores = []
        
        for chunk in chunks:
            words = chunk.content.lower().split()
            if len(words) < 10:
                scores.append(0.6)  # Too short to evaluate properly
                continue
            
            # Simple coherence: keyword repetition
            word_counts = Counter(w for w in words if len(w) > 3)
            most_common = word_counts.most_common(5)
            
            coherence = 0
            for word, count in most_common:
                if count > 1:
                    coherence += min(count / len(words), 0.1)
            
            scores.append(min(coherence * 10, 1.0))
        
        return statistics.mean(scores) if scores else 0.5
    
    def _evaluate_context(self, chunks: List[Chunk]) -> float:
        """Evaluate contextual information quality"""
        scores = []
        
        for chunk in chunks:
            context_score = 0
            
            # Check metadata completeness
            if chunk.metadata.section_title:
                context_score += 0.3
            if chunk.metadata.section_path:
                context_score += 0.3
            if chunk.metadata.chunk_type.value != 'text':  # Has specific type
                context_score += 0.2
            if chunk.metadata.source_file:
                context_score += 0.1
            if chunk.metadata.has_code or chunk.metadata.has_table:
                context_score += 0.1
            
            scores.append(min(context_score, 1.0))
        
        return statistics.mean(scores) if scores else 0.0
    
    def _evaluate_size_distribution(self, chunks: List[Chunk]) -> float:
        """Evaluate if chunk sizes are well-distributed"""
        sizes = [chunk.char_count for chunk in chunks]
        
        if not sizes:
            return 0.0
        
        # Optimal range: 300-1200 characters
        optimal_count = sum(1 for size in sizes if 300 <= size <= 1200)
        too_small = sum(1 for size in sizes if size < 100)
        too_large = sum(1 for size in sizes if size > 2000)
        
        # Score based on distribution
        optimal_ratio = optimal_count / len(sizes)
        penalty_ratio = (too_small + too_large) / len(sizes)
        
        return max(0, optimal_ratio - penalty_ratio * 0.5)
    
    def _evaluate_content_preservation(self, chunks: List[Chunk], original_sections: List[Section]) -> float:
        """Evaluate how well original content is preserved"""
        if not original_sections:
            return 1.0
        
        # Calculate total original content
        original_chars = sum(
            len(section.content or "") + 
            sum(len(cb.content) for cb in section.code_blocks) +
            sum(len(str(table.headers) + str(table.rows)) for table in section.tables)
            for section in original_sections
        )
        
        # Calculate total chunked content (excluding headers and metadata)
        chunked_chars = sum(chunk.char_count for chunk in chunks)
        
        if original_chars == 0:
            return 1.0
        
        # Some expansion is expected due to formatting and headers
        preservation_ratio = chunked_chars / original_chars
        
        # Score: close to 1.0-1.3 is good (some expansion expected)
        if 0.9 <= preservation_ratio <= 1.3:
            return 1.0
        elif 0.8 <= preservation_ratio < 0.9:
            return 0.8
        elif 1.3 < preservation_ratio <= 1.5:
            return 0.8
        else:
            return max(0.3, 1 - abs(preservation_ratio - 1.0))
    
    def _evaluate_overlap_quality(self, chunks: List[Chunk]) -> float:
        """Evaluate quality of overlap between chunks"""
        if len(chunks) <= 1:
            return 1.0
        
        scores = []
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Check if chunks are from same section
            if (current_chunk.metadata.section_id == next_chunk.metadata.section_id and
                current_chunk.metadata.overlap_with_next > 0):
                
                # Good overlap: provides context without too much redundancy
                overlap_chars = current_chunk.metadata.overlap_with_next
                overlap_ratio = overlap_chars / current_chunk.char_count
                
                if 0.05 <= overlap_ratio <= 0.25:  # 5-25% overlap is good
                    scores.append(1.0)
                elif overlap_ratio < 0.05:
                    scores.append(0.7)  # Too little overlap
                else:
                    scores.append(0.5)  # Too much overlap
            else:
                scores.append(0.8)  # No overlap expected
        
        return statistics.mean(scores) if scores else 1.0
    
    def _empty_evaluation(self) -> Dict[str, float]:
        """Return evaluation for empty chunks"""
        return {
            'completeness_score': 0.0,
            'coherence_score': 0.0,
            'context_score': 0.0,
            'size_distribution_score': 0.0,
            'content_preservation_score': 0.0,
            'overlap_score': 0.0,
            'overall_quality_score': 0.0
        }


class ChunkerBenchmark:
    """Benchmark different chunkers against each other"""
    
    def __init__(self):
        self.console = Console()
        self.evaluator = ChunkQualityEvaluator()
        
        # Initialize chunkers with default config
        self.base_config = ChunkingPresets.for_documentation()
        self.chunkers = {
            'basic_text': TextChunker(self.base_config),
            'advanced_text': AdvancedTextChunker(self.base_config),
            'code_aware': CodeAwareChunker(self.base_config),
            'section_based': SectionBasedChunker(self.base_config),
            'hybrid': HybridChunker(self.base_config)
        }
    
    def benchmark_chunkers(
        self, 
        sections: List[Section], 
        source_file: str = "test.html",
        doc_index: int = 1
    ) -> ComparisonResult:
        """Benchmark all chunkers on given sections"""
        
        self.console.print(Panel.fit(
            f"[bold cyan]Benchmarking Chunkers[/bold cyan]\n"
            f"Testing {len(self.chunkers)} chunkers on {len(sections)} sections",
            border_style="cyan"
        ))
        
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
        ) as progress:
            
            task = progress.add_task("Benchmarking chunkers...", total=len(self.chunkers))
            
            for chunker_name, chunker in self.chunkers.items():
                progress.update(task, description=f"Testing {chunker_name}...")
                
                try:
                    start_time = time.time()
                    
                    # Run chunker
                    if hasattr(chunker, 'chunk_sections') and len(sections) > 1:
                        chunks = chunker.chunk_sections(sections, source_file, doc_index)
                    else:
                        chunks = []
                        for section in sections:
                            section_chunks = chunker.chunk_section(section, source_file, doc_index)
                            chunks.extend(section_chunks)
                    
                    processing_time = time.time() - start_time
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(
                        chunks, sections, chunker_name, processing_time
                    )
                    
                    results[chunker_name] = metrics
                    
                except Exception as e:
                    logger.error(f"Error benchmarking {chunker_name}: {e}")
                    results[chunker_name] = self._create_error_metrics(chunker_name, str(e))
                
                progress.advance(task)
        
        # Analyze results
        comparison = self._analyze_comparison_results(results)
        
        # Display results
        self._display_benchmark_results(comparison)
        
        return comparison
    
    def _calculate_metrics(
        self, 
        chunks: List[Chunk], 
        original_sections: List[Section],
        chunker_name: str,
        processing_time: float
    ) -> ChunkingMetrics:
        """Calculate comprehensive metrics for chunks"""
        
        if not chunks:
            return self._create_empty_metrics(chunker_name, processing_time)
        
        # Basic statistics
        chunk_sizes = [chunk.char_count for chunk in chunks]
        total_chunks = len(chunks)
        avg_chunk_size = statistics.mean(chunk_sizes)
        min_chunk_size = min(chunk_sizes)
        max_chunk_size = max(chunk_sizes)
        std_chunk_size = statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0
        
        # Size distribution
        small_chunks = sum(1 for size in chunk_sizes if size < 300)
        medium_chunks = sum(1 for size in chunk_sizes if 300 <= size <= 1000)
        large_chunks = sum(1 for size in chunk_sizes if size > 1000)
        
        # Content analysis
        total_original_chars = sum(
            len(section.content or "") + 
            sum(len(cb.content) for cb in section.code_blocks)
            for section in original_sections
        )
        total_chunked_chars = sum(chunk.char_count for chunk in chunks)
        content_preservation_ratio = (
            total_chunked_chars / total_original_chars 
            if total_original_chars > 0 else 0
        )
        
        # Overlap analysis
        overlaps = [
            chunk.metadata.overlap_with_next 
            for chunk in chunks 
            if chunk.metadata.overlap_with_next > 0
        ]
        avg_overlap = statistics.mean(overlaps) if overlaps else 0
        overlap_efficiency = min(avg_overlap / 100, 1.0)  # Normalize
        
        # Quality evaluation
        quality_scores = self.evaluator.evaluate_chunks(chunks, original_sections)
        
        # Performance
        chunks_per_second = total_chunks / processing_time if processing_time > 0 else 0
        
        return ChunkingMetrics(
            total_chunks=total_chunks,
            avg_chunk_size=avg_chunk_size,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            std_chunk_size=std_chunk_size,
            small_chunks_count=small_chunks,
            medium_chunks_count=medium_chunks,
            large_chunks_count=large_chunks,
            total_original_chars=total_original_chars,
            total_chunked_chars=total_chunked_chars,
            content_preservation_ratio=content_preservation_ratio,
            avg_overlap=avg_overlap,
            overlap_efficiency=overlap_efficiency,
            completeness_score=quality_scores['completeness_score'],
            coherence_score=quality_scores['coherence_score'],
            context_score=quality_scores['context_score'],
            overall_quality_score=quality_scores['overall_quality_score'],
            processing_time_seconds=processing_time,
            chunks_per_second=chunks_per_second,
            strategy_used=chunker_name,
            config_parameters={
                'max_chunk_size': self.base_config.max_chunk_size,
                'chunk_overlap': self.base_config.chunk_overlap,
                'min_chunk_size': self.base_config.min_chunk_size
            }
        )
    
    def _analyze_comparison_results(self, results: Dict[str, ChunkingMetrics]) -> ComparisonResult:
        """Analyze and compare results from different chunkers"""
        
        if not results:
            return ComparisonResult({}, "", 0.0, [], {})
        
        # Calculate composite scores for ranking
        composite_scores = {}
        for name, metrics in results.items():
            # Weighted composite score
            composite_score = (
                metrics.overall_quality_score * 0.5 +
                min(metrics.chunks_per_second / 10, 1.0) * 0.2 +  # Performance (normalized)
                (1 - abs(metrics.content_preservation_ratio - 1.0)) * 0.2 +  # Content preservation
                (metrics.medium_chunks_count / metrics.total_chunks) * 0.1  # Good size distribution
            )
            composite_scores[name] = composite_score
        
        # Find best chunker
        best_chunker = max(composite_scores.items(), key=lambda x: x[1])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        
        # Detailed analysis
        detailed_analysis = {
            'composite_scores': composite_scores,
            'quality_ranking': sorted(
                [(name, metrics.overall_quality_score) for name, metrics in results.items()],
                key=lambda x: x[1], reverse=True
            ),
            'performance_ranking': sorted(
                [(name, metrics.chunks_per_second) for name, metrics in results.items()],
                key=lambda x: x[1], reverse=True
            ),
            'size_analysis': {
                name: {
                    'avg_size': metrics.avg_chunk_size,
                    'size_std': metrics.std_chunk_size,
                    'size_distribution': (metrics.small_chunks_count, metrics.medium_chunks_count, metrics.large_chunks_count)
                }
                for name, metrics in results.items()
            }
        }
        
        return ComparisonResult(
            chunker_metrics=results,
            best_chunker=best_chunker[0],
            best_score=best_chunker[1],
            recommendations=recommendations,
            detailed_analysis=detailed_analysis
        )
    
    def _generate_recommendations(self, results: Dict[str, ChunkingMetrics]) -> List[str]:
        """Generate recommendations based on benchmark results"""
        recommendations = []
        
        # Find best performers in different categories
        best_quality = max(results.items(), key=lambda x: x[1].overall_quality_score)
        best_performance = max(results.items(), key=lambda x: x[1].chunks_per_second)
        best_preservation = min(results.items(), key=lambda x: abs(x[1].content_preservation_ratio - 1.0))
        
        recommendations.append(f"Best Overall Quality: {best_quality[0]} (score: {best_quality[1].overall_quality_score:.3f})")
        recommendations.append(f"Best Performance: {best_performance[0]} ({best_performance[1].chunks_per_second:.1f} chunks/sec)")
        recommendations.append(f"Best Content Preservation: {best_preservation[0]} (ratio: {best_preservation[1].content_preservation_ratio:.3f})")
        
        # Content-specific recommendations
        avg_code_ratio = statistics.mean([
            metrics.total_chunked_chars / max(metrics.total_original_chars, 1)
            for metrics in results.values()
        ])
        
        if avg_code_ratio > 1.2:
            recommendations.append("Content appears code-heavy - consider CodeAware or Hybrid chunker")
        
        # Performance recommendations
        slow_chunkers = [name for name, metrics in results.items() if metrics.chunks_per_second < 1.0]
        if slow_chunkers:
            recommendations.append(f"Slow chunkers detected: {', '.join(slow_chunkers)} - consider optimization")
        
        return recommendations
    
    def _display_benchmark_results(self, comparison: ComparisonResult):
        """Display benchmark results in formatted tables"""
        
        # Main results table
        table = Table(title="Chunker Benchmark Results", show_header=True)
        table.add_column("Chunker", style="cyan")
        table.add_column("Quality Score", style="green")
        table.add_column("Total Chunks", style="blue")
        table.add_column("Avg Size", style="blue")
        table.add_column("Performance", style="yellow")
        table.add_column("Content Ratio", style="magenta")
        
        for name, metrics in comparison.chunker_metrics.items():
            table.add_row(
                name,
                f"{metrics.overall_quality_score:.3f}",
                str(metrics.total_chunks),
                f"{metrics.avg_chunk_size:.0f}",
                f"{metrics.chunks_per_second:.1f} c/s",
                f"{metrics.content_preservation_ratio:.2f}"
            )
        
        self.console.print(table)
        
        # Best chunker announcement
        self.console.print(f"\n[bold green]🏆 Best Chunker: {comparison.best_chunker}[/bold green]")
        self.console.print(f"[green]Composite Score: {comparison.best_score:.3f}[/green]")
        
        # Recommendations
        self.console.print("\n[bold yellow]📋 Recommendations:[/bold yellow]")
        for rec in comparison.recommendations:
            self.console.print(f"  • {rec}")
    
    def _create_empty_metrics(self, chunker_name: str, processing_time: float) -> ChunkingMetrics:
        """Create empty metrics for failed chunking"""
        return ChunkingMetrics(
            total_chunks=0,
            avg_chunk_size=0.0,
            min_chunk_size=0,
            max_chunk_size=0,
            std_chunk_size=0.0,
            small_chunks_count=0,
            medium_chunks_count=0,
            large_chunks_count=0,
            total_original_chars=0,
            total_chunked_chars=0,
            content_preservation_ratio=0.0,
            avg_overlap=0.0,
            overlap_efficiency=0.0,
            completeness_score=0.0,
            coherence_score=0.0,
            context_score=0.0,
            overall_quality_score=0.0,
            processing_time_seconds=processing_time,
            chunks_per_second=0.0,
            strategy_used=chunker_name,
            config_parameters={}
        )
    
    def _create_error_metrics(self, chunker_name: str, error_msg: str) -> ChunkingMetrics:
        """Create error metrics for failed chunker"""
        return self._create_empty_metrics(chunker_name, 0.0)


class ParameterOptimizer:
    """Optimize chunking parameters for specific content types"""
    
    def __init__(self):
        self.console = Console()
        self.evaluator = ChunkQualityEvaluator()
    
    def optimize_parameters(
        self, 
        sections: List[Section],
        chunker_class,
        parameter_ranges: Dict[str, List[Any]],
        max_iterations: int = 20
    ) -> Tuple[ChunkingConfig, float]:
        """Optimize parameters using grid search"""
        
        self.console.print(Panel.fit(
            f"[bold cyan]Parameter Optimization[/bold cyan]\n"
            f"Optimizing {chunker_class.__name__} parameters",
            border_style="cyan"
        ))
        
        best_config = None
        best_score = 0.0
        results = []
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(parameter_ranges)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
        ) as progress:
            
            task = progress.add_task("Optimizing parameters...", total=min(len(param_combinations), max_iterations))
            
            for i, params in enumerate(param_combinations[:max_iterations]):
                # Create config with these parameters
                config = ChunkingConfig(**params)
                
                try:
                    # Test with this config
                    chunker = chunker_class(config)
                    chunks = []
                    
                    for section in sections:
                        section_chunks = chunker.chunk_section(section, "test.html", 1)
                        chunks.extend(section_chunks)
                    
                    # Evaluate quality
                    quality_scores = self.evaluator.evaluate_chunks(chunks, sections)
                    overall_score = quality_scores['overall_quality_score']
                    
                    results.append({
                        'params': params,
                        'score': overall_score,
                        'chunks_count': len(chunks)
                    })
                    
                    if overall_score > best_score:
                        best_score = overall_score
                        best_config = config
                    
                except Exception as e:
                    logger.warning(f"Failed to test params {params}: {e}")
                
                progress.advance(task)
        
        # Display results
        self._display_optimization_results(results, best_config, best_score)
        
        return best_config, best_score
    
    def _generate_param_combinations(self, parameter_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters"""
        import itertools
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def _display_optimization_results(self, results: List[Dict], best_config: ChunkingConfig, best_score: float):
        """Display optimization results"""
        
        # Sort results by score
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # Top results table
        table = Table(title="Top Parameter Combinations", show_header=True)
        table.add_column("Rank", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Max Size", style="blue")
        table.add_column("Overlap", style="blue")
        table.add_column("Min Size", style="blue")
        table.add_column("Chunks", style="yellow")
        
        for i, result in enumerate(sorted_results[:10]):
            params = result['params']
            table.add_row(
                str(i + 1),
                f"{result['score']:.3f}",
                str(params.get('max_chunk_size', 'N/A')),
                str(params.get('chunk_overlap', 'N/A')),
                str(params.get('min_chunk_size', 'N/A')),
                str(result['chunks_count'])
            )
        
        self.console.print(table)
        
        # Best configuration
        self.console.print(f"\n[bold green]🎯 Best Configuration:[/bold green]")
        self.console.print(f"[green]Score: {best_score:.3f}[/green]")
        if best_config:
            self.console.print(f"[blue]Max Chunk Size: {best_config.max_chunk_size}[/blue]")
            self.console.print(f"[blue]Chunk Overlap: {best_config.chunk_overlap}[/blue]")
            self.console.print(f"[blue]Min Chunk Size: {best_config.min_chunk_size}[/blue]")


class ChunkingTestSuite:
    """Main test suite for all chunking functionality"""
    
    def __init__(self):
        self.console = Console()
        self.benchmark = ChunkerBenchmark()
        self.optimizer = ParameterOptimizer()
    
    def run_comprehensive_tests(self, test_data_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        
        self.console.print(Panel.fit(
            "[bold cyan]QuantConnect Chunking Test Suite[/bold cyan]\n"
            "Comprehensive testing và optimization của tất cả chunking strategies",
            border_style="cyan"
        ))
        
        results = {}
        
        # Load test data
        test_sections = self._load_test_data(test_data_dir)
        
        if not test_sections:
            self.console.print("[red]No test data available![/red]")
            return results
        
        # 1. Benchmark all chunkers
        self.console.print("\n[yellow]1. Benchmarking All Chunkers[/yellow]")
        benchmark_results = self.benchmark.benchmark_chunkers(test_sections)
        results['benchmark'] = benchmark_results
        
        # 2. Parameter optimization for best chunker
        self.console.print("\n[yellow]2. Parameter Optimization[/yellow]")
        best_chunker_name = benchmark_results.best_chunker
        
        if best_chunker_name == 'hybrid':
            chunker_class = HybridChunker
        elif best_chunker_name == 'code_aware':
            chunker_class = CodeAwareChunker
        elif best_chunker_name == 'section_based':
            chunker_class = SectionBasedChunker
        else:
            chunker_class = AdvancedTextChunker
        
        # Define parameter ranges for optimization
        param_ranges = {
            'max_chunk_size': [800, 1000, 1200, 1500],
            'chunk_overlap': [100, 150, 200, 250],
            'min_chunk_size': [100, 150, 200]
        }
        
        optimal_config, optimal_score = self.optimizer.optimize_parameters(
            test_sections, chunker_class, param_ranges
        )
        
        results['optimization'] = {
            'best_chunker': best_chunker_name,
            'optimal_config': optimal_config.to_dict() if optimal_config else None,
            'optimal_score': optimal_score
        }
        
        # 3. Generate final recommendations
        recommendations = self._generate_final_recommendations(results)
        results['recommendations'] = recommendations
        
        # Display final summary
        self._display_final_summary(results)
        
        # Save results
        self._save_test_results(results)
        
        return results
    
    def _load_test_data(self, test_data_dir: Optional[Path]) -> List[Section]:
        """Load test data from parsed JSON files"""
        
        if test_data_dir is None:
            # Try to find parsed data
            processed_dirs = list(settings.processed_data_path.glob("batch_*"))
            if processed_dirs:
                test_data_dir = max(processed_dirs, key=lambda p: p.stat().st_mtime)
            else:
                return []
        
        # Load sections from JSON files
        sections = []
        json_files = list(test_data_dir.glob("*_parsed.json"))
        
        for json_file in json_files[:2]:  # Limit to 2 files for testing
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for section_dict in data.get('sections', [])[:5]:  # Limit sections per file
                    section = Section(
                        id=section_dict['id'],
                        title=section_dict['title'],
                        level=section_dict['level'],
                        content=section_dict.get('content', ''),
                        section_number=section_dict.get('section_number'),
                        breadcrumb=section_dict.get('breadcrumb'),
                        parent_id=section_dict.get('parent_id')
                    )
                    sections.append(section)
                    
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        self.console.print(f"[green]Loaded {len(sections)} test sections[/green]")
        return sections
    
    def _generate_final_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate final recommendations based on all test results"""
        recommendations = []
        
        if 'benchmark' in results:
            benchmark = results['benchmark']
            recommendations.append(f"Recommended chunker: {benchmark.best_chunker}")
            recommendations.extend(benchmark.recommendations)
        
        if 'optimization' in results and results['optimization']['optimal_config']:
            opt_config = results['optimization']['optimal_config']
            recommendations.append(f"Optimal max_chunk_size: {opt_config.get('max_chunk_size', 1000)}")
            recommendations.append(f"Optimal chunk_overlap: {opt_config.get('chunk_overlap', 150)}")
        
        # General recommendations
        recommendations.append("For code-heavy content: Use CodeAware or Hybrid chunker")
        recommendations.append("For structured documentation: Use SectionBased chunker")
        recommendations.append("For mixed content: Use Hybrid chunker")
        recommendations.append("For API reference: Use smaller chunk sizes (800-1000 chars)")
        
        return recommendations
    
    def _display_final_summary(self, results: Dict[str, Any]):
        """Display final test summary"""
        
        self.console.print("\n" + "="*60)
        self.console.print("[bold green]🎉 Test Suite Complete![/bold green]")
        self.console.print("="*60)
        
        if 'benchmark' in results:
            best_chunker = results['benchmark'].best_chunker
            best_score = results['benchmark'].best_score
            self.console.print(f"[green]Best Chunker: {best_chunker} (score: {best_score:.3f})[/green]")
        
        if 'optimization' in results:
            opt_score = results['optimization']['optimal_score']
            self.console.print(f"[blue]Optimized Score: {opt_score:.3f}[/blue]")
        
        self.console.print("\n[yellow]Final Recommendations:[/yellow]")
        for rec in results.get('recommendations', []):
            self.console.print(f"  ✓ {rec}")
    
    def _save_test_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        
        output_dir = settings.processed_data_path / "test_results"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"chunking_test_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for key, value in results.items():
            if hasattr(value, 'to_dict'):
                serializable_results[key] = value.to_dict()
            elif isinstance(value, dict):
                serializable_results[key] = {
                    k: v.to_dict() if hasattr(v, 'to_dict') else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
            
            self.console.print(f"\n[green]Results saved to: {output_file}[/green]")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


# Main function
def main():
    """Main testing function"""
    test_suite = ChunkingTestSuite()
    results = test_suite.run_comprehensive_tests()
    return results


if __name__ == "__main__":
    main()