"""
Comprehensive Code Validation và Testing Guide cho QuantConnect RAG System.
Kiểm tra logic của tất cả components và test với real HTML files.
"""

import sys
import json
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing.html_parser import QuantConnectHTMLParser, Section
from src.data_processing.parser_utils import count_documents_in_html
from src.data_processing.text_chunker import TextChunker, AdvancedTextChunker
from src.data_processing.code_aware_chunker import CodeAwareChunker
from src.data_processing.section_based_chunker import SectionBasedChunker
from src.data_processing.hybrid_chunker import HybridChunker
from src.data_processing.chunk_models import ChunkingConfig, ChunkingPresets
from src.utils.logger import logger
from config.config import settings

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich import print as rprint


@dataclass
class ValidationResult:
    """Result of validation test"""
    component: str
    test_name: str
    passed: bool
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0


class CodeValidator:
    """Validates all code components systematically"""
    
    def __init__(self):
        self.console = Console()
        self.results: List[ValidationResult] = []
        
    def run_full_validation(self) -> List[ValidationResult]:
        """Run complete validation suite"""
        
        self.console.print(Panel.fit(
            "[bold cyan]QuantConnect RAG System - Code Validation[/bold cyan]\n"
            "Comprehensive testing của tất cả components với real data",
            border_style="cyan"
        ))
        
        # Clear previous results
        self.results = []
        
        # Test HTML Parser
        self._test_html_parser()
        
        # Test Chunkers
        self._test_chunkers()
        
        # Test Integration
        self._test_integration()
        
        # Display results
        self._display_validation_results()
        
        return self.results
    
    def _test_html_parser(self):
        """Test HTML Parser component"""
        self.console.print("\n[yellow]🔍 Testing HTML Parser...[/yellow]")
        
        # Test 1: Basic HTML parsing
        self._test_basic_html_parsing()
        
        # Test 2: Document counting
        self._test_document_counting()
        
        # Test 3: Section extraction
        self._test_section_extraction()
        
        # Test 4: Code block extraction
        self._test_code_block_extraction()
        
        # Test 5: Error handling
        self._test_parser_error_handling()
    
    def _test_basic_html_parsing(self):
        """Test basic HTML parsing functionality"""
        test_name = "Basic HTML Parsing"
        
        try:
            start_time = datetime.now()
            
            # Find a test HTML file
            html_files = list(settings.raw_html_path.glob("*.html"))
            if not html_files:
                self.results.append(ValidationResult(
                    component="HTML Parser",
                    test_name=test_name,
                    passed=False,
                    error_message="No HTML files found in raw_html_path"
                ))
                return
            
            test_file = html_files[0]
            
            # Test parsing
            parser = QuantConnectHTMLParser(test_file)
            sections = parser.parse(target_document_index=1)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Validation checks
            if not sections:
                raise ValueError("No sections extracted from HTML")
            
            # Check section structure
            for section in sections[:3]:  # Check first 3 sections
                if not section.id:
                    raise ValueError(f"Section missing ID: {section.title}")
                if not section.title:
                    raise ValueError(f"Section missing title: {section.id}")
            
            self.results.append(ValidationResult(
                component="HTML Parser",
                test_name=test_name,
                passed=True,
                details={
                    "file_tested": test_file.name,
                    "sections_extracted": len(sections),
                    "sample_sections": [
                        {"id": s.id, "title": s.title, "level": s.level}
                        for s in sections[:3]
                    ]
                },
                execution_time=execution_time
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                component="HTML Parser",
                test_name=test_name,
                passed=False,
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
    
    def _test_document_counting(self):
        """Test document counting functionality"""
        test_name = "Document Counting"
        
        try:
            start_time = datetime.now()
            
            html_files = list(settings.raw_html_path.glob("*.html"))
            if not html_files:
                raise ValueError("No HTML files found")
            
            test_file = html_files[0]
            doc_count = count_documents_in_html(test_file)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Validation
            if doc_count < 1:
                raise ValueError("Document count should be at least 1")
            
            # Test with all files
            all_counts = {}
            for html_file in html_files:
                all_counts[html_file.name] = count_documents_in_html(html_file)
            
            self.results.append(ValidationResult(
                component="HTML Parser",
                test_name=test_name,
                passed=True,
                details={
                    "test_file": test_file.name,
                    "document_count": doc_count,
                    "all_file_counts": all_counts
                },
                execution_time=execution_time
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                component="HTML Parser",
                test_name=test_name,
                passed=False,
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
    
    def _test_section_extraction(self):
        """Test section extraction and hierarchy"""
        test_name = "Section Extraction & Hierarchy"
        
        try:
            start_time = datetime.now()
            
            html_files = list(settings.raw_html_path.glob("*.html"))
            test_file = html_files[0]
            
            parser = QuantConnectHTMLParser(test_file)
            sections = parser.parse(target_document_index=1)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Validation checks
            if not sections:
                raise ValueError("No sections extracted")
            
            # Check hierarchy
            levels = [section.level for section in sections]
            if not any(level > 1 for level in levels):
                raise ValueError("No hierarchical structure found")
            
            # Check parent-child relationships
            parent_child_pairs = [
                (s.id, s.parent_id) for s in sections if s.parent_id
            ]
            
            # Check breadcrumbs
            sections_with_breadcrumbs = [
                s for s in sections if s.breadcrumb
            ]
            
            self.results.append(ValidationResult(
                component="HTML Parser",
                test_name=test_name,
                passed=True,
                details={
                    "total_sections": len(sections),
                    "levels_found": sorted(set(levels)),
                    "parent_child_pairs": len(parent_child_pairs),
                    "sections_with_breadcrumbs": len(sections_with_breadcrumbs),
                    "sample_hierarchy": [
                        {
                            "id": s.id,
                            "title": s.title,
                            "level": s.level,
                            "parent_id": s.parent_id,
                            "breadcrumb": s.breadcrumb
                        }
                        for s in sections[:5]
                    ]
                },
                execution_time=execution_time
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                component="HTML Parser",
                test_name=test_name,
                passed=False,
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
    
    def _test_code_block_extraction(self):
        """Test code block extraction"""
        test_name = "Code Block Extraction"
        
        try:
            start_time = datetime.now()
            
            html_files = list(settings.raw_html_path.glob("*.html"))
            test_file = html_files[0]
            
            parser = QuantConnectHTMLParser(test_file)
            sections = parser.parse(target_document_index=1)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Find sections with code
            sections_with_code = [s for s in sections if s.code_blocks]
            total_code_blocks = sum(len(s.code_blocks) for s in sections)
            
            # Analyze code languages
            languages = []
            for section in sections:
                for code_block in section.code_blocks:
                    languages.append(code_block.language)
            
            language_counts = {}
            for lang in languages:
                language_counts[lang] = languages.count(lang)
            
            # Sample code blocks
            sample_code_blocks = []
            for section in sections_with_code[:3]:
                for code_block in section.code_blocks[:2]:
                    sample_code_blocks.append({
                        "section_id": section.id,
                        "language": code_block.language,
                        "content_preview": code_block.content[:100] + "..." if len(code_block.content) > 100 else code_block.content
                    })
            
            self.results.append(ValidationResult(
                component="HTML Parser",
                test_name=test_name,
                passed=True,
                details={
                    "sections_with_code": len(sections_with_code),
                    "total_code_blocks": total_code_blocks,
                    "language_distribution": language_counts,
                    "sample_code_blocks": sample_code_blocks
                },
                execution_time=execution_time
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                component="HTML Parser",
                test_name=test_name,
                passed=False,
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
    
    def _test_parser_error_handling(self):
        """Test parser error handling"""
        test_name = "Parser Error Handling"
        
        try:
            start_time = datetime.now()
            
            # Test with non-existent file
            fake_file = Path("non_existent_file.html")
            parser = QuantConnectHTMLParser(fake_file)
            
            try:
                sections = parser.parse()
                raise ValueError("Should have failed with non-existent file")
            except FileNotFoundError:
                pass  # Expected
            
            # Test with invalid document index
            html_files = list(settings.raw_html_path.glob("*.html"))
            if html_files:
                test_file = html_files[0]
                parser = QuantConnectHTMLParser(test_file)
                
                try:
                    sections = parser.parse(target_document_index=999)
                    raise ValueError("Should have failed with invalid document index")
                except (IndexError, ValueError):
                    pass  # Expected
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.results.append(ValidationResult(
                component="HTML Parser",
                test_name=test_name,
                passed=True,
                details={"error_handling": "Proper exceptions raised"},
                execution_time=execution_time
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                component="HTML Parser",
                test_name=test_name,
                passed=False,
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
    
    def _test_chunkers(self):
        """Test all chunker components"""
        self.console.print("\n[yellow]🔧 Testing Chunkers...[/yellow]")
        
        # Get test data
        test_sections = self._get_test_sections()
        if not test_sections:
            self.console.print("[red]No test sections available for chunker testing[/red]")
            return
        
        # Test each chunker
        chunkers = {
            "Basic Text Chunker": TextChunker,
            "Advanced Text Chunker": AdvancedTextChunker,
            "Code-Aware Chunker": CodeAwareChunker,
            "Section-Based Chunker": SectionBasedChunker,
            "Hybrid Chunker": HybridChunker
        }
        
        config = ChunkingPresets.for_documentation()
        
        for chunker_name, chunker_class in chunkers.items():
            self._test_single_chunker(chunker_name, chunker_class, config, test_sections)
    
    def _test_single_chunker(self, chunker_name: str, chunker_class, config: ChunkingConfig, test_sections: List[Section]):
        """Test a single chunker"""
        try:
            start_time = datetime.now()
            
            chunker = chunker_class(config)
            
            # Test with single section
            test_section = test_sections[0]
            chunks = chunker.chunk_section(test_section, "test.html", 1)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Validation checks
            if not chunks:
                raise ValueError("No chunks produced")
            
            # Check chunk structure
            for chunk in chunks:
                if not chunk.chunk_id:
                    raise ValueError("Chunk missing ID")
                if not chunk.content:
                    raise ValueError("Chunk missing content")
                if not chunk.metadata:
                    raise ValueError("Chunk missing metadata")
            
            # Check sizes
            chunk_sizes = [chunk.char_count for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            
            # Check overlaps
            overlaps = [chunk.metadata.overlap_with_next for chunk in chunks[:-1]]
            
            self.results.append(ValidationResult(
                component="Chunkers",
                test_name=chunker_name,
                passed=True,
                details={
                    "chunks_produced": len(chunks),
                    "avg_chunk_size": avg_size,
                    "size_range": [min(chunk_sizes), max(chunk_sizes)],
                    "overlaps": overlaps,
                    "chunk_types": [chunk.metadata.chunk_type.value for chunk in chunks],
                    "sample_content": chunks[0].content[:100] + "..." if chunks[0].content else ""
                },
                execution_time=execution_time
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                component="Chunkers",
                test_name=chunker_name,
                passed=False,
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
    
    def _test_integration(self):
        """Test integration between components"""
        self.console.print("\n[yellow]🔗 Testing Integration...[/yellow]")
        
        self._test_parser_to_chunker_pipeline()
        self._test_multiple_files_processing()
        self._test_memory_usage()
    
    def _test_parser_to_chunker_pipeline(self):
        """Test full pipeline from parser to chunker"""
        test_name = "Parser to Chunker Pipeline"
        
        try:
            start_time = datetime.now()
            
            # Find test file
            html_files = list(settings.raw_html_path.glob("*.html"))
            if not html_files:
                raise ValueError("No HTML files found")
            
            test_file = html_files[0]
            
            # Parse HTML
            parser = QuantConnectHTMLParser(test_file)
            sections = parser.parse(target_document_index=1)
            
            if not sections:
                raise ValueError("No sections parsed")
            
            # Chunk sections
            config = ChunkingPresets.for_documentation()
            chunker = HybridChunker(config)
            
            all_chunks = []
            for section in sections[:3]:  # Test first 3 sections
                chunks = chunker.chunk_section(section, test_file.name, 1)
                all_chunks.extend(chunks)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Validation
            if not all_chunks:
                raise ValueError("No chunks produced from pipeline")
            
            # Check metadata consistency
            source_files = set(chunk.metadata.source_file for chunk in all_chunks)
            if len(source_files) != 1:
                raise ValueError("Inconsistent source file metadata")
            
            self.results.append(ValidationResult(
                component="Integration",
                test_name=test_name,
                passed=True,
                details={
                    "file_processed": test_file.name,
                    "sections_processed": len(sections[:3]),
                    "total_chunks": len(all_chunks),
                    "pipeline_stages": ["HTML Parse", "Section Extract", "Chunking"],
                    "avg_chunks_per_section": len(all_chunks) / 3
                },
                execution_time=execution_time
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                component="Integration",
                test_name=test_name,
                passed=False,
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
    
    def _test_multiple_files_processing(self):
        """Test processing multiple files"""
        test_name = "Multiple Files Processing"
        
        try:
            start_time = datetime.now()
            
            html_files = list(settings.raw_html_path.glob("*.html"))[:2]  # Test first 2 files
            
            if len(html_files) < 2:
                raise ValueError("Need at least 2 HTML files for testing")
            
            results = {}
            
            for html_file in html_files:
                # Parse
                parser = QuantConnectHTMLParser(html_file)
                sections = parser.parse(target_document_index=1)
                
                # Chunk
                config = ChunkingPresets.for_documentation()
                chunker = HybridChunker(config)
                
                chunks = []
                for section in sections[:2]:  # Process first 2 sections per file
                    section_chunks = chunker.chunk_section(section, html_file.name, 1)
                    chunks.extend(section_chunks)
                
                results[html_file.name] = {
                    "sections": len(sections),
                    "chunks": len(chunks)
                }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.results.append(ValidationResult(
                component="Integration",
                test_name=test_name,
                passed=True,
                details={
                    "files_processed": len(html_files),
                    "processing_results": results,
                    "total_processing_time": execution_time
                },
                execution_time=execution_time
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                component="Integration",
                test_name=test_name,
                passed=False,
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
    
    def _test_memory_usage(self):
        """Test memory usage during processing"""
        test_name = "Memory Usage"
        
        try:
            import psutil
            import os
            
            start_time = datetime.now()
            process = psutil.Process(os.getpid())
            
            # Get initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process a file
            html_files = list(settings.raw_html_path.glob("*.html"))
            if html_files:
                test_file = html_files[0]
                parser = QuantConnectHTMLParser(test_file)
                sections = parser.parse(target_document_index=1)
                
                config = ChunkingPresets.for_documentation()
                chunker = HybridChunker(config)
                
                for section in sections:
                    chunks = chunker.chunk_section(section, test_file.name, 1)
            
            # Get final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.results.append(ValidationResult(
                component="Integration",
                test_name=test_name,
                passed=True,
                details={
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "memory_increase_mb": memory_increase,
                    "acceptable_increase": memory_increase < 500  # Less than 500MB increase
                },
                execution_time=execution_time
            ))
            
        except ImportError:
            self.results.append(ValidationResult(
                component="Integration",
                test_name=test_name,
                passed=False,
                error_message="psutil not available for memory testing"
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                component="Integration",
                test_name=test_name,
                passed=False,
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
    
    def _get_test_sections(self) -> List[Section]:
        """Get test sections for chunker testing"""
        try:
            html_files = list(settings.raw_html_path.glob("*.html"))
            if not html_files:
                return []
            
            test_file = html_files[0]
            parser = QuantConnectHTMLParser(test_file)
            sections = parser.parse(target_document_index=1)
            
            return sections[:3]  # Return first 3 sections
            
        except Exception as e:
            logger.error(f"Failed to get test sections: {e}")
            return []
    
    def _display_validation_results(self):
        """Display validation results in formatted table"""
        
        # Summary table
        table = Table(title="Validation Results Summary", show_header=True)
        table.add_column("Component", style="cyan")
        table.add_column("Test", style="blue")
        table.add_column("Status", style="green")
        table.add_column("Time (s)", style="yellow")
        table.add_column("Details", style="magenta")
        
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            details = "OK" if result.passed else (result.error_message or "Unknown error")
            
            table.add_row(
                result.component,
                result.test_name,
                status,
                f"{result.execution_time:.3f}",
                details[:50] + "..." if len(details) > 50 else details
            )
        
        self.console.print(table)
        
        # Summary stats
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        self.console.print(f"\n[bold]Summary:[/bold]")
        self.console.print(f"[green]✅ Passed: {passed_tests}[/green]")
        self.console.print(f"[red]❌ Failed: {failed_tests}[/red]")
        self.console.print(f"[blue]📊 Total: {total_tests}[/blue]")
        
        # Show failed tests details
        if failed_tests > 0:
            self.console.print(f"\n[red]Failed Tests Details:[/red]")
            for result in self.results:
                if not result.passed:
                    self.console.print(f"[red]• {result.component} - {result.test_name}:[/red]")
                    self.console.print(f"  Error: {result.error_message}")
    
    def save_validation_report(self, output_path: Optional[Path] = None):
        """Save validation report to file"""
        if output_path is None:
            output_path = settings.processed_data_path / "validation_report.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(self.results),
                "passed_tests": sum(1 for r in self.results if r.passed),
                "failed_tests": sum(1 for r in self.results if not r.passed)
            },
            "results": [
                {
                    "component": r.component,
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "error_message": r.error_message,
                    "details": r.details,
                    "execution_time": r.execution_time
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.console.print(f"\n[green]Validation report saved to: {output_path}[/green]")


def create_test_html_file():
    """Create a minimal test HTML file for testing"""
    test_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test Document</title>
</head>
<body>
    <h1>Test Document</h1>
    <p>This is a test document.</p>
</body>
</html>
<p style="page-break-after: always;">&nbsp;</p>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Main Document</title>
</head>
<body>
    <h3>Table of Content</h3>
    <nav>
        <ul>
            <li><a href="#1" class="toc-h1">1 Getting Started</a></li>
            <li><a href="#1.1" class="toc-h2">1.1 Installation</a></li>
            <li><a href="#2" class="toc-h1">2 Advanced Topics</a></li>
        </ul>
    </nav>
    
    <p class='page-breadcrumb'>Getting Started</p>
    <div class='page-heading'>
        <section id="1">
            <h1>Getting Started</h1>
        </section>
    </div>
    <html>
        <body>
            <p>This section covers the basics of getting started with QuantConnect.</p>
            <p>We'll walk through installation, setup, and your first algorithm.</p>
        </body>
    </html>
    
    <p class='page-breadcrumb'>Getting Started > Installation</p>
    <div class='page-heading'>
        <section id="1.1">
            <h1>Installation</h1>
        </section>
    </div>
    <html>
        <body>
            <p>Follow these steps to install QuantConnect:</p>
            <pre><code class="python">
pip install quantconnect
from QuantConnect import *

class MyAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)
            </code></pre>
        </body>
    </html>
    
    <p class='page-breadcrumb'>Advanced Topics</p>
    <div class='page-heading'>
        <section id="2">
            <h1>Advanced Topics</h1>
        </section>
    </div>
    <html>
        <body>
            <p>This section covers advanced topics like custom indicators and portfolio optimization.</p>
        </body>
    </html>
</body>
</html>'''
    
    test_file = settings.raw_html_path / "test_document.html"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_html)
    
    return test_file


def quick_validation_check():
    """Quick validation check for immediate feedback"""
    console = Console()
    
    console.print(Panel.fit(
        "[bold cyan]Quick Validation Check[/bold cyan]\n"
        "Fast check của core functionality",
        border_style="cyan"
    ))
    
    # Check if directories exist
    console.print("\n[yellow]1. Checking directories...[/yellow]")
    
    if not settings.raw_html_path.exists():
        console.print(f"[red]❌ Raw HTML path not found: {settings.raw_html_path}[/red]")
        console.print("[yellow]Creating test HTML file...[/yellow]")
        test_file = create_test_html_file()
        console.print(f"[green]✅ Created test file: {test_file}[/green]")
    else:
        console.print(f"[green]✅ Raw HTML path exists: {settings.raw_html_path}[/green]")
    
    settings.processed_data_path.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✅ Processed data path ready: {settings.processed_data_path}[/green]")
    
    # Check HTML files
    console.print("\n[yellow]2. Checking HTML files...[/yellow]")
    html_files = list(settings.raw_html_path.glob("*.html"))
    console.print(f"[blue]Found {len(html_files)} HTML files:[/blue]")
    for html_file in html_files:
        file_size = html_file.stat().st_size / 1024 / 1024  # MB
        console.print(f"  • {html_file.name} ({file_size:.1f} MB)")
    
    # Quick parse test
    if html_files:
        console.print("\n[yellow]3. Quick parse test...[/yellow]")
        try:
            test_file = html_files[0]
            
            # Test document counting
            doc_count = count_documents_in_html(test_file)
            console.print(f"[green]✅ Document count: {doc_count}[/green]")
            
            # Test parsing
            parser = QuantConnectHTMLParser(test_file)
            sections = parser.parse(target_document_index=1 if doc_count > 1 else 0)
            console.print(f"[green]✅ Sections extracted: {len(sections)}[/green]")
            
            if sections:
                sample_section = sections[0]
                console.print(f"[blue]Sample section: {sample_section.id} - {sample_section.title}[/blue]")
                
                # Test chunking
                config = ChunkingPresets.for_documentation()
                chunker = HybridChunker(config)
                chunks = chunker.chunk_section(sample_section, test_file.name, 1)
                console.print(f"[green]✅ Chunks created: {len(chunks)}[/green]")
                
            console.print("[green]🎉 Quick validation passed![/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Quick validation failed: {e}[/red]")
            console.print("[yellow]Running full validation recommended[/yellow]")
    
    else:
        console.print("[red]No HTML files found for testing[/red]")


def main():
    """Main validation function"""
    console = Console()
    
    console.print(Panel.fit(
        "[bold cyan]QuantConnect RAG System Validation[/bold cyan]\n"
        "Choose validation type",
        border_style="cyan"
    ))
    
    while True:
        console.print("\n[cyan]Validation Options:[/cyan]")
        console.print("1. Quick validation check")
        console.print("2. Full validation suite")
        console.print("3. Create test HTML file")
        console.print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ")
        
        if choice == '1':
            quick_validation_check()
        elif choice == '2':
            validator = CodeValidator()
            results = validator.run_full_validation()
            validator.save_validation_report()
        elif choice == '3':
            test_file = create_test_html_file()
            console.print(f"[green]Test HTML file created: {test_file}[/green]")
        elif choice == '4':
            break
        else:
            console.print("[red]Invalid choice![/red]")


if __name__ == "__main__":
    main()