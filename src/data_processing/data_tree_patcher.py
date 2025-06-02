"""
DataTreePatcher - Patch HTML files with real data-tree content
Pre-processes HTML files to replace data-tree placeholders with actual content
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup, Tag
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing.data_tree_resolver import DataTreeResolver
from src.utils.logger import logger
from config.config import settings
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel


class DataTreePatcher:
    """
    Patches HTML files by replacing data-tree placeholders with real content
    """

    def __init__(self):
        self.resolver = DataTreeResolver()
        self.console = Console()
        self.patched_count = 0
        self.failed_count = 0
        self.total_data_trees = 0

    def patch_html_file(self, html_file: Path, output_file: Optional[Path] = None) -> Dict:
        """
        Patch a single HTML file with data-tree content

        Args:
            html_file: Input HTML file path
            output_file: Output file path (default: overwrite input)

        Returns:
            Dict with processing statistics
        """
        if not html_file.exists():
            raise FileNotFoundError(f"HTML file not found: {html_file}")

        # Read HTML content
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Find all data-tree elements
        data_tree_elements = self._find_data_tree_elements(html_content)

        if not data_tree_elements:
            logger.debug(f"No data-tree elements found in {html_file.name}")
            return {
                'file': html_file.name,
                'data_trees_found': 0,
                'data_trees_patched': 0,
                'data_trees_failed': 0,
                'status': 'no_data_trees'
            }

        logger.info(f"Found {len(data_tree_elements)} data-tree elements in {html_file.name}")

        # Patch each data-tree element
        patched_content = html_content
        patched_count = 0
        failed_count = 0

        for data_tree_value, original_element in data_tree_elements:
            try:
                # Resolve content for both languages
                resolved_content = self._create_language_aware_content(data_tree_value)

                if resolved_content:
                    # Replace the empty data-tree element with resolved content
                    patched_element = self._create_patched_element(data_tree_value, resolved_content)
                    patched_content = patched_content.replace(original_element, patched_element)
                    patched_count += 1
                    logger.debug(f"✅ Patched {data_tree_value}")
                else:
                    failed_count += 1
                    logger.warning(f"❌ Failed to resolve {data_tree_value}")

            except Exception as e:
                failed_count += 1
                logger.error(f"Error patching {data_tree_value}: {str(e)}")

        # Write patched content
        output_path = output_file or html_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(patched_content)

        # Update global counters
        self.patched_count += patched_count
        self.failed_count += failed_count
        self.total_data_trees += len(data_tree_elements)

        return {
            'file': html_file.name,
            'data_trees_found': len(data_tree_elements),
            'data_trees_patched': patched_count,
            'data_trees_failed': failed_count,
            'status': 'completed',
            'output_file': str(output_path)
        }

    def _find_data_tree_elements(self, html_content: str) -> List[Tuple[str, str]]:
        """
        Find all data-tree elements in HTML content

        Returns:
            List of (data_tree_value, original_element_html) tuples
        """
        # Pattern to match data-tree elements
        pattern = r'<div\s+data-tree\s*=\s*["\']([^"\']+)["\'][^>]*>.*?</div>'

        matches = []
        for match in re.finditer(pattern, html_content, re.DOTALL | re.IGNORECASE):
            data_tree_value = match.group(1)
            original_element = match.group(0)
            matches.append((data_tree_value, original_element))

        # Also check for self-closing div tags
        pattern_self_closing = r'<div\s+data-tree\s*=\s*["\']([^"\']+)["\'][^>]*/?>'

        for match in re.finditer(pattern_self_closing, html_content, re.IGNORECASE):
            data_tree_value = match.group(1)
            original_element = match.group(0)

            # Only add if not already found in the first pattern
            if not any(dt_val == data_tree_value for dt_val, _ in matches):
                matches.append((data_tree_value, original_element))

        return matches

    def _create_language_aware_content(self, data_tree_value: str) -> Optional[str]:
        """
        Create content that supports both Python and C# with language switching
        """
        # Resolve for both languages
        python_content = self.resolver.resolve_data_tree(data_tree_value, "python")
        csharp_content = self.resolver.resolve_data_tree(data_tree_value, "csharp")

        if not python_content and not csharp_content:
            return None

        # If only one language is available, use it for both
        if not python_content:
            python_content = csharp_content
        if not csharp_content:
            csharp_content = python_content

        # Create language-aware HTML structure
        content_parts = [
            '<div class="base-expandable-type section-example-container">',
            '    <div class="base-expandable-header">',
            '        <div class="base-expandable-link">',
            '            <div class="blue-text-action language-buttons" style="justify-content: space-between;">',
            f'                {self._extract_type_name(data_tree_value)}',
            '                <span>Select Language: <button class="lang-csharp">C#</button><button class="lang-python">Python</button></span>',
            '            </div>',
            '            <div class="python" data-tree-language="python" data-tree="root">',
            '                <div class="inner-tree-container section-code-wrapper">',
            f'                    {python_content}',
            '                </div>',
            '            </div>',
            '            <div class="csharp" data-tree-language="csharp" data-tree="root" style="display: none;">',
            '                <div class="inner-tree-container section-code-wrapper">',
            f'                    {csharp_content}',
            '                </div>',
            '            </div>',
            '        </div>',
            '    </div>',
            '</div>'
        ]

        return '\n'.join(content_parts)

    def _extract_type_name(self, data_tree_value: str) -> str:
        """Extract type name from full type path"""
        return data_tree_value.split('.')[-1]

    def _create_patched_element(self, data_tree_value: str, resolved_content: str) -> str:
        """
        Create the patched HTML element
        """
        return f'<div data-tree="{data_tree_value}">\n{resolved_content}\n</div>'

    def process_all_html_files(
            self,
            html_dir: Path,
            pattern: str = "*.html",
            output_dir: Optional[Path] = None,
            backup: bool = True
    ) -> Dict:
        """
        Process all HTML files in a directory

        Args:
            html_dir: Directory containing HTML files
            pattern: File pattern to match (default: "*.html")
            output_dir: Output directory (default: overwrite originals)
            backup: Whether to create backup files

        Returns:
            Processing statistics
        """
        if not html_dir.exists():
            raise FileNotFoundError(f"HTML directory not found: {html_dir}")

        # Find all HTML files
        html_files = list(html_dir.glob(pattern))

        if not html_files:
            logger.warning(f"No HTML files found in {html_dir} with pattern {pattern}")
            return {
                'files_found': 0,
                'files_processed': 0,
                'total_data_trees': 0,
                'total_patched': 0,
                'total_failed': 0
            }

        self.console.print(Panel.fit(
            f"[bold cyan]DataTree Patcher[/bold cyan]\n"
            f"Processing {len(html_files)} HTML files from {html_dir}",
            border_style="cyan"
        ))

        # Reset counters
        self.patched_count = 0
        self.failed_count = 0
        self.total_data_trees = 0

        # Create output directory if needed
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Process files with progress bar
        results = []

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console
        ) as progress:

            task = progress.add_task("Processing HTML files...", total=len(html_files))

            for html_file in html_files:
                progress.update(task, description=f"Processing {html_file.name}")

                try:
                    # Create backup if requested
                    if backup and not output_dir:
                        backup_file = html_file.with_suffix('.html.backup')
                        if not backup_file.exists():
                            with open(html_file, 'r', encoding='utf-8') as f:
                                backup_content = f.read()
                            with open(backup_file, 'w', encoding='utf-8') as f:
                                f.write(backup_content)

                    # Determine output file
                    if output_dir:
                        output_file = output_dir / html_file.name
                    else:
                        output_file = None  # Overwrite original

                    # Process file
                    result = self.patch_html_file(html_file, output_file)
                    results.append(result)

                except Exception as e:
                    logger.error(f"Error processing {html_file.name}: {str(e)}")
                    results.append({
                        'file': html_file.name,
                        'status': 'error',
                        'error': str(e)
                    })

                progress.advance(task)

        # Display results
        self._display_results(results)

        return {
            'files_found': len(html_files),
            'files_processed': len([r for r in results if r.get('status') != 'error']),
            'total_data_trees': self.total_data_trees,
            'total_patched': self.patched_count,
            'total_failed': self.failed_count,
            'resolver_stats': self.resolver.get_cache_stats(),
            'results': results
        }

    def _display_results(self, results: List[Dict]):
        """Display processing results in a nice table"""

        # Summary table
        table = Table(title="Processing Results", show_header=True)
        table.add_column("File", style="cyan")
        table.add_column("Data Trees", style="blue")
        table.add_column("Patched", style="green")
        table.add_column("Failed", style="red")
        table.add_column("Status", style="yellow")

        for result in results:
            file_name = result.get('file', 'Unknown')
            data_trees = str(result.get('data_trees_found', 0))
            patched = str(result.get('data_trees_patched', 0))
            failed = str(result.get('data_trees_failed', 0))
            status = result.get('status', 'unknown')

            # Color code status
            if status == 'completed':
                status_display = "[green]✅ Completed[/green]"
            elif status == 'no_data_trees':
                status_display = "[yellow]⚪ No data-trees[/yellow]"
            elif status == 'error':
                status_display = "[red]❌ Error[/red]"
            else:
                status_display = status

            table.add_row(file_name, data_trees, patched, failed, status_display)

        self.console.print(table)

        # Summary stats
        self.console.print(f"\n[bold]Summary:[/bold]")
        self.console.print(f"[green]✅ Total patched: {self.patched_count}[/green]")
        self.console.print(f"[red]❌ Total failed: {self.failed_count}[/red]")
        self.console.print(f"[blue]📊 Total data-trees: {self.total_data_trees}[/blue]")

        # Resolver stats
        resolver_stats = self.resolver.get_cache_stats()
        self.console.print(f"[cyan]🔄 API calls: {resolver_stats['total_calls']} "
                           f"(Success rate: {resolver_stats['success_rate']:.1%})[/cyan]")


# Main function
def main():
    """Main function to run data-tree patching"""

    console = Console()

    console.print(Panel.fit(
        "[bold cyan]QuantConnect DataTree Patcher[/bold cyan]\n"
        "Pre-process HTML files to resolve data-tree placeholders",
        border_style="cyan"
    ))

    while True:
        console.print("\n[cyan]Choose an option:[/cyan]")
        console.print("1. Patch all HTML files in raw_html_path")
        console.print("2. Patch specific file")
        console.print("3. Test with QuantConnect.Resolution")
        console.print("4. Exit")

        choice = input("\nEnter choice (1-4): ")

        if choice == '1':
            patcher = DataTreePatcher()
            stats = patcher.process_all_html_files(
                html_dir=settings.raw_html_path,
                backup=True
            )
            console.print(f"\n[green]✅ Processing complete![/green]")

        elif choice == '2':
            file_name = input("Enter HTML file name: ")
            file_path = settings.raw_html_path / file_name

            if file_path.exists():
                patcher = DataTreePatcher()
                result = patcher.patch_html_file(file_path)
                console.print(f"[green]✅ Processed {file_name}[/green]")
                console.print(f"Data trees patched: {result['data_trees_patched']}")
            else:
                console.print(f"[red]File not found: {file_path}[/red]")

        elif choice == '3':
            resolver = DataTreeResolver()
            console.print("Testing with QuantConnect.Resolution...")

            result = resolver.resolve_data_tree("QuantConnect.Resolution", "python")
            if result:
                console.print("[green]✅ Test successful![/green]")
                console.print("Content preview:")
                console.print(result[:300] + "..." if len(result) > 300 else result)
            else:
                console.print("[red]❌ Test failed![/red]")

        elif choice == '4':
            break
        else:
            console.print("[red]Invalid choice![/red]")


if __name__ == "__main__":
    main()