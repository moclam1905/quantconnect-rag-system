#!/usr/bin/env python3
"""
Main script to patch HTML files with data-tree content
Run this before parsing HTML files to ensure data-tree elements are resolved

Usage:
    python patch_data_trees.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing.data_tree_patcher import DataTreePatcher
from config.config import settings
from rich.console import Console
from rich.panel import Panel


def main():
    """Main function"""
    console = Console()

    console.print(Panel.fit(
        "[bold cyan]QuantConnect HTML DataTree Patcher[/bold cyan]\n"
        f"Processing HTML files in: {settings.raw_html_path}\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        border_style="cyan"
    ))

    # Check if raw HTML directory exists
    if not settings.raw_html_path.exists():
        console.print(f"[red]❌ HTML directory not found: {settings.raw_html_path}[/red]")
        console.print("[yellow]Please ensure HTML files are in the correct directory[/yellow]")
        return False

    # Count HTML files
    html_files = list(settings.raw_html_path.glob("*.html"))
    if not html_files:
        console.print(f"[red]❌ No HTML files found in {settings.raw_html_path}[/red]")
        return False

    console.print(f"[green]📁 Found {len(html_files)} HTML files to process[/green]")

    # Ask for confirmation
    console.print("\n[yellow]This will modify HTML files in place (backups will be created).[/yellow]")
    confirm = input("Continue? (y/N): ")

    if confirm.lower() != 'y':
        console.print("[yellow]Operation cancelled.[/yellow]")
        return False

    try:
        # Create patcher and process files
        patcher = DataTreePatcher()

        stats = patcher.process_all_html_files(
            html_dir=settings.raw_html_path,
            pattern="*.html",
            backup=True  # Always create backups
        )

        # Display final summary
        console.print("\n" + "=" * 60)
        console.print("[bold green]🎉 Patching Complete![/bold green]")
        console.print("=" * 60)

        console.print(f"[blue]📊 Files processed: {stats['files_processed']}/{stats['files_found']}[/blue]")
        console.print(f"[green]✅ Data-trees patched: {stats['total_patched']}[/green]")
        console.print(f"[red]❌ Data-trees failed: {stats['total_failed']}[/red]")

        resolver_stats = stats.get('resolver_stats', {})
        if resolver_stats:
            console.print(f"[cyan]🌐 API success rate: {resolver_stats.get('success_rate', 0):.1%}[/cyan]")

        # Show next steps
        console.print("\n[yellow]📋 Next Steps:[/yellow]")
        console.print("1. Run batch_process_documents.py to parse enhanced HTML files")
        console.print("2. Check backup files (.html.backup) if you need to restore originals")
        console.print("3. Enhanced HTML files now contain resolved data-tree content")

        return True

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Operation interrupted by user[/yellow]")
        return False

    except Exception as e:
        console.print(f"\n[red]❌ Error during processing: {str(e)}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return False


def test_single_resolution():
    """Test function for QuantConnect.Resolution"""
    console = Console()

    console.print(Panel.fit(
        "[bold cyan]Testing DataTree Resolution[/bold cyan]\n"
        "Testing with QuantConnect.Resolution",
        border_style="cyan"
    ))

    try:
        from src.data_processing.data_tree_resolver import DataTreeResolver

        resolver = DataTreeResolver()

        # Test Python version
        console.print("[yellow]Testing Python version...[/yellow]")
        python_result = resolver.resolve_data_tree("QuantConnect.Resolution", "python")

        if python_result:
            console.print("[green]✅ Python resolution successful![/green]")
            console.print("Preview:")
            preview = python_result[:200] + "..." if len(python_result) > 200 else python_result
            console.print(f"[dim]{preview}[/dim]")
        else:
            console.print("[red]❌ Python resolution failed[/red]")

        # Test C# version
        console.print("\n[yellow]Testing C# version...[/yellow]")
        csharp_result = resolver.resolve_data_tree("QuantConnect.Resolution", "csharp")

        if csharp_result:
            console.print("[green]✅ C# resolution successful![/green]")
        else:
            console.print("[red]❌ C# resolution failed[/red]")

        # Show cache stats
        stats = resolver.get_cache_stats()
        console.print(f"\n[blue]Cache stats: {stats}[/blue]")

        return python_result is not None or csharp_result is not None

    except Exception as e:
        console.print(f"[red]❌ Test failed: {str(e)}[/red]")
        return False


if __name__ == "__main__":
    console = Console()

    # Check if this is a test run
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        success = test_single_resolution()
        sys.exit(0 if success else 1)

    # Normal operation
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)