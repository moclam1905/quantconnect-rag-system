"""
Script kiểm tra xem môi trường đã được setup đúng chưa
Chạy script này để đảm bảo mọi thứ hoạt động trước khi bắt đầu coding
"""

import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Add src to path để import được các modules
sys.path.append(str(Path(__file__).parent))

console = Console()

def check_python_version():
    """Kiểm tra phiên bản Python"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (Cần >= 3.8)"

def check_imports():
    """Kiểm tra các thư viện đã cài đặt"""
    libraries = [
        ("langchain", "LangChain Core"),
        ("langchain_community", "LangChain Community"),
        ("langchain_google_genai", "LangChain Google GenAI"),
        ("chromadb", "ChromaDB Vector Database"),
        ("bs4", "BeautifulSoup4"),
        ("fastapi", "FastAPI"),
        ("loguru", "Loguru Logger"),
        ("dotenv", "Python Dotenv")
    ]
    
    results = []
    for lib_import, lib_name in libraries:
        try:
            __import__(lib_import)
            results.append((lib_name, True, "✓ Installed"))
        except ImportError as e:
            results.append((lib_name, False, f"✗ Missing: {str(e)}"))
    
    return results

def check_environment():
    """Kiểm tra file .env và các biến môi trường"""
    checks = []
    
    # Kiểm tra file .env tồn tại
    env_file = Path(".env")
    if env_file.exists():
        checks.append((".env file", True, "✓ Found"))
    else:
        checks.append((".env file", False, "✗ Not found"))
        return checks
    
    # Load config và kiểm tra
    try:
        from config.config import settings
        
        # Kiểm tra API key
        if settings.validate_api_key():
            checks.append(("Google API Key", True, "✓ Configured"))
        else:
            checks.append(("Google API Key", False, "✗ Not set or invalid"))
        
        # Kiểm tra các paths
        checks.append(("ChromaDB Path", True, f"✓ {settings.chroma_db_path}"))
        checks.append(("Raw HTML Path", True, f"✓ {settings.raw_html_path}"))
        
    except Exception as e:
        checks.append(("Config Loading", False, f"✗ Error: {str(e)}"))
    
    return checks

def check_directories():
    """Kiểm tra cấu trúc thư mục"""
    required_dirs = [
        "src/data_processing",
        "src/rag_service", 
        "src/agents",
        "src/utils",
        "data/raw_html",
        "data/processed",
        "data/vector_db",
        "config",
        "logs"
    ]
    
    results = []
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            results.append((dir_path, True, "✓ Exists"))
        else:
            results.append((dir_path, False, "✗ Missing"))
    
    return results

def check_html_files():
    """Kiểm tra các file HTML cần xử lý"""
    required_files = [
        "Quantconnect-Lean-Cli.html",
        "Quantconnect-Lean-Engine.html",
        "Quantconnect-Research-Environment.html",
        "Quantconnect-Writing-Algorithms.html"
    ]
    
    results = []
    html_dir = Path("data/raw_html")
    
    for file_name in required_files:
        file_path = html_dir / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            results.append((file_name, True, f"✓ {size_mb:.1f} MB"))
        else:
            results.append((file_name, False, "✗ Not found"))
    
    return results

def main():
    """Chạy tất cả các kiểm tra và hiển thị kết quả"""
    console.print(Panel.fit(
        "[bold cyan]QuantConnect RAG System - Environment Check[/bold cyan]",
        border_style="cyan"
    ))
    
    all_checks_passed = True
    
    # 1. Python Version
    table = Table(title="Python Version Check", show_header=True)
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    
    passed, version = check_python_version()
    all_checks_passed &= passed
    table.add_row("Python Version", version)
    console.print(table)
    console.print()
    
    # 2. Libraries
    table = Table(title="Required Libraries", show_header=True)
    table.add_column("Library", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")
    
    for lib_name, passed, details in check_imports():
        all_checks_passed &= passed
        status_style = "green" if passed else "red"
        table.add_row(lib_name, "✓" if passed else "✗", f"[{status_style}]{details}[/{status_style}]")
    console.print(table)
    console.print()
    
    # 3. Environment Configuration
    table = Table(title="Environment Configuration", show_header=True)
    table.add_column("Configuration", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")
    
    for config_name, passed, details in check_environment():
        all_checks_passed &= passed
        status_style = "green" if passed else "red"
        table.add_row(config_name, "✓" if passed else "✗", f"[{status_style}]{details}[/{status_style}]")
    console.print(table)
    console.print()
    
    # 4. Directory Structure
    table = Table(title="Directory Structure", show_header=True)
    table.add_column("Directory", style="cyan")
    table.add_column("Status", style="green")
    
    for dir_name, passed, details in check_directories():
        all_checks_passed &= passed
        status_style = "green" if passed else "red"
        table.add_row(dir_name, f"[{status_style}]{details}[/{status_style}]")
    console.print(table)
    console.print()
    
    # 5. HTML Files
    table = Table(title="QuantConnect HTML Files", show_header=True)
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Size")
    
    for file_name, passed, details in check_html_files():
        # Không bắt buộc phải có HTML files ngay từ đầu
        status_style = "green" if passed else "yellow"
        table.add_row(file_name, "✓" if passed else "⚠", f"[{status_style}]{details}[/{status_style}]")
    console.print(table)
    console.print()
    
    # Final Summary
    if all_checks_passed:
        console.print(Panel.fit(
            "[bold green]✓ All checks passed! Environment is ready.[/bold green]",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            "[bold red]✗ Some checks failed. Please fix the issues above.[/bold red]",
            border_style="red"
        ))
        
        # Gợi ý sửa lỗi
        console.print("\n[yellow]Suggestions:[/yellow]")
        console.print("1. Ensure you've run: pip install -r requirements.txt")
        console.print("2. Copy .env.example to .env and add your Google API key")
        console.print("3. Place HTML files in data/raw_html/ directory")
        console.print("4. Run: python check_setup.py again")

if __name__ == "__main__":
    main()