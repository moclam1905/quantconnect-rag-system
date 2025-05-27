"""
Configuration loader cho QuantConnect RAG System
Sử dụng pydantic để validate và type-check các config values
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os
from pathlib import Path

def find_env_file(start: Path, filename: str = ".env") -> Path:
    current = start.resolve()
    while not (current / filename).exists():
        if current == current.parent:
            raise FileNotFoundError(f"{filename} not found from {start} upward.")
        current = current.parent
    return current / filename

ENV_PATH = find_env_file(Path(__file__))

class Settings(BaseSettings):
    """
    Quản lý tất cả các cấu hình của hệ thống.
    Tự động load từ file .env và có thể override bằng environment variables.
    """

    # Google Gemini API Configuration
    google_api_key: str = Field(..., description="Google Gemini API key")
    gemini_model: str = Field(default="gemini-1.5-pro", description="Model Gemini để dùng")
    embedding_model: str = Field(default="models/text-embedding-004", description="Model embedding của Google")

    # ChromaDB Settings
    chroma_db_path: Path = Field(default=Path("./data/vector_db"), description="Đường dẫn lưu ChromaDB")
    chroma_collection_name: str = Field(default="quantconnect_docs", description="Tên collection trong ChromaDB")

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")
    log_file: Path = Field(default=Path("./logs/rag_system.log"), description="Đường dẫn file log")

    # RAG Configuration
    chunk_size: int = Field(default=1000, description="Kích thước mỗi chunk text")
    chunk_overlap: int = Field(default=200, description="Số ký tự overlap giữa các chunks")
    top_k_results: int = Field(default=5, description="Số lượng chunks trả về khi search")

    # Cache Settings
    enable_cache: bool = Field(default=True, description="Bật/tắt cache system")
    cache_ttl_hours: int = Field(default=24, description="Thời gian cache tồn tại (giờ)")

    # API Settings
    api_host: str = Field(default="localhost", description="API host")
    api_port: int = Field(default=8000, description="API port")

    # Data Paths
    raw_html_path: Path = Field(default=Path("./data/raw_html"), description="Thư mục chứa HTML files")
    processed_data_path: Path = Field(default=Path("./data/processed"), description="Thư mục chứa processed data")

    class Config:
        env_file = str(ENV_PATH)
        env_file_encoding = 'utf-8'
        # Cho phép đọc từ cả .env file và environment variables
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Tạo các thư mục cần thiết nếu chưa tồn tại
        self._create_directories()

    def _create_directories(self):
        """Tự động tạo các thư mục cần thiết"""
        directories = [
            self.chroma_db_path,
            self.log_file.parent,
            self.raw_html_path,
            self.processed_data_path
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def validate_api_key(self) -> bool:
        """Kiểm tra API key có hợp lệ không"""
        return self.google_api_key !=  ""

# Singleton instance để sử dụng trong toàn bộ app
settings = Settings()

if __name__ == "__main__":
    # Test load config
    print("=== Configuration Loaded ===")
    print(f"Gemini Model: {settings.gemini_model}")
    print(f"ChromaDB Path: {settings.chroma_db_path}")
    print(f"Chunk Size: {settings.chunk_size}")
