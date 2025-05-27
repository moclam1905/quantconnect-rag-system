"""
Logger utility cho QuantConnect RAG System
Sử dụng loguru để có logging đẹp và dễ debug
"""

from loguru import logger
import sys
from pathlib import Path
from config.config import settings

def setup_logger():
    """
    Cấu hình logger cho toàn bộ hệ thống.
    - Output ra cả console và file
    - Format đẹp với màu sắc trên console
    - Rotation tự động cho file logs
    """
    
    # Remove default logger
    logger.remove()
    
    # Format cho console output - có màu sắc và dễ đọc
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    # Format cho file output - không có màu
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}"
    )
    
    # Add console handler
    logger.add(
        sys.stdout,
        format=console_format,
        level=settings.log_level,
        colorize=True,
        enqueue=True  # Thread-safe
    )
    
    # Add file handler với rotation
    logger.add(
        settings.log_file,
        format=file_format,
        level=settings.log_level,
        rotation="10 MB",  # Rotate khi file đạt 10MB
        retention="7 days",  # Giữ logs trong 7 ngày
        compression="zip",  # Compress old logs
        enqueue=True  # Thread-safe
    )
    
    # Log khởi động
    logger.info("="*50)
    logger.info("QuantConnect RAG System Logger Initialized")
    logger.info(f"Log Level: {settings.log_level}")
    logger.info(f"Log File: {settings.log_file}")
    logger.info("="*50)
    
    return logger

# Initialize logger khi import module này
setup_logger()

# Export logger instance để sử dụng trong các modules khác
__all__ = ['logger']