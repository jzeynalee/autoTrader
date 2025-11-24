# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

import datetime
import logging 
import os

import logging
from pathlib import Path

def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> logging.Logger:
    """
    Configure file-based logging for regime detection system.
    
    Args:
        log_dir: Directory for log files (created if doesn't exist)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"regime_detection_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger('RegimeDetection')
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-30s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Optional: Console handler for errors only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info("="*80)
    logger.info("Regime Detection Logging System Initialized")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")
    logger.info("="*80)
    
    return logger