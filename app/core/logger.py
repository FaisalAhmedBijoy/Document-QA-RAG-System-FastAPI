import os
import logging
from datetime import datetime

def configure_logging(logger_name: str = "fastapi_app"):
    logger = logging.getLogger(logger_name)

    # Avoid duplicate handlers during reload
    if logger.handlers:
        return logger

    # Ensure log directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Generate file name with current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"log_file_{current_date}.log"
    log_file_path = os.path.join(log_dir, log_filename)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File handler with daily log file
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)

    return logger
