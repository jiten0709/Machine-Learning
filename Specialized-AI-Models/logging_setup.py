"""
This module sets up a standardized logging configuration for the Nutshell.io application.
It defines a get_logger function that can be used across different modules to create loggers with both console and optional file handlers.
"""


import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str = None, level=logging.DEBUG):
   """
   Returns a logger with console + optional file handler.
  
   Args:
       name: Logger name (usually __name__)
       log_file: Optional filename in logs/ dir (e.g., "adapters.log")
       level: Logging level
   """
   logger = logging.getLogger(name)
  
   # Avoid adding duplicate handlers if called multiple times
   if logger.hasHandlers():
       return logger
  
   logger.setLevel(level)
   formatter = logging.Formatter(
       "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
       datefmt='%Y-%m-%d %H:%M:%S'
   )
  
   # Console handler
   console_handler = logging.StreamHandler(sys.stdout)
   console_handler.setLevel(level)
   console_handler.setFormatter(formatter)
   logger.addHandler(console_handler)
  
   # File handler (if specified)
   if log_file:
       log_dir = Path("logs")
       log_dir.mkdir(parents=True, exist_ok=True)
       file_handler = logging.FileHandler(log_dir / log_file, mode='w')  # 'a' for append
       file_handler.setLevel(level)
       file_handler.setFormatter(formatter)
       logger.addHandler(file_handler)
  
   return logger