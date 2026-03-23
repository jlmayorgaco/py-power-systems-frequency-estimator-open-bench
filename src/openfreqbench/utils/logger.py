"""
openfreqbench/utils/logger.py

Structured logging utility for the OpenFreqBench framework.
"""

from __future__ import annotations

import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()

def get_logger(name: str = "openfreqbench", level: int = logging.INFO) -> logging.Logger:
    """Obtain a Rich-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = RichHandler(
            console=console, 
            rich_tracebacks=True,
            show_path=False,
            log_time_format="[%X]"
        )
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger()
