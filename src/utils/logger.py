"""utils/logger.py — Structured logging with Rich."""
import logging
from pathlib import Path

def setup_logging(level: str = "INFO", log_file: str | None = None):
    handlers = []
    try:
        from rich.logging import RichHandler
        handlers.append(RichHandler(rich_tracebacks=True, show_path=False, markup=True))
    except ImportError:
        handlers.append(logging.StreamHandler())

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s", datefmt="[%X]", handlers=handlers,
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
