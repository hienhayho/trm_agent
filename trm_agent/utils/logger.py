"""Custom logger with color support and distributed environment awareness."""

import logging
import sys
from typing import Optional

from trm_agent.utils.ddp import is_main_process


class ColorCode:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Level colors
    DEBUG = "\033[36m"  # Cyan
    INFO = "\033[32m"  # Green
    WARNING = "\033[33m"  # Yellow
    ERROR = "\033[31m"  # Red
    CRITICAL = "\033[35m"  # Magenta

    # Other colors
    TIME = "\033[90m"  # Gray
    LOCATION = "\033[34m"  # Blue


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and location info."""

    LEVEL_COLORS = {
        logging.DEBUG: ColorCode.DEBUG,
        logging.INFO: ColorCode.INFO,
        logging.WARNING: ColorCode.WARNING,
        logging.ERROR: ColorCode.ERROR,
        logging.CRITICAL: ColorCode.CRITICAL,
    }

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        # Time format: h-m-s d-m-y
        time_str = self.formatTime(record, "%H-%M-%S %d-%m-%y")

        # Level name
        level_name = record.levelname.lower()

        # Location: file_name.function:line
        location = f"{record.module}.{record.funcName}:{record.lineno}"

        # Content
        content = record.getMessage()

        if self.use_color:
            level_color = self.LEVEL_COLORS.get(record.levelno, ColorCode.RESET)
            formatted = (
                f"{ColorCode.TIME}{time_str}{ColorCode.RESET} - "
                f"{level_color}{ColorCode.BOLD}{level_name}{ColorCode.RESET} - "
                f"{ColorCode.LOCATION}[{location}]{ColorCode.RESET} - "
                f"{content}"
            )
        else:
            formatted = f"{time_str} - {level_name} - [{location}] - {content}"

        # Handle exception info
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            formatted += f"\n{record.exc_text}"

        return formatted


def get_logger(
    name: str = "trm_agent",
    level: int = logging.INFO,
    use_color: bool = True,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name
        level: Logging level
        use_color: Whether to use colored output
        log_file: Optional file path to write logs

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Only add handlers on main process
    if is_main_process():
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter(use_color=use_color))
        logger.addHandler(console_handler)

        # File handler without colors (optional)
        if log_file:
            from pathlib import Path

            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(ColoredFormatter(use_color=False))
            logger.addHandler(file_handler)
    else:
        # Add null handler for non-main processes
        logger.addHandler(logging.NullHandler())

    return logger


# Default logger instance
logger = get_logger()
