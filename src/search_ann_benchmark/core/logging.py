"""Logging configuration for search-ann-benchmark."""

import logging
import sys

LOGGER_NAME = "search_ann_benchmark"


def setup_logging(debug: bool = False) -> logging.Logger:
    """Set up logging configuration.

    Args:
        debug: If True, set log level to DEBUG. Otherwise, INFO.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(LOGGER_NAME)
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Format with timestamp
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Sub-logger name (e.g., "runner", "docker").
              If None, returns the root package logger.

    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)
