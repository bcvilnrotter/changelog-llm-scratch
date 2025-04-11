"""
Changelog package initialization.
"""

import os
import logging

# Use SQLite-based logger by default, but allow reverting to JSON if needed
USE_SQLITE = True

if USE_SQLITE:
    try:
        from src.changelog.db_logger import ChangelogLogger
        logger = logging.getLogger(__name__)
        logger.info("Using SQLite-based changelog logger")
    except ImportError:
        from src.changelog.logger import ChangelogLogger
        logger = logging.getLogger(__name__)
        logger.warning("SQLite logger not available, falling back to JSON-based logger")
else:
    from src.changelog.logger import ChangelogLogger
    logger = logging.getLogger(__name__)
    logger.info("Using JSON-based changelog logger")

__all__ = ["ChangelogLogger"]