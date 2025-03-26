#!/usr/bin/env python3
"""
Migration script to convert existing changelog.json to SQLite database.
This is a wrapper around src/db/migrate_to_sqlite.py for easy access.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the migration.
    """
    try:
        from src.db.migrate_to_sqlite import main as migrate_main
        migrate_main()
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you're running this script from the project root.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()