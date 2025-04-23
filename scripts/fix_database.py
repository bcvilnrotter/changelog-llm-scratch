#!/usr/bin/env python3
"""
Script to fix the database schema by adding missing columns.
"""

import sqlite3
import logging
import sys
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_database_schema(db_path="data/changelog.db", debug=False):
    """
    Fix the database schema by adding missing columns.
    
    Args:
        db_path: Path to the database file
        debug: Enable debug logging
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Connect to the database
        logger.info(f"Connecting to database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get current schema of training_metadata table
        cursor.execute("PRAGMA table_info(training_metadata)")
        columns = [col[1] for col in cursor.fetchall()]
        logger.info(f"Current columns in training_metadata: {columns}")
        
        # Check if training_timestamp column exists
        if 'training_timestamp' not in columns:
            logger.info("Adding missing column 'training_timestamp' to training_metadata table")
            cursor.execute("ALTER TABLE training_metadata ADD COLUMN training_timestamp TEXT")
            logger.info("Column 'training_timestamp' added successfully")
        else:
            logger.info("Column 'training_timestamp' already exists")
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        logger.info("Database schema fixed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing database schema: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix database schema by adding missing columns"
    )
    parser.add_argument(
        "--db-path",
        default="data/changelog.db",
        help="Path to the changelog database"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    success = fix_database_schema(
        db_path=args.db_path,
        debug=args.debug
    )
    
    if success:
        logger.info("Database schema fixed successfully!")
        sys.exit(0)
    else:
        logger.error("Failed to fix database schema.")
        sys.exit(1)

if __name__ == "__main__":
    main()
