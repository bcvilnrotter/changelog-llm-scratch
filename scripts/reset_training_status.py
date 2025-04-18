#!/usr/bin/env python3
"""
Script to reset the training status in the changelog database.
This script deletes all entries from the training_metadata table and recreates them
with used_in_training=0.
"""

import sys
import logging
import sqlite3
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.db.changelog_db import ChangelogDB

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reset_training_status(db_path="data/changelog.db", debug=False):
    """
    Reset the training status of all pages in the database.
    
    Args:
        db_path: Path to the changelog database
        debug: Enable debug logging
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # First, get some stats before reset for reporting
        db = ChangelogDB(db_path, debug=debug)
        
        # Get all pages
        all_pages = db.get_main_pages()
        total_pages = len(all_pages)
        
        # Get unused pages
        unused_pages = db.get_unused_pages()
        unused_count = len(unused_pages)
        used_count = total_pages - unused_count
        
        logger.info(f"Database stats before reset:")
        logger.info(f"- Total pages: {total_pages}")
        logger.info(f"- Pages marked as used in training: {used_count}")
        logger.info(f"- Pages not yet used in training: {unused_count}")
        
        # Connect to the database directly
        logger.info(f"Connecting to database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if training_metadata table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_metadata'")
        if not cursor.fetchone():
            logger.warning("No training_metadata table found in database. Nothing to reset.")
            conn.close()
            return True
        
        # Get count of training metadata entries
        cursor.execute("SELECT COUNT(*) FROM training_metadata")
        metadata_count = cursor.fetchone()[0]
        logger.info(f"Found {metadata_count} training metadata entries")
        
        # Get all entry IDs
        cursor.execute("SELECT id FROM entries")
        entry_ids = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(entry_ids)} entries in the database")
        
        # Delete all entries from training_metadata
        logger.info("Deleting all entries from training_metadata table...")
        cursor.execute("DELETE FROM training_metadata")
        
        # Create new training_metadata entries with used_in_training=0
        logger.info("Creating new training_metadata entries with used_in_training=0...")
        for entry_id in entry_ids:
            cursor.execute('''
                INSERT INTO training_metadata (entry_id, used_in_training)
                VALUES (?, 0)
            ''', (entry_id,))
        
        # Verify reset
        cursor.execute("SELECT COUNT(*) FROM training_metadata WHERE used_in_training = 0")
        new_count = cursor.fetchone()[0]
        logger.info(f"Training metadata entries after reset: {new_count}")
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        logger.info(f"Successfully reset training status for all {new_count} entries")
        logger.info(f"All {total_pages} pages are now marked as unused and available for training")
        
        return True
    
    except Exception as e:
        logger.error(f"Error resetting training status: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reset training status in the changelog database"
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
    
    success = reset_training_status(
        db_path=args.db_path,
        debug=args.debug
    )
    
    if success:
        logger.info("Training status reset successfully!")
        sys.exit(0)
    else:
        logger.error("Failed to reset training status.")
        sys.exit(1)

if __name__ == "__main__":
    main()
