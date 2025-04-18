#!/usr/bin/env python3
"""
Script to reset the training status in the changelog database using direct SQL commands.
"""

import sqlite3
import logging
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reset_db(db_path="data/changelog.db"):
    """Reset the training status in the database."""
    try:
        # Connect to the database
        logger.info(f"Connecting to database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get count of entries
        cursor.execute("SELECT COUNT(*) FROM entries")
        entry_count = cursor.fetchone()[0]
        logger.info(f"Found {entry_count} entries in the database")
        
        # Get count of training metadata entries
        cursor.execute("SELECT COUNT(*) FROM training_metadata")
        metadata_count = cursor.fetchone()[0]
        logger.info(f"Found {metadata_count} training metadata entries")
        
        # Delete all entries from training_metadata
        logger.info("Deleting all entries from training_metadata table...")
        cursor.execute("DELETE FROM training_metadata")
        
        # Get all entry IDs
        cursor.execute("SELECT id FROM entries")
        entry_ids = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(entry_ids)} entries in the database")
        
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
        return True
    
    except Exception as e:
        logger.error(f"Error resetting database: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
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
    
    success = reset_db(db_path=args.db_path)
    
    if success:
        logger.info("Database reset successfully!")
    else:
        logger.error("Failed to reset database.")

if __name__ == "__main__":
    main()
