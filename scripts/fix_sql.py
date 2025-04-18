#!/usr/bin/env python3
"""
Script to fix the SQL query issue in the database.
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

def get_main_pages_fixed(db_path="data/changelog.db"):
    """
    Get all non-revision pages with fixed SQL query.
    
    Returns:
        List[Dict]: List of main page entries (excluding revisions)
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Use a simpler query without the problematic columns
        query = '''
            SELECT e.*, tm.used_in_training
            FROM entries e
            LEFT JOIN training_metadata tm ON e.id = tm.entry_id
            WHERE e.is_revision = 0
        '''
        
        logger.info(f"Executing fixed query: {query}")
        cursor.execute(query)
        
        pages = []
        for row in cursor.fetchall():
            page = dict(row)
            # If training_metadata values are NULL, set defaults
            if page.get("used_in_training") is None:
                page["used_in_training"] = 0
            if page.get("training_timestamp") is None:
                page["training_timestamp"] = None
            if page.get("model_checkpoint") is None:
                page["model_checkpoint"] = None
            if page.get("average_loss") is None:
                page["average_loss"] = None
            if page.get("relative_loss") is None:
                page["relative_loss"] = None
            pages.append(page)
        
        logger.info(f"Found {len(pages)} main pages")
        conn.close()
        
        return pages
    except Exception as e:
        logger.error(f"Error in get_main_pages_fixed: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        # Return an empty list to avoid breaking the caller
        return []

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix SQL query issue in the database"
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
    
    # Get main pages with fixed query
    pages = get_main_pages_fixed(args.db_path)
    
    # Print results
    logger.info(f"Found {len(pages)} main pages")
    
    # Print first 5 pages
    if len(pages) > 0:
        logger.info("First 5 pages:")
        for i, page in enumerate(pages[:5]):
            logger.info(f"Page {i+1}: {page.get('title', 'No title')} (ID: {page.get('page_id', 'No ID')})")

if __name__ == "__main__":
    main()
