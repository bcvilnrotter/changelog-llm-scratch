#!/usr/bin/env python3
"""
Migration script to convert existing changelog.json to SQLite database.
This script reads the JSON file and populates the SQLite database with its data.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Add src directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.db.db_schema import init_db
from src.db.db_utils import (
    log_page, mark_used_in_training
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_json_data(json_path: str) -> Optional[Dict]:
    """
    Load data from the changelog JSON file.
    
    Args:
        json_path (str): Path to the changelog JSON file
    
    Returns:
        dict: The loaded JSON data
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Changelog file {json_path} not found. Creating a new database.")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {json_path}. Invalid JSON format.")
        return None

def migrate_json_to_sqlite(json_data: Dict) -> None:
    """
    Migrate data from JSON format to SQLite database.
    
    Args:
        json_data (dict): The JSON data to migrate
    """
    if not json_data or 'entries' not in json_data:
        logger.warning("No valid entries found in JSON data.")
        return
    
    entries = json_data['entries']
    logger.info(f"Found {len(entries)} entries to migrate.")
    
    # Process all entries first to create the basic records
    migrated_pages = 0
    
    for entry in entries:
        try:
            # Extract basic entry data
            title = entry['title']
            page_id = entry['page_id']
            revision_id = entry['revision_id']
            content_hash = entry['content_hash']
            action = entry.get('action', 'added')
            is_revision = entry.get('is_revision', False)
            parent_id = entry.get('parent_id')
            revision_number = entry.get('revision_number')
            
            # Create the entry in the database
            log_page(
                title, 
                page_id, 
                revision_id, 
                content_hash, 
                action, 
                is_revision, 
                parent_id, 
                revision_number
            )
            
            migrated_pages += 1
            
            if migrated_pages % 100 == 0:
                logger.info(f"Migrated {migrated_pages} entries...")
                
        except Exception as e:
            logger.error(f"Error migrating entry: {e}")
            continue
    
    logger.info(f"Successfully migrated {migrated_pages} entries.")
    
    # Now process training metadata in a second pass
    logger.info("Migrating training metadata...")
    migrated_metadata = 0
    
    for entry in entries:
        try:
            page_id = entry['page_id']
            
            # Only process entries that have been used in training
            training_metadata = entry.get('training_metadata', {})
            if not training_metadata.get('used_in_training', False):
                continue
            
            # Extract training metadata
            model_checkpoint = training_metadata.get('model_checkpoint')
            if not model_checkpoint:
                continue
                
            # Prepare metrics
            metrics_dict = {}
            metrics_dict[page_id] = {
                'average_loss': training_metadata.get('average_loss'),
                'relative_loss': training_metadata.get('relative_loss'),
                'token_impact': training_metadata.get('token_impact')
            }
            
            # Update the entry with training metadata
            mark_used_in_training([page_id], model_checkpoint, metrics_dict)
            
            migrated_metadata += 1
            
        except Exception as e:
            # Get page_id from the current entry context, or use 'unknown' if not available
            current_page_id = entry.get('page_id', 'unknown')
            logger.error(f"Error migrating training metadata for {current_page_id}: {e}")
            continue
    
    logger.info(f"Successfully migrated training metadata for {migrated_metadata} entries.")

def main():
    """
    Main function to run the migration.
    """
    parser = argparse.ArgumentParser(
        description="Migrate changelog data from JSON to SQLite"
    )
    parser.add_argument(
        "--json-path",
        default="data/changelog.json",
        help="Path to the JSON changelog file (default: data/changelog.json)"
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to the SQLite database file (default: data/changelog.db)"
    )
    args = parser.parse_args()
    
    logger.info("Starting migration process")
    
    # Initialize the SQLite database
    init_db(args.db_path)
    
    # Load JSON data
    logger.info(f"Loading JSON data from {args.json_path}")
    json_data = load_json_data(args.json_path)
    
    if json_data:
        # Migrate data to SQLite
        migrate_json_to_sqlite(json_data)
        logger.info("Migration completed successfully")
    else:
        logger.info("No data to migrate. Created empty database.")

if __name__ == "__main__":
    main()