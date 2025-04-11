#!/usr/bin/env python3
"""
Migration script to convert existing changelog.json to SQLite database.
This is a direct implementation of the migration functionality.
"""

import json
import os
import sys
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to the original changelog.json file
CHANGELOG_JSON_PATH = os.environ.get("CHANGELOG_JSON_PATH", "changelog.json")
CHANGELOG_DB_PATH = os.environ.get("CHANGELOG_DB_PATH", "changelog.db")

def get_db_connection():
    """Create and return a connection to the SQLite database."""
    conn = sqlite3.connect(CHANGELOG_DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

def init_db():
    """Initialize the database with the required schema."""
    logger.info(f"Initializing database at {CHANGELOG_DB_PATH}")
    conn = get_db_connection()
    
    try:
        cursor = conn.cursor()
        
        # Create entries table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            page_id TEXT NOT NULL,
            revision_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            action TEXT DEFAULT 'added',
            timestamp TEXT DEFAULT (datetime('now')),
            is_revision BOOLEAN DEFAULT 0,
            parent_id TEXT,
            revision_number INTEGER,
            UNIQUE(page_id, revision_id)
        )
        ''')
        
        # Create training_metadata table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id INTEGER NOT NULL,
            used_in_training BOOLEAN DEFAULT 0,
            model_checkpoint TEXT,
            average_loss REAL,
            relative_loss REAL,
            FOREIGN KEY (entry_id) REFERENCES entries (id)
        )
        ''')
        
        # Create token_impact table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS token_impact (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metadata_id INTEGER NOT NULL,
            token TEXT NOT NULL,
            impact REAL NOT NULL,
            FOREIGN KEY (metadata_id) REFERENCES training_metadata (id)
        )
        ''')
        
        conn.commit()
        logger.info("Database initialization completed successfully")
    except sqlite3.Error as e:
        conn.rollback()
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        conn.close()

def load_json_data(json_path):
    """
    Load data from the changelog JSON file.
    
    Args:
        json_path (str): Path to the changelog JSON file
    
    Returns:
        dict: The loaded JSON data
    """
    logger.info(f"Loading JSON data from {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON data with {len(data.get('entries', []))} entries")
        return data
    except FileNotFoundError:
        logger.warning(f"Changelog file {json_path} not found. Creating a new database.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        return None

def migrate_json_to_sqlite(json_data):
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
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
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
                is_revision = 1 if entry.get('is_revision', False) else 0
                parent_id = entry.get('parent_id')
                revision_number = entry.get('revision_number')
                
                # Extract timestamp if available
                timestamp = entry.get('timestamp', datetime.now().isoformat())
                
                # Insert entry and get the ID (lastrowid will be available if INSERT occurred)
                cursor.execute('''
                INSERT OR REPLACE INTO entries 
                (title, page_id, revision_id, content_hash, action, timestamp, is_revision, parent_id, revision_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    title, page_id, revision_id, content_hash, action, timestamp,
                    is_revision, parent_id, revision_number
                ))
                
                # Get the entry ID
                entry_id = cursor.lastrowid
                
                # If lastrowid is not available (e.g., due to OR IGNORE), query for the ID
                if not entry_id:
                    cursor.execute(
                        'SELECT id FROM entries WHERE page_id = ? AND revision_id = ?', 
                        (page_id, revision_id)
                    )
                    row = cursor.fetchone()
                    if row:
                        entry_id = row[0]  # Access by index (SQLite row is tuple-like)
                    else:
                        logger.error(f"Failed to get entry ID for page {page_id}, revision {revision_id}")
                        continue
                
                # Handle training metadata if present
                training_metadata = entry.get('training_metadata', {})
                if training_metadata:
                    used_in_training = 1 if training_metadata.get('used_in_training', False) else 0
                    model_checkpoint = training_metadata.get('model_checkpoint')
                    average_loss = training_metadata.get('average_loss')
                    relative_loss = training_metadata.get('relative_loss')
                    
                    # Insert training metadata
                    cursor.execute('''
                    INSERT INTO training_metadata 
                    (entry_id, used_in_training, model_checkpoint, average_loss, relative_loss)
                    VALUES (?, ?, ?, ?, ?)
                    ''', (
                        entry_id, used_in_training, model_checkpoint, average_loss, relative_loss
                    ))
                    
                    # If used in training and has token impact, insert those
                    if used_in_training and 'token_impact' in training_metadata:
                        # Get the metadata ID
                        metadata_id = cursor.lastrowid
                        
                        # Process token impact
                        token_impact = training_metadata['token_impact']
                        for token, impact in token_impact.items():
                            try:
                                cursor.execute('''
                                INSERT INTO token_impact (metadata_id, token, impact)
                                VALUES (?, ?, ?)
                                ''', (metadata_id, token, float(impact)))
                            except (ValueError, TypeError):
                                logger.warning(f"Invalid impact value for token {token}: {impact}")
                
                migrated_pages += 1
                
                if migrated_pages % 100 == 0:
                    logger.info(f"Migrated {migrated_pages} entries...")
                    conn.commit()  # Intermediate commit for large datasets
                    
            except Exception as e:
                logger.error(f"Error migrating entry: {e}")
                continue
        
        conn.commit()
        logger.info(f"Successfully migrated {migrated_pages} entries.")
        
    except sqlite3.Error as e:
        conn.rollback()
        logger.error(f"Database error during migration: {e}")
        raise
    finally:
        conn.close()

def main():
    """
    Main function to run the migration.
    """
    logger.info("Starting migration process")
    try:
        # Initialize the SQLite database
        init_db()
        
        # Load JSON data
        json_data = load_json_data(CHANGELOG_JSON_PATH)
        
        if json_data:
            # Migrate data to SQLite
            migrate_json_to_sqlite(json_data)
            logger.info("Migration completed successfully")
        else:
            logger.info("No data to migrate. Created empty database.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
