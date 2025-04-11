"""
SQLite database schema for the changelog-llm project.
This module defines the database schema and provides utility functions
for database initialization.
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Create and return a connection to the SQLite database.
    
    Args:
        db_path (str, optional): Path to the database file
        
    Returns:
        sqlite3.Connection: A connection to the SQLite database
    """
    if db_path is None:
        # Default to a database file next to this module
        parent_dir = Path(__file__).resolve().parent.parent.parent
        db_path = os.path.join(parent_dir, "data", "changelog.db")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to the database and enable foreign keys
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    
    return conn

def init_db(db_path: Optional[str] = None) -> None:
    """
    Initialize the database with the required schema.
    This function creates the necessary tables if they don't exist.
    
    Args:
        db_path (str, optional): Path to the database file
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        page_id TEXT NOT NULL UNIQUE,
        revision_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        action TEXT NOT NULL,
        is_revision BOOLEAN NOT NULL,
        parent_id TEXT,
        revision_number INTEGER,
        FOREIGN KEY (parent_id) REFERENCES entries (page_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS training_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id INTEGER NOT NULL,
        used_in_training BOOLEAN NOT NULL DEFAULT 0,
        training_timestamp TEXT,
        model_checkpoint TEXT,
        average_loss REAL,
        relative_loss REAL,
        FOREIGN KEY (entry_id) REFERENCES entries (id) ON DELETE CASCADE
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS token_impacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metadata_id INTEGER NOT NULL,
        total_tokens INTEGER NOT NULL,
        FOREIGN KEY (metadata_id) REFERENCES training_metadata (id) ON DELETE CASCADE
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS top_tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        token_impact_id INTEGER NOT NULL,
        token_id INTEGER NOT NULL,
        position INTEGER NOT NULL,
        impact REAL NOT NULL,
        context_start INTEGER NOT NULL,
        context_end INTEGER NOT NULL,
        FOREIGN KEY (token_impact_id) REFERENCES token_impacts (id) ON DELETE CASCADE
    )
    ''')
    
    # Create indices for faster querying
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entries_page_id ON entries (page_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entries_parent_id ON entries (parent_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_metadata_entry_id ON training_metadata (entry_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_token_impacts_metadata_id ON token_impacts (metadata_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_top_tokens_token_impact_id ON top_tokens (token_impact_id)')
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()