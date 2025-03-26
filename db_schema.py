"""
SQLite database schema for the changelog-llm project.
This module defines the database schema and provides utility functions
for database initialization.
"""
import os
import sqlite3
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database file location
DB_PATH = os.environ.get("CHANGELOG_DB_PATH", "changelog.db")

# Schema version for future migrations
SCHEMA_VERSION = 1

def get_db_connection():
    """
    Create and return a connection to the SQLite database.
    
    Returns:
        sqlite3.Connection: A connection to the SQLite database
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initialize the database with the required schema.
    This function creates the necessary tables if they don't exist.
    """
    logger.info(f"Initializing database at {DB_PATH}")
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Create metadata table to track schema version
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        ''')
        
        # Create training_runs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT NOT NULL,
            end_time TEXT,
            model_name TEXT NOT NULL,
            base_model TEXT NOT NULL,
            status TEXT NOT NULL,
            hyperparameters TEXT NOT NULL,  -- JSON string for flexibility
            metrics TEXT,                   -- JSON string for metrics
            git_commit TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create training_examples table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            input_text TEXT NOT NULL,
            target_text TEXT NOT NULL,
            example_type TEXT NOT NULL,
            metadata TEXT,                  -- JSON string for additional metadata
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES training_runs (id)
        )
        ''')
        
        # Create outputs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            input_text TEXT NOT NULL,
            output_text TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT,                  -- JSON string for additional metadata
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES training_runs (id)
        )
        ''')
        
        # Set schema version if it doesn't exist
        cursor.execute('''
        INSERT OR IGNORE INTO metadata (key, value) VALUES (?, ?)
        ''', ('schema_version', str(SCHEMA_VERSION)))
        
        conn.commit()
        logger.info("Database initialization completed successfully")
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    # Initialize the database when this script is run directly
    init_db()
