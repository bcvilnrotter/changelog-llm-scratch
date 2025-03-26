#!/usr/bin/env python3
"""
Simple test script to verify database creation and initialization.
"""
import os
import logging
from db_schema import init_db, get_db_connection, DB_PATH

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_init_db():
    """Test database initialization and verify the file is created."""
    logger.info(f"Testing database initialization at path: {DB_PATH}")
    
    # Initialize the database
    init_db()
    
    # Check if file exists
    if os.path.exists(DB_PATH):
        logger.info(f"SUCCESS: Database file created at {DB_PATH}")
        file_size = os.path.getsize(DB_PATH)
        logger.info(f"Database file size: {file_size} bytes")
    else:
        logger.error(f"FAILURE: Database file not created at {DB_PATH}")

def test_basic_query():
    """Test a basic query to verify database works."""
    if not os.path.exists(DB_PATH):
        logger.error(f"Database file not found at {DB_PATH}")
        return
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM metadata")
        rows = cursor.fetchall()
        logger.info(f"Metadata table contains {len(rows)} rows")
        for row in rows:
            logger.info(f"  {dict(row)}")
    except Exception as e:
        logger.error(f"Query error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    test_init_db()
    test_basic_query()