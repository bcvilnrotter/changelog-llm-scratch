#!/usr/bin/env python3
"""
Simple test script to verify database creation and initialization.
"""

import os
import json
import logging
import sys
import sqlite3
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

def test_init_db():
    """Test database initialization and verify the file is created."""
    logger.info("Testing database initialization...")
    
    # Import the db_schema module and initialize the database
    try:
        from src.db.db_schema import init_db
        
        # Create a test database in a temporary location
        test_db_path = os.path.join(project_root, "data", "test_changelog.db")
        
        # Remove the file if it already exists
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        
        # Initialize the database
        init_db(test_db_path)
        
        # Verify the file was created
        if os.path.exists(test_db_path):
            logger.info(f"Database created successfully at {test_db_path}")
        else:
            logger.error(f"Database file was not created at {test_db_path}")
            return False
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False
    
    return True

def test_basic_query():
    """Test a basic query to verify database works."""
    logger.info("Testing basic database query...")
    
    # Import the db_schema module
    try:
        from src.db.db_schema import get_db_connection
        
        # Connect to the test database
        test_db_path = os.path.join(project_root, "data", "test_changelog.db")
        
        if not os.path.exists(test_db_path):
            logger.error(f"Test database does not exist at {test_db_path}")
            return False
        
        # Get a connection
        conn = get_db_connection(test_db_path)
        cursor = conn.cursor()
        
        # Get a list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if tables:
            logger.info("Tables in database:")
            for table in tables:
                logger.info(f"  - {table['name']}")
        else:
            logger.error("No tables found in the database")
            return False
        
        # Close the connection
        conn.close()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error querying database: {e}")
        return False
    
    return True

def test_logger_interface():
    """Test the ChangelogLogger interface with SQLite backend."""
    logger.info("Testing ChangelogLogger with SQLite backend...")
    
    try:
        # Import the logger
        from src.changelog import ChangelogLogger
        
        # Create a logger instance
        test_db_path = os.path.join(project_root, "data", "test_changelog.db")
        changelog = ChangelogLogger(test_db_path)
        
        # Test logging a page
        entry = changelog.log_page(
            title="Test Page",
            page_id="test123",
            revision_id="rev1",
            content="This is a test page content",
            action="added"
        )
        
        if entry:
            logger.info(f"Successfully added page: {entry['title']} (ID: {entry['page_id']})")
        else:
            logger.error("Failed to add page to the changelog")
            return False
        
        # Test retrieving the page
        history = changelog.get_page_history("test123")
        if history:
            logger.info(f"Successfully retrieved page history: {len(history)} entries")
        else:
            logger.error("Failed to retrieve page history")
            return False
        
        # Test updating training metadata
        changelog.mark_used_in_training(
            page_ids=["test123"],
            model_checkpoint="test_checkpoint",
            training_metrics={
                "test123": {
                    "average_loss": 1.234,
                    "relative_loss": 0.567,
                    "token_impact": {
                        "top_tokens": [
                            {
                                "token_id": 42,
                                "position": 10,
                                "impact": 3.14,
                                "context": [8, 12]
                            }
                        ],
                        "total_tokens": 100
                    }
                }
            }
        )
        
        # Verify the metadata was updated
        history = changelog.get_page_history("test123")
        if not history:
            logger.error("Failed to retrieve page history after updating metadata")
            return False
        
        entry = history[0]
        if entry["training_metadata"]["used_in_training"]:
            logger.info("Successfully updated training metadata")
        else:
            logger.error("Failed to update training metadata")
            return False
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error testing logger interface: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    success = True
    
    if not test_init_db():
        logger.error("Database initialization test failed")
        success = False
    
    if not test_basic_query():
        logger.error("Basic query test failed")
        success = False
    
    if not test_logger_interface():
        logger.error("Logger interface test failed")
        success = False
    
    if success:
        logger.info("All tests passed!")
    else:
        logger.error("Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()