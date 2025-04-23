#!/usr/bin/env python3
"""
Script to prepare the environment for training by fixing common issues.
This script should be run before training to ensure everything is set up correctly.
"""

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

def prepare_training(db_path="data/changelog.db", debug=False):
    """
    Prepare the environment for training by fixing common issues.
    
    Args:
        db_path: Path to the database file
        debug: Enable debug logging
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Import the fix functions
        from scripts.fix_database import fix_database_schema
        from scripts.fix_tokenizer import fix_tokenizer
        
        # Fix database schema
        logger.info("Fixing database schema...")
        if not fix_database_schema(db_path, debug):
            logger.error("Failed to fix database schema")
            return False
        
        # Fix tokenizer
        logger.info("Fixing tokenizer...")
        tokenizer_path = "models/tokenizer"
        try:
            fix_tokenizer(tokenizer_path=tokenizer_path, debug=debug)
            logger.info("Tokenizer fixed successfully")
        except Exception as e:
            logger.error(f"Failed to fix tokenizer: {str(e)}")
            logger.error("Continuing anyway as tokenizer validation is built into the tokenizer.py file")
        
        logger.info("Training environment prepared successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error preparing training environment: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare the environment for training by fixing common issues"
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
    
    success = prepare_training(
        db_path=args.db_path,
        debug=args.debug
    )
    
    if success:
        logger.info("Training environment prepared successfully!")
        sys.exit(0)
    else:
        logger.error("Failed to prepare training environment.")
        sys.exit(1)

if __name__ == "__main__":
    main()
