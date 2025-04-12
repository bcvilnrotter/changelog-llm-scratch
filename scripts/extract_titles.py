#!/usr/bin/env python3
"""
Script to extract page titles from the changelog database for fetching Wikipedia content.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_appropriate_logger(changelog_path, debug=False):
    """Get the appropriate logger based on the file extension."""
    path = Path(changelog_path)
    if path.suffix.lower() == '.db':
        logger.info(f"Using ChangelogDB for {changelog_path}")
        from src.db.changelog_db import ChangelogDB
        return ChangelogDB(changelog_path, debug=debug)
    else:
        logger.info(f"Using ChangelogLogger for {changelog_path}")
        from src.changelog.logger import ChangelogLogger
        return ChangelogLogger(changelog_path)

def extract_titles(
    changelog_path: str = "data/changelog.db",
    output_path: str = "titles.json",
    debug: bool = False
) -> None:
    """
    Extract page titles from the changelog database.
    
    Args:
        changelog_path: Path to the changelog database
        output_path: Path to save the JSON file with titles
        debug: Enable debug logging
    """
    logger.info("Extracting page titles from changelog database...")
    
    # Get appropriate logger
    changelog = get_appropriate_logger(changelog_path, debug=debug)
    
    # Get all main pages from the changelog
    all_pages = changelog.get_main_pages()
    logger.info(f"Found {len(all_pages)} pages in changelog")
    
    # Log the actual entries for debugging
    if debug:
        logger.debug(f"Entries from database: {all_pages[:5]} (showing first 5)")
    
    # Extract titles
    titles = []
    for entry in all_pages:
        if 'title' in entry:
            title = entry['title']
            # Handle case where title is a byte string
            if isinstance(title, bytes):
                title = title.decode('utf-8', errors='replace')
                if debug:
                    logger.debug(f"Decoded byte string title: {title}")
            titles.append(title)
    
    logger.info(f"Extracted {len(titles)} titles")
    
    # Save titles to JSON file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(titles, f)
        logger.info(f"Titles saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving titles to {output_path}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Extract page titles from the changelog database"
    )
    parser.add_argument(
        "--changelog-path",
        default="data/changelog.db",
        help="Path to the changelog database"
    )
    parser.add_argument(
        "--output",
        default="titles.json",
        help="Path to save the JSON file with titles"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()
    
    extract_titles(
        changelog_path=args.changelog_path,
        output_path=args.output,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
