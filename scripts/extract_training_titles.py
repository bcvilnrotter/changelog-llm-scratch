#!/usr/bin/env python3
"""
Script to extract titles for training, prioritizing unused pages from the database
and filling the remainder with random Wikipedia pages.
"""

import argparse
import json
import logging
import sys
import requests
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add src directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.db.changelog_db import ChangelogDB

def get_random_wikipedia_titles(count, language="en"):
    """Fetch random Wikipedia page titles."""
    api_url = f"https://{language}.wikipedia.org/w/api.php"
    
    # Wikipedia API limits to 500 random pages per request
    batch_size = 500
    all_titles = []
    
    for i in range(0, count, batch_size):
        batch_count = min(batch_size, count - i)
        params = {
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": "0",  # Main namespace only
            "grnlimit": str(batch_count)
        }
        
        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "query" in data and "pages" in data["query"]:
                batch_titles = [page["title"] for page in data["query"]["pages"].values()]
                all_titles.extend(batch_titles)
                logger.info(f"Fetched {len(batch_titles)} random titles")
            else:
                logger.warning(f"Unexpected API response: {data}")
            
            # Be nice to the Wikipedia API
            if i + batch_size < count:
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error fetching random titles: {str(e)}")
    
    return all_titles

def extract_training_titles(output_path="titles.json", target_count=1000, debug=False):
    """
    Extract titles for training, prioritizing unused pages from the database
    and filling the remainder with random Wikipedia pages.
    """
    db_path = "data/changelog.db"
    db = ChangelogDB(db_path, debug=debug)
    
    # Get all unused pages
    unused_pages = db.get_unused_pages()
    logger.info(f"Found {len(unused_pages)} unused pages in the database")
    
    # Create a dictionary mapping page_ids to their entries
    # This ensures we handle each unique page_id only once
    page_id_map = {}
    for page in unused_pages:
        page_id = page["page_id"]
        # Convert bytes to string if needed
        if isinstance(page_id, bytes):
            page_id = page_id.decode('utf-8')
        
        # Only add if not already in the map (prioritize first occurrence)
        if page_id not in page_id_map:
            page_id_map[page_id] = page
    
    logger.info(f"Found {len(page_id_map)} unique page IDs")
    
    # Separate main pages from revisions
    main_page_ids = [pid for pid in page_id_map.keys() if '_' not in str(pid)]
    revision_page_ids = [pid for pid in page_id_map.keys() if '_' in str(pid)]
    
    logger.info(f"Found {len(main_page_ids)} unique main pages and {len(revision_page_ids)} unique revisions")
    
    # Prioritize main pages, then add revisions up to the target count
    prioritized_page_ids = main_page_ids + revision_page_ids
    prioritized_page_ids = prioritized_page_ids[:target_count]
    
    # Extract titles from database pages
    db_titles = []
    db_page_id_to_title = {}  # For tracking which page_id corresponds to which title
    
    for page_id in prioritized_page_ids:
        page = page_id_map[page_id]
        if "title" in page and page["title"]:
            title = page["title"]
            # Convert bytes to string if needed
            if isinstance(title, bytes):
                title = title.decode('utf-8')
            
            # Filter out single-character titles and very short titles
            if len(title) >= 3 and not title.startswith('"') and not title.startswith("'"):
                db_titles.append(title)
                db_page_id_to_title[page_id] = title
            elif debug:
                logger.debug(f"Skipping too short or invalid title: '{title}' for page_id {page_id}")
    
    logger.info(f"Extracted {len(db_titles)} titles from {len(prioritized_page_ids)} unique page IDs")
    
    # If we have fewer than target_count titles, fetch additional random ones
    additional_needed = target_count - len(db_titles)
    
    if additional_needed > 0:
        logger.info(f"Fetching {additional_needed} additional random titles")
        
        # Get all existing page titles from the database to avoid duplicates
        all_db_pages = db.get_main_pages()
        existing_titles = set()
        for p in all_db_pages:
            if "title" in p and p["title"]:
                title = p["title"]
                if isinstance(title, bytes):
                    title = title.decode('utf-8')
                existing_titles.add(title)
        
        # Fetch more random titles than needed to account for potential duplicates
        random_titles = get_random_wikipedia_titles(additional_needed * 2)
        
        # Filter out titles that already exist in the database
        new_titles = [t for t in random_titles if t not in existing_titles][:additional_needed]
        logger.info(f"Found {len(new_titles)} new random titles not in database")
        
        # Combine database titles with new random titles
        all_titles = db_titles + new_titles
    else:
        all_titles = db_titles
    
    logger.info(f"Final title count: {len(all_titles)}")
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_titles, f)
    logger.info(f"Saved {len(all_titles)} titles to {output_path}")
    
    # Also save the page_id to title mapping for reference
    mapping_path = output_path.replace('.json', '_mapping.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        # Convert keys to strings for JSON serialization
        serializable_mapping = {str(k): v for k, v in db_page_id_to_title.items()}
        json.dump(serializable_mapping, f)
    logger.info(f"Saved page_id to title mapping to {mapping_path}")
    
    return len(all_titles)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract titles for training")
    parser.add_argument("--output", default="titles.json", help="Output JSON file")
    parser.add_argument("--count", type=int, default=1000, help="Target number of titles")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    extract_training_titles(args.output, args.count, args.debug)
