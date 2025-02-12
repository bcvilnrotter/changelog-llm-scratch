#!/usr/bin/env python3
"""
Script to fetch Wikipedia pages and track them in the changelog.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import requests

# Add src directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.changelog.logger import ChangelogLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WikipediaFetcher:
    """Handles fetching and tracking Wikipedia pages."""

    def __init__(
        self,
        changelog_path: str = "data/changelog.json",
        raw_data_path: str = "data/raw",
        language: str = "en"
    ):
        """
        Initialize the Wikipedia fetcher.

        Args:
            changelog_path: Path to changelog file
            raw_data_path: Directory to store raw page content
            language: Wikipedia language version
        """
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.headers = {
            'User-Agent': 'ChangelogLLM/1.0 (https://github.com/yourusername/changelog-llm; your@email.com) Python/3.10',
            'Accept': 'application/json'
        }
        self.changelog = ChangelogLogger(changelog_path)
        self.raw_data_path = Path(raw_data_path)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

    def _save_raw_content(self, page_id: str, content: str) -> None:
        """Save raw page content to file."""
        file_path = self.raw_data_path / f"{page_id}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def fetch_page(self, title: str) -> Optional[Dict]:
        """
        Fetch a single Wikipedia page and track it in changelog.

        Args:
            title: Page title to fetch

        Returns:
            Changelog entry if successful, None if failed
        """
        try:
            # Get page info
            params = {
                "action": "query",
                "format": "json",
                "prop": "info|revisions",
                "rvprop": "content|ids",
                "rvslots": "main",
                "titles": title
            }
            
            response = requests.get(self.api_url, params=params, headers=self.headers)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()
            
            if "error" in data:
                logger.error(f"API Error: {data['error'].get('info', 'Unknown error')}")
                return None
            
            if "query" in data and "pages" in data["query"]:
                # Get the first (and only) page
                page = next(iter(data["query"]["pages"].values()))
                
                if "missing" in page:
                    logger.error(f"Page not found: {title}")
                    return None
                
                page_id = str(page["pageid"])
                revision = page["revisions"][0]
                content = revision["slots"]["main"]["*"]
                revision_id = str(revision["revid"])
                
                # Check if page needs updating
                if self.changelog.check_updates(page_id, revision_id):
                    action = "added"
                    history = self.changelog.get_page_history(page_id)
                    if history:
                        action = "updated"
                    
                    # Save raw content
                    self._save_raw_content(page_id, content)
                    
                    # Log to changelog
                    entry = self.changelog.log_page(
                        title=page["title"],
                        page_id=page_id,
                        revision_id=revision_id,
                        content=content,
                        action=action
                    )
                    
                    logger.info(
                        f"{action.capitalize()} page: {page['title']} "
                        f"(ID: {page_id}, Rev: {revision_id})"
                    )
                    return entry
                
                logger.info(f"Page already up to date: {title}")
                return None
            
            logger.error(f"Error fetching {title}: Unexpected API response")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching {title}: {str(e)}")
            return None

    def fetch_category(
        self,
        category: str,
        limit: Optional[int] = None,
        recursive: bool = False
    ) -> List[Dict]:
        """
        Fetch all pages in a category.

        Args:
            category: Category name
            limit: Maximum number of pages to fetch
            recursive: Whether to fetch pages from subcategories

        Returns:
            List of successful changelog entries
        """
        try:
            # Format category name
            category_name = f"Category:{category}" if not category.startswith("Category:") else category
            logger.info(f"Fetching members of {category_name}")
            
            # Parameters for API request
            params = {
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": category_name,
                "cmlimit": str(limit if limit else 50),
                "cmtype": "page"  # Only get pages, not subcategories
            }
            
            pages = []
            
            # Get pages from category
            response = requests.get(self.api_url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                logger.error(f"API Error: {data['error'].get('info', 'Unknown error')}")
                return []
            
            if "query" in data and "categorymembers" in data["query"]:
                members = data["query"]["categorymembers"]
                logger.info(f"Found {len(members)} pages in category")
                pages.extend(member["title"] for member in members)
            
            if recursive:
                # Get subcategories
                logger.info("Fetching subcategories")
                subcat_params = {
                    "action": "query",
                    "format": "json",
                    "list": "categorymembers",
                    "cmtitle": category_name,
                    "cmlimit": "10",
                    "cmtype": "subcat"
                }
                
                response = requests.get(self.api_url, params=subcat_params, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                
                if "error" in data:
                    logger.error(f"API Error: {data['error'].get('info', 'Unknown error')}")
                    return []
                
                if "query" in data and "categorymembers" in data["query"]:
                    subcats = data["query"]["categorymembers"]
                    logger.info(f"Found {len(subcats)} subcategories")
                    
                    # Get pages from each subcategory
                    if subcats:
                        pages_per_subcat = (limit if limit else 50) // len(subcats)
                        for subcat in subcats:
                            subcat_title = subcat["title"]
                            logger.info(f"Fetching from subcategory: {subcat_title}")
                            
                            subcat_params = {
                                "action": "query",
                                "format": "json",
                                "list": "categorymembers",
                                "cmtitle": subcat_title,
                                "cmlimit": str(pages_per_subcat),
                                "cmtype": "page"
                            }
                            
                            response = requests.get(self.api_url, params=subcat_params, headers=self.headers)
                            response.raise_for_status()
                            data = response.json()
                            
                            if "error" in data:
                                logger.error(f"API Error: {data['error'].get('info', 'Unknown error')}")
                                continue
                            
                            if "query" in data and "categorymembers" in data["query"]:
                                members = data["query"]["categorymembers"]
                                pages.extend(member["title"] for member in members)
            
            # Remove duplicates while preserving order
            pages = list(dict.fromkeys(pages))
            logger.info(f"Found {len(pages)} unique pages")
            
            # Fetch pages up to limit
            entries = []
            page_limit = pages[:limit] if limit else pages
            logger.info(f"Attempting to fetch {len(page_limit)} pages")
            for title in page_limit:
                logger.info(f"Fetching page: {title}")
                entry = self.fetch_page(title)
                if entry:
                    entries.append(entry)
            
            return entries
            
        except Exception as e:
            logger.error(f"Error fetching category {category}: {str(e)}")
            return []

def main():
    parser = argparse.ArgumentParser(
        description="Fetch Wikipedia pages and track them in changelog"
    )
    parser.add_argument(
        "--titles",
        nargs="+",
        help="Specific page titles to fetch"
    )
    parser.add_argument(
        "--category",
        help="Category to fetch pages from"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of pages to fetch"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Fetch pages from subcategories"
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Wikipedia language version (default: en)"
    )
    args = parser.parse_args()

    if not (args.titles or args.category):
        parser.error("Must specify either --titles or --category")

    fetcher = WikipediaFetcher(language=args.language)
    
    if args.titles:
        for title in args.titles:
            fetcher.fetch_page(title)
    
    if args.category:
        fetcher.fetch_category(
            args.category,
            limit=args.limit,
            recursive=args.recursive
        )

if __name__ == "__main__":
    main()
