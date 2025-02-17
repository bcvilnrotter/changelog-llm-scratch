#!/usr/bin/env python3
"""
Script to fetch Wikipedia pages and track them in the changelog.
"""

import argparse
import json
import logging
import os
import sys
import time
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
        language: str = "en",
        batch_size: int = 10
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
        self.batch_size = batch_size
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        self.changelog = ChangelogLogger(changelog_path)
        self.raw_data_path = Path(raw_data_path)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

    def _save_raw_content(self, page_id: str, content: str, revision_id: Optional[str] = None) -> None:
        """Save raw page content to file."""
        filename = f"{page_id}.txt" if revision_id is None else f"{page_id}_{revision_id}.txt"
        file_path = self.raw_data_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def _make_request(self, params: Dict) -> Optional[Dict]:
        """Make a rate-limited request to the Wikipedia API."""
        # Ensure at least 1 second between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        try:
            response = requests.get(self.api_url, params=params, headers=self.headers)
            response.raise_for_status()
            self.last_request_time = time.time()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            return None

    def _fetch_revisions(self, page_id: str, title: str) -> List[Dict]:
        """Fetch the last 5 revisions of a page."""
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions",
            "rvprop": "content|ids",
            "rvslots": "main",
            "rvlimit": "5",  # Get last 5 revisions
            "titles": title
        }

        data = self._make_request(params)
        if not data or "query" not in data or "pages" not in data["query"]:
            return []

        page = next(iter(data["query"]["pages"].values()))
        if "revisions" not in page:
            return []

        entries = []
        for i, rev in enumerate(page["revisions"], 1):
            content = rev["slots"]["main"]["*"]
            revision_id = str(rev["revid"])

            # Save revision content
            self._save_raw_content(page_id, content, revision_id)

            # Log revision to changelog
            entry = self.changelog.log_revision(
                title=title,
                page_id=f"{page_id}_{revision_id}",  # Unique ID for revision
                revision_id=revision_id,
                content=content,
                parent_id=page_id,
                revision_number=i  # 1 is most recent
            )
            entries.append(entry)

        return entries

    def fetch_page(self, title: str, fetch_revisions: bool = True) -> Optional[Dict]:
        """
        Fetch a single Wikipedia page and track it in changelog.

        Args:
            title: Page title to fetch

        Returns:
            Changelog entry if successful, None if failed
        """
        try:
            params = {
                "action": "query",
                "format": "json",
                "prop": "info|revisions",
                "rvprop": "content|ids",
                "rvslots": "main",
                "titles": title
            }
            
            data = self._make_request(params)
            if not data:
                return None
            
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

                    # Fetch revisions if requested
                    if fetch_revisions:
                        revision_entries = self._fetch_revisions(page_id, title)
                        if revision_entries:
                            logger.info(f"Added {len(revision_entries)} revisions for {title}")

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
            data = self._make_request(params)
            if not data:
                return []
            
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
            
            # Process pages in batches
            entries = []
            page_limit = pages[:limit] if limit else pages
            total_pages = len(page_limit)
            logger.info(f"Attempting to fetch {total_pages} pages")

            for i in range(0, total_pages, self.batch_size):
                batch = page_limit[i:i + self.batch_size]
                logger.info(f"Processing batch {i//self.batch_size + 1}/{(total_pages-1)//self.batch_size + 1}")
                
                for title in batch:
                    logger.info(f"Fetching page: {title}")
                    entry = self.fetch_page(title)
                    if entry:
                        entries.append(entry)
                
                # Add a small delay between batches
                if i + self.batch_size < total_pages:
                    time.sleep(2)  # 2 second pause between batches
            
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of pages to process in each batch (default: 10)"
    )
    args = parser.parse_args()

    if not (args.titles or args.category):
        parser.error("Must specify either --titles or --category")

    fetcher = WikipediaFetcher(language=args.language, batch_size=args.batch_size)
    
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
