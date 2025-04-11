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

# Import the appropriate logger based on the file extension
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_appropriate_logger(changelog_path, debug=False):
    """
    Get the appropriate logger based on the file extension.
    
    Args:
        changelog_path: Path to the changelog file
        debug: Enable debug logging
        
    Returns:
        The appropriate logger instance
    """
    path = Path(changelog_path)
    if path.suffix.lower() == '.db':
        logger.info(f"Using ChangelogDB for {changelog_path}")
        from src.db.changelog_db import ChangelogDB
        return ChangelogDB(changelog_path, debug=debug)
    else:
        logger.info(f"Using ChangelogLogger for {changelog_path}")
        from src.changelog.logger import ChangelogLogger
        return ChangelogLogger(changelog_path)

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
        changelog_path: str = "data/changelog.db",
        raw_data_path: str = "data/raw",
        language: str = "en",
        batch_size: int = 10,
        debug: bool = False
    ):
        """
        Initialize the Wikipedia fetcher.

        Args:
            changelog_path: Path to changelog file
            raw_data_path: Directory to store raw page content
            language: Wikipedia language version
            batch_size: Number of pages to process in each batch
            debug: Enable debug logging
        """
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.headers = {
            'User-Agent': 'ChangelogLLM/1.0 (https://github.com/yourusername/changelog-llm; your@email.com) Python/3.10',
            'Accept': 'application/json'
        }
        self.batch_size = batch_size
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        self.debug = debug
        self.changelog = get_appropriate_logger(changelog_path, debug=debug)
        self.raw_data_path = Path(raw_data_path)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

    def _save_raw_content(self, page_id: str, content: str, revision_id: Optional[str] = None) -> None:
        """Save raw page content to file."""
        filename = f"{page_id}.txt" if revision_id is None else f"{page_id}_{revision_id}.txt"
        file_path = self.raw_data_path / filename
        try:
            # Try to encode content as UTF-8, replacing any characters that can't be encoded
            encoded_content = content.encode('utf-8', errors='replace').decode('utf-8')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(encoded_content)
        except Exception as e:
            logger.error(f"Error saving content for {page_id}: {str(e)}")

    def _make_request(self, params: Dict) -> Optional[Dict]:
        """Make a rate-limited request to the Wikipedia API."""
        # Ensure at least 1 second between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        try:
            # Move API request details to debug level
            if self.debug:
                logger.debug(f"Making request to {self.api_url} with params: {params}")
            
            response = requests.get(self.api_url, params=params, headers=self.headers)
            response.raise_for_status()
            self.last_request_time = time.time()
            
            if self.debug:
                logger.debug(f"Received response with status code: {response.status_code}")
            
            # Explicitly decode the response with error handling
            try:
                # First try to decode as UTF-8
                if self.debug:
                    logger.debug("Attempting to decode response as UTF-8")
                content = response.content.decode('utf-8')
            except UnicodeDecodeError as e:
                # If that fails, use a more forgiving approach
                logger.warning(f"UTF-8 decode error: {str(e)}")
                if self.debug:
                    logger.debug("Falling back to UTF-8 with replacement characters")
                content = response.content.decode('utf-8', errors='replace')
                logger.warning(f"Had to use replacement characters when decoding response")
            
            # Parse the JSON from the decoded content
            if self.debug:
                logger.debug("Parsing JSON response")
            result = json.loads(content)
            if self.debug:
                logger.debug("Successfully parsed JSON response")
            return result
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            return None

    def _fetch_revisions(self, page_id: str, title: str) -> List[Dict]:
        """Fetch the last 5 revisions of a page."""
        if self.debug:
            logger.debug(f"Fetching revisions for page: {title} (ID: {page_id})")
            
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
            if self.debug:
                logger.debug(f"No revision data found for {title}")
            return []

        page = next(iter(data["query"]["pages"].values()))
        if "revisions" not in page:
            if self.debug:
                logger.debug(f"No revisions found for {title}")
            return []

        entries = []
        for i, rev in enumerate(page["revisions"], 1):
            if self.debug:
                logger.debug(f"Processing revision {i} (ID: {rev['revid']}) for {title}")
                
            content = rev["slots"]["main"]["*"]
            revision_id = str(rev["revid"])

            # Save revision content
            self._save_raw_content(page_id, content, revision_id)

            # Log revision to changelog with encoding handling
            try:
                # Ensure content is properly encoded
                encoded_content = content.encode('utf-8', errors='replace').decode('utf-8')
                
                # Create a unique ID for this revision
                revision_page_id = f"{page_id}_{revision_id}"
                
                # Log the revision to the changelog
                entry = self.changelog.log_revision(
                    title=title,
                    page_id=revision_page_id,
                    revision_id=revision_id,
                    content=encoded_content,
                    parent_id=page_id,
                    revision_number=i  # 1 is most recent
                )
            except Exception as e:
                logger.error(f"Error logging revision {revision_id} for {title} to changelog: {str(e)}")
                continue
                
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
            # Single line to indicate start of fetch
            logger.info(f"Fetching Wikipedia page: {title}")
            
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
                logger.error(f"No data returned from API for page: {title}")
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
                if self.debug:
                    logger.debug(f"Checking if page {page_id} needs updating")
                
                try:
                    # Check if the page needs updating
                    needs_update = self.changelog.check_updates(page_id, revision_id)
                    if self.debug:
                        logger.debug(f"Page {page_id} needs updating: {needs_update}")
                except Exception as e:
                    logger.error(f"Error checking updates for page {page_id}: {str(e)}")
                    if self.debug:
                        logger.debug(f"Exception type: {type(e).__name__}")
                    # If there's an error, assume we need to update the page
                    needs_update = True
                
                if needs_update:
                    action = "added"
                    try:
                        if self.debug:
                            logger.debug(f"Getting page history for {page_id}")
                        # Get page history to determine if this is an update or a new page
                        history = self.changelog.get_page_history(page_id)
                        if history:
                            action = "updated"
                            if self.debug:
                                logger.debug(f"Page {page_id} exists, updating")
                        else:
                            action = "added"
                            if self.debug:
                                logger.debug(f"Page {page_id} is new, adding")
                    except Exception as e:
                        logger.error(f"Error getting page history for {page_id}: {str(e)}")
                        if self.debug:
                            logger.debug(f"Exception type: {type(e).__name__}")
                        # If there's an error, assume we're adding a new page
                        action = "added"
                    
                    # Save raw content - move to debug level
                    if self.debug:
                        logger.debug(f"Saving raw content for page {page_id}")
                    
                    try:
                        self._save_raw_content(page_id, content)
                        if self.debug:
                            logger.debug(f"Raw content saved successfully")
                    except Exception as e:
                        logger.error(f"Error saving raw content for {page_id}: {str(e)}")
                        return None
                    
                    # Log to changelog with encoding handling
                    if self.debug:
                        logger.debug(f"Logging page {page_id} to changelog")
                    
                    try:
                        # Ensure content is properly encoded
                        encoded_content = content.encode('utf-8', errors='replace').decode('utf-8')
                        
                        entry = self.changelog.log_page(
                            title=page["title"],
                            page_id=page_id,
                            revision_id=revision_id,
                            content=encoded_content,
                            action=action
                        )
                    except Exception as e:
                        logger.error(f"Error logging page {title} to changelog: {str(e)}")
                        if self.debug:
                            logger.debug(f"Exception type: {type(e).__name__}")
                            logger.debug(f"Exception traceback: {sys.exc_info()[2]}")
                        return None
                    
                    # Single line to indicate successful page addition
                    logger.info(f"Added page: {page['title']} (ID: {page_id}, Rev: {revision_id})")

                    # Fetch revisions if requested
                    if fetch_revisions:
                        revision_entries = self._fetch_revisions(page_id, title)
                        if revision_entries:
                            # Single line to indicate revisions added
                            logger.info(f"Added {len(revision_entries)} revisions for {title}")

                    return entry
                
                logger.info(f"Page already up to date: {title}")
                return None
            
            logger.error(f"Error fetching {title}: Unexpected API response")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching {title}: {str(e)}")
            return None

    def _get_subcategories(self, category_name: str, limit: Optional[int] = None) -> List[str]:
        """Get subcategories of a category."""
        if self.debug:
            logger.debug(f"Getting subcategories for {category_name}")
            
        subcat_params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category_name,
            "cmlimit": "10",
            "cmtype": "subcat"
        }
        
        data = self._make_request(subcat_params)
        if not data or "error" in data or "query" not in data or "categorymembers" not in data["query"]:
            return []
        
        subcats = data["query"]["categorymembers"]
        return [subcat["title"] for subcat in subcats]
    
    def _get_category_pages(self, category_name: str, limit: Optional[int] = None) -> List[str]:
        """Get pages in a category."""
        if self.debug:
            logger.debug(f"Getting pages for subcategory {category_name}")
            
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category_name,
            "cmlimit": str(limit if limit else 50),
            "cmtype": "page"
        }
        
        data = self._make_request(params)
        if not data or "error" in data or "query" not in data or "categorymembers" not in data["query"]:
            return []
        
        members = data["query"]["categorymembers"]
        return [member["title"] for member in members]
    
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
            logger.info(f"Fetching Wikipedia category: {category_name}")
            
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
                logger.error(f"No data returned from API for category: {category_name}")
                return []
            
            if "error" in data:
                logger.error(f"API Error: {data['error'].get('info', 'Unknown error')}")
                return []
            
            if "query" in data and "categorymembers" in data["query"]:
                members = data["query"]["categorymembers"]
                logger.info(f"Found {len(members)} pages in category {category_name}")
                pages.extend(member["title"] for member in members)
            
            # Handle recursive fetching if requested
            if recursive:
                subcats = self._get_subcategories(category_name, limit)
                if subcats:
                    logger.info(f"Found {len(subcats)} subcategories in {category_name}")
                    
                    # Get pages from each subcategory
                    pages_per_subcat = (limit if limit else 50) // len(subcats)
                    for subcat in subcats:
                        subcat_pages = self._get_category_pages(subcat, pages_per_subcat)
                        if subcat_pages:
                            pages.extend(subcat_pages)
            
            # Remove duplicates while preserving order
            pages = list(dict.fromkeys(pages))
            
            # Process pages in batches
            entries = []
            page_limit = pages[:limit] if limit else pages
            total_pages = len(page_limit)
            logger.info(f"Fetching {total_pages} unique pages from category {category_name}")
            
            success_count = 0
            for i in range(0, total_pages, self.batch_size):
                batch = page_limit[i:i + self.batch_size]
                batch_num = i//self.batch_size + 1
                total_batches = (total_pages-1)//self.batch_size + 1
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} pages)")
                
                batch_entries = []
                for title in batch:
                    entry = self.fetch_page(title)
                    if entry:
                        batch_entries.append(entry)
                
                # Report batch results
                success_count += len(batch_entries)
                entries.extend(batch_entries)
                logger.info(f"Batch {batch_num} complete: {len(batch_entries)}/{len(batch)} pages fetched successfully")
                
                # Add a small delay between batches
                if i + self.batch_size < total_pages:
                    time.sleep(2)  # 2 second pause between batches
            
            # Final summary
            logger.info(f"Category fetch complete: {success_count}/{total_pages} pages fetched successfully")
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
        help="JSON-encoded list of page titles",
        type=str
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
    parser.add_argument(
        "--changelog-path",
        default="data/changelog.db",
        help="Path to the changelog database file (default: data/changelog.db)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    if not (args.titles or args.category):
        parser.error("Must specify either --titles or --category")

    fetcher = WikipediaFetcher(
        changelog_path=args.changelog_path,
        language=args.language, 
        batch_size=args.batch_size,
        debug=args.debug
    )
    
    if args.titles:
        try:
            titles = json.loads(args.titles)
            if not isinstance(titles,list):
                raise ValueError("Titles must be a JSON list")
        except json.JSONDecodeError:
            print("⚠️ Error: Failed to parse JSON input for titles.",file=sys.stderr)
            sys.exit(1)

        for title in titles:
            fetcher.fetch_page(title)
    
    if args.category:
        fetcher.fetch_category(
            args.category,
            limit=args.limit,
            recursive=args.recursive
        )

if __name__ == "__main__":
    main()
