"""
SQLite-based changelog logger for tracking Wikipedia page operations and training metadata.
This replaces the JSON-based logger with a SQLite implementation.
"""

import logging
import traceback
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from src.db.changelog_db import ChangelogDB

# Configure logging
logger = logging.getLogger(__name__)

class ChangelogLogger:
    """
    Manages the changelog for Wikipedia page operations and training metadata using SQLite.
    
    This is a compatibility layer that provides the same interface as the original
    JSON-based ChangelogLogger, but uses SQLite for storage.
    """
    
    def __init__(self, changelog_path: Union[str, Path] = "data/changelog.json"):
        """
        Initialize the changelog logger.

        Args:
            changelog_path: Path to the changelog JSON file (ignored, kept for compatibility)
        """
        # Convert JSON path to DB path if needed (data/changelog.json -> data/changelog.db)
        path = Path(changelog_path)
        db_path = str(path if path.suffix == '.db' else path.with_suffix('.db'))
        logger.info(f"Initializing ChangelogLogger with database path: {db_path}")
        self.db = ChangelogDB(db_path)
    
    def _compute_hash(self, content: str) -> str:
        """
        Compute SHA-256 hash of content.

        Args:
            content: String content to hash

        Returns:
            Hexadecimal string of content hash
        """
        return self.db._compute_hash(content)
    
    def log_page(
        self,
        title: str,
        page_id: str,
        revision_id: str,
        content: str,
        action: str = "added"
    ) -> Dict:
        """
        Log a Wikipedia page operation.

        Args:
            title: Page title
            page_id: Wikipedia page ID
            revision_id: Wikipedia revision ID
            content: Page content
            action: Operation type (added/updated/removed)

        Returns:
            The created changelog entry
        """
        try:
            logger.info(f"Logging page: title={title}, page_id={page_id}, revision_id={revision_id}, action={action}")
            # Log content length and sample for debugging
            content_length = len(content) if content else 0
            content_sample = content[:100] + "..." if content and len(content) > 100 else content
            logger.info(f"Content length: {content_length}, Sample: {content_sample}")
            
            result = self.db.log_page(title, page_id, revision_id, content, action)
            logger.info(f"Page logged successfully: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in log_page: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return a minimal entry to avoid breaking the caller
            return {"page_id": page_id, "title": title, "revision_id": revision_id, "action": action, "error": str(e)}
    
    def get_page_history(self, page_id: str) -> List[Dict]:
        """
        Get all changelog entries for a specific page.

        Args:
            page_id: Wikipedia page ID

        Returns:
            List of changelog entries for the page
        """
        try:
            logger.info(f"Getting page history for page_id={page_id}")
            result = self.db.get_page_history(page_id)
            logger.info(f"Found {len(result)} entries in page history")
            return result
        except Exception as e:
            logger.error(f"Error in get_page_history: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return an empty list to avoid breaking the caller
            return []
    
    def check_updates(self, page_id: str, revision_id: str) -> bool:
        """
        Check if a page needs updating based on revision ID.

        Args:
            page_id: Wikipedia page ID
            revision_id: Current revision ID to check

        Returns:
            True if page needs updating, False otherwise
        """
        try:
            logger.info(f"Checking updates for page_id={page_id}, revision_id={revision_id}")
            result = self.db.check_updates(page_id, revision_id)
            logger.info(f"Update check result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in check_updates: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # If there's an error, assume we need to update the page
            return True
    
    def mark_used_in_training(
        self,
        page_ids: List[str],
        model_checkpoint: str,
        training_metrics: Optional[Dict] = None
    ) -> None:
        """
        Mark pages as used in training with associated model checkpoint.

        Args:
            page_ids: List of page IDs used in training
            model_checkpoint: Hash or identifier of the model checkpoint
        """
        self.db.mark_used_in_training(page_ids, model_checkpoint, training_metrics)
    
    def get_unused_pages(self) -> List[Dict]:
        """
        Get all pages that haven't been used in training.

        Returns:
            List of changelog entries for unused pages (including revisions)
        """
        return self.db.get_unused_pages()
    
    def get_page_revisions(self, page_id: str) -> List[Dict]:
        """
        Get all revision entries for a page.

        Args:
            page_id: Wikipedia page ID

        Returns:
            List of revision entries for the page, sorted by revision_number
        """
        return self.db.get_page_revisions(page_id)
    
    def get_main_pages(self) -> List[Dict]:
        """
        Get all non-revision pages.

        Returns:
            List of main page entries (excluding revisions)
        """
        return self.db.get_main_pages()
    
    def remove_unused_entries(self) -> None:
        """
        Remove all entries that haven't been used in training.
        """
        self.db.remove_unused_entries()
    
    def log_revision(
        self,
        title: str,
        page_id: str,
        revision_id: str,
        content: str,
        parent_id: str,
        revision_number: int
    ) -> Dict:
        """
        Log a revision of a Wikipedia page.

        Args:
            title: Page title
            page_id: Wikipedia page ID
            revision_id: Wikipedia revision ID
            content: Page content
            parent_id: ID of the parent page
            revision_number: Revision number (1-5, with 1 being most recent)

        Returns:
            The created changelog entry
        """
        try:
            logger.info(f"Logging revision: title={title}, page_id={page_id}, revision_id={revision_id}, parent_id={parent_id}, revision_number={revision_number}")
            # Log content length and sample for debugging
            content_length = len(content) if content else 0
            content_sample = content[:100] + "..." if content and len(content) > 100 else content
            logger.info(f"Content length: {content_length}, Sample: {content_sample}")
            
            result = self.db.log_revision(title, page_id, revision_id, content, parent_id, revision_number)
            logger.info(f"Revision logged successfully: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in log_revision: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return a minimal entry to avoid breaking the caller
            return {
                "page_id": page_id, 
                "title": title, 
                "revision_id": revision_id,
                "parent_id": parent_id,
                "revision_number": revision_number,
                "is_revision": True,
                "error": str(e)
            }
