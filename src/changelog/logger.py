"""
Core changelog functionality for tracking Wikipedia page operations and training metadata.
"""

import json
import hashlib
import datetime
from pathlib import Path
from typing import Dict, Optional, List, Union

class ChangelogLogger:
    """
    Manages the changelog for Wikipedia page operations and training metadata.
    
    The changelog tracks:
    - Page retrievals and updates
    - Content hashes for integrity verification
    - Training usage metadata
    """

    def __init__(self, changelog_path: Union[str, Path] = "data/changelog.json"):
        """
        Initialize the changelog logger.

        Args:
            changelog_path: Path to the changelog JSON file
        """
        self.changelog_path = Path(changelog_path)
        self._ensure_changelog_exists()

    def _ensure_changelog_exists(self) -> None:
        """Create changelog file if it doesn't exist."""
        if not self.changelog_path.exists():
            self.changelog_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_changelog({"entries": []})

    def _read_changelog(self) -> Dict:
        """Read the current changelog."""
        with open(self.changelog_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _write_changelog(self, data: Dict) -> None:
        """Write data to the changelog file."""
        with open(self.changelog_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _compute_hash(self, content: str) -> str:
        """
        Compute SHA-256 hash of content.

        Args:
            content: String content to hash

        Returns:
            Hexadecimal string of content hash
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

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
        if action not in ["added", "updated", "removed"]:
            raise ValueError("Action must be one of: added, updated, removed")

        entry = {
            "title": title,
            "page_id": page_id,
            "revision_id": revision_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "content_hash": self._compute_hash(content),
            "action": action,
            "training_metadata": {
                "used_in_training": False,
                "training_timestamp": None,
                "model_checkpoint": None
            }
        }

        changelog = self._read_changelog()
        changelog["entries"].append(entry)
        self._write_changelog(changelog)
        return entry

    def get_page_history(self, page_id: str) -> List[Dict]:
        """
        Get all changelog entries for a specific page.

        Args:
            page_id: Wikipedia page ID

        Returns:
            List of changelog entries for the page
        """
        changelog = self._read_changelog()
        return [
            entry for entry in changelog["entries"]
            if entry["page_id"] == page_id
        ]

    def check_updates(self, page_id: str, revision_id: str) -> bool:
        """
        Check if a page needs updating based on revision ID.

        Args:
            page_id: Wikipedia page ID
            revision_id: Current revision ID to check

        Returns:
            True if page needs updating, False otherwise
        """
        history = self.get_page_history(page_id)
        if not history:
            return True
        
        latest_entry = max(
            history,
            key=lambda x: datetime.datetime.fromisoformat(
                x["timestamp"].rstrip("Z")
            )
        )
        return latest_entry["revision_id"] != revision_id

    def mark_used_in_training(
        self,
        page_ids: List[str],
        model_checkpoint: str
    ) -> None:
        """
        Mark pages as used in training with associated model checkpoint.

        Args:
            page_ids: List of page IDs used in training
            model_checkpoint: Hash or identifier of the model checkpoint
        """
        changelog = self._read_changelog()
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

        for entry in changelog["entries"]:
            if entry["page_id"] in page_ids:
                entry["training_metadata"].update({
                    "used_in_training": True,
                    "training_timestamp": timestamp,
                    "model_checkpoint": model_checkpoint
                })

        self._write_changelog(changelog)

    def get_unused_pages(self) -> List[Dict]:
        """
        Get all pages that haven't been used in training.

        Returns:
            List of changelog entries for unused pages
        """
        changelog = self._read_changelog()
        return [
            entry for entry in changelog["entries"]
            if not entry["training_metadata"]["used_in_training"]
        ]
