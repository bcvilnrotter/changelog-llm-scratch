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

    def __init__(self, changelog_path: Union[str, Path] = "data/changelog.db"):
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
            "is_revision": False,
            "parent_id": None,
            "revision_number": None,
            "training_metadata": {
                "used_in_training": False,
                "training_timestamp": None,
                "model_checkpoint": None,
                "average_loss": None,
                "relative_loss": None,
                "token_impact": None
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
        model_checkpoint: str,
        training_metrics: Optional[Dict] = None
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
                metadata_update = {
                    "used_in_training": True,
                    "training_timestamp": timestamp,
                    "model_checkpoint": model_checkpoint
                }
                
                # Add training metrics if provided
                if training_metrics and entry["page_id"] in training_metrics:
                    page_metrics = training_metrics[entry["page_id"]]
                    token_impact = page_metrics.get("token_impact")
                    
                    if token_impact and isinstance(token_impact, dict):
                        critical_tokens = token_impact.get("critical_tokens", [])
                        
                        # Get top 10 highest impact tokens
                        sorted_tokens = sorted(critical_tokens, key=lambda x: abs(x["impact"]), reverse=True)[:10]
                        top_tokens = []
                        
                        for token in sorted_tokens:
                            position = token["position"]
                            # Get 2 tokens before and after for context
                            context_start = max(0, position - 2)
                            context_end = min(token_impact.get("total_tokens", position + 3), position + 3)
                            
                            top_tokens.append({
                                "token_id": token["token_id"],
                                "position": position,
                                "impact": float(token["impact"]),
                                "context": [context_start, context_end]
                            })
                        
                        # Create token impact data
                        compressed_token_impact = {
                            "top_tokens": top_tokens,
                            "total_tokens": token_impact.get("total_tokens", 0)
                        }
                    else:
                        compressed_token_impact = None
                    
                    metadata_update.update({
                        "average_loss": float(page_metrics.get("average_loss")),
                        "relative_loss": float(page_metrics.get("relative_loss")),
                        "token_impact": compressed_token_impact
                    })
                else:
                    metadata_update.update({
                        "average_loss": float(page_metrics.get("average_loss")),
                        "relative_loss": float(page_metrics.get("relative_loss")),
                        "token_impact": None
                    })
                
                entry["training_metadata"].update(metadata_update)

        self._write_changelog(changelog)

    def get_unused_pages(self) -> List[Dict]:
        """
        Get all pages that haven't been used in training.

        Returns:
            List of changelog entries for unused pages (including revisions)
        """
        changelog = self._read_changelog()
        return [
            entry for entry in changelog["entries"]
            if not entry["training_metadata"]["used_in_training"]
        ]

    def get_page_revisions(self, page_id: str) -> List[Dict]:
        """
        Get all revision entries for a page.

        Args:
            page_id: Wikipedia page ID

        Returns:
            List of revision entries for the page, sorted by revision_number
        """
        changelog = self._read_changelog()
        revisions = [
            entry for entry in changelog["entries"]
            if entry["is_revision"] and entry["parent_id"] == page_id
        ]
        return sorted(revisions, key=lambda x: x["revision_number"])

    def get_main_pages(self) -> List[Dict]:
        """
        Get all non-revision pages.

        Returns:
            List of main page entries (excluding revisions)
        """
        changelog = self._read_changelog()
        return [
            entry for entry in changelog["entries"]
            if not entry["is_revision"]
        ]

    def remove_unused_entries(self) -> None:
        """Remove all entries that haven't been used in training."""
        changelog = self._read_changelog()
        changelog["entries"] = [
            entry for entry in changelog["entries"]
            if entry["training_metadata"]["used_in_training"]
        ]
        self._write_changelog(changelog)

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
        entry = {
            "title": title,
            "page_id": page_id,
            "revision_id": revision_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "content_hash": self._compute_hash(content),
            "action": "added",
            "is_revision": True,
            "parent_id": parent_id,
            "revision_number": revision_number,
            "training_metadata": {
                "used_in_training": False,
                "training_timestamp": None,
                "model_checkpoint": None,
                "average_loss": None,
                "relative_loss": None,
                "token_impact": None
            }
        }

        changelog = self._read_changelog()
        changelog["entries"].append(entry)
        self._write_changelog(changelog)
        return entry
