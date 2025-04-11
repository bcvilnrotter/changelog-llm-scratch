"""
Utility functions for the changelog system.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

def load_json(file_path: Path) -> Dict:
    """
    Safely load a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON content as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Dict, file_path: Path) -> None:
    """
    Safely save data to a JSON file.

    Args:
        data: Dictionary to save
        file_path: Path to save JSON file

    Creates parent directories if they don't exist.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def filter_entries_by_date(
    entries: List[Dict],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict]:
    """
    Filter changelog entries by date range.

    Args:
        entries: List of changelog entries
        start_date: ISO format date string (YYYY-MM-DD)
        end_date: ISO format date string (YYYY-MM-DD)

    Returns:
        Filtered list of entries
    """
    filtered = entries

    if start_date:
        filtered = [
            entry for entry in filtered
            if entry["timestamp"][:10] >= start_date
        ]

    if end_date:
        filtered = [
            entry for entry in filtered
            if entry["timestamp"][:10] <= end_date
        ]

    return filtered

def group_entries_by_action(entries: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group changelog entries by action type.

    Args:
        entries: List of changelog entries

    Returns:
        Dictionary with entries grouped by action
    """
    grouped = {"added": [], "updated": [], "removed": []}
    
    for entry in entries:
        action = entry["action"]
        if action in grouped:
            grouped[action].append(entry)
    
    return grouped

def get_training_statistics(entries: List[Dict]) -> Dict:
    """
    Calculate training usage statistics from changelog entries.

    Args:
        entries: List of changelog entries

    Returns:
        Dictionary containing training statistics:
        - total_pages: Total number of pages
        - pages_used: Number of pages used in training
        - pages_unused: Number of pages not used in training
        - usage_percentage: Percentage of pages used in training
    """
    total = len(entries)
    used = sum(1 for entry in entries if entry["training_metadata"]["used_in_training"])
    
    return {
        "total_pages": total,
        "pages_used": used,
        "pages_unused": total - used,
        "usage_percentage": (used / total * 100) if total > 0 else 0
    }
