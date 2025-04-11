#!/usr/bin/env python3
"""
Script to check if pages are being marked as used in training.
This script uses the ChangelogLogger to access the SQLite database.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add src directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.changelog.db_logger import ChangelogLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_latest_training_metadata(changelog: ChangelogLogger) -> Optional[Dict[str, Any]]:
    """
    Get the most recent training metadata from the changelog.
    
    Args:
        changelog: ChangelogLogger instance
    
    Returns:
        The most recent training metadata or None if not found
    """
    # Get all pages that have been used in training
    all_pages = changelog.get_main_pages()
    
    # Filter to only include pages used in training
    trained_pages = [
        page for page in all_pages 
        if page.get("used_in_training")
    ]
    
    if not trained_pages:
        return None
    
    # Find the most recent training timestamp
    latest_page = max(
        trained_pages,
        key=lambda x: x.get("training_timestamp", "") or ""
    )
    
    # Extract training metadata
    metadata = {
        "training_timestamp": latest_page.get("training_timestamp"),
        "model_checkpoint": latest_page.get("model_checkpoint"),
        "average_loss": latest_page.get("average_loss"),
        "relative_loss": latest_page.get("relative_loss")
    }
    
    return metadata

def main():
    changelog_path = Path("data/changelog.db")
    
    if not changelog_path.exists():
        print("\nStatus: No changelog.db file found yet.")
        print("This is normal if:")
        print("1. Training is still in its first epoch")
        print("2. Training hasn't reached the point of saving metrics")
        print("\nPlease check again after training progresses further.")
        return
    
    print("\nChecking training metrics in changelog.db...")
    print("(This file exists and contains Wikipedia page entries)")
    
    # Initialize the changelog logger
    changelog = ChangelogLogger(changelog_path)
    
    # Get all pages
    all_pages = changelog.get_main_pages()
    unused_pages = changelog.get_unused_pages()
    
    # Calculate used pages
    total_pages = len(all_pages)
    total_unused = len(unused_pages)
    total_used = total_pages - total_unused
    
    print(f"\nResults:")
    print(f"- Total pages in changelog: {total_pages}")
    print(f"- Pages marked as used in training: {total_used}")
    print(f"- Pages not yet used in training: {total_unused}")
    
    if total_used == 0:
        print("\nNote: No pages have been marked as used in training.")
        print("This could mean:")
        print("1. Training is still in progress and hasn't saved metrics yet")
        print("2. The training process hasn't reached the point of updating the changelog")
        print("3. There might be an issue with the training metrics collection")
    
    # Get latest training metadata
    latest_metadata = get_latest_training_metadata(changelog)
    
    if latest_metadata:
        print("\nMost recent training metadata:")
        print(f"Training timestamp: {latest_metadata.get('training_timestamp', 'Unknown')}")
        print(f"Model checkpoint: {latest_metadata.get('model_checkpoint', 'Unknown')}")
        if latest_metadata.get('average_loss') is not None:
            print(f"Average loss: {latest_metadata['average_loss']}")
        if latest_metadata.get('relative_loss') is not None:
            print(f"Relative loss improvement: {latest_metadata['relative_loss'] * 100:.2f}%")

if __name__ == "__main__":
    main()
