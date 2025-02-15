#!/usr/bin/env python3
"""
Script to validate changelog.json file structure and token impact format.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_token_impact(token_impact: Dict[str, Any]) -> bool:
    """Validate token impact structure."""
    if not isinstance(token_impact, dict):
        return False
    
    # Print token impact structure for debugging
    print(f"Token impact structure: {token_impact}")
    
    required_fields = {"top_tokens", "total_tokens"}
    if not all(field in token_impact for field in required_fields):
        print(f"Missing required fields. Found: {token_impact.keys()}")
        return False
    
    if not isinstance(token_impact["top_tokens"], list):
        print("top_tokens is not a list")
        return False
        
    # Check each token entry
    for token in token_impact["top_tokens"]:
        if not isinstance(token, dict):
            print(f"Token entry is not a dict: {token}")
            return False
        token_fields = {"token_id", "position", "impact", "context"}
        if not all(field in token for field in token_fields):
            print(f"Token missing required fields. Found: {token.keys()}")
            return False
        if not isinstance(token["token_id"], int):
            print(f"Invalid token_id format: {token['token_id']}")
            return False
        if not isinstance(token["position"], int):
            print(f"Invalid position format: {token['position']}")
            return False
        if not isinstance(token["impact"], float):
            print(f"Invalid impact format: {token['impact']}")
            return False
        if not isinstance(token["context"], list) or len(token["context"]) != 2:
            print(f"Invalid context format: {token['context']}")
            return False
    
    return True

def analyze_changelog(changelog_path: str) -> None:
    """Analyze changelog.json without loading entire content."""
    path = Path(changelog_path)
    if not path.exists():
        logger.error(f"Changelog file not found: {changelog_path}")
        return

    # Get file size
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"Changelog file size: {size_mb:.2f} MB")

    # Process file in chunks to count entries and sizes
    total_entries = 0
    entries_with_training = 0
    entries_with_token_impact = 0
    total_token_impacts = 0
    token_impact_sizes = []
    entry_sizes = []  # Track size of each entry

    with open(path, 'r', encoding='utf-8') as f:
        # Read opening brace
        f.read(1)
        
        # Skip until "entries" array
        while f.read(1) != '[':
            pass

        # Process entries
        depth = 1  # We're inside the array
        current_entry = ""
        
        while depth > 0:
            char = f.read(1)
            if not char:
                break
                
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 1:  # End of an entry
                    total_entries += 1
                    
                    # Parse single entry and get its size
                    entry_str = current_entry + '}'
                    entry_size = len(entry_str.encode('utf-8'))  # Size in bytes
                    entry_sizes.append(entry_size)
                    
                    entry = json.loads(entry_str)
                    
                    # Check training metadata
                    training_meta = entry.get("training_metadata", {})
                    if training_meta.get("used_in_training"):
                        entries_with_training += 1
                        
                        # Check token impact
                        token_impact = training_meta.get("token_impact")
                        if token_impact:
                            entries_with_token_impact += 1
                            if validate_token_impact(token_impact):
                                # Count total tokens
                                total_token_impacts += len(token_impact["top_tokens"])
                                
                                # Calculate size of token impact data
                                impact_size = len(str(token_impact).encode('utf-8'))
                                token_impact_sizes.append(impact_size)
                    
                    current_entry = ""
                    continue
            
            if depth > 1:  # Inside an entry
                current_entry += char

    # Calculate statistics
    avg_impact_size = sum(token_impact_sizes) / len(token_impact_sizes) if token_impact_sizes else 0
    avg_entry_size = sum(entry_sizes) / len(entry_sizes) if entry_sizes else 0
    total_entry_size = sum(entry_sizes)
    
    # Print analysis
    logger.info(f"Total entries: {total_entries}")
    logger.info(f"Entries with training data: {entries_with_training}")
    logger.info(f"Entries with token impact: {entries_with_token_impact}")
    logger.info(f"Average entry size: {avg_entry_size:.2f} bytes ({avg_entry_size/1024:.2f} KB)")
    logger.info(f"Total entries size: {total_entry_size/1024:.2f} KB ({total_entry_size/(1024*1024):.2f} MB)")
    if entries_with_token_impact > 0:
        logger.info(f"Average tokens per impact: {total_token_impacts / entries_with_token_impact:.2f}")
        logger.info(f"Average token impact size: {avg_impact_size:.2f} bytes ({avg_impact_size/1024:.2f} KB)")
        logger.info(f"Total token impact storage: {sum(token_impact_sizes)/(1024*1024):.2f} MB")

if __name__ == "__main__":
    analyze_changelog("data/changelog.json")
