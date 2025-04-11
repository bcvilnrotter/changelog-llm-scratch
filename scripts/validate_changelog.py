#!/usr/bin/env python3
"""
Script to validate changelog.db file structure and token impact format.
This script uses SQLite API to directly access the database.
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Add src directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.db.db_schema import get_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_token_impact(token_impact_id: int, conn: sqlite3.Connection) -> bool:
    """
    Validate token impact structure in the database.
    
    Args:
        token_impact_id: ID of the token impact record
        conn: SQLite connection
    
    Returns:
        True if valid, False otherwise
    """
    cursor = conn.cursor()
    
    # Get token impact record
    cursor.execute(
        "SELECT * FROM token_impacts WHERE id = ?", 
        (token_impact_id,)
    )
    impact_record = cursor.fetchone()
    
    if not impact_record:
        logger.error(f"Token impact record {token_impact_id} not found")
        return False
    
    # Get top tokens for this impact
    cursor.execute(
        "SELECT * FROM top_tokens WHERE token_impact_id = ?",
        (token_impact_id,)
    )
    top_tokens = cursor.fetchall()
    
    # Print token impact structure for debugging
    logger.info(f"Token impact {token_impact_id} has {len(top_tokens)} top tokens")
    logger.info(f"Total tokens: {impact_record['total_tokens']}")
    
    # Validate top tokens
    for token in top_tokens:
        # Check required fields
        required_fields = ["token_id", "position", "impact", "context_start", "context_end"]
        for field in required_fields:
            if field not in token.keys():
                logger.error(f"Token missing required field: {field}")
                return False
        
        # Validate field types
        if not isinstance(token["token_id"], int):
            logger.error(f"Invalid token_id format: {token['token_id']}")
            return False
        if not isinstance(token["position"], int):
            logger.error(f"Invalid position format: {token['position']}")
            return False
        
        # Check context range
        if token["context_start"] > token["context_end"]:
            logger.error(f"Invalid context range: {token['context_start']} > {token['context_end']}")
            return False
    
    return True

def get_db_stats(db_path: str) -> Dict[str, Any]:
    """
    Get statistics about the database.
    
    Args:
        db_path: Path to the database file
    
    Returns:
        Dictionary of statistics
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    stats = {}
    
    # Get file size
    stats["file_size_mb"] = Path(db_path).stat().st_size / (1024 * 1024)
    
    # Count entries
    cursor.execute("SELECT COUNT(*) FROM entries")
    stats["total_entries"] = cursor.fetchone()[0]
    
    # Count entries with training data
    cursor.execute("""
        SELECT COUNT(*) FROM training_metadata 
        WHERE used_in_training = 1
    """)
    stats["entries_with_training"] = cursor.fetchone()[0]
    
    # Count entries with token impact
    cursor.execute("""
        SELECT COUNT(*) FROM token_impacts
    """)
    stats["entries_with_token_impact"] = cursor.fetchone()[0]
    
    # Get average entry size (estimate based on page_id and content_hash)
    cursor.execute("""
        SELECT AVG(LENGTH(page_id) + LENGTH(content_hash)) FROM entries
    """)
    avg_entry_size = cursor.fetchone()[0] or 0
    stats["avg_entry_size_bytes"] = avg_entry_size
    
    # Count total token impacts
    cursor.execute("""
        SELECT COUNT(*) FROM top_tokens
    """)
    stats["total_token_impacts"] = cursor.fetchone()[0]
    
    # Get average tokens per impact
    if stats["entries_with_token_impact"] > 0:
        stats["avg_tokens_per_impact"] = stats["total_token_impacts"] / stats["entries_with_token_impact"]
    else:
        stats["avg_tokens_per_impact"] = 0
    
    # Get average token impact size (estimate based on token_id, position, impact)
    cursor.execute("""
        SELECT AVG(LENGTH(token_id) + LENGTH(position) + LENGTH(impact)) FROM top_tokens
    """)
    avg_impact_size = cursor.fetchone()[0] or 0
    stats["avg_impact_size_bytes"] = avg_impact_size
    
    # Estimate total token impact storage
    stats["total_token_impact_mb"] = (avg_impact_size * stats["total_token_impacts"]) / (1024 * 1024)
    
    conn.close()
    return stats

def validate_db_schema(db_path: str) -> bool:
    """
    Validate the database schema.
    
    Args:
        db_path: Path to the database file
    
    Returns:
        True if valid, False otherwise
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Check if required tables exist
    required_tables = [
        "entries", 
        "training_metadata", 
        "token_impacts", 
        "top_tokens"
    ]
    
    for table in required_tables:
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
        if not cursor.fetchone():
            logger.error(f"Required table '{table}' not found")
            conn.close()
            return False
    
    # Check if required indices exist
    required_indices = [
        "idx_entries_page_id",
        "idx_entries_parent_id",
        "idx_training_metadata_entry_id",
        "idx_token_impacts_metadata_id",
        "idx_top_tokens_token_impact_id"
    ]
    
    for index in required_indices:
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='index' AND name='{index}'")
        if not cursor.fetchone():
            logger.error(f"Required index '{index}' not found")
            conn.close()
            return False
    
    # Validate token impacts if they exist
    cursor.execute("SELECT id FROM token_impacts LIMIT 10")
    impact_ids = [row[0] for row in cursor.fetchall()]
    
    for impact_id in impact_ids:
        if not validate_token_impact(impact_id, conn):
            logger.error(f"Invalid token impact: {impact_id}")
            conn.close()
            return False
    
    conn.close()
    return True

def analyze_changelog(changelog_path: str) -> None:
    """
    Analyze the changelog database.
    
    Args:
        changelog_path: Path to the database file
    """
    path = Path(changelog_path)
    if not path.exists():
        logger.error(f"Changelog file not found: {changelog_path}")
        return
    
    # Validate database schema
    logger.info("Validating database schema...")
    if validate_db_schema(changelog_path):
        logger.info("Database schema is valid")
    else:
        logger.error("Database schema validation failed")
        return
    
    # Get database statistics
    logger.info("Analyzing database statistics...")
    stats = get_db_stats(changelog_path)
    
    # Print analysis
    logger.info(f"Changelog file size: {stats['file_size_mb']:.2f} MB")
    logger.info(f"Total entries: {stats['total_entries']}")
    logger.info(f"Entries with training data: {stats['entries_with_training']}")
    logger.info(f"Entries with token impact: {stats['entries_with_token_impact']}")
    logger.info(f"Average entry size: {stats['avg_entry_size_bytes']:.2f} bytes ({stats['avg_entry_size_bytes']/1024:.2f} KB)")
    
    if stats['entries_with_token_impact'] > 0:
        logger.info(f"Average tokens per impact: {stats['avg_tokens_per_impact']:.2f}")
        logger.info(f"Average token impact size: {stats['avg_impact_size_bytes']:.2f} bytes ({stats['avg_impact_size_bytes']/1024:.2f} KB)")
        logger.info(f"Total token impact storage: {stats['total_token_impact_mb']:.2f} MB")

if __name__ == "__main__":
    analyze_changelog("data/changelog.db")
