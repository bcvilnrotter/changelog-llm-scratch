#!/usr/bin/env python3
"""
Script to train a tokenizer on all available data from the changelog.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.training.tokenizer import SimpleTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_appropriate_logger(changelog_path, debug=False):
    """Get the appropriate logger based on the file extension."""
    path = Path(changelog_path)
    if path.suffix.lower() == '.db':
        logger.info(f"Using ChangelogDB for {changelog_path}")
        from src.db.changelog_db import ChangelogDB
        return ChangelogDB(changelog_path, debug=debug)
    else:
        logger.info(f"Using ChangelogLogger for {changelog_path}")
        from src.changelog.logger import ChangelogLogger
        return ChangelogLogger(changelog_path)

def train_tokenizer(
    output_path: str = "models/tokenizer",
    changelog_path: str = "data/changelog.db",
    raw_data_path: str = "data/raw",
    vocab_size: int = 10000,
    min_frequency: int = 2,
    debug: bool = False
) -> None:
    """
    Train a tokenizer on all available data from the changelog.
    
    Args:
        output_path: Path to save the tokenizer
        changelog_path: Path to changelog database
        raw_data_path: Path to raw data directory
        vocab_size: Size of vocabulary
        min_frequency: Minimum frequency for tokens
        debug: Enable debug logging
    """
    logger.info("Training tokenizer on all available data...")
    
    # Get appropriate logger
    changelog = get_appropriate_logger(changelog_path, debug=debug)
    raw_data_path = Path(raw_data_path)
    
    # Get ALL pages from the changelog
    all_pages = changelog.get_all_pages()
    logger.info(f"Found {len(all_pages)} total pages in changelog")
    
    # Read training data
    texts = []
    for entry in all_pages:
        file_path = raw_data_path / f"{entry['page_id']}.txt"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            continue
    
    logger.info(f"Training tokenizer on {len(texts)} pages...")
    
    # Initialize and train tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.train(
        texts=texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency
    )
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Tokenizer trained and saved to {output_path}")
    logger.info(f"Vocabulary size: {len(tokenizer)}")

def main():
    parser = argparse.ArgumentParser(
        description="Train a tokenizer on all available data from the changelog"
    )
    parser.add_argument(
        "--output-path",
        default="models/tokenizer",
        help="Path to save the tokenizer"
    )
    parser.add_argument(
        "--changelog-path",
        default="data/changelog.db",
        help="Path to changelog database"
    )
    parser.add_argument(
        "--raw-data-path",
        default="data/raw",
        help="Path to raw data directory"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Size of vocabulary"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency for tokens"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()
    
    train_tokenizer(
        output_path=args.output_path,
        changelog_path=args.changelog_path,
        raw_data_path=args.raw_data_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
