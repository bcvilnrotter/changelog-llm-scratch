#!/usr/bin/env python3
"""
Script to fix the tokenizer by cleaning up invalid merges.
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

def fix_tokenizer(
    tokenizer_path: str = "models/tokenizer",
    output_path: str = None,
    debug: bool = False
) -> None:
    """
    Fix the tokenizer by cleaning up invalid merges.
    
    Args:
        tokenizer_path: Path to the tokenizer directory
        output_path: Path to save the fixed tokenizer (defaults to tokenizer_path)
        debug: Enable debug logging
    """
    if output_path is None:
        output_path = tokenizer_path
    
    # Load the tokenizer
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = SimpleTokenizer.from_pretrained(tokenizer_path)
    
    # Count invalid merges
    invalid_merges = [m for m in tokenizer.merges if not (isinstance(m, tuple) and len(m) >= 2)]
    logger.info(f"Found {len(invalid_merges)} invalid merges out of {len(tokenizer.merges)} total merges")
    
    if debug:
        # Print some examples of invalid merges
        logger.debug(f"Examples of invalid merges: {invalid_merges[:10]}")
    
    # Filter out invalid merges
    original_count = len(tokenizer.merges)
    tokenizer.merges = [m for m in tokenizer.merges if isinstance(m, tuple) and len(m) >= 2]
    logger.info(f"Removed {original_count - len(tokenizer.merges)} invalid merges")
    
    # Save the fixed tokenizer
    logger.info(f"Saving fixed tokenizer to {output_path}")
    tokenizer.save_pretrained(output_path)
    logger.info(f"Tokenizer fixed and saved successfully")
    logger.info(f"Vocabulary size: {len(tokenizer)}")
    logger.info(f"Number of merges: {len(tokenizer.merges)}")

def main():
    parser = argparse.ArgumentParser(
        description="Fix the tokenizer by cleaning up invalid merges"
    )
    parser.add_argument(
        "--tokenizer-path",
        default="models/tokenizer",
        help="Path to the tokenizer directory"
    )
    parser.add_argument(
        "--output-path",
        help="Path to save the fixed tokenizer (defaults to tokenizer-path)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    fix_tokenizer(
        tokenizer_path=args.tokenizer_path,
        output_path=args.output_path,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
