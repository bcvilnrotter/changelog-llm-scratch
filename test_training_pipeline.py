#!/usr/bin/env python3
"""
Test script to run a small-scale version of the training pipeline with debugging enabled.
This helps diagnose issues with the GitHub Actions workflow.
"""

import argparse
import json
import logging
import sys
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum information
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test the training pipeline with a small dataset")
    parser.add_argument("--num-pages", type=int, default=5, help="Number of Wikipedia pages to fetch (default: 5)")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching pages (use existing files)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (only fetch pages)")
    args = parser.parse_args()
    
    # 1. Define test titles (or fetch random ones)
    if not args.skip_fetch:
        logger.info(f"Step 1: Preparing to fetch {args.num_pages} Wikipedia pages")
        
        # These are just example titles - you can replace with any Wikipedia titles
        test_titles = [
            "Python (programming language)",
            "Machine learning",
            "Natural language processing",
            "Artificial intelligence",
            "Deep learning"
        ]
        
        # Limit to the requested number of pages
        test_titles = test_titles[:args.num_pages]
        
        # Save titles to a temporary file
        with open("test_titles.json", "w") as f:
            json.dump(test_titles, f)
        
        # Create raw data directory if it doesn't exist
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        
        # 2. Fetch the pages with debug enabled
        logger.info("Step 2: Downloading pages with fetch_wikipedia.py")
        result = subprocess.run([
            sys.executable, 
            "scripts/fetch_wikipedia.py", 
            "--titles", json.dumps(test_titles),
            "--debug"
        ], capture_output=True, text=True)
        
        logger.info(f"fetch_wikipedia.py stdout:\n{result.stdout}")
        if result.stderr:
            logger.error(f"fetch_wikipedia.py stderr:\n{result.stderr}")
    else:
        logger.info("Skipping page fetch (--skip-fetch flag used)")
    
    # 3. List files in data/raw to verify download
    logger.info("Step 3: Verifying downloaded files")
    raw_files = list(Path("data/raw").glob("*.txt"))
    logger.info(f"Files in data/raw: {[f.name for f in raw_files]}")
    logger.info(f"Total files: {len(raw_files)}")
    
    if len(raw_files) == 0:
        logger.warning("No files found in data/raw directory!")
        if not args.skip_train:
            logger.error("Cannot proceed with training without data files")
            return
    
    # 4. Train the model with minimal steps
    if not args.skip_train:
        logger.info("Step 4: Training model with minimal steps")
        train_cmd = [
            sys.executable,
            "scripts/train_llm.py",
            "--model-path", "models/final/",
            "--d-model", "256",
            "--num-heads", "4",
            "--num-layers", "4",
            "--max-length", "512",
            "--batch-size", "4",
            "--learning-rate", "1e-4",
            "--num-epochs", "1",  # Just 1 epoch for testing
            "--min-pages", "1",   # Allow training with just 1 page
            "--debug"
        ]
        
        logger.info(f"Running command: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd, capture_output=True, text=True)
        
        logger.info(f"train_llm.py stdout:\n{result.stdout}")
        if result.stderr:
            logger.error(f"train_llm.py stderr:\n{result.stderr}")
        
        # 5. Check training status
        logger.info("Step 5: Checking training status")
        result = subprocess.run([
            sys.executable,
            "scripts/check_training_status.py"
        ], capture_output=True, text=True)
        
        logger.info(f"check_training_status.py output:\n{result.stdout}")
        if result.stderr:
            logger.error(f"check_training_status.py stderr:\n{result.stderr}")
    else:
        logger.info("Skipping training (--skip-train flag used)")
    
    logger.info("Test complete!")

if __name__ == "__main__":
    main()
