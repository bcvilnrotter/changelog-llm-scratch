#!/usr/bin/env python3
"""
Script to resize a model's embedding layer to match a new tokenizer's vocabulary size
and reset the training status in the changelog database.
"""

import sys
import json
import logging
import sqlite3
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import torch
from src.training.transformer import CustomTransformer
from src.training.tokenizer import SimpleTokenizer
from src.db.changelog_db import ChangelogDB

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resize_model_embeddings(model_path="models/final", tokenizer_path="models/tokenizer", debug=False):
    """
    Resize a model's embedding layer to match a tokenizer's vocabulary size.
    
    Args:
        model_path: Path to the model directory
        tokenizer_path: Path to the tokenizer directory
        debug: Enable debug logging
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the tokenizer first
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = SimpleTokenizer.from_pretrained(tokenizer_path)
        new_vocab_size = len(tokenizer)
        logger.info(f"Tokenizer vocabulary size: {new_vocab_size}")
        
        # Load model configuration
        config_path = Path(model_path) / "config.json"
        logger.info(f"Loading model config from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        old_vocab_size = config.get("vocab_size", 5000)  # Default to 5000 if not specified
        embedding_dim = config.get("d_model", 256)  # Default to 256 if not specified
        
        logger.info(f"Model vocabulary size: {old_vocab_size}")
        logger.info(f"Model embedding dimension: {embedding_dim}")
        
        if old_vocab_size == new_vocab_size:
            logger.info("Vocabulary sizes already match. No resizing needed.")
            return True
        
        # Load the model state dict
        state_dict_path = Path(model_path) / "pytorch_model.bin"
        logger.info(f"Loading model state dict from {state_dict_path}")
        state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
        
        # Create a new model with the new vocabulary size
        logger.info("Creating new model with updated vocabulary size")
        new_model = CustomTransformer(
            vocab_size=new_vocab_size,
            d_model=embedding_dim,
            num_heads=config.get("num_heads", 4),
            num_layers=config.get("num_layers", 4),
            d_ff=config.get("d_ff", 512),
            max_seq_length=config.get("max_seq_length", 512)
        )
        
        # Copy weights for shared parameters
        logger.info("Copying weights for shared parameters")
        for name, param in new_model.named_parameters():
            if name in state_dict:
                if name == "embedding.weight":
                    # Copy embedding weights for tokens that exist in both vocabularies
                    min_vocab_size = min(old_vocab_size, new_vocab_size)
                    logger.info(f"Copying embedding weights for {min_vocab_size} tokens")
                    param.data[:min_vocab_size] = state_dict[name][:min_vocab_size]
                elif name == "final_layer.weight":
                    # Copy output layer weights for tokens that exist in both vocabularies
                    min_vocab_size = min(old_vocab_size, new_vocab_size)
                    logger.info(f"Copying output layer weights for {min_vocab_size} tokens")
                    param.data[:min_vocab_size, :] = state_dict[name][:min_vocab_size, :]
                elif name == "final_layer.bias":
                    # Copy output layer bias for tokens that exist in both vocabularies
                    min_vocab_size = min(old_vocab_size, new_vocab_size)
                    logger.info(f"Copying output layer bias for {min_vocab_size} tokens")
                    param.data[:min_vocab_size] = state_dict[name][:min_vocab_size]
                else:
                    # Copy other parameters directly
                    if param.shape == state_dict[name].shape:
                        param.data.copy_(state_dict[name])
                    else:
                        logger.warning(f"Shape mismatch for {name}, skipping")
        
        # Update the model's config
        new_model.config["vocab_size"] = new_vocab_size
        if debug:
            logger.debug(f"Updated model config: {new_model.config}")
        
        # Save the updated model
        logger.info(f"Saving updated model to {model_path}")
        new_model.save_pretrained(model_path)
        
        # Also save the tokenizer to the model directory for consistency
        logger.info(f"Saving tokenizer to {model_path}")
        
        # Save tokenizer files manually to avoid issues with malformed merges
        vocab_file = Path(model_path) / "vocab.json"
        merges_file = Path(model_path) / "merges.txt"
        config_file = Path(model_path) / "tokenizer_config.json"
        
        # Save vocab
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(tokenizer.vocab, f, ensure_ascii=False)
            
        # Save merges with validation
        with open(merges_file, 'w', encoding='utf-8') as f:
            for merge in tokenizer.merges:
                # Validate that merge is a tuple with at least 2 elements
                if isinstance(merge, tuple) and len(merge) >= 2:
                    f.write(f"{merge[0]} {merge[1]}\n")
                else:
                    logger.warning(f"Skipping invalid merge: {merge}")
        
        # Save config
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(tokenizer.get_config(), f, ensure_ascii=False, indent=2)
        
        logger.info("Model embedding layer successfully resized!")
        return True
    
    except Exception as e:
        logger.error(f"Error resizing model embeddings: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def reset_training_status(db_path="data/changelog.db", debug=False):
    """
    Reset the training status of all pages in the database.
    
    Args:
        db_path: Path to the changelog database
        debug: Enable debug logging
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # First, get some stats before reset for reporting
        db = ChangelogDB(db_path, debug=debug)
        
        # Get all pages
        all_pages = db.get_main_pages()
        total_pages = len(all_pages)
        
        # Get unused pages
        unused_pages = db.get_unused_pages()
        unused_count = len(unused_pages)
        used_count = total_pages - unused_count
        
        logger.info(f"Database stats before reset:")
        logger.info(f"- Total pages: {total_pages}")
        logger.info(f"- Pages marked as used in training: {used_count}")
        logger.info(f"- Pages not yet used in training: {unused_count}")
        
        # Connect to the database directly
        logger.info(f"Connecting to database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if training_metadata table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_metadata'")
        if not cursor.fetchone():
            logger.warning("No training_metadata table found in database. Nothing to reset.")
            conn.close()
            return True
        
        # Get count of training metadata entries
        cursor.execute("SELECT COUNT(*) FROM training_metadata")
        metadata_count = cursor.fetchone()[0]
        logger.info(f"Found {metadata_count} training metadata entries")
        
        # Reset training status by deleting all entries from training_metadata
        logger.info("Resetting training status...")
        cursor.execute("DELETE FROM training_metadata")
        conn.commit()
        
        # Verify reset
        cursor.execute("SELECT COUNT(*) FROM training_metadata")
        new_count = cursor.fetchone()[0]
        logger.info(f"Training metadata entries after reset: {new_count}")
        
        # Close connection
        conn.close()
        
        logger.info(f"Successfully reset training status for {metadata_count} entries")
        logger.info(f"All {total_pages} pages are now marked as unused and available for training")
        
        return True
    
    except Exception as e:
        logger.error(f"Error resetting training status: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Resize model embeddings and reset training status"
    )
    parser.add_argument(
        "--model-path",
        default="models/final",
        help="Path to the model directory"
    )
    parser.add_argument(
        "--tokenizer-path",
        default="models/tokenizer",
        help="Path to the tokenizer directory"
    )
    parser.add_argument(
        "--db-path",
        default="data/changelog.db",
        help="Path to the changelog database"
    )
    parser.add_argument(
        "--skip-resize",
        action="store_true",
        help="Skip resizing the model embeddings"
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Skip resetting the training status"
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
    
    success = True
    
    # Resize model embeddings if not skipped
    if not args.skip_resize:
        logger.info("Step 1: Resizing model embeddings")
        resize_success = resize_model_embeddings(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            debug=args.debug
        )
        if not resize_success:
            logger.error("Failed to resize model embeddings")
            success = False
    else:
        logger.info("Skipping model embedding resize (--skip-resize flag used)")
    
    # Reset training status if not skipped
    if not args.skip_reset:
        logger.info("Step 2: Resetting training status")
        reset_success = reset_training_status(
            db_path=args.db_path,
            debug=args.debug
        )
        if not reset_success:
            logger.error("Failed to reset training status")
            success = False
    else:
        logger.info("Skipping training status reset (--skip-reset flag used)")
    
    # Final status
    if success:
        logger.info("All operations completed successfully!")
        logger.info("You can now run training with your new tokenizer.")
    else:
        logger.error("One or more operations failed. Please check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
