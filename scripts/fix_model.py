#!/usr/bin/env python3
"""
Script to fix the model's embedding layer to match the tokenizer's vocabulary size.
This script creates a new model with the correct vocabulary size and copies the weights
from the old model.
"""

import sys
import json
import logging
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import torch
from src.training.transformer import CustomTransformer
from src.training.tokenizer import SimpleTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_model(model_path="models/final", tokenizer_path="models/tokenizer", debug=False):
    """
    Fix the model's embedding layer to match the tokenizer's vocabulary size.
    
    Args:
        model_path: Path to the model directory
        tokenizer_path: Path to the tokenizer directory
        debug: Enable debug logging
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the tokenizer
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
        
        # Force resizing even if the vocabulary sizes match in the config
        logger.info("Forcing model resizing to ensure weights match the vocabulary size")
        
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
                    # Get the actual size of the weights
                    actual_old_size = state_dict[name].shape[0]
                    logger.info(f"Actual embedding weight size: {actual_old_size}")
                    
                    # Copy embedding weights for tokens that exist in both vocabularies
                    min_vocab_size = min(actual_old_size, new_vocab_size)
                    logger.info(f"Copying embedding weights for {min_vocab_size} tokens")
                    param.data[:min_vocab_size] = state_dict[name][:min_vocab_size]
                elif name == "final_layer.weight":
                    # Get the actual size of the weights
                    actual_old_size = state_dict[name].shape[0]
                    logger.info(f"Actual final layer weight size: {actual_old_size}")
                    
                    # Copy output layer weights for tokens that exist in both vocabularies
                    min_vocab_size = min(actual_old_size, new_vocab_size)
                    logger.info(f"Copying output layer weights for {min_vocab_size} tokens")
                    param.data[:min_vocab_size, :] = state_dict[name][:min_vocab_size, :]
                elif name == "final_layer.bias":
                    # Get the actual size of the weights
                    actual_old_size = state_dict[name].shape[0]
                    logger.info(f"Actual final layer bias size: {actual_old_size}")
                    
                    # Copy output layer bias for tokens that exist in both vocabularies
                    min_vocab_size = min(actual_old_size, new_vocab_size)
                    logger.info(f"Copying output layer bias for {min_vocab_size} tokens")
                    param.data[:min_vocab_size] = state_dict[name][:min_vocab_size]
                else:
                    # Copy other parameters directly
                    if param.shape == state_dict[name].shape:
                        param.data.copy_(state_dict[name])
                    else:
                        logger.warning(f"Shape mismatch for {name}, skipping")
        
        # The model doesn't have a config attribute, it's created during save_pretrained
        # No need to update it manually
        
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
        logger.error(f"Error fixing model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix model embedding layer to match tokenizer vocabulary size"
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
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    success = fix_model(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        debug=args.debug
    )
    
    if success:
        logger.info("Model fixed successfully!")
        sys.exit(0)
    else:
        logger.error("Failed to fix model.")
        sys.exit(1)

if __name__ == "__main__":
    main()
