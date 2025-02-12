#!/usr/bin/env python3
"""
Script to train a transformer-based LLM using Wikipedia data from the changelog.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed
from src.training.transformer import CustomTransformer
from src.training.tokenizer import SimpleTokenizer
from src.changelog.logger import ChangelogLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WikipediaDataset(Dataset):
    """Dataset for Wikipedia pages from changelog."""

    def __init__(
        self,
        raw_data_path: Path,
        page_ids: List[str],
        tokenizer: SimpleTokenizer,
        max_length: int = 512
    ):
        """
        Initialize dataset.

        Args:
            raw_data_path: Path to raw data directory
            page_ids: List of page IDs to include
            tokenizer: Custom tokenizer
            max_length: Maximum sequence length
        """
        self.raw_data_path = raw_data_path
        self.page_ids = page_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.page_ids)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        """Get tokenized page content."""
        page_id = self.page_ids[idx]
        file_path = self.raw_data_path / f"{page_id}.txt"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Tokenize content
        tokens = self.tokenizer._tokenize(content)[:self.max_length]
        input_ids = [self.tokenizer._convert_token_to_id(t) for t in tokens]

        # Return raw input_ids for data collator to handle padding
        return {"input_ids": input_ids}

class LLMTrainer:
    """Handles LLM training using Wikipedia data."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        changelog_path: str = "data/changelog.json",
        raw_data_path: str = "data/raw",
        output_dir: str = "models",
        max_length: int = 512,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
        seed: int = 42,
        vocab_size: int = 5000,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512
    ):
        """
        Initialize trainer.

        Args:
            model_path: Path to existing model to continue training from, or None to start fresh
            changelog_path: Path to changelog file
            raw_data_path: Path to raw data directory
            output_dir: Directory to save model checkpoints
            max_length: Maximum sequence length
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            seed: Random seed
            vocab_size: Size of vocabulary for custom tokenizer
            d_model: Model dimension for custom transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
        """
        self.model_path = model_path
        self.changelog = ChangelogLogger(changelog_path)
        self.raw_data_path = Path(raw_data_path)
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.seed = seed
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff

        # Set random seed
        set_seed(seed)
        
        # Initialize or load model and tokenizer
        if model_path and Path(model_path).exists():
            # Continue training from existing model
            logger.info(f"Loading existing model from {model_path}")
            self.tokenizer = SimpleTokenizer.from_pretrained(model_path)
            self.model = CustomTransformer.from_pretrained(model_path)
            logger.info(f"Model loaded successfully. Vocabulary size: {len(self.tokenizer)}")
        else:
            # Start fresh model from scratch
            logger.info("Initializing new tokenizer...")
            self.tokenizer = SimpleTokenizer()
            
            # Get initial data to train tokenizer
            unused_pages = self.changelog.get_unused_pages()
            logger.info(f"Training tokenizer on {min(100, len(unused_pages))} pages...")
            
            # Read training data
            texts = []
            for entry in unused_pages[:100]:
                file_path = self.raw_data_path / f"{entry['page_id']}.txt"
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {e}")
                    continue
            
            if not texts:
                raise ValueError("No training data found for tokenizer")
            
            # Train tokenizer directly on texts
            logger.info("Training tokenizer...")
            self.tokenizer.train(
                texts=texts,
                vocab_size=vocab_size,
                min_frequency=2
            )
            logger.info(f"Tokenizer trained. Vocabulary size: {len(self.tokenizer)}")
            
            # Initialize fresh model
            logger.info("Initializing new transformer model...")
            self.model = CustomTransformer(
                vocab_size=len(self.tokenizer),
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                d_ff=d_ff,
                max_seq_length=max_length
            )
            logger.info("Model initialized successfully")

    def _compute_checkpoint_hash(self) -> str:
        """Compute hash of model state."""
        state_dict = self.model.state_dict()
        hasher = hashlib.sha256()
        
        # Sort keys for consistent ordering
        for key in sorted(state_dict.keys()):
            hasher.update(state_dict[key].cpu().numpy().tobytes())
        
        return hasher.hexdigest()

    def _collate_fn(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        # Get all input_ids
        input_ids = [example["input_ids"] for example in examples]
        
        # Find max length in batch
        max_len = max(len(ids) for ids in input_ids)
        
        # Pad input_ids and create attention masks
        padded_input_ids = []
        attention_mask = []
        for ids in input_ids:
            padding_length = max_len - len(ids)
            padded_input_ids.append(ids + [self.tokenizer._convert_token_to_id(self.tokenizer.pad_token)] * padding_length)
            attention_mask.append([1] * len(ids) + [0] * padding_length)
        
        # Convert to tensors
        batch = {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(attention_mask)
        }
        
        return batch

    def train(
        self,
        val_split: float = 0.1,
        min_pages: int = 100
    ) -> None:
        """
        Train the model on unused Wikipedia pages.

        Args:
            val_split: Validation split ratio
            min_pages: Minimum number of pages required
        """
        # Get unused pages
        unused_pages = self.changelog.get_unused_pages()
        if len(unused_pages) < min_pages:
            raise ValueError(
                f"Not enough unused pages. Found {len(unused_pages)}, "
                f"need at least {min_pages}"
            )

        # Get page IDs
        page_ids = [entry["page_id"] for entry in unused_pages]

        # Split into train/val
        np.random.shuffle(page_ids)
        split_idx = int(len(page_ids) * (1 - val_split))
        train_ids = page_ids[:split_idx]
        val_ids = page_ids[split_idx:]

        logger.info(f"Training on {len(train_ids)} pages")
        logger.info(f"Validating on {len(val_ids)} pages")

        # Create datasets
        train_dataset = WikipediaDataset(
            self.raw_data_path,
            train_ids,
            self.tokenizer,
            self.max_length
        )
        val_dataset = WikipediaDataset(
            self.raw_data_path,
            val_ids,
            self.tokenizer,
            self.max_length
        )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = input_ids.clone()

                # Forward pass
                logits = self.model(x=input_ids, attention_mask=attention_mask)
                
                # Calculate loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=self.tokenizer._convert_token_to_id(self.tokenizer.pad_token)
                )

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.num_epochs} | "
                        f"Batch {batch_idx}/{len(train_loader)} | "
                        f"Loss: {loss.item():.4f}"
                    )

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # Save final model
        final_output_dir = self.output_dir / "final"
        self.model.save_pretrained(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)

        # Compute checkpoint hash
        checkpoint_hash = self._compute_checkpoint_hash()

        # Update changelog
        self.changelog.mark_used_in_training(page_ids, checkpoint_hash)

        logger.info(f"Training complete. Model saved to {final_output_dir}")
        logger.info(f"Checkpoint hash: {checkpoint_hash}")
        
        # Test model with a simple prompt
        test_output = self.generate_text("The quick brown fox", max_length=50)
        logger.info(f"Test generation:\n{test_output}")

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """Generate text from a prompt."""
        self.model.eval()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Generate text using custom model
        output_ids = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Decode output tokens
        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

def load_model(model_path: str) -> LLMTrainer:
    """Load a trained model."""
    if not Path(model_path).exists():
        raise ValueError(f"Model path {model_path} does not exist")
        
    # Create trainer instance with model path
    trainer = LLMTrainer(model_path=model_path)
    return trainer

def main():
    parser = argparse.ArgumentParser(
        description="Train LLM on Wikipedia data from changelog"
    )
    parser.add_argument(
        "--load-model",
        help="Load and test a trained model from the specified path"
    )
    parser.add_argument(
        "--test-prompt",
        default="The quick brown fox",
        help="Prompt to test the model with"
    )
    parser.add_argument(
        "--model-path",
        help="Path to existing model to continue training from, or none to start fresh"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=5000,
        help="Vocabulary size for custom tokenizer"
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Model dimension for custom transformer"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=512,
        help="Feed-forward dimension"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--min-pages",
        type=int,
        default=100,
        help="Minimum number of pages required"
    )
    args = parser.parse_args()

    if args.load_model:
        # Load and test model
        trainer = load_model(args.load_model)
        output = trainer.generate_text(args.test_prompt, max_length=100)
        print(f"\nInput prompt: {args.test_prompt}")
        print(f"Generated text:\n{output}\n")
    else:
        # Train model (either fresh or continue training)
        trainer = LLMTrainer(
            model_path=args.model_path,  # None for fresh, path for continue
            max_length=args.max_length,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            seed=args.seed,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            d_ff=args.d_ff
        )

        trainer.train(
            val_split=args.val_split,
            min_pages=args.min_pages
        )

if __name__ == "__main__":
    main()
