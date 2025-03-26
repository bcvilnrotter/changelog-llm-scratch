"""
Training script for changelog-llm that uses SQLite database for logging.
This script replaces the previous JSON-based implementation.
"""
import os
import sys
import logging
import json
import argparse
from datetime import datetime
import subprocess
import git
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from datasets import Dataset
from changelog_db import ChangelogDB

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_git_commit():
    """
    Get the current git commit hash.
    
    Returns:
        str: Git commit hash or None if not in a git repository
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        logger.warning("Not in a git repository, commit hash not available")
        return None

def preprocess_function(examples, tokenizer, max_length=512):
    """
    Preprocess training examples by tokenizing them.
    
    Args:
        examples (dict): Dictionary of examples
        tokenizer: Hugging Face tokenizer
        max_length (int): Maximum sequence length
    
    Returns:
        dict: Processed examples
    """
    # Combine input and target for training
    texts = []
    for i in range(len(examples["input"])):
        input_text = examples["input"][i]
        target_text = examples["target"][i]
        texts.append(f"{input_text}{target_text}")
    
    # Tokenize the texts
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return tokenized

def get_examples_from_file(file_path):
    """
    Load training examples from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file with examples
    
    Returns:
        list: List of example dictionaries
    """
    with open(file_path, 'r') as f:
        examples = json.load(f)
    return examples

def train_model(args):
    """
    Train a model using the provided arguments and log to the database.
    
    Args:
        args: Command line arguments
    
    Returns:
        int: Training run ID
    """
    # Initialize the database
    changelog_db = ChangelogDB()
    
    # Get current git commit
    git_commit = get_git_commit()
    
    # Load base model and tokenizer
    logger.info(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    
    # Define hyperparameters
    hyperparameters = {
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps
    }
    
    # Create new training run in the database
    run_id = changelog_db.create_run(
        model_name=args.model_name,
        base_model=args.base_model,
        hyperparameters=hyperparameters,
        git_commit=git_commit
    )
    logger.info(f"Created training run with ID: {run_id}")
    
    # Load training examples
    logger.info(f"Loading examples from: {args.examples_file}")
    examples = get_examples_from_file(args.examples_file)
    
    # Add examples to the database
    changelog_db.add_examples(run_id, examples)
    
    # Create a dataset for training
    dataset_dict = {"input": [], "target": []}
    for example in examples:
        dataset_dict["input"].append(example.get("input", ""))
        dataset_dict["target"].append(example.get("target", ""))
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Preprocess the dataset
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=["input", "target"]
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{args.model_name}",
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    try:
        # Start training
        logger.info("Starting training")
        trainer.train()
        
        # Save the model
        logger.info(f"Saving model to ./models/{args.model_name}")
        model.save_pretrained(f"./models/{args.model_name}")
        tokenizer.save_pretrained(f"./models/{args.model_name}")
        
        # Update training run with results
        metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
        changelog_db.update_run_status(run_id, "completed", metrics)
        
        # Generate and log sample outputs if requested
        if args.test_prompts:
            generate_test_outputs(run_id, model, tokenizer, args.test_prompts, changelog_db)
        
        logger.info(f"Training completed successfully. Run ID: {run_id}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        changelog_db.update_run_status(run_id, "failed", {"error": str(e)})
        raise
    
    return run_id

def generate_test_outputs(run_id, model, tokenizer, test_prompts_file, changelog_db):
    """
    Generate test outputs from the model and log them to the database.
    
    Args:
        run_id (int): Training run ID
        model: Trained model
        tokenizer: Tokenizer
        test_prompts_file (str): Path to file with test prompts
        changelog_db: ChangelogDB instance
    """
    logger.info(f"Generating test outputs for run {run_id}")
    
    try:
        with open(test_prompts_file, 'r') as f:
            prompts = json.load(f)
        
        for prompt in prompts:
            input_text = prompt["input"]
            
            # Generate output
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            output_ids = model.generate(
                inputs.input_ids,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Log the output to the database
            metadata = prompt.get("metadata", {})
            changelog_db.add_output(run_id, input_text, output_text, metadata)
            
        logger.info(f"Generated and logged {len(prompts)} test outputs")
        
    except Exception as e:
        logger.error(f"Error generating test outputs: {e}")

def main():
    """
    Main function to parse arguments and start training.
    """
    parser = argparse.ArgumentParser(description="Train a language model and log to SQLite database")
    parser.add_argument("--model-name", type=str, required=True, help="Name for the trained model")
    parser.add_argument("--base-model", type=str, required=True, help="Base model to fine-tune")
    parser.add_argument("--examples-file", type=str, required=True, help="JSON file with training examples")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--test-prompts", type=str, help="JSON file with test prompts")
    
    args = parser.parse_args()
    
    try:
        run_id = train_model(args)
        logger.info(f"Training completed. Run ID: {run_id}")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
