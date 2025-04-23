# Troubleshooting Guide

This document provides solutions for common issues encountered in the changelog-llm-scratch project.

## Database Schema Issues

### Missing Columns in Database

If you encounter an error like `Failed to save intermediate metrics: no such column: training_timestamp`, it means the database schema is missing required columns. This can happen if the database was created with an older schema version.

**Solution:**

Run the fix_database.py script to add missing columns:

```bash
python scripts/fix_database.py
```

This script will check for missing columns in the training_metadata table and add them if needed.

## Tokenizer Issues

### IndexError in Tokenizer Save Method

If you encounter an error like `IndexError: tuple index out of range` when saving the tokenizer, it means there are invalid merges in the tokenizer's merges list.

**Solution:**

The tokenizer has been updated to handle invalid merges gracefully in three ways:

1. When loading merges from a file, it validates each merge and skips invalid ones
2. When saving merges to a file, it validates each merge and skips invalid ones
3. A new script `fix_tokenizer.py` has been added to clean up invalid merges in an existing tokenizer

To fix an existing tokenizer with invalid merges, run:

```bash
python scripts/fix_tokenizer.py --tokenizer-path path/to/your/tokenizer
```

The `prepare_training.py` script now automatically runs this fix as part of the preparation process.

## Preparing for Training

To ensure that your environment is properly set up for training, you can run the prepare_training.py script:

```bash
python scripts/prepare_training.py
```

This script will:
1. Fix the database schema by adding any missing columns
2. Ensure the tokenizer is properly configured to handle invalid merges

The GitHub Actions workflow has been updated to run this script automatically before training.

## GitHub Actions Workflow

The daily_training.yml workflow has been updated to include a new step that runs the prepare_training.py script before training. This ensures that the training environment is properly set up and prevents common issues from causing the workflow to fail.

## Manual Fixes

If you need to manually fix specific issues:

### Fix Database Schema

```bash
python scripts/fix_database.py --db-path path/to/your/database.db
```

### Fix Model

If you need to fix issues with the model's embedding layer or tokenizer:

```bash
python scripts/fix_model.py --model-path path/to/your/model --tokenizer-path path/to/your/tokenizer
```

## Debugging

Add the `--debug` flag to any of the scripts to enable debug logging:

```bash
python scripts/prepare_training.py --debug
```

This will provide more detailed information about what the script is doing, which can be helpful for troubleshooting.
