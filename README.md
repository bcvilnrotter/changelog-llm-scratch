# Changelog LLM

A transparent, open-source language model trained on Wikipedia data with full changelog tracking. This project implements a system for fetching Wikipedia content, tracking all data operations in a changelog, and training a transformer-based language model with complete data provenance.

> **Note**: This project was built and tested using the [Cline VSCode Extension](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.cline), an AI-powered coding assistant.

## Features

- **Transparent Data Collection**: Every Wikipedia page retrieved is logged with metadata
- **Data Provenance**: Full tracking of which pages are used in training
- **Changelog System**: Records all data operations with content hashes
- **Modular Architecture**: Separate components for data collection and training
- **Extensible Design**: Easy to modify for different data sources or model architectures
- **Custom Transformer**: Option to train a small transformer model from scratch

## Project Structure

```
changelog-llm/
├── src/
│   ├── changelog/         # Changelog system
│   ├── data/             # Data handling
│   ├── training/         # Training components
│   │   ├── transformer.py # Custom transformer implementation
│   │   └── tokenizer.py  # Custom tokenizer
│   └── utils/            # Utility functions
├── scripts/
│   ├── fetch_wikipedia.py # Wikipedia data fetcher
│   └── train_llm.py      # Model training script
├── data/
│   ├── raw/              # Raw Wikipedia pages
│   ├── processed/        # Processed training data
│   └── changelog.json    # Changelog records
└── models/               # Saved model checkpoints
```

## Training Approach

The training process is automated to run daily via GitHub Actions, collecting 50 new pages each day. The initial training focuses on foundational content to establish basic language patterns, with subsequent days expanding the knowledge base while maintaining a balanced category distribution.

### Daily Training Process

Every day at 2 AM UTC, the GitHub Actions workflow:
1. Fetches 5 new pages from each of the following categories:
   - Basic concepts
   - Children's stories
   - Common English words
   - Basic geography
   - Simple machines
   - Elementary mathematics
   - Basic physics
   - Common occupations
   - Everyday life
   - Natural phenomena

2. Trains the model on the newly collected data
3. Saves the model checkpoint
4. Updates the changelog
5. Commits and pushes changes to the repository

The workflow maintains a rolling 7-day history of model checkpoints as artifacts, allowing for progress tracking and model comparison.

### Manual Training

For local development or manual training, you can use:
```bash
# Fetch initial training data
# Initial training data (50 pages total):
python scripts/fetch_wikipedia.py --category "Elementary_mathematics" --limit 15  # Basic math concepts
python scripts/fetch_wikipedia.py --category "Simple_English_words" --limit 15   # Common vocabulary
python scripts/fetch_wikipedia.py --category "Fairy_tales" --limit 10           # Simple narratives
python scripts/fetch_wikipedia.py --category "Earth_basic_concepts" --limit 10  # Basic geography/science

# Train the model
python scripts/train_llm.py --from-scratch \
    --vocab-size 5000 \
    --d-model 256 \
    --num-heads 4 \
    --num-layers 4 \
    --max-length 512 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --min-pages 50
```

## Model Architecture

The custom transformer implementation (`src/training/transformer.py`) includes:
- Multi-head self-attention mechanism
- Positional encoding
- Feed-forward networks with ReLU activation
- Layer normalization
- Dropout for regularization
- Text generation capabilities with top-k and nucleus sampling

Key parameters:
- Vocabulary size: 5000 tokens
- Model dimension: 256
- Number of attention heads: 4
- Number of transformer layers: 4
- Feed-forward dimension: 512
- Maximum sequence length: 512
- Batch size: 4
- Learning rate: 1e-4

## Tokenizer

The custom tokenizer (`src/training/tokenizer.py`) implements:
- Byte-level BPE tokenization
- Dynamic vocabulary building from training data
- Special token handling ([PAD], [BOS], [EOS], [UNK])
- HuggingFace compatibility for seamless integration

## Changelog System

The changelog (`data/changelog.json`) tracks:
- Page title
- Wikipedia page ID
- Revision ID
- Timestamp of retrieval
- Content hash
- Action (added/updated/removed)
- Training usage metadata

Example changelog entry:
```json
{
  "title": "Machine Learning",
  "page_id": "12345",
  "revision_id": "98765",
  "timestamp": "2024-02-12T15:30:00Z",
  "content_hash": "sha256_hash",
  "action": "added",
  "training_metadata": {
    "used_in_training": true,
    "training_timestamp": "2024-02-12T16:45:00Z",
    "model_checkpoint": "checkpoint_hash"
  }
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
