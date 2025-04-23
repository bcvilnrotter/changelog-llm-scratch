---
title: Changelog LLM Chatbot
emoji: 🤖
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.50.2
python_version: "3.10"
app_file: app.py
pinned: false
---

# Changelog LLM Chatbot

This is a custom transformer model trained on Wikipedia data, focusing on basic concepts and common knowledge.

## Model Details
- Custom transformer architecture
- Trained on curated Wikipedia articles
- Updated weekly with new training data

## Usage
Simply type your message in the chat interface and press enter. The model will generate a response based on its training.

Example queries:
- Tell me about basic physics concepts
- Explain how simple machines work
- What are some common English words?

## Updates
This Space is automatically updated every Sunday at midnight UTC with the latest model weights and improvements.

## Technical Details
- Model is updated weekly from the latest training runs
- Uses a custom transformer architecture
- Trained on carefully curated Wikipedia articles
- Optimized for educational content and basic concepts
- Uses SQLite database for tracking training data and model performance

## Database Migration
The training data tracking system has been migrated from JSON files to SQLite:

1. **Why SQLite?**
   - Better performance for large datasets
   - Structured query capabilities
   - Transaction support
   - Reduced memory footprint

2. **Migration Process**
   - Run `python migrate_to_sqlite.py` to convert existing JSON data to SQLite
   - The original JSON format is preserved for backward compatibility
   - All new data is stored in the SQLite database by default

3. **Using the Database**
   - The ChangelogDB class provides a high-level interface similar to the original JSON-based logger
   - For direct database access, use the functions in `db_utils.py`
   - Run `python test_db.py` to verify the database functionality

## Troubleshooting

If you encounter any issues with the training process or database, please refer to the [Troubleshooting Guide](TROUBLESHOOTING.md) for solutions to common problems.

The guide covers:
- Database schema issues
- Tokenizer problems
- Preparing for training
- GitHub Actions workflow
- Manual fixes and debugging
