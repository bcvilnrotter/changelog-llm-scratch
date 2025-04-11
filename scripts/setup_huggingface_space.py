#!/usr/bin/env python3
"""
One-time setup script for creating a Hugging Face Space for the Changelog LLM Chatbot.
This script should be run manually after setting up the HF_TOKEN and HF_SPACE_NAME.
"""

import os
import sys
from huggingface_hub import create_repo, HfApi
from pathlib import Path
import shutil

def setup_space():
    # Get environment variables
    hf_token = os.environ.get('HF_TOKEN')
    space_name = os.environ.get('HF_SPACE_NAME')
    
    if not hf_token or not space_name:
        print("Error: Please set HF_TOKEN and HF_SPACE_NAME environment variables")
        print("Example (PowerShell):")
        print("  $env:HF_TOKEN = 'your_token_here'")
        print("  $env:HF_SPACE_NAME = 'username/space-name'")
        sys.exit(1)
    
    try:
        # Create temporary directory
        temp_dir = Path("temp_space_content")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()
        
        # Create the Space if it doesn't exist
        print(f"Ensuring Space exists: {space_name}")
        try:
            create_repo(
                space_name,
                repo_type="space",
                space_sdk="gradio",
                token=hf_token
            )
            print("Space created successfully!")
        except Exception as e:
            if "You already created this space repo" in str(e):
                print("Space already exists, proceeding with update...")
            else:
                raise
        
        # Initialize API
        api = HfApi(token=hf_token)
        
        # Create package structure
        training_dir = temp_dir / "training"
        training_dir.mkdir()
        (training_dir / "__init__.py").touch()
        
        # Copy necessary files
        base_path = Path(__file__).parent.parent
        shutil.copy2(base_path / "src/training/transformer.py", training_dir / "transformer.py")
        shutil.copy2(base_path / "src/training/tokenizer.py", training_dir / "tokenizer.py")
        shutil.copy2(base_path / "src/gradio_app.py", temp_dir / "app.py")
        shutil.copy2(base_path / "requirements.txt", temp_dir / "requirements.txt")
        
        # Copy model files
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        for file in (base_path / "models/final").glob("*"):
            shutil.copy2(file, model_dir / file.name)
        
        # Create README with metadata
        readme_content = """---
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
"""
        
        # Write README
        (temp_dir / "README.md").write_text(readme_content, encoding='utf-8')
        
        # Delete existing repo if it exists
        repo_dir = Path("temp_repo")
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        
        # Clone the repository fresh
        print("\nCloning Space repository...")
        repo_url = f"https://huggingface.co/spaces/{space_name}"
        os.system(f'git clone {repo_url} temp_repo')
        
        # Copy new files
        print("\nCopying new files...")
        for file in temp_dir.rglob("*"):
            if file.is_file():
                relative_path = str(file.relative_to(temp_dir)).replace('\\', '/')
                dest_path = repo_dir / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, dest_path)
                print(f"Copied {relative_path}")
                
                # Also copy to the root of the repo (without temp_repo prefix)
                root_dest_path = Path(relative_path)
                root_dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, root_dest_path)
        
        # Setup and push changes
        print("\nPushing changes...")
        os.chdir(repo_dir)
        os.system('git lfs install')
        os.system('git lfs track "*.bin"')  # Track large model files
        os.system('git add .gitattributes')
        os.system('git add .')
        os.system('git config --global user.email "github-actions[bot]@users.noreply.github.com"')
        os.system('git config --global user.name "github-actions[bot]"')
        os.system('git commit -m "Update Space with latest files"')
        os.system(f'git push https://oauth2:{hf_token}@huggingface.co/spaces/{space_name} main --force')
        os.chdir('..')
        
        print("\nSetup completed successfully!")
        print(f"Visit your Space at: https://huggingface.co/spaces/{space_name}")
        print("Please allow a few minutes for the Space to build and start running.")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        sys.exit(1)

if __name__ == "__main__":
    setup_space()
