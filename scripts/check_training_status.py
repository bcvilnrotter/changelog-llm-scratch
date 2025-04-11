#!/usr/bin/env python3
"""
Script to check if pages are being marked as used in training.
"""

import sys
import json
from pathlib import Path

def process_changelog_chunk(chunk):
    """Process a chunk of the changelog to count training usage."""
    used = 0
    unused = 0
    training_metadata = None
    last_training_time = None
    
    try:
        # Try to find complete JSON objects in the chunk
        start = chunk.find('"training_metadata":{')
        while start != -1:
            # Find the end of the training_metadata object
            brace_count = 1
            pos = start + len('"training_metadata":{')
            while brace_count > 0 and pos < len(chunk):
                if chunk[pos] == '{':
                    brace_count += 1
                elif chunk[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            if brace_count == 0:
                metadata_str = chunk[start:pos]
                try:
                    # Extract just the training_metadata object
                    metadata = json.loads('{' + metadata_str + '}')
                    training_time = metadata['training_metadata'].get('training_timestamp')
                    
                    if metadata['training_metadata'].get('used_in_training'):
                        used += 1
                        # Keep track of most recent training
                        if training_time and (not last_training_time or training_time > last_training_time):
                            last_training_time = training_time
                            training_metadata = metadata['training_metadata']
                    else:
                        unused += 1
                except json.JSONDecodeError:
                    pass
            
            start = chunk.find('"training_metadata":{', pos)
    except Exception as e:
        print(f"Error processing chunk: {e}")
    
    return used, unused, training_metadata, last_training_time

def main():
    changelog_path = Path("data/changelog.db")
    
    if not changelog_path.exists():
        print("\nStatus: No changelog.db file found yet.")
        print("This is normal if:")
        print("1. Training is still in its first epoch")
        print("2. Training hasn't reached the point of saving metrics")
        print("\nPlease check again after training progresses further.")
        return
    
    print("\nChecking training metrics in changelog.db...")
    print("(This file exists and contains Wikipedia page entries)")
    
    total_used = 0
    total_unused = 0
    latest_metadata = None
    latest_training_time = None
    entries_found = False
    
    # Process the file in chunks
    chunk_size = 1024 * 1024  # 1MB chunks
    with open(changelog_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
                
            used, unused, metadata, training_time = process_changelog_chunk(chunk)
            total_used += used
            total_unused += unused
            if training_time and (not latest_training_time or training_time > latest_training_time):
                latest_training_time = training_time
                latest_metadata = metadata
    
    print(f"\nResults:")
    print(f"- Pages marked as used in training: {total_used}")
    print(f"- Pages not yet used in training: {total_unused}")
    
    if total_used == 0 and total_unused == 0:
        print("\nNote: No training metadata entries found.")
        print("This could mean:")
        print("1. Training is still in progress and hasn't saved metrics yet")
        print("2. The training process hasn't reached the point of updating the changelog")
        print("3. There might be an issue with the training metrics collection")
    
    if latest_metadata:
        print("\nMost recent training metadata:")
        print(f"Training timestamp: {latest_metadata.get('training_timestamp', 'Unknown')}")
        print(f"Model checkpoint: {latest_metadata.get('model_checkpoint', 'Unknown')}")
        if 'average_loss' in latest_metadata:
            print(f"Average loss: {latest_metadata['average_loss']}")
        if 'relative_loss' in latest_metadata:
            print(f"Relative loss improvement: {latest_metadata['relative_loss'] * 100:.2f}%")

if __name__ == "__main__":
    main()
