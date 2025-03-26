"""
Migration script to convert existing changelog.json to SQLite database.
This script reads the JSON file and populates the SQLite database with its data.
"""
import json
import os
import logging
import sqlite3
from datetime import datetime
from db_schema import init_db, get_db_connection

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Path to the original changelog.json file
CHANGELOG_JSON_PATH = os.environ.get("CHANGELOG_JSON_PATH", "changelog.json")

def load_json_data(json_path):
    """
    Load data from the changelog JSON file.
    
    Args:
        json_path (str): Path to the changelog JSON file
    
    Returns:
        dict: The loaded JSON data
    """
    logger.info(f"Loading JSON data from {json_path}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON data with {len(data)} entries")
        return data
    except FileNotFoundError:
        logger.warning(f"Changelog file {json_path} not found. Creating a new database.")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        raise

def migrate_json_to_sqlite(json_data):
    """
    Migrate data from JSON format to SQLite database.
    
    Args:
        json_data (dict): The JSON data to migrate
    """
    logger.info("Starting migration from JSON to SQLite")
    
    # Initialize the database
    init_db()
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Process each training run
        for run_id, run_data in json_data.items():
            # Extract training run information
            hyperparameters = json.dumps(run_data.get('hyperparameters', {}))
            metrics = json.dumps(run_data.get('metrics', {}))
            status = run_data.get('status', 'unknown')
            model_name = run_data.get('model_name', 'unknown')
            base_model = run_data.get('base_model', 'unknown')
            start_time = run_data.get('start_time', datetime.now().isoformat())
            end_time = run_data.get('end_time')
            git_commit = run_data.get('git_commit')
            
            # Insert training run
            cursor.execute('''
            INSERT INTO training_runs 
            (id, start_time, end_time, model_name, base_model, status, hyperparameters, metrics, git_commit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                int(run_id) if run_id.isdigit() else None,
                start_time,
                end_time,
                model_name,
                base_model,
                status,
                hyperparameters,
                metrics,
                git_commit
            ))
            
            # Get the new run_id if it was auto-generated
            if not run_id.isdigit():
                run_id = cursor.lastrowid
            else:
                run_id = int(run_id)
            
            # Process training examples
            examples = run_data.get('examples', [])
            for example in examples:
                input_text = example.get('input', '')
                target_text = example.get('target', '')
                example_type = example.get('type', 'unknown')
                metadata = json.dumps(example.get('metadata', {}))
                
                cursor.execute('''
                INSERT INTO training_examples
                (run_id, input_text, target_text, example_type, metadata)
                VALUES (?, ?, ?, ?, ?)
                ''', (run_id, input_text, target_text, example_type, metadata))
            
            # Process model outputs
            outputs = run_data.get('outputs', [])
            for output in outputs:
                input_text = output.get('input', '')
                output_text = output.get('output', '')
                timestamp = output.get('timestamp', datetime.now().isoformat())
                metadata = json.dumps(output.get('metadata', {}))
                
                cursor.execute('''
                INSERT INTO model_outputs
                (run_id, input_text, output_text, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
                ''', (run_id, input_text, output_text, timestamp, metadata))
        
        conn.commit()
        logger.info("Migration completed successfully")
    except sqlite3.Error as e:
        conn.rollback()
        logger.error(f"Database error during migration: {e}")
        raise
    finally:
        conn.close()

def main():
    """
    Main function to run the migration.
    """
    logger.info("Starting migration process")
    try:
        json_data = load_json_data(CHANGELOG_JSON_PATH)
        if json_data:
            migrate_json_to_sqlite(json_data)
            logger.info("Migration completed successfully")
        else:
            logger.info("No data to migrate. Created empty database.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    main()
