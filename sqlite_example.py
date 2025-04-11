#!/usr/bin/env python3
"""
Example script demonstrating how to use the SQLite-based ChangelogDB.
This shows how to perform common operations with the SQLite database.
"""

import os
import sqlite3
import hashlib
import datetime
import json
from typing import Dict, List, Optional, Any

class ChangelogDB:
    """A class to interact with the changelog SQLite database."""
    
    def __init__(self, db_path=None):
        """
        Initialize the database connection.
        
        Args:
            db_path (str, optional): Path to the SQLite database file
        """
        self.db_path = db_path or os.environ.get("CHANGELOG_DB_PATH", "changelog.db")
    
    def _get_connection(self):
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _compute_hash(self, content: str) -> str:
        """
        Compute a SHA-256 hash of content.
        
        Args:
            content (str): Content to hash
        
        Returns:
            str: Hexadecimal hash of content
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def log_page(self, title: str, page_id: str, revision_id: str, 
                 content: str, action: str = "added") -> Dict:
        """
        Log a Wikipedia page operation.
        
        Args:
            title (str): Page title
            page_id (str): Wikipedia page ID
            revision_id (str): Wikipedia revision ID
            content (str): Page content
            action (str): Operation type (added/updated/removed)
        
        Returns:
            dict: The created changelog entry
        """
        content_hash = self._compute_hash(content)
        timestamp = datetime.datetime.now().isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Insert entry
            cursor.execute('''
            INSERT INTO entries 
            (title, page_id, revision_id, content_hash, action, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (title, page_id, revision_id, content_hash, action, timestamp))
            
            # Get entry ID
            entry_id = cursor.lastrowid
            
            # Create empty training metadata
            cursor.execute('''
            INSERT INTO training_metadata (entry_id, used_in_training)
            VALUES (?, 0)
            ''', (entry_id,))
            
            conn.commit()
            
            # Return entry as dictionary
            return {
                'id': entry_id,
                'title': title,
                'page_id': page_id,
                'revision_id': revision_id,
                'content_hash': content_hash,
                'action': action,
                'timestamp': timestamp,
                'is_revision': False,
                'training_metadata': {
                    'used_in_training': False
                }
            }
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def log_revision(self, title: str, page_id: str, revision_id: str, 
                     content: str, parent_id: str, revision_number: int) -> Dict:
        """
        Log a revision of a Wikipedia page.
        
        Args:
            title (str): Page title
            page_id (str): Wikipedia page ID
            revision_id (str): Wikipedia revision ID
            content (str): Page content
            parent_id (str): ID of the parent page
            revision_number (int): Revision number (1-5, with 1 being most recent)
        
        Returns:
            dict: The created changelog entry
        """
        content_hash = self._compute_hash(content)
        timestamp = datetime.datetime.now().isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Insert entry as revision
            cursor.execute('''
            INSERT INTO entries 
            (title, page_id, revision_id, content_hash, action, timestamp, is_revision, parent_id, revision_number)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            ''', (title, page_id, revision_id, content_hash, 'added', timestamp, parent_id, revision_number))
            
            # Get entry ID
            entry_id = cursor.lastrowid
            
            # Create empty training metadata
            cursor.execute('''
            INSERT INTO training_metadata (entry_id, used_in_training)
            VALUES (?, 0)
            ''', (entry_id,))
            
            conn.commit()
            
            # Return entry as dictionary
            return {
                'id': entry_id,
                'title': title,
                'page_id': page_id,
                'revision_id': revision_id,
                'content_hash': content_hash,
                'action': 'added',
                'timestamp': timestamp,
                'is_revision': True,
                'parent_id': parent_id,
                'revision_number': revision_number,
                'training_metadata': {
                    'used_in_training': False
                }
            }
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_page_history(self, page_id: str) -> List[Dict]:
        """
        Get all changelog entries for a specific page.
        
        Args:
            page_id (str): Wikipedia page ID
        
        Returns:
            list: List of changelog entries for the page
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Get entries for this page ID
            cursor.execute('''
            SELECT e.*, tm.used_in_training, tm.model_checkpoint, tm.average_loss, tm.relative_loss
            FROM entries e
            LEFT JOIN training_metadata tm ON e.id = tm.entry_id
            WHERE e.page_id = ?
            ORDER BY e.timestamp DESC
            ''', (page_id,))
            
            entries = []
            for row in cursor.fetchall():
                entry = dict(row)
                
                # Format training metadata
                training_metadata = {
                    'used_in_training': bool(entry.pop('used_in_training', 0)),
                    'model_checkpoint': entry.pop('model_checkpoint', None),
                    'average_loss': entry.pop('average_loss', None),
                    'relative_loss': entry.pop('relative_loss', None)
                }
                
                # Get token impacts if used in training
                if training_metadata['used_in_training']:
                    token_impact = self._get_token_impact(entry['id'])
                    if token_impact:
                        training_metadata['token_impact'] = token_impact
                
                entry['training_metadata'] = training_metadata
                entry['is_revision'] = bool(entry['is_revision'])
                entries.append(entry)
            
            return entries
        finally:
            conn.close()
    
    def _get_token_impact(self, entry_id: int) -> Optional[Dict[str, float]]:
        """
        Get token impact data for an entry.
        
        Args:
            entry_id (int): Entry ID
        
        Returns:
            dict: Token impact dictionary or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT t.token, t.impact
            FROM token_impact t
            JOIN training_metadata tm ON t.metadata_id = tm.id
            WHERE tm.entry_id = ?
            ''', (entry_id,))
            
            rows = cursor.fetchall()
            if not rows:
                return None
            
            return {row['token']: row['impact'] for row in rows}
        finally:
            conn.close()
    
    def mark_used_in_training(self, page_ids: List[str], model_checkpoint: str, 
                              training_metrics: Optional[Dict] = None) -> None:
        """
        Mark pages as used in training with associated model checkpoint.
        
        Args:
            page_ids (list): List of page IDs used in training
            model_checkpoint (str): Hash or identifier of the model checkpoint
            training_metrics (dict, optional): Dictionary of training metrics by page ID
        """
        if training_metrics is None:
            training_metrics = {}
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            for page_id in page_ids:
                # Get the entry ID for this page
                cursor.execute('SELECT id FROM entries WHERE page_id = ?', (page_id,))
                row = cursor.fetchone()
                if not row:
                    continue
                
                entry_id = row['id']
                
                # Get metrics for this page
                metrics = training_metrics.get(page_id, {})
                average_loss = metrics.get('average_loss')
                relative_loss = metrics.get('relative_loss')
                
                # Update training metadata
                cursor.execute('''
                UPDATE training_metadata
                SET used_in_training = 1, 
                    model_checkpoint = ?,
                    average_loss = ?,
                    relative_loss = ?
                WHERE entry_id = ?
                ''', (model_checkpoint, average_loss, relative_loss, entry_id))
                
                # Get training metadata ID
                cursor.execute('SELECT id FROM training_metadata WHERE entry_id = ?', (entry_id,))
                metadata_id = cursor.fetchone()['id']
                
                # Add token impact data if available
                token_impact = metrics.get('token_impact', {})
                for token, impact in token_impact.items():
                    cursor.execute('''
                    INSERT OR REPLACE INTO token_impact (metadata_id, token, impact)
                    VALUES (?, ?, ?)
                    ''', (metadata_id, token, impact))
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_unused_pages(self) -> List[Dict]:
        """
        Get all pages that haven't been used in training.
        
        Returns:
            list: List of changelog entries for unused pages
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT e.*
            FROM entries e
            JOIN training_metadata tm ON e.id = tm.entry_id
            WHERE tm.used_in_training = 0
            ORDER BY e.timestamp DESC
            ''')
            
            entries = []
            for row in cursor.fetchall():
                entry = dict(row)
                entry['training_metadata'] = {'used_in_training': False}
                entry['is_revision'] = bool(entry['is_revision'])
                entries.append(entry)
            
            return entries
        finally:
            conn.close()

def main():
    """Demonstrate how to use the ChangelogDB class."""
    # Create a database connection
    db = ChangelogDB()
    
    print("Testing SQLite-based ChangelogDB...")
    
    try:
        # Log a new page
        print("\nAdding a new page...")
        entry = db.log_page(
            title="SQLite Test Page",
            page_id="sqlite-test-1",
            revision_id="rev1",
            content="This is a test page for SQLite migration.",
            action="added"
        )
        print(f"Added page with ID: {entry['page_id']}")
        
        # Log a revision
        print("\nAdding a revision...")
        revision = db.log_revision(
            title="SQLite Test Page (Revision)",
            page_id="sqlite-test-1-rev",
            revision_id="rev2",
            content="This is a revision of the test page.",
            parent_id="sqlite-test-1",
            revision_number=1
        )
        print(f"Added revision with ID: {revision['page_id']}")
        
        # Get page history
        print("\nGetting page history...")
        history = db.get_page_history("sqlite-test-1")
        print(f"Found {len(history)} entries in history")
        
        # Get unused pages
        print("\nGetting unused pages...")
        unused = db.get_unused_pages()
        print(f"Found {len(unused)} unused pages")
        
        # Mark as used in training
        print("\nMarking pages as used in training...")
        db.mark_used_in_training(
            page_ids=["sqlite-test-1"],
            model_checkpoint="test-model-1",
            training_metrics={
                "sqlite-test-1": {
                    "average_loss": 0.25,
                    "relative_loss": 0.5,
                    "token_impact": {
                        "test_token_1": 0.8,
                        "test_token_2": 0.6
                    }
                }
            }
        )
        print("Marked page as used in training")
        
        # Get page history again to see the updated training metadata
        print("\nGetting updated page history...")
        history = db.get_page_history("sqlite-test-1")
        print("Training metadata for first entry:")
        print(json.dumps(history[0]['training_metadata'], indent=2))
        
        # Get unused pages again (should have one less)
        print("\nGetting unused pages again...")
        unused_after = db.get_unused_pages()
        print(f"Found {len(unused_after)} unused pages (should be one less than before)")
        
        print("\nSQLite migration example completed successfully!")
    
    except Exception as e:
        print(f"Error during demonstration: {e}")

if __name__ == "__main__":
    main()