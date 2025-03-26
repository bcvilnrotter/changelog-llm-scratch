#!/usr/bin/env python3
"""
Flask application for the Changelog-LLM project.
This provides a simple web interface to demonstrate the SQLite migration.
"""

import os
import json
import sqlite3
from pathlib import Path
from flask import Flask, render_template, jsonify, request

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")

# Database path
db_path = os.path.join(os.path.dirname(__file__), "changelog.db")

def get_db_connection():
    """Get a database connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create entries table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            page_id TEXT UNIQUE NOT NULL,
            revision_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            action TEXT DEFAULT 'added',
            is_revision BOOLEAN DEFAULT 0,
            parent_id TEXT,
            revision_number INTEGER
        )
    ''')
    
    # Create training metadata table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id INTEGER NOT NULL,
            used_in_training BOOLEAN DEFAULT 0,
            training_timestamp TEXT,
            model_checkpoint TEXT,
            average_loss REAL,
            relative_loss REAL,
            FOREIGN KEY (entry_id) REFERENCES entries (id) ON DELETE CASCADE
        )
    ''')
    
    # Create training runs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            base_model TEXT NOT NULL,
            hyperparameters TEXT NOT NULL,
            git_commit TEXT,
            status TEXT DEFAULT 'running',
            timestamp TEXT NOT NULL,
            metrics TEXT DEFAULT '{}'
        )
    ''')
    
    # Create training examples table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            input_text TEXT NOT NULL,
            target_text TEXT NOT NULL,
            example_type TEXT DEFAULT 'general',
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (run_id) REFERENCES training_runs (id) ON DELETE CASCADE
        )
    ''')
    
    # Create model outputs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            input_text TEXT NOT NULL,
            output_text TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (run_id) REFERENCES training_runs (id) ON DELETE CASCADE
        )
    ''')
    
    # Create token impacts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS token_impacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metadata_id INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            FOREIGN KEY (metadata_id) REFERENCES training_metadata (id) ON DELETE CASCADE
        )
    ''')
    
    # Create top tokens table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS top_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_impact_id INTEGER NOT NULL,
            token_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            impact REAL NOT NULL,
            context_start INTEGER,
            context_end INTEGER,
            FOREIGN KEY (token_impact_id) REFERENCES token_impacts (id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

# Always initialize the database to ensure tables exist
init_db()

@app.route('/')
def index():
    """Render the main page."""
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create a basic index.html if it doesn't exist
    index_html_path = os.path.join('templates', 'index.html')
    if not os.path.exists(index_html_path):
        with open(index_html_path, 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Changelog-LLM Database</title>
    <link rel="stylesheet" href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css">
    <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
    <div class="container mt-4">
        <h1>Changelog-LLM Database</h1>
        <p>This is a simple web interface for the Changelog-LLM SQLite database.</p>
        
        <div class="card mb-4">
            <div class="card-header">SQLite Migration Demo</div>
            <div class="card-body">
                <p>This project has successfully migrated from a JSON-based changelog to a SQLite database.</p>
                <p>The database is located at: <code>changelog.db</code></p>
                
                <hr>
                
                <h4>Database Structure</h4>
                <div id="tables" class="mb-4">Loading tables...</div>
                
                <h4>Add Sample Data</h4>
                <form id="sample-form" class="mb-4">
                    <div class="mb-3">
                        <label for="title" class="form-label">Page Title</label>
                        <input type="text" class="form-control" id="title" placeholder="Wikipedia Page Title">
                    </div>
                    <div class="mb-3">
                        <label for="page-id" class="form-label">Page ID</label>
                        <input type="text" class="form-control" id="page-id" placeholder="page123">
                    </div>
                    <div class="mb-3">
                        <label for="revision-id" class="form-label">Revision ID</label>
                        <input type="text" class="form-control" id="revision-id" placeholder="rev456">
                    </div>
                    <div class="mb-3">
                        <label for="content" class="form-label">Content</label>
                        <textarea class="form-control" id="content" rows="3" placeholder="Page content"></textarea>
                    </div>
                    <button type="submit" class="btn btn-primary">Add Sample Page</button>
                </form>
                
                <h4>Pages</h4>
                <div id="pages">Loading pages...</div>
            </div>
        </div>
    </div>
    
    <script>
        // Fetch database tables
        fetch('/api/tables')
            .then(response => response.json())
            .then(data => {
                const tablesEl = document.getElementById('tables');
                if (data.length === 0) {
                    tablesEl.innerHTML = 'No tables found.';
                    return;
                }
                
                let html = '<ul class="list-group">';
                data.forEach(table => {
                    html += `<li class="list-group-item">${table}</li>`;
                });
                html += '</ul>';
                tablesEl.innerHTML = html;
            })
            .catch(error => {
                document.getElementById('tables').innerHTML = `Error: ${error.message}`;
            });
        
        // Fetch pages
        function loadPages() {
            fetch('/api/pages')
                .then(response => response.json())
                .then(data => {
                    const pagesEl = document.getElementById('pages');
                    if (data.length === 0) {
                        pagesEl.innerHTML = 'No pages found.';
                        return;
                    }
                    
                    let html = '<div class="list-group">';
                    data.forEach(page => {
                        html += `
                            <div class="list-group-item">
                                <h5>${page.title}</h5>
                                <p>Page ID: ${page.page_id}</p>
                                <p>Revision ID: ${page.revision_id}</p>
                                <p>Date: ${page.timestamp}</p>
                                <p>Content Hash: ${page.content_hash}</p>
                            </div>
                        `;
                    });
                    html += '</div>';
                    pagesEl.innerHTML = html;
                })
                .catch(error => {
                    document.getElementById('pages').innerHTML = `Error: ${error.message}`;
                });
        }
        
        loadPages();
        
        // Handle form submission
        document.getElementById('sample-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const data = {
                title: document.getElementById('title').value,
                page_id: document.getElementById('page-id').value,
                revision_id: document.getElementById('revision-id').value,
                content: document.getElementById('content').value
            };
            
            fetch('/api/add-page', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                alert(result.message);
                loadPages();
                // Clear the form
                document.getElementById('sample-form').reset();
            })
            .catch(error => {
                alert('Error: ' + error.message);
            });
        });
    </script>
</body>
</html>
            ''')
    
    return render_template('index.html')

@app.route('/api/tables')
def get_tables():
    """Get a list of database tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row['name'] for row in cursor.fetchall()]
    
    conn.close()
    return jsonify(tables)

@app.route('/api/pages')
def get_pages():
    """Get all pages."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM entries ORDER BY timestamp DESC
    ''')
    
    pages = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify(pages)

@app.route('/api/add-page', methods=['POST'])
def add_page():
    """Add a sample page."""
    data = request.json
    
    if not all(k in data for k in ('title', 'page_id', 'revision_id', 'content')):
        return jsonify({'success': False, 'message': 'Missing required fields'}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Generate hash
        import hashlib
        content_hash = hashlib.sha256(data['content'].encode('utf-8')).hexdigest()
        
        # Generate timestamp
        import datetime
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        
        # Insert new page
        cursor.execute('''
            INSERT INTO entries (
                title, page_id, revision_id, timestamp, content_hash, action
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            data['title'],
            data['page_id'],
            data['revision_id'],
            timestamp,
            content_hash,
            'added'
        ))
        
        entry_id = cursor.lastrowid
        
        # Create initial training metadata record
        cursor.execute('''
            INSERT INTO training_metadata (entry_id, used_in_training)
            VALUES (?, 0)
        ''', (entry_id,))
        
        conn.commit()
        return jsonify({'success': True, 'message': 'Page added successfully'})
    
    except sqlite3.IntegrityError:
        conn.rollback()
        return jsonify({'success': False, 'message': 'Page ID already exists'}), 400
    
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
    
    finally:
        conn.close()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)