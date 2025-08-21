import sqlite3
import os

DB_NAME = "rag_app.db"


# ---------- Connection ----------
def get_db_connection():
    # Separate connection per call is fine for SQLite (esp. with FastAPI)
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


# ---------- Application logs ----------
def create_application_logs():
    conn = get_db_connection()
    conn.execute(
        '''CREATE TABLE IF NOT EXISTS application_logs
           (id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_query TEXT,
            gpt_response TEXT,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)'''
    )
    conn.commit()
    conn.close()


def insert_application_logs(session_id, user_query, gpt_response, model):
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
        (session_id, user_query, gpt_response, model),
    )
    conn.commit()
    conn.close()


def get_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at',
        (session_id,),
    )
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['gpt_response']},
        ])
    conn.close()
    return messages


# ---------- Document store ----------
def create_document_store():
    """
    Base DDL includes `path` so fresh DBs have it from the start.
    """
    conn = get_db_connection()
    conn.execute(
        '''CREATE TABLE IF NOT EXISTS document_store
           (id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            path TEXT)'''
    )
    conn.commit()
    conn.close()


def migrate_document_store():
    """
    Adds missing columns/indexes for existing DBs created before `path` existed.
    Safe to run multiple times.
    """
    conn = get_db_connection()
    # Add `path` column if not present
    try:
        conn.execute("ALTER TABLE document_store ADD COLUMN path TEXT")
    except Exception:
        pass
    # Helpful index for fast lookups; make UNIQUE if you want to forbid duplicate paths
    conn.execute("CREATE INDEX IF NOT EXISTS idx_document_store_path ON document_store(path)")
    conn.commit()
    conn.close()


def insert_document_record(filename: str) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('INSERT INTO document_store (filename) VALUES (?)', (filename,))
    file_id = cur.lastrowid
    conn.commit()
    conn.close()
    return file_id


def insert_document_record_with_path(filename: str, path: str) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO document_store (filename, path) VALUES (?, ?)',
        (filename, os.path.abspath(path)),
    )
    file_id = cur.lastrowid
    conn.commit()
    conn.close()
    return file_id


def delete_document_record(file_id: int) -> bool:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    conn.commit()
    changed = cur.rowcount
    conn.close()
    return changed > 0


def get_all_documents():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_document_by_path(path: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        'SELECT id, filename, path, upload_timestamp FROM document_store WHERE path = ?',
        (os.path.abspath(path),),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_documents_with_path():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, filename, path, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ---------- One-time init (safe to import multiple times) ----------
create_application_logs()
create_document_store()
migrate_document_store()
