"""SQLite database for users, assessments, and model metrics."""
import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'responses.db')
PENDING_COUNT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'pending_count.txt')


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            form_data TEXT,
            stress_level TEXT,
            confidence REAL,
            probabilities TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            responses TEXT NOT NULL,
            stress_level TEXT NOT NULL,
            stress_code INTEGER NOT NULL,
            confidence REAL,
            probabilities TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1 REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_response(form_data, stress_level, confidence, probabilities):
    init_db()
    conn = get_connection()
    conn.execute(
        "INSERT INTO responses (form_data, stress_level, confidence, probabilities) VALUES (?, ?, ?, ?)",
        (json.dumps(form_data), stress_level, confidence, json.dumps(probabilities))
    )
    conn.commit()
    conn.close()


def save_user_assessment(user_id, form_data, stress_level, stress_code, confidence, probabilities):
    init_db()
    conn = get_connection()
    conn.execute(
        """INSERT INTO user_assessments (user_id, responses, stress_level, stress_code, confidence, probabilities)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (user_id, json.dumps(form_data), stress_level, stress_code, confidence, json.dumps(probabilities))
    )
    conn.commit()
    conn.close()


def get_user_assessments(user_id, limit=50):
    init_db()
    conn = get_connection()
    cur = conn.execute(
        """SELECT id, responses, stress_level, stress_code, confidence, probabilities, created_at
           FROM user_assessments WHERE user_id = ? ORDER BY created_at DESC LIMIT ?""",
        (user_id, limit)
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            'id': r[0], 'responses': json.loads(r[1]) if r[1] else {},
            'stress_level': r[2], 'stress_code': r[3], 'confidence': r[4],
            'probabilities': json.loads(r[5]) if r[5] else {},
            'created_at': r[6]
        }
        for r in rows
    ]


def get_previous_assessment(user_id, exclude_id=None):
    """Get second-most-recent assessment for comparison (exclude current one)."""
    init_db()
    conn = get_connection()
    if exclude_id:
        cur = conn.execute(
            """SELECT id, responses, stress_level, stress_code, confidence, probabilities, created_at
               FROM user_assessments WHERE user_id = ? AND id != ? ORDER BY created_at DESC LIMIT 1""",
            (user_id, exclude_id)
        )
    else:
        cur = conn.execute(
            """SELECT id, responses, stress_level, stress_code, confidence, probabilities, created_at
               FROM user_assessments WHERE user_id = ? ORDER BY created_at DESC LIMIT 1 OFFSET 1""",
            (user_id,)
        )
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            'id': row[0], 'responses': json.loads(row[1]) if row[1] else {},
            'stress_level': row[2], 'stress_code': row[3], 'confidence': row[4],
            'probabilities': json.loads(row[5]) if row[5] else {},
            'created_at': row[6]
        }
    return None


def get_latest_assessment(user_id):
    init_db()
    conn = get_connection()
    cur = conn.execute(
        """SELECT id, responses, stress_level, stress_code, confidence, probabilities, created_at
           FROM user_assessments WHERE user_id = ? ORDER BY created_at DESC LIMIT 1""",
        (user_id,)
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            'id': row[0], 'responses': json.loads(row[1]) if row[1] else {},
            'stress_level': row[2], 'stress_code': row[3], 'confidence': row[4],
            'probabilities': json.loads(row[5]) if row[5] else {},
            'created_at': row[6]
        }
    return None


def save_model_metrics(model_name, metrics):
    init_db()
    conn = get_connection()
    conn.execute(
        """INSERT INTO model_metrics (model_name, accuracy, precision, recall, f1)
           VALUES (?, ?, ?, ?, ?)""",
        (model_name, metrics.get('accuracy'), metrics.get('precision'),
         metrics.get('recall'), metrics.get('f1'))
    )
    conn.commit()
    conn.close()


def get_pending_retrain_count():
    if not os.path.exists(PENDING_COUNT_PATH):
        return 0
    try:
        with open(PENDING_COUNT_PATH) as f:
            return int(f.read().strip())
    except Exception:
        return 0


def increment_pending_count():
    os.makedirs(os.path.dirname(PENDING_COUNT_PATH), exist_ok=True)
    count = get_pending_retrain_count() + 1
    with open(PENDING_COUNT_PATH, 'w') as f:
        f.write(str(count))
    return count


def reset_pending_count():
    if os.path.exists(PENDING_COUNT_PATH):
        os.remove(PENDING_COUNT_PATH)
