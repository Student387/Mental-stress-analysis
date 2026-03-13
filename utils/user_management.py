"""User authentication and management utilities."""
import sqlite3
import uuid
from werkzeug.security import generate_password_hash, check_password_hash


def get_db():
    """Get database connection."""
    from utils.database import get_connection
    return get_connection()


def init_users():
    """Create users table if not exists."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def create_user(username, email, password):
    """Register new user. Returns (user, error)."""
    init_users()
    user_id = str(uuid.uuid4())
    password_hash = generate_password_hash(password, method='pbkdf2:sha256')
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?)",
            (user_id, username.strip(), email.strip().lower(), password_hash)
        )
        conn.commit()
        return {'id': user_id, 'username': username, 'email': email}, None
    except sqlite3.IntegrityError as e:
        return None, "Username or email already exists"
    finally:
        conn.close()


def get_user_by_credentials(username_or_email, password):
    """Authenticate user. Returns user dict or None."""
    init_users()
    conn = get_db()
    cur = conn.execute(
        "SELECT id, username, email, password_hash FROM users WHERE username = ? OR email = ?",
        (username_or_email.strip(), username_or_email.strip().lower())
    )
    row = cur.fetchone()
    conn.close()
    if row and check_password_hash(row[3], password):
        return {'id': row[0], 'username': row[1], 'email': row[2]}
    return None


def get_user_by_id(user_id):
    """Get user by id."""
    init_users()
    conn = get_db()
    cur = conn.execute(
        "SELECT id, username, email FROM users WHERE id = ?",
        (user_id,)
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return {'id': row[0], 'username': row[1], 'email': row[2]}
    return None
