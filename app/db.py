"""
db - SQLite 数据库访问层（用户 + 历史评论）

使用 Python 内置 sqlite3，无需额外依赖。
"""
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / 'tmf.db'


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """建表（幂等）"""
    conn = get_conn()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        );
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            sentiment TEXT,
            aspect TEXT,
            sentiment_conf REAL,
            aspect_conf REAL,
            used_fallback INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        );
    ''')
    conn.commit()
    conn.close()


def create_user(username, password_hash) -> bool:
    """注册用户；用户名已存在返回 False"""
    conn = get_conn()
    try:
        conn.execute(
            'INSERT INTO users (username, password_hash) VALUES (?, ?)',
            (username, password_hash),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user(username):
    """按用户名查用户"""
    conn = get_conn()
    row = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()
    return row


def add_history(user_id, text, sentiment, aspect, sentiment_conf, aspect_conf, used_fallback):
    conn = get_conn()
    conn.execute(
        'INSERT INTO history (user_id, text, sentiment, aspect, sentiment_conf, aspect_conf, used_fallback) '
        'VALUES (?, ?, ?, ?, ?, ?, ?)',
        (user_id, text, sentiment, aspect, sentiment_conf, aspect_conf, 1 if used_fallback else 0),
    )
    conn.commit()
    conn.close()


def get_history(user_id, limit=50):
    conn = get_conn()
    rows = conn.execute(
        'SELECT * FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?',
        (user_id, limit),
    ).fetchall()
    conn.close()
    return rows
