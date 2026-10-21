"""SQLite database layer (Section 6.3 / Table 6 of the report).

Tables: users, lessons, quiz_results, progress  (+ login_log for login activity).
"""
import hashlib
import hmac
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

import pandas as pd

from config import (ADMIN_EMAIL, ADMIN_PASSWORD, DATA_DIR, DB_PATH, DEMO_PASSWORD,
                    DIFFICULTIES, PASS_MARK)
from lesson_content import LESSONS, TOPICS

TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def _now() -> datetime:
    return datetime.now().replace(microsecond=0)


@contextmanager
def get_conn():
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return f"pbkdf2${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt, digest = stored.split("$")
        new = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 120_000)
        return hmac.compare_digest(new.hex(), digest)
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'trainee',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lessons (
    lesson_id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    content TEXT NOT NULL,
    UNIQUE (topic, difficulty)
);
CREATE TABLE IF NOT EXISTS quiz_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    lesson_id INTEGER NOT NULL REFERENCES lessons(lesson_id),
    score REAL NOT NULL,
    time_spent REAL NOT NULL,
    attempts INTEGER NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (
    user_id INTEGER PRIMARY KEY REFERENCES users(user_id),
    lessons_completed REAL NOT NULL DEFAULT 0,
    predicted_level TEXT,
    confidence REAL,
    last_updated TEXT
);
CREATE TABLE IF NOT EXISTS login_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    timestamp TEXT NOT NULL
);
"""


def init_db(seed: bool = True) -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        for topic in TOPICS:
            for diff in DIFFICULTIES:
                conn.execute("INSERT OR IGNORE INTO lessons (topic, difficulty, content) VALUES (?,?,?)",
                             (topic, diff, LESSONS[(topic, diff)]["content"]))
        has_admin = conn.execute("SELECT 1 FROM users WHERE role='admin'").fetchone()
    if not has_admin:
        create_user("Administrator", ADMIN_EMAIL, ADMIN_PASSWORD, "admin")
        if seed:
            seed_demo_data()


# --------------------------------------------------------------------------- #
# Users and login
# --------------------------------------------------------------------------- #
def create_user(name: str, email: str, password: str, role: str = "trainee", created_at=None):
    name, email = name.strip(), email.strip().lower()
    if not name or "@" not in email or len(password) < 4:
        return False, "Enter a name, a valid email and a password of at least 4 characters."
    try:
        with get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?,?,?,?,?)",
                (name, email, hash_password(password), role, (created_at or _now()).strftime(TS_FORMAT)))
            if role == "trainee":
                conn.execute("INSERT INTO progress (user_id, lessons_completed) VALUES (?, 0)", (cur.lastrowid,))
        return True, cur.lastrowid
    except sqlite3.IntegrityError:
        return False, "This email is already registered."


def authenticate(email: str, password: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    if row and verify_password(password, row["password_hash"]):
        log_login(row["user_id"])
        return dict(row)
    return None


def log_login(user_id: int, when: datetime = None) -> None:
    with get_conn() as conn:
        conn.execute("INSERT INTO login_log (user_id, timestamp) VALUES (?,?)",
                     (user_id, (when or _now()).strftime(TS_FORMAT)))


# --------------------------------------------------------------------------- #
# Lessons and quiz results
# --------------------------------------------------------------------------- #
def lesson_id(topic: str, difficulty: str) -> int:
    with get_conn() as conn:
        return conn.execute("SELECT lesson_id FROM lessons WHERE topic=? AND difficulty=?",
                            (topic, difficulty)).fetchone()[0]


def save_quiz_result(user_id: int, topic: str, difficulty: str, score: float,
                     time_spent_min: float, when: datetime = None) -> int:
    """Store one quiz attempt. `attempts` = how many times this user took this lesson's quiz."""
    lid = lesson_id(topic, difficulty)
    with get_conn() as conn:
        prev = conn.execute("SELECT COUNT(*) FROM quiz_results WHERE user_id=? AND lesson_id=?",
                            (user_id, lid)).fetchone()[0]
        conn.execute("INSERT INTO quiz_results (user_id, lesson_id, score, time_spent, attempts, timestamp) "
                     "VALUES (?,?,?,?,?,?)",
                     (user_id, lid, round(score, 1), round(time_spent_min, 2), prev + 1,
                      (when or _now()).strftime(TS_FORMAT)))
    return prev + 1


def get_results(user_id: int) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT r.result_id, l.topic, l.difficulty, r.score, r.time_spent, r.attempts, r.timestamp "
            "FROM quiz_results r JOIN lessons l USING (lesson_id) WHERE r.user_id = ? "
            "ORDER BY r.timestamp, r.result_id", conn, params=(user_id,))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_logins(user_id: int) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT timestamp FROM login_log WHERE user_id=? ORDER BY timestamp",
                               conn, params=(user_id,))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_user(user_id: int) -> dict:
    with get_conn() as conn:
        return dict(conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone())


# --------------------------------------------------------------------------- #
# Features for the ANN (Table 2)
# --------------------------------------------------------------------------- #
def passed_topics(results: pd.DataFrame) -> int:
    if results.empty:
        return 0
    return int(results[results["score"] >= PASS_MARK]["topic"].nunique())


def compute_features(user_id: int):
    """Turn a trainee's stored activity into the 7 ANN features. Returns None if no quiz was taken."""
    results = get_results(user_id)
    if results.empty:
        return None
    now = _now()
    logins = get_logins(user_id)
    created = datetime.strptime(get_user(user_id)["created_at"], TS_FORMAT)

    window_days = 28
    recent = logins[logins["timestamp"] >= pd.Timestamp(now - timedelta(days=window_days))]
    age_days = max((now - created).days, 0)
    weeks = max(min(age_days, window_days) / 7, 1.0)
    days_idle = (now - logins["timestamp"].max().to_pydatetime()).days if not logins.empty else 0

    return {
        "avg_quiz_score": round(float(results["score"].mean()), 1),
        "last_quiz_score": round(float(results["score"].iloc[-1]), 1),
        "time_spent": round(float(results["time_spent"].sum()), 1),
        "attempts_per_quiz": round(len(results) / results["topic"].add(results["difficulty"]).nunique(), 2),
        "lessons_completed": round(passed_topics(results) / len(TOPICS) * 100, 1),
        "login_frequency": round(len(recent) / weeks, 1),
        "days_since_last_login": int(max(days_idle, 0)),
    }


# --------------------------------------------------------------------------- #
# Progress table
# --------------------------------------------------------------------------- #
def save_progress(user_id: int, lessons_completed: float, level: str, confidence: float) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO progress (user_id, lessons_completed, predicted_level, confidence, last_updated) "
            "VALUES (?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET lessons_completed=excluded.lessons_completed, "
            "predicted_level=excluded.predicted_level, confidence=excluded.confidence, "
            "last_updated=excluded.last_updated",
            (user_id, lessons_completed, level, confidence, _now().strftime(TS_FORMAT)))


def get_progress(user_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM progress WHERE user_id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def list_trainees() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT u.user_id, u.name, u.email, u.created_at, p.lessons_completed, p.predicted_level, "
            "p.confidence, p.last_updated FROM users u LEFT JOIN progress p USING (user_id) "
            "WHERE u.role = 'trainee' ORDER BY u.name", conn)


# --------------------------------------------------------------------------- #
# Demo data (three trainees with history, so the admin view is not empty)
# --------------------------------------------------------------------------- #
DEMO_TRAINEES = {
    # name, email, (days_ago, topic, difficulty, score, minutes), login days-ago list
    "Ali Khan (demo: weak)": dict(
        email="ali@demo.com",
        logins=[20, 14, 9],
        quizzes=[(19, "Python Basics", "Easy", 40, 22), (19, "Python Basics", "Easy", 50, 18),
                 (14, "Python Basics", "Easy", 55, 20), (9, "Control Flow", "Easy", 35, 25),
                 (9, "Control Flow", "Easy", 45, 15)]),
    "Sara Ahmed (demo: average)": dict(
        email="sara@demo.com",
        logins=[20, 17, 14, 11, 9, 6, 3, 1],
        quizzes=[(20, "Python Basics", "Medium", 70, 25), (17, "Python Basics", "Medium", 75, 20),
                 (14, "Control Flow", "Medium", 65, 30), (9, "Functions", "Medium", 50, 28),
                 (6, "Functions", "Medium", 65, 22), (3, "Data Structures", "Medium", 60, 26)]),
    "Hamza Raza (demo: strong)": dict(
        email="hamza@demo.com",
        logins=[20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1, 0],
        quizzes=[(20, "Python Basics", "Hard", 100, 24), (16, "Control Flow", "Hard", 100, 30),
                 (12, "Functions", "Hard", 75, 33), (8, "Data Structures", "Hard", 100, 28),
                 (3, "Functions", "Hard", 100, 18)]),
}


def seed_demo_data() -> None:
    now = _now()
    for name, d in DEMO_TRAINEES.items():
        ok, uid = create_user(name, d["email"], DEMO_PASSWORD, "trainee", created_at=now - timedelta(days=21))
        if not ok:
            continue
        for days_ago in d["logins"]:
            log_login(uid, now - timedelta(days=days_ago, hours=2))
        for days_ago, topic, diff, score, minutes in d["quizzes"]:
            save_quiz_result(uid, topic, diff, score, minutes, now - timedelta(days=days_ago, hours=2))
