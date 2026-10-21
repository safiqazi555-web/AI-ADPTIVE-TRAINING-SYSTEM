"""Shared settings for the AI-Based Adaptive Training System."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
DB_PATH = DATA_DIR / "training_system.db"
DATASET_PATH = DATA_DIR / "trainee_dataset.csv"

# The 7 input features of the ANN (Table 2 of the report)
FEATURES = [
    "avg_quiz_score",        # average quiz score (0-100)
    "last_quiz_score",       # most recent quiz score (0-100)
    "time_spent",            # total minutes spent on lessons
    "attempts_per_quiz",     # average attempts needed per quiz
    "lessons_completed",     # percentage of topics completed (0-100)
    "login_frequency",       # logins per week
    "days_since_last_login", # days since the trainee was last active
]
FEATURE_LABELS = {
    "avg_quiz_score": "Average quiz score",
    "last_quiz_score": "Last quiz score",
    "time_spent": "Time spent (min)",
    "attempts_per_quiz": "Attempts per quiz",
    "lessons_completed": "Lessons completed (%)",
    "login_frequency": "Logins per week",
    "days_since_last_login": "Days since last login",
}

TARGET = "level"
LEVELS = ["Weak", "Average", "Strong"]          # class index 0, 1, 2
DIFFICULTIES = ["Easy", "Medium", "Hard"]
LEVEL_TO_DIFFICULTY = {"Weak": "Easy", "Average": "Medium", "Strong": "Hard"}

PASS_MARK = 60          # quiz score (%) needed to pass a lesson
REVISE_MARK = 50        # below this the topic is sent back for revision
RANDOM_STATE = 42

# Default accounts created on first start (change them if you like)
ADMIN_EMAIL = "admin@demo.com"
ADMIN_PASSWORD = "admin123"
DEMO_PASSWORD = "demo123"
