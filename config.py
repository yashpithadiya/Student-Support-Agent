"""Configuration. Secrets come from environment variables or a local .env file
(the course notebooks used Colab Secrets / getpass; this is the local equivalent)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# --- Model (same provider/model as the course notebooks) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = os.environ.get("MODEL", "openai/gpt-oss-20b")

# --- Agent loop limits ---
MAX_TURNS = int(os.environ.get("MAX_TURNS", "6"))          # termination guard (Unit I)
HISTORY_TURNS = int(os.environ.get("HISTORY_TURNS", "3"))  # recent user/assistant pairs replayed
MEMORY_TOP_K = 3                                           # max memory items per request
POLICY_TOP_K = 2                                           # max policy chunks per search

# --- Local data (all fictional) ---
DATA_DIR = BASE_DIR / "data"
STUDENTS_CSV = DATA_DIR / "sample_students.csv"
POLICY_MD = DATA_DIR / "student_policy.md"
GOLDEN_DATASET = DATA_DIR / "golden_dataset.json"
DB_PATH = Path(os.environ.get("AGENT_DB_PATH", BASE_DIR / "data" / "agent.db"))  # gitignored
TRACE_FILE = Path(os.environ.get("TRACE_FILE", BASE_DIR / "traces" / "traces.jsonl"))  # gitignored


def validate_config() -> list[str]:
    """Return a list of human-readable setup problems (empty list = OK)."""
    problems = []
    if not GROQ_API_KEY:
        problems.append(
            "GROQ_API_KEY is not set. Copy `.env.example` to `.env`, paste your key "
            "(free at https://console.groq.com) and restart."
        )
    for p in (STUDENTS_CSV, POLICY_MD):
        if not p.exists():
            problems.append(f"Missing data file: {p}")
    return problems
