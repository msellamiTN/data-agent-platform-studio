import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
SAMPLE_SALES_CSV = DATA_DIR / "sales_demo.csv"

MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "384"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_SQL_ROWS = 100

WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
