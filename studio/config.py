import os
from pathlib import Path

from dotenv import load_dotenv

from .runtime import positive_float_setting, positive_int_setting

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)
DATA_DIR = PROJECT_ROOT / "data"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
PIPELINE_ROOT = WORKSPACE_DIR / "pipeline"
CURRENT_PIPELINE_DIR = PIPELINE_ROOT / "current"
PIPELINE_MANIFEST = CURRENT_PIPELINE_DIR / "manifest.json"
SAMPLE_SALES_CSV = DATA_DIR / "sales_demo.csv"

# One local model shared by both CrewAI agents.
MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_NEW_TOKENS = positive_int_setting(os.getenv("HF_MAX_NEW_TOKENS"), "HF_MAX_NEW_TOKENS", 384)
MAX_UPLOAD_MB = positive_int_setting(os.getenv("MAX_UPLOAD_MB"), "MAX_UPLOAD_MB", 25)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_DATA_ROWS = positive_int_setting(os.getenv("MAX_DATA_ROWS"), "MAX_DATA_ROWS", 250_000)
MAX_DATA_COLUMNS = positive_int_setting(os.getenv("MAX_DATA_COLUMNS"), "MAX_DATA_COLUMNS", 250)
MAX_BUSINESS_CONTEXT_CHARS = positive_int_setting(
    os.getenv("MAX_BUSINESS_CONTEXT_CHARS"), "MAX_BUSINESS_CONTEXT_CHARS", 4_000
)
MAX_ANALYSIS_QUESTION_CHARS = positive_int_setting(
    os.getenv("MAX_ANALYSIS_QUESTION_CHARS"), "MAX_ANALYSIS_QUESTION_CHARS", 2_000
)
MAX_SQL_ROWS = 100
MAX_SQL_EXECUTION_SECONDS = positive_float_setting(
    os.getenv("MAX_SQL_EXECUTION_SECONDS"),
    "MAX_SQL_EXECUTION_SECONDS",
    5.0,
)

WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
