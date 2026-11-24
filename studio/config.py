import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
PIPELINE_ROOT = WORKSPACE_DIR / "pipeline"
CURRENT_PIPELINE_DIR = PIPELINE_ROOT / "current"
PIPELINE_MANIFEST = CURRENT_PIPELINE_DIR / "manifest.json"
SAMPLE_SALES_CSV = DATA_DIR / "sales_demo.csv"

MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "384"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_SQL_ROWS = 100

# Provider credentials are read only from environment variables / Colab Secrets.
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
DATABRICKS_GENIE_SPACE_ID = os.getenv("DATABRICKS_GENIE_SPACE_ID", "")

SNOWFLAKE_HOST = os.getenv("SNOWFLAKE_HOST", "").rstrip("/")
SNOWFLAKE_TOKEN = os.getenv("SNOWFLAKE_TOKEN", "")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE", "")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "")
SNOWFLAKE_AGENT_NAME = os.getenv("SNOWFLAKE_AGENT_NAME", "")

WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
