"""Global configuration for OptPilot."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

# Together AI
TOGETHER_API_KEY = os.environ.get("TOGETHER_AI_API_KEY", "")
TOGETHER_BASE_URL = "https://api.together.xyz/v1"

# Models
SYSTEM_MODEL = os.environ.get("OPTPILOT_SYSTEM_MODEL", "zai-org/GLM-5")
JUDGE_MODEL = os.environ.get("OPTPILOT_JUDGE_MODEL", "moonshotai/Kimi-K2.5")
TARGET_MODEL = "openai/gpt-oss-120b"       # Target MAS agents

# Paths
MAST_DATA_CACHE = Path.home() / ".cache/huggingface/hub/datasets--mcemri--MAST-Data/snapshots"
RESULTS_DIR = _PROJECT_ROOT / "results"
LIBRARY_DIR = _PROJECT_ROOT / "library_store"
DAG_DIR = _PROJECT_ROOT / "dags"            # MASDAG definition files
PROJECT_ROOT = _PROJECT_ROOT

RESULTS_DIR.mkdir(exist_ok=True)
LIBRARY_DIR.mkdir(exist_ok=True)
DAG_DIR.mkdir(exist_ok=True)
