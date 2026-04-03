"""Global configuration for OptPilot."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

# Together AI
TOGETHER_API_KEY = (
    os.environ.get("TOGETHER_AI_API_KEY")
    or os.environ.get("together_ai_api")
    or ""
)
TOGETHER_BASE_URL = "https://api.together.xyz/v1"

# Models
SYSTEM_MODEL = os.environ.get("OPTPILOT_SYSTEM_MODEL", "MiniMaxAI/MiniMax-M2.5")
TARGET_MODEL = os.environ.get("OPTPILOT_TARGET_MODEL", "MiniMaxAI/MiniMax-M2.5")

# LLM rate limiting
LLM_DEFAULT_RPM = int(os.environ.get("OPTPILOT_LLM_DEFAULT_RPM", "3000"))
LLM_DEFAULT_MAX_CONCURRENCY = int(
    os.environ.get("OPTPILOT_LLM_DEFAULT_MAX_CONCURRENCY", "256")
)
LLM_GLM5_RPM = int(os.environ.get("OPTPILOT_LLM_GLM5_RPM", str(LLM_DEFAULT_RPM)))
LLM_GLM5_MAX_CONCURRENCY = int(
    os.environ.get("OPTPILOT_LLM_GLM5_MAX_CONCURRENCY", "48")
)
LLM_MINIMAX_M2_5_RPM = int(
    os.environ.get("OPTPILOT_LLM_MINIMAX_M2_5_RPM", str(LLM_DEFAULT_RPM))
)
LLM_MINIMAX_M2_5_MAX_CONCURRENCY = int(
    os.environ.get("OPTPILOT_LLM_MINIMAX_M2_5_MAX_CONCURRENCY", "96")
)

# Worker defaults
DIAGNOSER_MAX_WORKERS = int(os.environ.get("OPTPILOT_DIAGNOSER_MAX_WORKERS", "64"))

# Jacobian configuration
JACOBIAN_TOP_K_PATTERNS = int(
    os.environ.get("OPTPILOT_JACOBIAN_TOP_K_PATTERNS", "3")
)
JACOBIAN_PATTERN_FAILURE_COOLDOWN_THRESHOLD = int(
    os.environ.get("OPTPILOT_JACOBIAN_PATTERN_FAILURE_COOLDOWN_THRESHOLD", "2")
)
JACOBIAN_PATTERN_COOLDOWN_ROUNDS = int(
    os.environ.get("OPTPILOT_JACOBIAN_PATTERN_COOLDOWN_ROUNDS", "1")
)
JACOBIAN_APPLIED_DECAY = float(
    os.environ.get("OPTPILOT_JACOBIAN_APPLIED_DECAY", "0.3")
)

# Paths
RESULTS_DIR = _PROJECT_ROOT / "results"
LIBRARY_DIR = _PROJECT_ROOT / "library_store"
DAG_DIR = _PROJECT_ROOT / "dags"
NEGATIVES_DIR = LIBRARY_DIR / "negatives"
JACOBIAN_DIR = LIBRARY_DIR / "jacobian"
RECIPES_DIR = LIBRARY_DIR / "recipes"
CATALOG_PATH = LIBRARY_DIR / "pattern_catalog.json"
PROJECT_ROOT = _PROJECT_ROOT

# Experience is stored globally. Topology differentiation is handled by
# FailureSignature embedding has_hub into the Jacobian row key
# (auto-detected from DAG structure via MASDAG.extract_topology_features()).

RESULTS_DIR.mkdir(exist_ok=True)
LIBRARY_DIR.mkdir(exist_ok=True)
NEGATIVES_DIR.mkdir(parents=True, exist_ok=True)
JACOBIAN_DIR.mkdir(parents=True, exist_ok=True)
RECIPES_DIR.mkdir(parents=True, exist_ok=True)
DAG_DIR.mkdir(exist_ok=True)
