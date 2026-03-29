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
JUDGE_MODEL = os.environ.get("OPTPILOT_JUDGE_MODEL", "zai-org/GLM-5")
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
OFFLINE_YAML_MAX_WORKERS = int(
    os.environ.get("OPTPILOT_OFFLINE_YAML_MAX_WORKERS", str(DIAGNOSER_MAX_WORKERS))
)

# Agentic budgets
SKILL_EVOLVE_MAX_TOKENS = int(
    os.environ.get("OPTPILOT_SKILL_EVOLVE_MAX_TOKENS", "16384")
)
SKILL_EVOLVE_NUM_CANDIDATES = int(
    os.environ.get("OPTPILOT_SKILL_EVOLVE_NUM_CANDIDATES", "1")
)
JACOBIAN_TOP_K_PATTERNS = int(
    os.environ.get("OPTPILOT_JACOBIAN_TOP_K_PATTERNS", "3")
)
SKILL_EVOLVE_MAX_TURNS = int(
    os.environ.get("OPTPILOT_SKILL_EVOLVE_MAX_TURNS", "30")
)
META_EVOLVE_MAX_TOKENS = int(
    os.environ.get("OPTPILOT_META_EVOLVE_MAX_TOKENS", "32768")
)
META_EVOLVE_MAX_TURNS = int(
    os.environ.get("OPTPILOT_META_EVOLVE_MAX_TURNS", "30")
)
META_EVOLVE_FAILURE_THRESHOLD = int(
    os.environ.get("OPTPILOT_META_EVOLVE_FAILURE_THRESHOLD", "3")
)
JACOBIAN_PATTERN_FAILURE_COOLDOWN_THRESHOLD = int(
    os.environ.get("OPTPILOT_JACOBIAN_PATTERN_FAILURE_COOLDOWN_THRESHOLD", "2")
)
JACOBIAN_PATTERN_COOLDOWN_ROUNDS = int(
    os.environ.get("OPTPILOT_JACOBIAN_PATTERN_COOLDOWN_ROUNDS", "1")
)
ONLINE_EVAL_RANDOM_SEED = int(
    os.environ.get("OPTPILOT_ONLINE_EVAL_RANDOM_SEED", "42")
)
SHADOW_EVAL_INTERVAL = int(
    os.environ.get("OPTPILOT_SHADOW_EVAL_INTERVAL", "5")
)
SHADOW_META_EVOLVE_THRESHOLD = int(
    os.environ.get("OPTPILOT_SHADOW_META_EVOLVE_THRESHOLD", "3")
)

# Paths
MAST_DATA_CACHE = Path.home() / ".cache/huggingface/hub/datasets--mcemri--MAST-Data/snapshots"
RESULTS_DIR = _PROJECT_ROOT / "results"
LIBRARY_DIR = _PROJECT_ROOT / "library_store"
OFFLINE_HINTS_DIR = LIBRARY_DIR / "offline_hints"
OFFLINE_SKILLS_DIR = LIBRARY_DIR / "offline_skills"
ONLINE_HINTS_DIR = LIBRARY_DIR / "online_hints"
ONLINE_SKILLS_DIR = LIBRARY_DIR / "online_skills"
DAG_DIR = _PROJECT_ROOT / "dags"            # MASDAG definition files
EVOLVED_SKILLS_DIR = LIBRARY_DIR / "evolved_skills"
NEGATIVES_DIR = LIBRARY_DIR / "negatives"
JACOBIAN_DIR = LIBRARY_DIR / "jacobian"
PROJECT_ROOT = _PROJECT_ROOT

RESULTS_DIR.mkdir(exist_ok=True)
LIBRARY_DIR.mkdir(exist_ok=True)
OFFLINE_HINTS_DIR.mkdir(parents=True, exist_ok=True)
OFFLINE_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
ONLINE_HINTS_DIR.mkdir(parents=True, exist_ok=True)
ONLINE_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
EVOLVED_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
NEGATIVES_DIR.mkdir(parents=True, exist_ok=True)
JACOBIAN_DIR.mkdir(parents=True, exist_ok=True)
DAG_DIR.mkdir(exist_ok=True)
