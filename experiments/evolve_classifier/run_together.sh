#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

TOGETHER_KEY="${TOGETHER_API_KEY:-${TOGETHER_AI_API_KEY:-${together_ai_api:-${OPENAI_API_KEY:-}}}}"
if [[ -z "$TOGETHER_KEY" ]]; then
  printf '%s\n' \
    "Missing Together API key." \
    "Set TOGETHER_API_KEY, TOGETHER_AI_API_KEY, or together_ai_api in $REPO_ROOT/.env."
  exit 1
fi

export TOGETHER_API_KEY="$TOGETHER_KEY"
export TOGETHER_AI_API_KEY="$TOGETHER_KEY"
export OPENAI_API_KEY="$TOGETHER_KEY"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.together.xyz/v1}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$OPENAI_API_BASE}"

cd "$SCRIPT_DIR"
export PYTHONPATH="${REPO_ROOT}/skydiscover${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m skydiscover.cli initial_program.py evaluator/evaluator.py --config config.yaml "$@"
