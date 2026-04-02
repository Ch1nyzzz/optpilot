# Repository Guidelines

## Project Structure

```
optpilot/
├── src/optpilot/              # Core library
│   ├── config.py              # Global configuration (env, models, paths, rate limits)
│   ├── models.py              # Core data models (MASTrace, FMProfile, EvolveResult, etc.)
│   ├── llm.py                 # LLM interface (Together AI, sync + async, rate limiting)
│   ├── dag/                   # DAG abstraction and execution
│   │   ├── core.py            # MASDAG, DAGNode, DAGEdge + topology feature detection
│   │   └── executor.py        # DAGExecutor: lightweight BFS workflow engine
│   ├── modules/               # Core modules
│   │   ├── base_runner.py     # MASRunner abstract base
│   │   ├── runner.py          # OptPilotRunner (DAG execution + benchmark scoring)
│   │   └── diagnoser.py       # FM diagnosis via LLM (6-group, concurrent)
│   ├── skills/                # Prior store and pattern analysis
│   │   ├── jacobian.py        # RepairJacobian matrix (failure×pattern → success rate)
│   │   ├── recipes.py         # Repair recipe library
│   │   └── repair_patterns.py # PatternCatalog + mutation classification
│   ├── data/                  # Benchmarks and taxonomy
│   │   ├── fm_taxonomy_6group.py  # 6-group FM taxonomy (A-F)
│   │   ├── benchmarks.py         # MMLU, AIME 2025, OlympiadBench
│   │   ├── benchmarks_humaneval.py # HumanEval (code generation)
│   │   ├── benchmarks_appworld.py
│   │   ├── benchmarks_gaia.py
│   │   └── benchmarks_swebench.py
│   └── tools/                 # External tool integrations per target MAS
│       ├── agentcoder_tools.py
│       ├── appworld_tools.py
│       ├── hyperagent_tools.py
│       └── magentic_tools.py
├── experiments/               # Experiment entry points
│   ├── run_openevolve.py      # Blind / prior-guided OpenEvolve
│   ├── analyze_openevolve_traces.py  # Prior extraction from traces
│   ├── openevolve_evaluator_multi.py # Multi-target fitness evaluator
│   ├── openevolve_initial_dag*.py    # Initial DAG builders per target
│   ├── run_ag2_mathchat_baseline.py  # Baseline measurement
│   └── openevolve_config.yaml        # SkyDiscover configuration
├── library_store/             # Global experience store
│   ├── jacobian/              # Repair effectiveness matrix
│   ├── recipes/               # Repair recipes (per FM group)
│   ├── negatives/             # Lessons from failed repairs
│   └── pattern_catalog.json   # Evolved pattern catalog
├── tests/                     # Regression tests (pytest)
└── memory_bank/               # Project documentation
```

## Build & Test

```bash
pip install -e .
python -m pytest tests/ -x -q

# Parallel comparison experiment
python -m experiments.run_openevolve --target-mas agentcoder --iterations 50 &
python -m experiments.run_openevolve --target-mas agentcoder --iterations 50 --with-priors &
wait

# Extract priors from guided run
python -m experiments.analyze_openevolve_traces \
    --openevolve-dir results/..._priors_artifacts/openevolve_output \
    --target-mas agentcoder
```

## Coding Style

- Python 3.11+, type hints throughout
- Filenames: lowercase with underscores
- Async-first for LLM calls and batch operations
- Keep `memory_bank/` updated when architecture changes

## Commit Guidelines

Short, imperative commit messages, e.g. `add prior extraction pipeline` or `fix evaluator task split`.
