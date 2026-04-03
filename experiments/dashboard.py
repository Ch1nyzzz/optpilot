"""
OptPilot Training Dashboard
============================
Visualize OpenEvolve training runs: DAG diffs, shadow gate traces,
test-set results, and evolution progress.

Usage:
    python -m experiments.dashboard [--port 8501] [--results-dir results/]
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

RESULTS_DIR = Path("results")


# ── Data loading helpers ─────────────────────────────────────────────

def discover_experiments(results_dir: Path) -> list[dict]:
    """Find all top-level result JSON files and return metadata."""
    experiments = []
    for f in sorted(results_dir.glob("*.json")):
        # Skip non-experiment files
        if "calibration" in f.name or "offline" in f.name or "baseline" in f.name or "skill_opt" in f.name:
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            if "experiment" not in data and "evolution" not in data:
                continue
            experiments.append({
                "filename": f.name,
                "path": str(f),
                "experiment": data.get("experiment", f.stem),
                "target_mas": data.get("target_mas", data.get("target_mas_name", "?")),
                "with_priors": data.get("with_priors", False),
                "model": data.get("model", "?"),
                "dag": data.get("dag", "?"),
                "iterations": data.get("iterations", "?"),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return experiments


def load_experiment(results_dir: Path, filename: str) -> dict:
    """Load full experiment JSON."""
    with open(results_dir / filename) as f:
        return json.load(f)


def load_dag_yaml(path: str) -> str:
    """Load a DAG YAML file as text."""
    p = Path(path)
    if p.exists():
        return p.read_text()
    return f"# File not found: {path}"


def compute_dag_diff(base_path: str, evolved_path: str) -> str:
    """Compute unified diff between base and evolved DAG YAMLs."""
    base = load_dag_yaml(base_path).splitlines(keepends=True)
    evolved = load_dag_yaml(evolved_path).splitlines(keepends=True)
    diff = difflib.unified_diff(
        base, evolved,
        fromfile="input.yaml (baseline)",
        tofile="evolved.yaml (shadow_selected / best)",
        lineterm="",
    )
    return "\n".join(diff)


def load_checkpoints(artifacts_dir: str) -> list[dict]:
    """Load checkpoint metrics sorted by iteration."""
    ckpt_dir = Path(artifacts_dir) / "openevolve_output" / "checkpoints"
    if not ckpt_dir.exists():
        return []
    checkpoints = []
    for d in sorted(ckpt_dir.iterdir()):
        info_file = d / "best_program_info.json"
        if not info_file.exists():
            continue
        with open(info_file) as f:
            info = json.load(f)
        m = re.search(r"checkpoint_(\d+)", d.name)
        iteration = int(m.group(1)) if m else 0
        metrics = info.get("metrics", {})
        checkpoints.append({
            "iteration": iteration,
            "checkpoint_name": d.name,
            "combined_score": metrics.get("combined_score", 0),
            "accuracy": metrics.get("accuracy", 0),
            "fm_A": metrics.get("fm_A", 0),
            "fm_B": metrics.get("fm_B", 0),
            "fm_C": metrics.get("fm_C", 0),
            "fm_D": metrics.get("fm_D", 0),
            "fm_E": metrics.get("fm_E", 0),
            "fm_F": metrics.get("fm_F", 0),
            "num_agents": metrics.get("num_agents", 0),
            "total_failures": metrics.get("total_failures", 0),
        })
    checkpoints.sort(key=lambda x: x["iteration"])
    return checkpoints


def load_checkpoint_diffs(artifacts_dir: str) -> list[dict]:
    """Load all programs from the MAP-Elites archive across checkpoints.

    Returns a list of unique programs sorted by iteration_found, each with
    a diff against the initial seed program (from the earliest checkpoint).
    """
    ckpt_dir = Path(artifacts_dir) / "openevolve_output" / "checkpoints"
    if not ckpt_dir.exists():
        return []

    # Collect all unique programs across all checkpoints
    seen_ids: set[str] = set()
    all_programs: list[dict] = []

    # Find the latest checkpoint to get the most complete archive
    ckpt_dirs = sorted(ckpt_dir.iterdir(), key=lambda d: int(re.search(r"(\d+)", d.name).group(1)) if re.search(r"(\d+)", d.name) else 0)

    for d in ckpt_dirs:
        programs_dir = d / "programs"
        if not programs_dir.exists():
            continue
        for pf in programs_dir.iterdir():
            if not pf.suffix == ".json":
                continue
            try:
                with open(pf) as f:
                    prog = json.load(f)
                pid = prog.get("id", pf.stem)
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                all_programs.append(prog)
            except (json.JSONDecodeError, KeyError):
                continue

    if not all_programs:
        return []

    # Sort by iteration_found
    all_programs.sort(key=lambda p: (p.get("iteration_found", 0), p.get("generation", 0)))

    # The first program (iteration_found=0 or lowest) is the seed
    seed_text = all_programs[0].get("solution", "")

    results = []
    for prog in all_programs:
        sol = prog.get("solution", "")
        metrics = prog.get("metrics", {})
        iteration = prog.get("iteration_found", 0)

        diff_lines = difflib.unified_diff(
            seed_text.splitlines(keepends=True),
            sol.splitlines(keepends=True),
            fromfile="seed (iter 0)",
            tofile=f"iter {iteration}",
            lineterm="",
        )
        diff_text = "\n".join(diff_lines)

        results.append({
            "iteration": iteration,
            "generation": prog.get("generation", 0),
            "program_id": prog.get("id", "?"),
            "parent_id": prog.get("parent_id"),
            "diff": diff_text,
            "has_changes": bool(diff_text.strip()),
            "combined_score": metrics.get("combined_score", 0),
            "accuracy": metrics.get("accuracy", 0),
            "num_agents": metrics.get("num_agents", 0),
        })

    return results


def load_traces(task_dir: str) -> list[dict]:
    """Load all task traces from a test/shadow directory."""
    d = Path(task_dir)
    if not d.exists():
        return []
    traces = []
    for td in sorted(d.iterdir(), key=lambda x: _task_sort_key(x.name)):
        if not td.is_dir() or not td.name.startswith("task_"):
            continue
        trace_json = td / "trace.json"
        trace_txt = td / "trace.txt"
        entry = {
            "task_id": td.name,
            "task_index": _task_sort_key(td.name),
        }
        if trace_json.exists():
            with open(trace_json) as f:
                entry.update(json.load(f))
        if trace_txt.exists():
            entry["trace_text"] = trace_txt.read_text()
        else:
            entry["trace_text"] = ""
        traces.append(entry)
    return traces


def _task_sort_key(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def load_shadow_candidates(artifacts_dir: str) -> dict:
    """Load shadow candidate evaluation results."""
    shadow_dir = Path(artifacts_dir) / "shadow_candidates"
    if not shadow_dir.exists():
        return {}
    result = {}
    for cand_dir in sorted(shadow_dir.iterdir()):
        if not cand_dir.is_dir():
            continue
        traces = load_traces(str(cand_dir))
        n_total = len(traces)
        n_correct = sum(1 for t in traces if t.get("task_success"))
        result[cand_dir.name] = {
            "n_total": n_total,
            "n_correct": n_correct,
            "accuracy": n_correct / n_total if n_total > 0 else 0,
            "traces": traces,
        }
    return result


# ── API Handler ──────────────────────────────────────────────────────

class DashboardHandler(SimpleHTTPRequestHandler):
    results_dir: Path = RESULTS_DIR

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._serve_html()
        elif path == "/api/experiments":
            self._json_response(discover_experiments(self.results_dir))
        elif path == "/api/experiment":
            filename = params.get("f", [None])[0]
            if not filename:
                self._json_response({"error": "missing ?f="}, 400)
                return
            try:
                data = load_experiment(self.results_dir, filename)
                self._json_response(data)
            except FileNotFoundError:
                self._json_response({"error": "not found"}, 404)
        elif path == "/api/checkpoints":
            artifacts = params.get("d", [None])[0]
            if not artifacts:
                self._json_response({"error": "missing ?d="}, 400)
                return
            self._json_response(load_checkpoints(artifacts))
        elif path == "/api/dag_diff":
            base = params.get("base", [None])[0]
            evolved = params.get("evolved", [None])[0]
            if not base or not evolved:
                self._json_response({"error": "missing ?base=&evolved="}, 400)
                return
            base_text = load_dag_yaml(base)
            evolved_text = load_dag_yaml(evolved)
            diff = compute_dag_diff(base, evolved)
            self._json_response({
                "base_text": base_text,
                "evolved_text": evolved_text,
                "diff": diff,
            })
        elif path == "/api/traces":
            d = params.get("d", [None])[0]
            if not d:
                self._json_response({"error": "missing ?d="}, 400)
                return
            self._json_response(load_traces(d))
        elif path == "/api/checkpoint_diffs":
            artifacts = params.get("d", [None])[0]
            if not artifacts:
                self._json_response({"error": "missing ?d="}, 400)
                return
            self._json_response(load_checkpoint_diffs(artifacts))
        elif path == "/api/shadow_candidates":
            d = params.get("d", [None])[0]
            if not d:
                self._json_response({"error": "missing ?d="}, 400)
                return
            self._json_response(load_shadow_candidates(d))
        else:
            self.send_error(404)

    def _json_response(self, data, code=200):
        body = json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        html = DASHBOARD_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, format, *args):
        # Quieter logging
        pass


# ── HTML Dashboard ───────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OptPilot Training Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
/* ── Theme ─────────────────────────────── */
:root {
  --bg: #0d1117; --bg2: #161b22; --bg3: #21262d; --bg4: #2d333b;
  --text: #e6edf3; --text-dim: #7d8590; --text-muted: #484f58;
  --accent: #58a6ff; --green: #3fb950; --red: #f85149; --orange: #d29922;
  --purple: #bc8cff; --cyan: #56d4dd; --yellow: #e3b341;
  --border: #30363d; --shadow: rgba(0,0,0,0.4);
  --diff-add-bg: rgba(63,185,80,0.15); --diff-del-bg: rgba(248,81,73,0.15);
  --diff-add-text: #3fb950; --diff-del-text: #f85149;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: 'Segoe UI',system-ui,-apple-system,sans-serif; background:var(--bg); color:var(--text); min-height:100vh; display:flex; }
a { color:var(--accent); text-decoration:none; }

/* ── Sidebar ───────────────────────────── */
.sidebar { width:300px; min-width:300px; background:var(--bg2); border-right:1px solid var(--border); display:flex; flex-direction:column; height:100vh; position:sticky; top:0; }
.sidebar-header { padding:16px; border-bottom:1px solid var(--border); }
.sidebar-header h1 { font-size:16px; font-weight:700; letter-spacing:0.5px; }
.sidebar-header h1 span { color:var(--accent); }
.sidebar-header p { font-size:11px; color:var(--text-dim); margin-top:4px; }
.exp-list { flex:1; overflow-y:auto; padding:8px; }
.exp-item { padding:10px 12px; border-radius:8px; cursor:pointer; margin-bottom:4px; border:1px solid transparent; transition:all 0.15s; }
.exp-item:hover { background:var(--bg3); border-color:var(--border); }
.exp-item.active { background:var(--bg3); border-color:var(--accent); }
.exp-item .name { font-size:13px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.exp-item .meta { font-size:11px; color:var(--text-dim); margin-top:2px; }
.exp-item .tag { display:inline-block; padding:1px 6px; border-radius:4px; font-size:10px; font-weight:600; margin-left:4px; }
.tag-blind { background:rgba(248,81,73,0.15); color:var(--red); }
.tag-priors { background:rgba(63,185,80,0.15); color:var(--green); }

/* ── Main ──────────────────────────────── */
.main { flex:1; overflow-y:auto; height:100vh; }
.main-empty { display:flex; align-items:center; justify-content:center; height:100%; color:var(--text-dim); font-size:15px; }

/* Tabs */
.tabs { display:flex; gap:0; border-bottom:1px solid var(--border); background:var(--bg2); position:sticky; top:0; z-index:10; }
.tab { padding:12px 20px; font-size:13px; font-weight:600; cursor:pointer; border-bottom:2px solid transparent; color:var(--text-dim); transition:all 0.15s; }
.tab:hover { color:var(--text); background:var(--bg3); }
.tab.active { color:var(--accent); border-bottom-color:var(--accent); }
.tab-content { display:none; padding:20px; }
.tab-content.active { display:block; }

/* ── Cards / Panels ────────────────────── */
.card { background:var(--bg2); border:1px solid var(--border); border-radius:10px; padding:16px; margin-bottom:16px; }
.card h3 { font-size:14px; font-weight:700; margin-bottom:12px; color:var(--text); }
.card h3 .badge { font-size:11px; font-weight:600; padding:2px 8px; border-radius:4px; margin-left:8px; }

/* Stats row */
.stats-row { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px; }
.stat-card { flex:1; min-width:140px; background:var(--bg3); border-radius:8px; padding:14px; text-align:center; }
.stat-card .label { font-size:11px; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.5px; }
.stat-card .value { font-size:24px; font-weight:700; margin-top:4px; }
.stat-card .sub { font-size:11px; color:var(--text-dim); margin-top:2px; }
.stat-green .value { color:var(--green); }
.stat-red .value { color:var(--red); }
.stat-blue .value { color:var(--accent); }
.stat-orange .value { color:var(--orange); }
.stat-purple .value { color:var(--purple); }

/* Delta indicator */
.delta { font-size:12px; font-weight:600; }
.delta.pos { color:var(--green); }
.delta.neg { color:var(--red); }
.delta.neutral { color:var(--text-dim); }

/* ── Diff view ─────────────────────────── */
.diff-container { display:flex; gap:0; border:1px solid var(--border); border-radius:8px; overflow:hidden; max-height:70vh; }
.diff-pane { flex:1; overflow:auto; font-family:'Cascadia Code','Fira Code',monospace; font-size:12px; line-height:1.6; }
.diff-pane-header { padding:8px 12px; background:var(--bg3); border-bottom:1px solid var(--border); font-size:11px; font-weight:600; color:var(--text-dim); position:sticky; top:0; z-index:1; }
.diff-pane pre { padding:8px 12px; white-space:pre-wrap; word-break:break-all; }
.diff-unified { font-family:'Cascadia Code','Fira Code',monospace; font-size:12px; line-height:1.7; overflow:auto; max-height:70vh; padding:12px; background:var(--bg3); border-radius:8px; }
.diff-line { padding:0 8px; }
.diff-add { background:var(--diff-add-bg); color:var(--diff-add-text); }
.diff-del { background:var(--diff-del-bg); color:var(--diff-del-text); }
.diff-hunk { color:var(--purple); font-weight:600; margin-top:8px; }
.diff-header { color:var(--text-dim); font-weight:600; }

/* ── Table ─────────────────────────────── */
table { width:100%; border-collapse:collapse; font-size:13px; }
th { text-align:left; padding:10px 12px; background:var(--bg3); border-bottom:1px solid var(--border); font-size:11px; text-transform:uppercase; letter-spacing:0.5px; color:var(--text-dim); font-weight:700; position:sticky; top:0; z-index:1; }
td { padding:10px 12px; border-bottom:1px solid var(--border); vertical-align:top; }
tr:hover td { background:var(--bg3); }
.success { color:var(--green); font-weight:600; }
.failure { color:var(--red); font-weight:600; }
.btn { padding:4px 10px; border-radius:5px; border:1px solid var(--border); background:var(--bg3); color:var(--accent); cursor:pointer; font-size:11px; font-weight:600; transition:all 0.15s; }
.btn:hover { background:var(--bg4); border-color:var(--accent); }

/* ── Modal ─────────────────────────────── */
.modal-overlay { position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.7); z-index:100; display:none; align-items:center; justify-content:center; }
.modal-overlay.show { display:flex; }
.modal { background:var(--bg2); border:1px solid var(--border); border-radius:12px; width:90vw; max-width:1100px; max-height:85vh; display:flex; flex-direction:column; box-shadow:0 8px 32px var(--shadow); }
.modal-header { padding:16px 20px; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; }
.modal-header h3 { font-size:15px; }
.modal-close { background:none; border:none; color:var(--text-dim); font-size:20px; cursor:pointer; padding:4px 8px; border-radius:4px; }
.modal-close:hover { background:var(--bg3); color:var(--text); }
.modal-body { flex:1; overflow:auto; padding:20px; }
.trace-content { font-family:'Cascadia Code','Fira Code',monospace; font-size:12px; line-height:1.6; white-space:pre-wrap; word-break:break-all; background:var(--bg3); padding:16px; border-radius:8px; max-height:60vh; overflow:auto; }

/* ── Shadow gate ───────────────────────── */
.candidate-card { background:var(--bg3); border:1px solid var(--border); border-radius:8px; padding:14px; margin-bottom:8px; display:flex; align-items:center; gap:16px; }
.candidate-card.selected { border-color:var(--green); background:rgba(63,185,80,0.05); }
.candidate-bar { flex:1; }
.candidate-bar .name { font-size:13px; font-weight:600; }
.candidate-bar .metrics { font-size:11px; color:var(--text-dim); margin-top:4px; }
.acc-bar-wrap { width:200px; }
.acc-bar-bg { height:8px; background:var(--bg4); border-radius:4px; overflow:hidden; }
.acc-bar-fill { height:100%; border-radius:4px; transition:width 0.3s; }

/* Sub-tabs for test results */
.sub-tabs { display:flex; gap:8px; margin-bottom:16px; }
.sub-tab { padding:6px 14px; border-radius:6px; font-size:12px; font-weight:600; cursor:pointer; background:var(--bg3); color:var(--text-dim); border:1px solid transparent; transition:all 0.15s; }
.sub-tab:hover { color:var(--text); }
.sub-tab.active { color:var(--accent); border-color:var(--accent); background:rgba(88,166,255,0.08); }

/* Chart placeholder */
.chart-area { width:100%; min-height:350px; }

/* Scrollbar */
::-webkit-scrollbar { width:8px; height:8px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--bg4); border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background:var(--text-muted); }

/* Loading */
.loading { text-align:center; padding:40px; color:var(--text-dim); }
.spinner { display:inline-block; width:20px; height:20px; border:2px solid var(--border); border-top-color:var(--accent); border-radius:50%; animation:spin 0.6s linear infinite; margin-right:8px; vertical-align:middle; }
@keyframes spin { to { transform:rotate(360deg); } }
</style>
</head>
<body>

<!-- Sidebar -->
<div class="sidebar">
  <div class="sidebar-header">
    <h1><span>OptPilot</span> Dashboard</h1>
    <p>Training Run Inspector</p>
  </div>
  <div class="exp-list" id="expList">
    <div class="loading"><span class="spinner"></span>Loading experiments...</div>
  </div>
</div>

<!-- Main -->
<div class="main" id="mainArea">
  <div class="main-empty" id="emptyState">Select an experiment from the sidebar</div>
  <div id="expView" style="display:none;">
    <div class="tabs" id="tabBar">
      <div class="tab active" data-tab="overview">Overview</div>
      <div class="tab" data-tab="dag-diff">DAG Diff</div>
      <div class="tab" data-tab="evolution">Evolution</div>
      <div class="tab" data-tab="shadow">Shadow Gate</div>
      <div class="tab" data-tab="test">Test Results</div>
    </div>

    <!-- Overview -->
    <div class="tab-content active" id="tab-overview"></div>

    <!-- DAG Diff -->
    <div class="tab-content" id="tab-dag-diff"></div>

    <!-- Evolution -->
    <div class="tab-content" id="tab-evolution"></div>

    <!-- Shadow Gate -->
    <div class="tab-content" id="tab-shadow"></div>

    <!-- Test Results -->
    <div class="tab-content" id="tab-test"></div>
  </div>
</div>

<!-- Trace Modal -->
<div class="modal-overlay" id="traceModal">
  <div class="modal">
    <div class="modal-header">
      <h3 id="traceModalTitle">Trace</h3>
      <button class="modal-close" onclick="closeTraceModal()">&times;</button>
    </div>
    <div class="modal-body">
      <div class="trace-content" id="traceModalBody"></div>
    </div>
  </div>
</div>

<script>
// ── State ─────────────────────────────────
let experiments = [];
let currentExp = null;   // full experiment data
let currentMeta = null;  // sidebar metadata

// ── Init ──────────────────────────────────
async function init() {
  experiments = await api('/api/experiments');
  renderExpList();
}

async function api(url) {
  const r = await fetch(url);
  return r.json();
}

// ── Sidebar ───────────────────────────────
function renderExpList() {
  const el = document.getElementById('expList');
  if (!experiments.length) { el.innerHTML = '<div class="loading">No experiments found</div>'; return; }
  el.innerHTML = experiments.map((e, i) => `
    <div class="exp-item" data-idx="${i}" onclick="selectExp(${i})">
      <div class="name">${e.target_mas} / ${e.dag}
        <span class="tag ${e.with_priors ? 'tag-priors' : 'tag-blind'}">${e.with_priors ? 'PRIORS' : 'BLIND'}</span>
      </div>
      <div class="meta">${e.iterations} iters &middot; ${e.model.split('/').pop()} &middot; ${e.filename.match(/\d{8}_\d{6}/)?.[0] || ''}</div>
    </div>
  `).join('');
}

async function selectExp(idx) {
  // Highlight sidebar
  document.querySelectorAll('.exp-item').forEach(el => el.classList.remove('active'));
  document.querySelector(`.exp-item[data-idx="${idx}"]`).classList.add('active');

  currentMeta = experiments[idx];
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('expView').style.display = 'block';

  // Show loading
  ['overview','dag-diff','evolution','shadow','test'].forEach(t => {
    document.getElementById('tab-'+t).innerHTML = '<div class="loading"><span class="spinner"></span>Loading...</div>';
  });

  currentExp = await api('/api/experiment?f=' + encodeURIComponent(currentMeta.filename));
  renderOverview();
  renderDagDiff();
  renderEvolution();
  renderShadow();
  renderTest();
}

// ── Tabs ──────────────────────────────────
document.getElementById('tabBar').addEventListener('click', e => {
  const tab = e.target.closest('.tab');
  if (!tab) return;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  tab.classList.add('active');
  document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
});

// ── Overview ──────────────────────────────
function renderOverview() {
  const d = currentExp;
  const evo = d.evolution || {};
  const shadow = d.shadow_selection || {};
  const tb = d.test_baseline || {};
  const tf = d.test_final || {};
  const baselineAcc = tb.overall_accuracy ?? 0;
  const finalAcc = tf.overall_accuracy ?? 0;
  const delta = finalAcc - baselineAcc;
  const deltaClass = delta > 0 ? 'pos' : delta < 0 ? 'neg' : 'neutral';
  const deltaStr = (delta > 0 ? '+' : '') + (delta * 100).toFixed(1) + '%';

  const bestMetrics = evo.best_metrics || {};

  document.getElementById('tab-overview').innerHTML = `
    <div class="card">
      <h3>Experiment Summary</h3>
      <div class="stats-row">
        <div class="stat-card stat-blue">
          <div class="label">Target MAS</div>
          <div class="value" style="font-size:18px">${d.target_mas || d.dag || '?'}</div>
          <div class="sub">${d.dag || ''}</div>
        </div>
        <div class="stat-card stat-orange">
          <div class="label">Iterations</div>
          <div class="value">${d.iterations || '?'}</div>
          <div class="sub">${d.eval_tasks_per_iteration || '?'} tasks/iter</div>
        </div>
        <div class="stat-card">
          <div class="label">Train Tasks</div>
          <div class="value">${d.n_train || '?'}</div>
        </div>
        <div class="stat-card">
          <div class="label">Test Tasks</div>
          <div class="value">${d.n_test || '?'}</div>
        </div>
        <div class="stat-card stat-purple">
          <div class="label">Mode</div>
          <div class="value" style="font-size:16px">${d.with_priors ? 'Prior-Guided' : 'Blind'}</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h3>Accuracy Comparison <span class="delta ${deltaClass}">${deltaStr}</span></h3>
      <div class="stats-row">
        <div class="stat-card stat-red">
          <div class="label">Baseline (test)</div>
          <div class="value">${(baselineAcc * 100).toFixed(1)}%</div>
          <div class="sub">${tb.per_benchmark ? Object.entries(tb.per_benchmark).map(([k,v])=> k+': '+v.correct+'/'+v.n).join(', ') : ''}</div>
        </div>
        <div class="stat-card stat-green">
          <div class="label">Final (test)</div>
          <div class="value">${(finalAcc * 100).toFixed(1)}%</div>
          <div class="sub">${tf.per_benchmark ? Object.entries(tf.per_benchmark).map(([k,v])=> k+': '+v.correct+'/'+v.n).join(', ') : ''}</div>
        </div>
        <div class="stat-card stat-blue">
          <div class="label">Best Train Score</div>
          <div class="value">${(evo.best_score ?? 0).toFixed(3)}</div>
          <div class="sub">initial: ${(evo.initial_score ?? 0).toFixed(3)}</div>
        </div>
        <div class="stat-card">
          <div class="label">Shadow Accuracy</div>
          <div class="value">${shadow.selected_shadow_accuracy != null ? (shadow.selected_shadow_accuracy * 100).toFixed(1) + '%' : 'N/A'}</div>
          <div class="sub">baseline: ${shadow.baseline_shadow_accuracy != null ? (shadow.baseline_shadow_accuracy * 100).toFixed(1) + '%' : 'N/A'}</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h3>Best Candidate FM Rates (Train)</h3>
      <div class="stats-row">
        ${['A','B','C','D','E','F'].map(g => `
          <div class="stat-card">
            <div class="label">FM ${g}</div>
            <div class="value" style="font-size:18px;color:${(bestMetrics['fm_'+g]||0) > 0.2 ? 'var(--red)' : (bestMetrics['fm_'+g]||0) > 0 ? 'var(--orange)' : 'var(--green)'}">${((bestMetrics['fm_'+g]||0)*100).toFixed(0)}%</div>
          </div>
        `).join('')}
      </div>
    </div>

    <div class="card">
      <h3>Timing</h3>
      <div class="stats-row">
        <div class="stat-card">
          <div class="label">Evolution</div>
          <div class="value" style="font-size:16px">${formatTime(evo.elapsed_s)}</div>
        </div>
        <div class="stat-card">
          <div class="label">Test Baseline</div>
          <div class="value" style="font-size:16px">${formatTime(tb.elapsed_s)}</div>
        </div>
        <div class="stat-card">
          <div class="label">Test Final</div>
          <div class="value" style="font-size:16px">${formatTime(tf.elapsed_s)}</div>
        </div>
      </div>
    </div>
  `;
}

function formatTime(s) {
  if (!s) return 'N/A';
  if (s < 60) return s.toFixed(0) + 's';
  if (s < 3600) return (s/60).toFixed(1) + 'm';
  return (s/3600).toFixed(1) + 'h';
}

// ── DAG Diff ──────────────────────────────
async function renderDagDiff() {
  const dv = currentExp.dag_versions_dir;
  if (!dv) { document.getElementById('tab-dag-diff').innerHTML = '<div class="loading">No DAG versions directory</div>'; return; }

  const basePath = dv + '/input.yaml';
  // Prefer shadow_selected.yaml, fall back to openevolve_best.yaml
  let evolvedName = 'shadow_selected.yaml';
  const data = await api(`/api/dag_diff?base=${encodeURIComponent(basePath)}&evolved=${encodeURIComponent(dv + '/' + evolvedName)}`);
  if (data.evolved_text && data.evolved_text.startsWith('# File not found')) {
    evolvedName = 'openevolve_best.yaml';
    const data2 = await api(`/api/dag_diff?base=${encodeURIComponent(basePath)}&evolved=${encodeURIComponent(dv + '/' + evolvedName)}`);
    renderDagDiffContent(data2, 'input.yaml', evolvedName);
  } else {
    renderDagDiffContent(data, 'input.yaml', evolvedName);
  }
}

function renderDagDiffContent(data, baseName, evolvedName) {
  const el = document.getElementById('tab-dag-diff');
  if (!data.diff || data.diff.trim() === '') {
    el.innerHTML = `<div class="card"><h3>No Changes</h3><p style="color:var(--text-dim)">The evolved DAG is identical to the baseline.</p></div>`;
    return;
  }

  // Render unified diff with syntax highlighting
  const diffLines = data.diff.split('\n').map(line => {
    if (line.startsWith('+++') || line.startsWith('---')) return `<div class="diff-line diff-header">${esc(line)}</div>`;
    if (line.startsWith('@@')) return `<div class="diff-line diff-hunk">${esc(line)}</div>`;
    if (line.startsWith('+')) return `<div class="diff-line diff-add">${esc(line)}</div>`;
    if (line.startsWith('-')) return `<div class="diff-line diff-del">${esc(line)}</div>`;
    return `<div class="diff-line">${esc(line)}</div>`;
  }).join('');

  el.innerHTML = `
    <div class="card">
      <h3>Unified Diff: ${baseName} → ${evolvedName}</h3>
      <div class="diff-unified">${diffLines}</div>
    </div>
    <div class="card">
      <h3>Side-by-Side</h3>
      <div class="diff-container">
        <div class="diff-pane">
          <div class="diff-pane-header">${baseName} (baseline)</div>
          <pre>${esc(data.base_text)}</pre>
        </div>
        <div class="diff-pane" style="border-left:1px solid var(--border)">
          <div class="diff-pane-header">${evolvedName} (evolved)</div>
          <pre>${esc(data.evolved_text)}</pre>
        </div>
      </div>
    </div>
  `;
}

function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ── Evolution ─────────────────────────────
async function renderEvolution() {
  const el = document.getElementById('tab-evolution');
  const artifacts = currentExp.artifacts_dir;
  if (!artifacts) { el.innerHTML = '<div class="loading">No artifacts directory</div>'; return; }

  const [checkpoints, ckptDiffs] = await Promise.all([
    api('/api/checkpoints?d=' + encodeURIComponent(artifacts)),
    api('/api/checkpoint_diffs?d=' + encodeURIComponent(artifacts)),
  ]);
  if (!checkpoints.length) { el.innerHTML = '<div class="card"><h3>No checkpoints found</h3></div>'; return; }

  // Store program diffs for the diff viewer
  window._ckptDiffs = ckptDiffs || [];

  el.innerHTML = `
    <div class="card">
      <h3>Training Score Over Iterations</h3>
      <div class="chart-area" id="chartScore"></div>
    </div>
    <div class="card">
      <h3>Training Accuracy Over Iterations</h3>
      <div class="chart-area" id="chartAcc"></div>
    </div>
    <div class="card">
      <h3>Failure Mode Rates Over Iterations</h3>
      <div class="chart-area" id="chartFM"></div>
    </div>
    <div class="card">
      <h3>Checkpoint Details</h3>
      <div style="overflow-x:auto">
        <table>
          <thead><tr><th>Iter</th><th>Score</th><th>Accuracy</th><th>Agents</th><th>Failures</th><th>FM A</th><th>FM B</th><th>FM C</th><th>FM D</th><th>FM E</th><th>FM F</th></tr></thead>
          <tbody>
            ${checkpoints.map(c => `<tr>
              <td>${c.iteration}</td>
              <td>${c.combined_score.toFixed(3)}</td>
              <td>${(c.accuracy*100).toFixed(1)}%</td>
              <td>${c.num_agents}</td>
              <td>${c.total_failures}</td>
              ${['A','B','C','D','E','F'].map(g => `<td style="color:${c['fm_'+g]>0.2?'var(--red)':c['fm_'+g]>0?'var(--orange)':'var(--green)'}">${(c['fm_'+g]*100).toFixed(0)}%</td>`).join('')}
            </tr>`).join('')}
          </tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h3>Evolved Programs (${ckptDiffs.length} variants) <span style="font-size:11px;color:var(--text-dim);font-weight:400">diff vs seed program</span></h3>
      ${ckptDiffs.length ? `
      <div style="overflow-x:auto;max-height:50vh;overflow-y:auto;">
        <table>
          <thead><tr><th>Iter</th><th>Gen</th><th>Score</th><th>Accuracy</th><th>Agents</th><th>Program ID</th><th>Diff</th></tr></thead>
          <tbody>
            ${ckptDiffs.map((d, i) => `<tr>
              <td>${d.iteration}</td>
              <td>${d.generation}</td>
              <td>${d.combined_score.toFixed(3)}</td>
              <td>${(d.accuracy*100).toFixed(1)}%</td>
              <td>${d.num_agents}</td>
              <td style="font-size:11px;color:var(--text-dim)">${d.program_id.substring(0,8)}...</td>
              <td>${d.has_changes ? `<button class="btn" onclick="showCkptDiff(${i})">View Diff</button>` : '<span style="color:var(--text-dim)">seed</span>'}</td>
            </tr>`).join('')}
          </tbody>
        </table>
      </div>` : '<p style="color:var(--text-dim)">No program variants found in checkpoints</p>'}
    </div>
  `;

  const iters = checkpoints.map(c => c.iteration);
  const plotLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e6edf3', size: 12 },
    xaxis: { title: 'Iteration', gridcolor: '#21262d', linecolor: '#30363d' },
    yaxis: { gridcolor: '#21262d', linecolor: '#30363d' },
    margin: { l:50, r:20, t:20, b:50 },
    legend: { orientation: 'h', y: -0.2 },
  };

  // Score chart
  Plotly.newPlot('chartScore', [{
    x: iters, y: checkpoints.map(c => c.combined_score),
    type: 'scatter', mode: 'lines+markers', name: 'Combined Score',
    line: { color: '#58a6ff', width: 2 }, marker: { size: 6 },
  }], { ...plotLayout, yaxis: { ...plotLayout.yaxis, title: 'Combined Score' } }, { responsive: true });

  // Accuracy chart
  Plotly.newPlot('chartAcc', [{
    x: iters, y: checkpoints.map(c => c.accuracy * 100),
    type: 'scatter', mode: 'lines+markers', name: 'Accuracy',
    line: { color: '#3fb950', width: 2 }, marker: { size: 6 },
  }], { ...plotLayout, yaxis: { ...plotLayout.yaxis, title: 'Accuracy (%)' } }, { responsive: true });

  // FM chart
  const fmColors = { A:'#f85149', B:'#d29922', C:'#bc8cff', D:'#58a6ff', E:'#56d4dd', F:'#e3b341' };
  const fmNames = { A:'Instruction', B:'Loop/Stuck', C:'Context Loss', D:'Communication', E:'Task Drift', F:'Verification' };
  Plotly.newPlot('chartFM',
    ['A','B','C','D','E','F'].map(g => ({
      x: iters, y: checkpoints.map(c => c['fm_'+g]*100),
      type: 'scatter', mode: 'lines+markers', name: `FM ${g}: ${fmNames[g]}`,
      line: { color: fmColors[g], width: 2 }, marker: { size: 5 },
    })),
    { ...plotLayout, yaxis: { ...plotLayout.yaxis, title: 'FM Rate (%)' } },
    { responsive: true }
  );
}

// ── Checkpoint Diff Viewer ────────────────
function showCkptDiff(idx) {
  const d = window._ckptDiffs[idx];
  if (!d || !d.diff) return;

  const diffLines = d.diff.split('\n').map(line => {
    if (line.startsWith('+++') || line.startsWith('---')) return `<div class="diff-line diff-header">${esc(line)}</div>`;
    if (line.startsWith('@@')) return `<div class="diff-line diff-hunk">${esc(line)}</div>`;
    if (line.startsWith('+')) return `<div class="diff-line diff-add">${esc(line)}</div>`;
    if (line.startsWith('-')) return `<div class="diff-line diff-del">${esc(line)}</div>`;
    return `<div class="diff-line">${esc(line)}</div>`;
  }).join('');

  document.getElementById('traceModalTitle').textContent =
    `Program Diff — iter ${d.iteration}, gen ${d.generation} (${d.program_id.substring(0,8)}...) — score: ${d.combined_score.toFixed(3)}, acc: ${(d.accuracy*100).toFixed(1)}%`;
  document.getElementById('traceModalBody').innerHTML =
    `<div class="diff-unified">${diffLines}</div>`;
  document.getElementById('traceModal').classList.add('show');
}

// ── Shadow Gate ───────────────────────────
async function renderShadow() {
  const el = document.getElementById('tab-shadow');
  const shadow = currentExp.shadow_selection || {};
  const artifacts = currentExp.artifacts_dir;

  if (!shadow.shadow_used && !shadow.candidates_evaluated) {
    el.innerHTML = '<div class="card"><h3>Shadow Gate Not Used</h3><p style="color:var(--text-dim)">This experiment did not use shadow gate validation.</p></div>';
    return;
  }

  const candidates = shadow.candidates_evaluated || [];
  const baselineAcc = shadow.baseline_shadow_accuracy ?? 0;
  const selectedName = shadow.selected_candidate || '';

  // Load shadow candidate traces
  let shadowCandData = {};
  if (artifacts) {
    shadowCandData = await api('/api/shadow_candidates?d=' + encodeURIComponent(artifacts));
  }

  // Also load shadow baseline traces
  let shadowBaselineTraces = [];
  if (artifacts) {
    shadowBaselineTraces = await api('/api/traces?d=' + encodeURIComponent(artifacts + '/shadow_baseline'));
  }

  el.innerHTML = `
    <div class="card">
      <h3>Shadow Gate Summary</h3>
      <div class="stats-row">
        <div class="stat-card stat-red">
          <div class="label">Baseline Shadow Acc</div>
          <div class="value">${(baselineAcc*100).toFixed(1)}%</div>
        </div>
        <div class="stat-card stat-green">
          <div class="label">Selected</div>
          <div class="value" style="font-size:14px">${selectedName}</div>
          <div class="sub">Acc: ${shadow.selected_shadow_accuracy != null ? (shadow.selected_shadow_accuracy*100).toFixed(1)+'%' : 'N/A'}</div>
        </div>
        <div class="stat-card stat-blue">
          <div class="label">Candidates</div>
          <div class="value">${candidates.length}</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h3>Candidate Comparison</h3>
      ${candidates.map(c => {
        const acc = c.shadow_accuracy ?? 0;
        const isSelected = c.name === selectedName;
        const barColor = isSelected ? 'var(--green)' : acc >= baselineAcc ? 'var(--accent)' : 'var(--red)';
        return `
        <div class="candidate-card ${isSelected ? 'selected' : ''}">
          <div class="candidate-bar">
            <div class="name">${c.name} ${isSelected ? '<span style="color:var(--green);font-size:11px">● SELECTED</span>' : ''} ${!c.valid ? '<span style="color:var(--red);font-size:11px">INVALID</span>' : ''}</div>
            <div class="metrics">
              Iter: ${c.iteration ?? '?'} · Train score: ${(c.train_combined_score??0).toFixed(3)} · Train acc: ${c.train_accuracy != null ? (c.train_accuracy*100).toFixed(1)+'%' : 'N/A'} · Shadow acc: ${(acc*100).toFixed(1)}%
            </div>
          </div>
          <div class="acc-bar-wrap">
            <div class="acc-bar-bg"><div class="acc-bar-fill" style="width:${acc*100}%;background:${barColor}"></div></div>
          </div>
          ${shadowCandData[c.name] ? `<button class="btn" onclick="showShadowTraces('${c.name}')">View Traces (${shadowCandData[c.name].n_total})</button>` : ''}
        </div>`;
      }).join('')}
    </div>

    <div class="card">
      <h3>Shadow Baseline Traces <span class="badge" style="background:rgba(248,81,73,0.15);color:var(--red)">${shadowBaselineTraces.length} tasks</span></h3>
      ${shadowBaselineTraces.length ? renderTraceTable(shadowBaselineTraces, 'shadow-baseline') : '<p style="color:var(--text-dim)">No shadow baseline traces found</p>'}
    </div>

    <div id="shadowCandTracesArea"></div>
  `;

  // Store for use in showShadowTraces
  window._shadowCandData = shadowCandData;
}

function showShadowTraces(name) {
  const data = window._shadowCandData[name];
  if (!data) return;
  const area = document.getElementById('shadowCandTracesArea');
  area.innerHTML = `
    <div class="card">
      <h3>Shadow Traces: ${name} <span class="badge" style="background:rgba(63,185,80,0.15);color:var(--green)">${data.n_correct}/${data.n_total} correct (${(data.accuracy*100).toFixed(1)}%)</span></h3>
      ${renderTraceTable(data.traces, 'shadow-' + name)}
    </div>
  `;
}

// ── Test Results ──────────────────────────
async function renderTest() {
  const el = document.getElementById('tab-test');
  const artifacts = currentExp.artifacts_dir;
  const tb = currentExp.test_baseline || {};
  const tf = currentExp.test_final || {};

  // Load traces
  let baselineTraces = [], finalTraces = [];
  if (artifacts) {
    [baselineTraces, finalTraces] = await Promise.all([
      api('/api/traces?d=' + encodeURIComponent(artifacts + '/test_baseline')),
      api('/api/traces?d=' + encodeURIComponent(artifacts + '/test_final')),
    ]);
  }

  // Also check for reeval runs
  let reevalSections = '';
  if (artifacts) {
    for (let run = 1; run <= 3; run++) {
      const [rb, rf] = await Promise.all([
        api('/api/traces?d=' + encodeURIComponent(artifacts + '/reeval_baseline_run' + run)),
        api('/api/traces?d=' + encodeURIComponent(artifacts + '/reeval_final_run' + run)),
      ]);
      if (rb.length || rf.length) {
        const rbCorr = rb.filter(t => t.task_success).length;
        const rfCorr = rf.filter(t => t.task_success).length;
        reevalSections += `
          <div class="sub-tab" data-subtab="reeval-b${run}">Reeval Baseline #${run} (${rbCorr}/${rb.length})</div>
          <div class="sub-tab" data-subtab="reeval-f${run}">Reeval Final #${run} (${rfCorr}/${rf.length})</div>
        `;
        window['_reeval_b'+run] = rb;
        window['_reeval_f'+run] = rf;
      }
    }
  }

  const baselineCorr = baselineTraces.filter(t => t.task_success).length;
  const finalCorr = finalTraces.filter(t => t.task_success).length;

  el.innerHTML = `
    <div class="card">
      <h3>Test Set Accuracy</h3>
      <div class="stats-row">
        <div class="stat-card stat-red">
          <div class="label">Baseline</div>
          <div class="value">${tb.overall_accuracy != null ? (tb.overall_accuracy*100).toFixed(1)+'%' : (baselineTraces.length ? (baselineCorr/baselineTraces.length*100).toFixed(1)+'%' : 'N/A')}</div>
          <div class="sub">${baselineCorr}/${baselineTraces.length} correct</div>
        </div>
        <div class="stat-card stat-green">
          <div class="label">Final (evolved)</div>
          <div class="value">${tf.overall_accuracy != null ? (tf.overall_accuracy*100).toFixed(1)+'%' : (finalTraces.length ? (finalCorr/finalTraces.length*100).toFixed(1)+'%' : 'N/A')}</div>
          <div class="sub">${finalCorr}/${finalTraces.length} correct</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h3>Task-Level Results</h3>
      <div class="sub-tabs" id="testSubTabs">
        <div class="sub-tab active" data-subtab="baseline">Baseline (${baselineCorr}/${baselineTraces.length})</div>
        <div class="sub-tab" data-subtab="final">Final (${finalCorr}/${finalTraces.length})</div>
        <div class="sub-tab" data-subtab="comparison">Comparison</div>
        ${reevalSections}
      </div>
      <div id="testSubContent"></div>
    </div>
  `;

  window._baselineTraces = baselineTraces;
  window._finalTraces = finalTraces;

  // Sub-tab click handler
  document.getElementById('testSubTabs').addEventListener('click', e => {
    const st = e.target.closest('.sub-tab');
    if (!st) return;
    document.querySelectorAll('#testSubTabs .sub-tab').forEach(s => s.classList.remove('active'));
    st.classList.add('active');
    renderTestSubTab(st.dataset.subtab);
  });

  renderTestSubTab('baseline');
}

function renderTestSubTab(name) {
  const el = document.getElementById('testSubContent');
  if (name === 'baseline') {
    el.innerHTML = renderTraceTable(window._baselineTraces, 'test-baseline');
  } else if (name === 'final') {
    el.innerHTML = renderTraceTable(window._finalTraces, 'test-final');
  } else if (name === 'comparison') {
    el.innerHTML = renderComparisonTable(window._baselineTraces, window._finalTraces);
  } else if (name.startsWith('reeval-')) {
    const key = '_' + name.replace('-','_');
    el.innerHTML = renderTraceTable(window[key] || [], name);
  }
}

// ── Trace Table Renderer ──────────────────
function renderTraceTable(traces, prefix) {
  if (!traces || !traces.length) return '<p style="color:var(--text-dim);padding:12px;">No traces found</p>';
  return `
    <div style="overflow-x:auto;max-height:60vh;overflow-y:auto;">
      <table>
        <thead><tr>
          <th>#</th><th>Task Key</th><th>Benchmark</th><th>Result</th><th>Score</th><th>Latency</th><th>Trace</th>
        </tr></thead>
        <tbody>
          ${traces.map((t, i) => `<tr>
            <td>${t.task_index ?? i}</td>
            <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${esc(t.task_key||t.task_prompt||'')}">${esc((t.task_key||t.task_prompt||'').substring(0,60))}</td>
            <td>${t.benchmark_name || t.benchmark || ''}</td>
            <td class="${t.task_success ? 'success' : 'failure'}">${t.task_success ? 'PASS' : 'FAIL'}</td>
            <td>${t.task_score != null ? t.task_score.toFixed(2) : ''}</td>
            <td>${t.latency_s != null ? t.latency_s.toFixed(1) + 's' : ''}</td>
            <td>${t.trace_text ? `<button class="btn" onclick="showTrace('${prefix}-${i}')">View</button>` : ''}</td>
          </tr>`).join('')}
        </tbody>
      </table>
    </div>
  `;
  // Store traces for modal access
}

function renderComparisonTable(baseline, final) {
  if (!baseline.length && !final.length) return '<p style="color:var(--text-dim);padding:12px;">No traces</p>';
  const maxLen = Math.max(baseline.length, final.length);
  let rows = '';
  let improved = 0, regressed = 0, same = 0;
  for (let i = 0; i < maxLen; i++) {
    const b = baseline[i] || {};
    const f = final[i] || {};
    const bPass = !!b.task_success;
    const fPass = !!f.task_success;
    let change = '';
    if (bPass && !fPass) { change = '<span style="color:var(--red);font-weight:700">▼ REGRESSED</span>'; regressed++; }
    else if (!bPass && fPass) { change = '<span style="color:var(--green);font-weight:700">▲ IMPROVED</span>'; improved++; }
    else { change = '<span style="color:var(--text-dim)">—</span>'; same++; }
    rows += `<tr>
      <td>${i}</td>
      <td style="max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${esc(b.task_key||b.task_prompt||'')}">${esc((b.task_key||b.task_prompt||'').substring(0,50))}</td>
      <td class="${bPass?'success':'failure'}">${bPass?'PASS':'FAIL'}</td>
      <td class="${fPass?'success':'failure'}">${fPass?'PASS':'FAIL'}</td>
      <td>${change}</td>
    </tr>`;
  }
  return `
    <div class="stats-row" style="margin-bottom:12px;">
      <div class="stat-card stat-green"><div class="label">Improved</div><div class="value">${improved}</div></div>
      <div class="stat-card stat-red"><div class="label">Regressed</div><div class="value">${regressed}</div></div>
      <div class="stat-card"><div class="label">Same</div><div class="value">${same}</div></div>
    </div>
    <div style="overflow-x:auto;max-height:60vh;overflow-y:auto;">
      <table>
        <thead><tr><th>#</th><th>Task</th><th>Baseline</th><th>Final</th><th>Change</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

// ── Trace Modal ───────────────────────────
function showTrace(key) {
  // Parse key: prefix-index
  const parts = key.split('-');
  const idx = parseInt(parts.pop());
  const prefix = parts.join('-');
  let traces;
  if (prefix === 'test-baseline') traces = window._baselineTraces;
  else if (prefix === 'test-final') traces = window._finalTraces;
  else if (prefix.startsWith('shadow-baseline')) traces = null; // loaded differently
  else if (prefix.startsWith('shadow-')) {
    const candName = prefix.replace('shadow-','');
    traces = window._shadowCandData?.[candName]?.traces;
  } else if (prefix.startsWith('reeval-')) {
    const k = '_' + prefix.replace(/-/g,'_');
    traces = window[k];
  }

  // Also check inline shadow baseline
  if (!traces) {
    // Try to find from all loaded data
    return;
  }

  const t = traces?.[idx];
  if (!t) return;

  document.getElementById('traceModalTitle').textContent =
    `${t.task_id || 'Task '+idx} — ${t.task_success ? 'PASS' : 'FAIL'} — ${t.benchmark_name || ''}`;
  document.getElementById('traceModalBody').textContent = t.trace_text || '(no trace text available)';
  document.getElementById('traceModal').classList.add('show');
}

function closeTraceModal() {
  document.getElementById('traceModal').classList.remove('show');
}

// Close modal on overlay click
document.getElementById('traceModal').addEventListener('click', e => {
  if (e.target === e.currentTarget) closeTraceModal();
});

// Close modal on Escape
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeTraceModal();
});

// ── Boot ──────────────────────────────────
init();
</script>
</body>
</html>
"""


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OptPilot Training Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to serve on")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}")
        sys.exit(1)

    DashboardHandler.results_dir = results_dir

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"OptPilot Dashboard running at http://localhost:{args.port}")
    print(f"Results directory: {results_dir}")
    print(f"Found {len(discover_experiments(results_dir))} experiments")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
        server.server_close()


if __name__ == "__main__":
    main()
