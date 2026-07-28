"""Data access + refresh for the regime dashboard. No Streamlit imports."""
import json
import os
import subprocess
import sys

import pandas as pd

SCHEMA_VERSION = 1
NOTEBOOK = "Macro_Regime_Analysis.ipynb"
# Columns the dashboard hard-depends on, per parquet. The notebook's export
# cell derives its file list from these keys; keep in sync with app.py reads.
REQUIRED_COLUMNS = {
    "gaps": ["g_gap", "p_gap", "pit_g", "pit_p",
             "growth_factor", "inflation_factor"],
    "labels": ["quad", "pit_quad", "gmm_cluster"],
    "probs": ["Q1_Goldilocks", "Q2_Overheating",
              "Q3_Stagflation", "Q4_Recession"],
    "returns": [],
    "backtest": ["PIT_MaxSharpe", "PIT_MinVar", "Oracle_MaxSharpe",
                 "Static_6040", "EqualWeight"],
}
_PARQUETS = list(REQUIRED_COLUMNS)


class SchemaError(Exception):
    """ui_data/ exists but does not match the expected contract."""


def load_bundle(ui_dir):
    """Load ui_data/. None -> empty state (no run yet). SchemaError -> bad data."""
    meta_path = os.path.join(ui_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    meta = json.load(open(meta_path, encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"ui_data schema {meta.get('schema_version')!r} != {SCHEMA_VERSION}; "
            "re-run the notebook to regenerate ui_data/.")
    bundle = {"meta": meta, "tables_path": os.path.join(ui_dir, "tables.xlsx")}
    for name in _PARQUETS:
        path = os.path.join(ui_dir, name + ".parquet")
        try:
            df = pd.read_parquet(path)
        except FileNotFoundError:
            raise SchemaError(f"missing {name}.parquet; re-run the notebook.")
        missing = set(REQUIRED_COLUMNS[name]) - set(df.columns)
        if missing:
            raise SchemaError(
                f"{name}.parquet lacks columns {sorted(missing)}; "
                "re-run the notebook to regenerate ui_data/.")
        bundle[name] = df
    if not os.path.exists(bundle["tables_path"]):
        raise SchemaError("missing tables.xlsx; re-run the notebook.")
    return bundle


def run_refresh(project_dir, timeout_s=1800, runner=subprocess.run):
    """Execute the notebook (exports ui_data/ via Cell M). Returns (ok, log_tail)."""
    out_dir = os.path.join(project_dir, "ui_data", "last_run")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
           "--execute", "--ExecutePreprocessor.timeout=1500",
           os.path.join(project_dir, NOTEBOOK),
           "--output-dir", out_dir, "--output", "last_refresh.ipynb"]
    try:
        proc = runner(cmd, cwd=project_dir, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, f"refresh timed out after {timeout_s}s"
    tail = (proc.stderr or b"").decode(errors="replace")[-2000:]
    return proc.returncode == 0, tail
