"""Data access + refresh for the regime dashboard. No Streamlit imports."""
import json
import os
import subprocess
import sys

import pandas as pd

SCHEMA_VERSION = 1
NOTEBOOK = "Macro_Regime_Analysis.ipynb"
_PARQUETS = ["gaps", "labels", "probs", "returns", "backtest"]


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
        if not os.path.exists(path):
            raise SchemaError(f"missing {name}.parquet; re-run the notebook.")
    for name in _PARQUETS:
        bundle[name] = pd.read_parquet(os.path.join(ui_dir, name + ".parquet"))
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
