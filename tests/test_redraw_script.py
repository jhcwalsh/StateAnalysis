"""scripts/redraw_doc_figures.py: the entrypoint's redraw of the doc figures from a published run."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redraw_doc_figures.py"


def test_script_redraws_from_a_published_run(published_dir):
    out, figs = published_dir
    (figs / "doc_pipeline.png").unlink(missing_ok=True)
    try:
        r = subprocess.run([sys.executable, str(SCRIPT), "--out-dir", str(out), "--figs-dir", str(figs)],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        assert (figs / "doc_pipeline.png").exists()
        s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert s["doc_figures"]["doc_pipeline"] == "doc_pipeline.png"
        assert "doc figures redrawn" in r.stdout
    finally:
        # Restore the session-scoped fixture regardless of the outcome above, so a failed
        # assertion here does not leak into every other test that shares published_dir.
        subprocess.run([sys.executable, str(SCRIPT), "--out-dir", str(out), "--figs-dir", str(figs)],
                       capture_output=True, text=True, cwd=ROOT)


def test_script_exits_1_without_a_published_run(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "--out-dir", str(tmp_path / "o"), "--figs-dir", str(tmp_path / "f")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1
    assert "no published run" in r.stderr


def test_script_reads_the_container_env(tmp_path, monkeypatch):
    # The entrypoint passes no flags; the defaults are the Dockerfile's REGIME_* variables.
    import os
    env = dict(os.environ, REGIME_OUTPUT_DIR=str(tmp_path / "o"), REGIME_FIGS_DIR=str(tmp_path / "f"))
    r = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 1 and str(tmp_path / "o") in r.stderr
