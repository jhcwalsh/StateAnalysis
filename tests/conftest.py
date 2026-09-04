"""Dashboard tests: one real engine run per session, no network, no Docker."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "regime_v2"
sys.path.insert(0, str(ENGINE))   # makes `regime_v2` (package) and `run` importable, as app.py does


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    from regime_v2 import assets
    def boom(*a, **k): raise RuntimeError("network disabled in tests")
    monkeypatch.setattr(assets, "_download_yfinance", boom)


@pytest.fixture(scope="session")
def published_dir(tmp_path_factory):
    # Costs ~90 s (one real engine run); mirrors regime_v2/tests/conftest.py's fixture — keep the two in step.
    import run as runmod
    from regime_v2 import acceptance
    root = tmp_path_factory.mktemp("published")
    out, figs = root / "output", root / "figs"
    mp = pytest.MonkeyPatch()
    mp.setattr(acceptance, "all_passed", lambda table: True)   # see regime_v2/tests/conftest.py for why
    try:
        rc = runmod.main([str(ENGINE / "data" / "fredmd_2026-07.csv"), "--wf-step", "24", "--skip-robustness",
                          "--skip-expanding", "--skip-placebo", "--returns-cache", str(ENGINE / "data" / "returns_fixture.parquet"),
                          "--out-dir", str(out), "--figs-dir", str(figs), "--data-sheet", str(root / "README.md")])
    finally:
        mp.undo()
    assert rc == 0
    return out, figs
