from pathlib import Path
import pytest

from regime_v2 import assets as _assets

VINTAGE = Path(__file__).resolve().parents[1] / "data" / "fredmd_2026-07.csv"
RETURNS = Path(__file__).resolve().parents[1] / "data" / "returns_fixture.parquet"


@pytest.fixture(scope="session")
def vintage_path() -> str:
    assert VINTAGE.exists(), f"pinned vintage missing: {VINTAGE}"
    return str(VINTAGE)


@pytest.fixture(scope="session")
def returns_path() -> str:
    assert RETURNS.exists(), f"returns fixture missing: {RETURNS}"
    return str(RETURNS)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Suite-wide guard: no test may hit the live yfinance download, whatever cache/fetch args it uses
    (or omits). Tests that need returns data pass `fetch=` directly or use the `returns_path` fixture,
    both of which bypass `_download_yfinance` entirely."""
    def _blocked(*a, **kw):
        raise RuntimeError("network disabled in tests")
    monkeypatch.setattr(_assets, "_download_yfinance", _blocked)


@pytest.fixture(scope="session")
def published_dir(tmp_path_factory):
    """One real driver run per session, with the asset stage on the pinned fixture cache.

    --wf-step 24 gives genuine (coarse) walk-forward labels cheaply; because a 24-month
    step cannot sample both the GFC and COVID acceptance windows, the gate is stubbed
    exactly as test_run.py does. Thresholds are tested for real in test_acceptance.py.
    """
    import run as runmod
    from regime_v2 import acceptance
    root = tmp_path_factory.mktemp("published")
    out, figs = root / "output", root / "figs"
    mp = pytest.MonkeyPatch()
    mp.setattr(acceptance, "all_passed", lambda table: True)
    try:
        rc = runmod.main([str(VINTAGE), "--wf-step", "24", "--skip-robustness",
                          "--skip-expanding", "--skip-placebo", "--returns-cache", str(RETURNS),
                          "--out-dir", str(out), "--figs-dir", str(figs), "--data-sheet", str(root / "README.md")])
    finally:
        mp.undo()
    assert rc == 0
    return out, figs
