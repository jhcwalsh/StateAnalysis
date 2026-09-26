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
    # Costs ~90 s (one real engine run); mirrors tests/conftest.py's fixture — keep the two in step.
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


@pytest.fixture(scope="session")
def make_archive():
    """Build a synthetic vintage archive from the pinned file: vintage YYYY-MM holds data rows
    through the end of the previous month (spec D11). `plant[v]` edits one vintage's raw frame."""
    import pandas as pd

    raw_all = pd.read_csv(VINTAGE, dtype=str, keep_default_na=False)
    dates = pd.to_datetime(raw_all["sasdate"].iloc[1:], errors="coerce")

    def build(root, vintages, plant=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for v in vintages:
            end = pd.Period(v, "M") - 1
            keep = (dates <= end.to_timestamp(how="end")).to_numpy()
            raw = pd.concat([raw_all.iloc[:1], raw_all.iloc[1:][keep]])
            if plant and v in plant:
                raw = plant[v](raw)
            raw.to_csv(root / f"fredmd_{v}.csv", index=False)
        return root

    return build
