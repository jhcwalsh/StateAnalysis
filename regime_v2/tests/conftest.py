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
