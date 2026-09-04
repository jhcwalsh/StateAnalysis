from pathlib import Path
import pytest

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
