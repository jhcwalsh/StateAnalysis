from pathlib import Path
import pytest

VINTAGE = Path(__file__).resolve().parents[1] / "data" / "fredmd_2026-07.csv"


@pytest.fixture(scope="session")
def vintage_path() -> str:
    assert VINTAGE.exists(), f"pinned vintage missing: {VINTAGE}"
    return str(VINTAGE)
