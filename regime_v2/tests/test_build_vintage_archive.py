import io, sys, zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_vintage_archive as B   # noqa: E402

FRED = b"sasdate,INDPRO\nTransform:,5\n1/1/2020,1.0\n"


def _zip(names):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n in names:
            z.writestr(n, FRED)
    return buf.getvalue()


def test_vintage_of_handles_the_three_filename_styles():
    assert B.vintage_of("2015-01.csv") == "2015-01"
    assert B.vintage_of("FRED-MD-2025m06.csv") == "2025-06"
    assert B.vintage_of("FRED-MD_2024m10.csv") == "2024-10"
    assert B.vintage_of("sub/FRED-MD_2024m10.csv") == "2024-10"
    assert B.vintage_of("ReadMe.txt") is None


def test_extract_zip_normalises_names_and_skips_existing(tmp_path):
    out = tmp_path / "v"
    out.mkdir(); (out / "fredmd_2015-01.csv").write_bytes(b"keep")
    written, rejected = B.extract_zip(_zip(["2015-01.csv", "FRED-MD-2015m02.csv", "ReadMe.txt"]), out)
    assert written == ["2015-02"]
    assert rejected == []
    assert (out / "fredmd_2015-01.csv").read_bytes() == b"keep"
    assert (out / "fredmd_2015-02.csv").read_bytes() == FRED


def test_extract_zip_rejects_a_correctly_named_but_corrupt_entry(tmp_path):
    """A zip entry whose name matches the vintage pattern but whose body isn't FRED-MD data
    (a holding page saved under the right name, corruption, etc.) must not be written, and must
    not silently count as covered."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("2015-01.csv", FRED)
        z.writestr("2015-02.csv", b"<html>not a vintage</html>")
    out = tmp_path / "v"
    written, rejected = B.extract_zip(buf.getvalue(), out)
    assert written == ["2015-01"]
    assert rejected == ["2015-02"]
    assert not (out / "fredmd_2015-02.csv").exists()
    assert (out / "fredmd_2015-01.csv").read_bytes() == FRED
    assert B.coverage(out) == {"first": "2015-01", "last": "2015-01", "missing": []}


def test_fetch_monthly_prefers_the_revised_file_and_skips_holding_pages(tmp_path):
    seen = []

    class R:
        def __init__(self, data): self.data = data
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self.data

    def fetch(url, timeout=60):
        seen.append(url)
        if "2026-rev-01" in url:
            return R(FRED)
        if "2026-02" in url:
            return R(b"<html>not yet</html>")
        raise OSError("404")

    got = B.fetch_monthly(["2026-01", "2026-02"], tmp_path, fetch)
    assert got == ["2026-01"] and (tmp_path / "fredmd_2026-01.csv").exists()
    assert not (tmp_path / "fredmd_2026-02.csv").exists()
    assert seen[0].endswith("2026-rev-01-md.csv")


def test_coverage_reports_gaps(tmp_path):
    for v in ("2020-01", "2020-02", "2020-04"):
        (tmp_path / f"fredmd_{v}.csv").write_bytes(FRED)
    assert B.coverage(tmp_path) == {"first": "2020-01", "last": "2020-04", "missing": ["2020-03"]}


def test_main_exit_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "ZIPS", [("z.zip", "2020-01", "2020-02")])
    calls = {}

    class R:
        def __init__(self, data): self.data = data
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self.data

    def fetch(url, timeout=60):
        calls[url] = 1
        if url.endswith("z.zip"):
            return R(_zip(["2020-01.csv", "2020-02.csv"]))
        return R(FRED)
    assert B.main(["--out", str(tmp_path), "--through", "2020-03"], fetch=fetch) == 0
    assert B.coverage(tmp_path)["missing"] == []

    def broken(url, timeout=60):
        raise OSError("boom")
    assert B.main(["--out", str(tmp_path / "b"), "--through", "2020-03"], fetch=broken) == 1
