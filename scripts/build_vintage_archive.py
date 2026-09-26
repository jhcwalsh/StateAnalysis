"""One-off: build the FRED-MD vintage archive the real-time-vintage stage reads
(spec 2026-09-25 §1). Idempotent: existing files are never overwritten.

    python scripts/build_vintage_archive.py --out regime_v2/data/vintages
    docker exec states python scripts/build_vintage_archive.py --out /app/var/vintages

Exit 0 on full monthly coverage from the first zip's start through --through; 2 when
months are missing (listed); 1 when a download fails (URL printed).
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "regime_v2"))
from run import fredmd_urls, looks_like_fredmd  # noqa: E402

BASE = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/"
ZIPS = [("historical_fred-md.zip", "1999-08", "2014-12"),
        ("historical-vintages-of-fred-md-2015-01-to-2025-12.zip", "2015-01", "2025-12")]
_NAME = re.compile(r"(\d{4})[-m_](\d{2})\.csv$", re.IGNORECASE)


def vintage_of(filename: str) -> str | None:
    m = _NAME.search(Path(filename).name)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def extract_zip(zip_bytes: bytes, out_dir) -> tuple[list[str], list[str]]:
    """Returns (written, rejected). A member whose name matches the vintage pattern but whose
    body does not pass `looks_like_fredmd` (a corrupt or mislabelled entry) is never written —
    it comes back in `rejected` instead, so a bad-but-correctly-named file can't silently count
    as covered."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    rejected = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            v = vintage_of(info.filename)
            if v is None or info.is_dir():
                continue
            dest = out_dir / f"fredmd_{v}.csv"
            if dest.exists():
                continue
            data = z.read(info)
            if not looks_like_fredmd(data):
                rejected.append(v)
                continue
            dest.write_bytes(data)
            written.append(v)
    return sorted(written), sorted(rejected)


def fetch_monthly(vintages, out_dir, fetch=urllib.request.urlopen) -> list[str]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    got = []
    for v in vintages:
        dest = out_dir / f"fredmd_{v}.csv"
        if dest.exists():
            got.append(v); continue
        for url in fredmd_urls(v):
            try:
                with fetch(url, timeout=60) as r:
                    data = r.read()
            except Exception:
                continue
            if looks_like_fredmd(data):
                dest.write_bytes(data); got.append(v); break
    return got


def coverage(out_dir) -> dict:
    have = sorted(vintage_of(p.name) for p in Path(out_dir).glob("fredmd_*.csv"))
    if not have:
        return {"first": None, "last": None, "missing": []}
    want = [str(p) for p in pd.period_range(have[0], have[-1], freq="M")]
    return {"first": have[0], "last": have[-1], "missing": [v for v in want if v not in have]}


def main(argv=None, fetch=urllib.request.urlopen) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "regime_v2" / "data" / "vintages"))
    ap.add_argument("--through", default=date.today().strftime("%Y-%m"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    all_rejected = []
    for name, first, last in ZIPS:
        need = [str(p) for p in pd.period_range(first, last, freq="M") if not (out / f"fredmd_{p}.csv").exists()]
        if not need:
            continue
        url = BASE + name
        try:
            with fetch(url, timeout=600) as r:
                data = r.read()
            written, rejected = extract_zip(data, out)
        except Exception as e:
            print(f"download failed: {url}: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        all_rejected += rejected
        print(f"{name}: wrote {len(written)} vintages")
    monthly_start = (pd.Period(ZIPS[-1][2], "M") + 1).strftime("%Y-%m")
    monthly = [str(p) for p in pd.period_range(monthly_start, a.through, freq="M")]
    got = fetch_monthly(monthly, out, fetch)
    print(f"monthly files: {len(got)} of {len(monthly)} through {a.through}")
    if all_rejected:
        print(f"rejected (not FRED-MD): {all_rejected}")
    cov = coverage(out)
    print(f"archive {cov['first']}..{cov['last']}; missing: {cov['missing'] or 'none'}")
    return 2 if cov["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
