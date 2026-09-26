"""Stage 1 — data layer.

Loads a FRED-MD vintage CSV, applies the McCracken–Ng t-code transformations,
removes outliers with the FRED-MD rule (|x - median| > k * IQR -> NaN) using
thresholds computed on estimation rows only, and returns the growth and
inflation blocks. `asof` truncates the raw panel before any statistic is
computed (D1, D8); `mask` is the COVID estimation window (D9).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COVID_MASK = ("2020-03-01", "2020-12-01")

GROWTH_BLOCK = [
    "INDPRO", "IPFINAL", "IPCONGD", "IPBUSEQ", "IPMANSICS", "CUMFNS",
    "PAYEMS", "USGOOD", "MANEMP", "SRVPRD", "USTPU", "HWI",
    "UNRATE", "CLAIMSx", "UEMPMEAN",
    "RETAILx", "DPCERA3M086SBEA", "CMRMTSPLx", "RPI", "W875RX1",
    "HOUST", "PERMIT",
]

INFLATION_BLOCK = [
    "CPIAUCSL", "CPIULFSL", "CUSR0000SA0L2", "CUSR0000SA0L5",
    "PCEPI", "DNDGRG3M086SBEA", "DSERRG3M086SBEA",
    "WPSFD49207", "WPSFD49502", "PPICMM",
    "CES0600000008", "CES2000000008", "CES3000000008",
]


PINNED_VINTAGE = Path(__file__).resolve().parents[1] / "data" / "fredmd_2026-07.csv"
VINTAGE_MIN_CORR = 0.98      # level correlation a series must keep with the pinned vintage on the overlap
_REFERENCE_CACHE: dict = {}
_WARNED: set = set()


class VintageError(ValueError):
    """The vintage disagrees with the pinned one: header misaligned or series redefined."""


def _read_raw(path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    return raw


def repair_header(raw: pd.DataFrame, ref: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Restore a header that lost exactly one name while the data kept the column.

    The 2026-08 FRED-MD file dropped `S&P div yield` from the header line only, so
    every later series was mislabelled by one and the last real column read as
    "Unnamed". When the field count matches the reference and the named columns are
    the reference's minus one name, the reference header is the correct one.
    """
    named = [c for c in raw.columns if not c.startswith("Unnamed")]
    ref_named = list(ref.columns)
    if raw.shape[1] == ref.shape[1] and raw.columns[-1].startswith("Unnamed"):
        missing = [c for c in ref_named if c not in named]
        if len(missing) == 1 and named == [c for c in ref_named if c != missing[0]]:
            # The t-code row is aligned with the (short) header, not with the data rows:
            # keep each series' t-code by name and take the dropped series' from the reference.
            tcodes_by_name = {c: raw.iloc[0][c] for c in named}
            tcodes_by_name[missing[0]] = ref.iloc[0][missing[0]]
            raw = raw.copy()
            raw.columns = ref_named
            raw.iloc[0, 1:] = [tcodes_by_name[c] for c in ref_named[1:]]
            return raw, (f"header repaired against the pinned vintage: '{missing[0]}' was dropped from the "
                         f"header line but its data column was kept, shifting every later series by one")
    return raw, None


def _levels(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # A trailing comma or a genuinely dropped series leaves "Unnamed" columns with NaN
    # t-codes; keep only named columns whose t-code parses.
    raw = raw.loc[:, [c for c in raw.columns if c and not c.startswith("Unnamed")]]
    tc = pd.to_numeric(raw.iloc[0, 1:], errors="coerce")
    keep = tc.notna().to_numpy()
    tcodes = tc[keep].astype(int)
    tcodes.index = raw.columns[1:][keep]
    df = raw.iloc[1:, :].loc[:, ["sasdate", *tcodes.index]].copy()
    df["sasdate"] = pd.to_datetime(df["sasdate"])
    df = df.set_index("sasdate").astype(float)
    df.index.name = "date"
    return df.dropna(how="all"), tcodes


def _reference(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = (str(path), os.path.getmtime(path))
    if key not in _REFERENCE_CACHE:
        raw = _read_raw(path)
        _REFERENCE_CACHE.clear()
        _REFERENCE_CACHE[key] = (raw, _levels(raw)[0])
    return _REFERENCE_CACHE[key]


def check_vintage(levels: pd.DataFrame, ref_levels: pd.DataFrame, series: list[str],
                  min_corr: float = VINTAGE_MIN_CORR, start: str = "1990-01-01") -> list[str]:
    """Series whose levels disagree with the reference on the overlap (corr < min_corr).

    Revisions barely move a level correlation; a misaligned header or a redefined
    series drops it well below 0.98.
    """
    bad = []
    j = levels.index.intersection(ref_levels.index)
    j = j[j >= pd.Timestamp(start)]
    for c in series:
        if c not in levels.columns or c not in ref_levels.columns:
            continue
        d = pd.concat([levels.loc[j, c], ref_levels.loc[j, c]], axis=1).dropna()
        if len(d) < 24:
            continue
        r = float(d.corr().iloc[0, 1])
        if not (r >= min_corr):
            bad.append(c)
    return bad


def load_fredmd(path: str, reference=PINNED_VINTAGE) -> tuple[pd.DataFrame, pd.Series]:
    """Return (raw levels, t-codes) from a FRED-MD monthly CSV.

    Any vintage other than the pinned one is repaired and checked against it (see
    `repair_header`, `check_vintage`); a vintage that still disagrees raises
    `VintageError` so a refresh fails and the last good outputs stay published (§12).
    A repair is recorded in `levels.attrs["vintage_note"]`.
    """
    raw = _read_raw(path)
    note = None
    use_ref = reference is not None and Path(reference).exists() and Path(reference).resolve() != Path(path).resolve()
    if use_ref:
        ref_raw, ref_levels = _reference(reference)
        raw, note = repair_header(raw, ref_raw)
    df, tcodes = _levels(raw)
    if use_ref:
        bad = check_vintage(df, ref_levels, GROWTH_BLOCK + INFLATION_BLOCK)
        if bad:
            raise VintageError(f"{os.path.basename(str(path))} disagrees with the pinned vintage on {bad}: "
                               "header misaligned or series redefined; refusing to publish")
        ref_tc = _levels(ref_raw)[1]
        changed = {c: (int(ref_tc[c]), int(tcodes[c])) for c in GROWTH_BLOCK + INFLATION_BLOCK
                   if c in ref_tc.index and c in tcodes.index and int(ref_tc[c]) != int(tcodes[c])}
        if changed:
            raise VintageError(f"{os.path.basename(str(path))} changes t-codes versus the pinned vintage "
                               f"{changed} (reference, new); update the pinned vintage deliberately instead")
        if note and str(path) not in _WARNED:      # once per vintage, not once per walk-forward step
            _WARNED.add(str(path))
            print(f"warning: {note}", file=sys.stderr)
    df.attrs["vintage_note"] = note
    return df, tcodes


# --- archive vintages (spec 2026-09-25 §2) --------------------------------------------------
ARCHIVE_RENAMES = {"PPIFGS": "WPSFD49207", "PPIFCG": "WPSFD49502"}   # same series, older names
ARCHIVE_DROP = {"CUUR0000SA0L2"}     # the NSA CPI stand-in of 2015-01..2017-03: never substituted
ANCHORS = ("INDPRO", "CPIAUCSL")


def _normalise_archive_raw(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the archive rename/drop treatment to a raw frame; used for both the vintage under
    test and its neighbour so the two can never see different column names for the same series
    (a rename applied to only one side would make `check_vintage`/the t-code comparison silently
    skip the renamed series instead of checking it)."""
    renamed = {old: new for old, new in ARCHIVE_RENAMES.items() if old in raw.columns and new not in raw.columns}
    raw = raw.rename(columns=renamed)
    raw = raw.drop(columns=[c for c in ARCHIVE_DROP if c in raw.columns])
    return raw, renamed


def load_archive_vintage(path: str, neighbour: str | None = None) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Return (levels, tcodes, report) for a historical FRED-MD vintage.

    Unlike `load_fredmd`, which checks a live vintage against the pinned one and refuses the
    whole file, an archive vintage is checked *per series* against its neighbouring vintage:
    a series that fails the level-correlation or t-code check is dropped and recorded, and the
    vintage is kept. The vintage is refused only when an anchor series is missing or failing,
    or when a block keeps fewer than half its series.
    """
    raw, renamed = _normalise_archive_raw(_read_raw(path))
    levels, tcodes = _levels(raw)
    block = GROWTH_BLOCK + INFLATION_BLOCK
    dropped_check: list[str] = []
    dropped_tcode: list[str] = []
    if neighbour is not None:
        n_raw, _ = _normalise_archive_raw(_read_raw(neighbour))
        n_levels, n_tcodes = _levels(n_raw)
        dropped_check = check_vintage(levels, n_levels, block)
        dropped_tcode = [c for c in block if c in tcodes.index and c in n_tcodes.index
                         and int(tcodes[c]) != int(n_tcodes[c]) and c not in dropped_check]
    for a in ANCHORS:
        if a not in levels.columns or a in dropped_check or a in dropped_tcode:
            raise VintageError(f"{os.path.basename(str(path))}: anchor series {a} is missing or fails the "
                               "neighbour check; refusing the vintage")
    drop = dropped_check + dropped_tcode
    levels = levels.drop(columns=drop)
    tcodes = tcodes.drop(index=drop)
    missing = sorted(c for c in block if c not in levels.columns)
    n_growth = sum(c in levels.columns for c in GROWTH_BLOCK)
    n_inflation = sum(c in levels.columns for c in INFLATION_BLOCK)
    for name, n, total in (("growth", n_growth, len(GROWTH_BLOCK)), ("inflation", n_inflation, len(INFLATION_BLOCK))):
        if n * 2 < total:
            raise VintageError(f"{os.path.basename(str(path))}: only {n} of {total} {name} series usable; "
                               "refusing the vintage")
    report = {"renamed": renamed, "dropped_check": dropped_check, "dropped_tcode": dropped_tcode,
              "missing": missing, "n_growth": n_growth, "n_inflation": n_inflation}
    levels.attrs["vintage_note"] = None
    return levels, tcodes, report


def transform(x: pd.Series, tcode: int) -> pd.Series:
    """McCracken–Ng transformation codes 1–7."""
    if tcode == 1:
        return x
    if tcode == 2:
        return x.diff()
    if tcode == 3:
        return x.diff().diff()
    if tcode == 4:
        return np.log(x)
    if tcode == 5:
        return np.log(x).diff()
    if tcode == 6:
        return np.log(x).diff().diff()
    if tcode == 7:
        return (x / x.shift(1) - 1.0).diff()
    raise ValueError(f"unknown tcode {tcode}")


def estimation_mask(index: pd.DatetimeIndex, mask: tuple[str, str] | None,
                    coverage: pd.Series | None = None, min_coverage: float = 0.5) -> pd.Series:
    """True where a month may be used for estimation.

    `mask` = (start, end) window excluded (D9). `coverage` = share of a
    block's series present each month; months below `min_coverage` are also
    excluded (D3: 2020-04 has 3 of 22 growth series).
    """
    m = pd.Series(True, index=index)
    if mask is not None:
        m[(index >= pd.Timestamp(mask[0])) & (index <= pd.Timestamp(mask[1]))] = False
    if coverage is not None:
        m &= coverage.reindex(index).fillna(0.0) >= min_coverage
    return m


def remove_outliers(df: pd.DataFrame, k: float, est_mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FRED-MD rule with median/IQR computed on estimation rows only.

    Returns (cleaned, flagged) where flagged marks the removed cells.
    """
    ref = df[est_mask.reindex(df.index).fillna(False).to_numpy()]
    med = ref.median()
    iqr = ref.quantile(0.75) - ref.quantile(0.25)
    # A zero IQR (more than half the estimation rows share one value, e.g. a coarsely
    # rounded early history whose second log-difference is exactly zero, as PPICMM in
    # the 2026-08 vintage on pre-2000 windows) makes the rule undefined: it would flag
    # every non-modal value and leave a zero-variance column. Skip such columns.
    flagged = ((df - med).abs() > k * iqr) & (iqr > 0)
    return df.mask(flagged), flagged


def build_blocks(path: str, k_outlier: float = 10.0, asof: str | None = None,
                 mask: tuple[str, str] | None = COVID_MASK, loader=None) -> dict:
    levels, tcodes_all = (loader or load_fredmd)(path)[:2]
    if asof is not None:
        levels = levels[levels.index <= pd.Timestamp(asof)]
    wanted = GROWTH_BLOCK + INFLATION_BLOCK
    cols = [c for c in wanted if c in levels.columns]
    missing = sorted(set(wanted) - set(cols))
    stat = pd.DataFrame({c: transform(levels[c], int(tcodes_all[c])) for c in cols})
    stat = stat.dropna(how="all")
    # coverage before outlier removal only counts raw availability; the thin-month
    # rule must see post-outlier coverage, so run the rule twice: first with the
    # COVID mask alone, then rebuild the mask with coverage.
    m0 = estimation_mask(stat.index, mask)
    cleaned, flagged = remove_outliers(stat, k_outlier, m0)
    g_cols = [c for c in GROWTH_BLOCK if c in cleaned]
    p_cols = [c for c in INFLATION_BLOCK if c in cleaned]
    coverage = pd.concat([cleaned[g_cols].notna().mean(axis=1),
                          cleaned[p_cols].notna().mean(axis=1)], axis=1).min(axis=1)
    est = estimation_mask(stat.index, mask, coverage)
    outliers = flagged.stack()
    outliers = outliers[outliers].reset_index()
    outliers.columns = ["date", "series", "removed"]
    return {
        "growth": cleaned[g_cols],
        "inflation": cleaned[p_cols],
        "outliers": outliers.drop(columns="removed"),
        "missing_series": missing,
        "tcodes": tcodes_all[cols],
        "estimation_mask": est,
    }
