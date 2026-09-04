"""Stage 1 — data layer.

Loads a FRED-MD vintage CSV, applies the McCracken–Ng t-code transformations,
removes outliers with the FRED-MD rule (|x - median| > k * IQR -> NaN) using
thresholds computed on estimation rows only, and returns the growth and
inflation blocks. `asof` truncates the raw panel before any statistic is
computed (D1, D8); `mask` is the COVID estimation window (D9).
"""
from __future__ import annotations

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


def load_fredmd(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return (raw levels, t-codes) from a FRED-MD monthly CSV."""
    raw = pd.read_csv(path)
    tcodes = raw.iloc[0, 1:].astype(int)
    tcodes.index = raw.columns[1:]
    df = raw.iloc[1:].copy()
    df["sasdate"] = pd.to_datetime(df["sasdate"])
    df = df.set_index("sasdate").astype(float)
    df.index.name = "date"
    return df.dropna(how="all"), tcodes


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
    flagged = (df - med).abs() > k * iqr
    return df.mask(flagged), flagged


def build_blocks(path: str, k_outlier: float = 10.0, asof: str | None = None,
                 mask: tuple[str, str] | None = COVID_MASK) -> dict:
    levels, tcodes_all = load_fredmd(path)
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
