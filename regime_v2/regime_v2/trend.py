"""Stage 2b — quasi-real-time gaps (D4, D5).

Every estimator uses only data <= t for the gap at t. The expanding
standardisation uses estimation rows only (D9) but scores every row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter


def hamilton_recursive(y: pd.Series, h: int = 24, p: int = 12, min_obs: int = 120) -> pd.DataFrame:
    y = y.dropna()
    vals = y.to_numpy()
    T = len(vals)
    trend = np.full(T, np.nan)
    lag_mat = np.column_stack([np.roll(vals, h + j) for j in range(p)])
    X_all = np.column_stack([np.ones(T), lag_mat])
    first = h + p - 1
    for t in range(max(first + min_obs, first + p + 2), T):
        X, Y = X_all[first:t + 1], vals[first:t + 1]
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        trend[t] = X[-1] @ beta
    return pd.DataFrame({"level": vals, "trend": trend, "gap_raw": vals - trend}, index=y.index)


def trailing_mean(y: pd.Series, window: int = 120, min_obs: int = 60) -> pd.DataFrame:
    y = y.dropna()
    tr = y.rolling(window, min_periods=min_obs).mean()
    return pd.DataFrame({"level": y, "trend": tr, "gap_raw": y - tr})


def trailing_median(y: pd.Series, window: int = 120, min_obs: int = 60) -> pd.DataFrame:
    y = y.dropna()
    tr = y.rolling(window, min_periods=min_obs).median()
    return pd.DataFrame({"level": y, "trend": tr, "gap_raw": y - tr})


def smoothed_trailing(y: pd.Series, smooth: int = 3, window: int = 120, min_obs: int = 60) -> pd.DataFrame:
    """3m MA of a *rate* series minus its one-sided long-run mean (default)."""
    y = y.dropna()
    sm = y.rolling(smooth, min_periods=1).mean()
    tr = sm.rolling(window, min_periods=min_obs).mean()
    return pd.DataFrame({"level": sm, "trend": tr, "gap_raw": sm - tr})


def onesided_hp(y: pd.Series, lamb: float = 129600.0, min_obs: int = 120) -> pd.DataFrame:
    y = y.dropna()
    vals = y.to_numpy()
    trend = np.full(len(vals), np.nan)
    for t in range(min_obs, len(vals)):
        _, tr = hpfilter(vals[: t + 1], lamb=lamb)
        trend[t] = tr[-1]
    return pd.DataFrame({"level": vals, "trend": trend, "gap_raw": vals - trend}, index=y.index)


def standardise_expanding(gap: pd.Series, est_mask: pd.Series, min_obs: int = 60) -> pd.Series:
    """Real-time z-score with the expanding std taken over estimation rows only.

    Masked rows do not update the std but are still scored with the last
    available std (ffill), so every month gets a value.
    """
    m = est_mask.reindex(gap.index).fillna(False)
    sd = gap.where(m).expanding(min_periods=min_obs).std().ffill()
    return gap / sd


_METHODS = {"hamilton": hamilton_recursive, "trailing_mean": trailing_mean,
            "trailing_median": trailing_median, "smoothed_trailing": smoothed_trailing,
            "onesided_hp": onesided_hp}


def make_gap(y: pd.Series, method: str, est_mask: pd.Series, min_obs_std: int = 60, **kw) -> pd.DataFrame:
    out = _METHODS[method](y, **kw)
    out["gap"] = standardise_expanding(out["gap_raw"], est_mask, min_obs_std)
    return out


def centred_trend_expost(y: pd.Series, smooth: int = 3, window: int = 120) -> pd.Series:
    """Two-sided comparator for fig 5 ONLY (D5). Never feeds a label."""
    y = y.dropna()
    sm = y.rolling(smooth, min_periods=1).mean()
    gap = sm - sm.rolling(window, center=True, min_periods=window // 2).mean()
    return gap / gap.std()


def revision_stats(first: pd.Series, final: pd.Series) -> dict:
    df = pd.concat([first, final], axis=1, keys=["first", "final"]).dropna()
    rev = df["final"] - df["first"]
    return {
        "corr_first_final": float(df.corr().iloc[0, 1]),
        "noise_to_signal_rmse": float(np.sqrt((rev ** 2).mean()) / df["final"].std()),
        "sign_agreement": float((np.sign(df["first"]) == np.sign(df["final"])).mean()),
        "n": int(len(df)),
    }
