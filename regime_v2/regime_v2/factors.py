"""Stage 2a — factor extraction (D3).

One-factor PCA per block with EM imputation of NaN cells. Loadings, block
standardisation and the factor's own standardisation use estimation rows
only (est_mask); every row is scored. Convergence is judged on the rank-1
reconstruction, which is invariant to the SVD sign flip that made the
prototype run to its iteration cap every time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _first_pc_loadings(X: np.ndarray) -> np.ndarray:
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[0]


def pca_factor_em(block: pd.DataFrame, anchor: str, est_mask: pd.Series,
                  n_iter: int = 50, tol: float = 1e-6, _trace: list | None = None
                  ) -> tuple[pd.DataFrame, pd.Series]:
    df = block.dropna(how="all")
    m = est_mask.reindex(df.index).fillna(False).to_numpy()
    mu, sd = df[m].mean(), df[m].std()
    Z = (df - mu) / sd
    obs = Z.notna().to_numpy()
    Zv = Z.to_numpy()
    X = np.where(obs, Zv, 0.0)
    prev_recon = None
    for _ in range(n_iter):
        load = _first_pc_loadings(X[m])
        scores = X @ load
        recon = np.outer(scores, load)
        X = np.where(obs, Zv, recon)
        if _trace is not None:
            _trace.append(1)
        if prev_recon is not None and np.abs(recon - prev_recon).max() < tol:
            break
        prev_recon = recon
    f = pd.Series(scores, index=df.index)
    sign = 1.0
    if anchor in df.columns:
        a = Z[anchor].to_numpy()
        ok = m & ~np.isnan(a)
        if np.corrcoef(f.to_numpy()[ok], a[ok])[0, 1] < 0:
            sign = -1.0
    f = sign * f
    f = (f - f[m].mean()) / f[m].std()
    out = pd.DataFrame({"factor": f, "diffusion": f.cumsum(),
                        "n_series": obs.sum(axis=1)}, index=df.index)
    return out, pd.Series(sign * load, index=df.columns, name="loading")


def pca_factor_expanding(block: pd.DataFrame, anchor: str, est_mask: pd.Series,
                         min_obs: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Endpoint factor: month t scored with loadings and moments from data <= t."""
    df = block.dropna(how="all")
    rows, loads = [], {}
    for t in range(min_obs - 1, len(df)):
        sub = df.iloc[: t + 1]
        f, l = pca_factor_em(sub, anchor, est_mask.reindex(sub.index).fillna(False))
        rows.append((sub.index[-1], f["factor"].iloc[-1], f["n_series"].iloc[-1]))
        loads[sub.index[-1]] = l
    ep = pd.DataFrame(rows, columns=["date", "factor", "n_series"]).set_index("date")
    ep = ep.reindex(df.index)
    ep["diffusion"] = ep["factor"].cumsum()
    return ep[["factor", "diffusion", "n_series"]], pd.DataFrame(loads).T.reindex(df.index)
