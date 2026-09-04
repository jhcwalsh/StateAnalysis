"""Stage 3 — regime classification (D6, D7, D10).

Named regimes only. Two classifier families on the same (growth gap,
inflation gap) input:
  quadrant_labels : deterministic sign rule with causal hysteresis (D10)
  fit_hmm4        : constrained 4-state HMM with symmetric fixed emissions
                    (primary) or free emissions (challenger)
  fit_gmm4        : Gaussian mixture challenger
Challengers report under descriptive names and are bridged to quadrants by
marginalisation (D7).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

REGIMES = ["Contraction", "Goldilocks", "Overheating", "Stagflation"]
SIGNS = {"Contraction": (-1, -1), "Goldilocks": (1, -1), "Overheating": (1, 1), "Stagflation": (-1, 1)}
COLORS = {"Contraction": "#3b5bdb", "Goldilocks": "#2b8a3e", "Overheating": "#f08c00", "Stagflation": "#e03131"}
_BY_SIGN = {v: k for k, v in SIGNS.items()}


def hysteretic_sign(series: pd.Series, theta: float) -> pd.Series:
    """Schmitt-trigger sign: flips + -> - only below -theta, - -> + only above +theta.

    Causal by construction (moved verbatim from regime_core.hysteretic_sign).
    """
    vals = series.to_numpy()
    if len(vals) == 0:
        return pd.Series([], index=series.index, dtype=int)
    state = 1 if vals[0] >= 0 else -1
    out = np.empty(len(vals), dtype=int)
    for i, v in enumerate(vals):
        if state > 0 and v < -theta:
            state = -1
        elif state < 0 and v > theta:
            state = 1
        out[i] = state
    return pd.Series(out, index=series.index)


def quadrant_labels(g: pd.Series, p: pd.Series, theta: float = 0.0) -> pd.Series:
    df = pd.concat([g, p], axis=1).dropna()
    sg = hysteretic_sign(df.iloc[:, 0], theta)
    sp = hysteretic_sign(df.iloc[:, 1], theta)
    lab = [_BY_SIGN[(int(a), int(b))] for a, b in zip(sg, sp)]
    return pd.Series(lab, index=df.index, name="quadrant")


def run_lengths(labels: pd.Series) -> pd.Series:
    runs = {r: [] for r in REGIMES}
    cur, n = None, 0
    for v in labels.dropna():
        if v == cur:
            n += 1
        else:
            if cur is not None and cur in runs:
                runs[cur].append(n)
            cur, n = v, 1
    if cur is not None and cur in runs:
        runs[cur].append(n)
    return pd.Series({r: (float(np.mean(v)) if v else np.nan) for r, v in runs.items()})


def expected_duration(tm: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0 / (1.0 - np.diag(tm.to_numpy())), index=tm.index)


def transition_table(transmat: np.ndarray, state_names: dict[int, str]) -> pd.DataFrame:
    """Rows = from-state, columns = to-state. Serialise with orient='index'."""
    names = [state_names[s] for s in range(len(state_names))]
    tm = pd.DataFrame(transmat, index=names, columns=names)
    order = [n for n in REGIMES if n in names] + [n for n in names if n not in REGIMES]
    return tm.loc[order, order]


from hmmlearn.hmm import GaussianHMM  # noqa: E402  (kept below the pure helpers on purpose)


def _aligned(g: pd.Series, p: pd.Series, est_mask: pd.Series):
    X = pd.concat([g.rename("g"), p.rename("p")], axis=1).dropna()
    m = est_mask.reindex(X.index).fillna(False).to_numpy()
    return X, m


def symmetric_means(g: pd.Series, p: pd.Series, est_mask: pd.Series) -> np.ndarray:
    """(±c_g, ±c_p) with c = mean absolute standardised gap over estimation rows (D6)."""
    X, m = _aligned(g, p, est_mask)
    c_g = float(X.loc[m, "g"].abs().mean())
    c_p = float(X.loc[m, "p"].abs().mean())
    return np.array([[SIGNS[r][0] * c_g, SIGNS[r][1] * c_p] for r in REGIMES])


def pooled_cov(g: pd.Series, p: pd.Series, means: np.ndarray, est_mask: pd.Series) -> np.ndarray:
    """Diagonal pooled within-quadrant covariance of residuals from the fixed means."""
    X, m = _aligned(g, p, est_mask)
    q = quadrant_labels(X["g"], X["p"], theta=0.0)
    resid = np.vstack([X.loc[m & (q == r).to_numpy()].to_numpy() - means[i] for i, r in enumerate(REGIMES)])
    return np.diag(resid.var(axis=0, ddof=1))


def forward_filter(X: np.ndarray, means: np.ndarray, covs: np.ndarray,
                   startprob: np.ndarray, transmat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Causal posterior P(state_t | x_1..x_t). Pure numpy; no hmmlearn internals."""
    K = len(means)
    ll = np.column_stack([multivariate_normal.logpdf(X, means[k], covs[k]) for k in range(K)])
    T = len(X)
    out = np.zeros((T, K))
    lp = np.log(startprob + 1e-300) + ll[0]
    out[0] = np.exp(lp - logsumexp(lp))
    for t in range(1, T):
        lp = np.log(out[t - 1] @ transmat + 1e-300) + ll[t]
        out[t] = np.exp(lp - logsumexp(lp))
    return out, ll


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of True as (start, stop) index pairs."""
    segs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            segs.append((start, i)); start = None
    if start is not None:
        segs.append((start, len(mask)))
    return segs


@dataclass
class HMMResult:
    labels_filtered: pd.Series
    labels_smoothed_expost: pd.Series
    probs_filtered: pd.DataFrame
    probs_smoothed_expost: pd.DataFrame
    model: object
    state_map: dict
    means: np.ndarray
    covs: np.ndarray
    transmat: pd.DataFrame
    emission_labels: pd.Series
    quadrant_profile: pd.DataFrame | None = None
    quadrant_probs_filtered: pd.DataFrame | None = None


def fit_hmm4(g: pd.Series, p: pd.Series, est_mask: pd.Series, persistence: float = 10.0,
             eps: float = 0.5, seed: int = 0, constrained: bool = True) -> HMMResult:
    X, m = _aligned(g, p, est_mask)
    Xv = X.to_numpy()
    prior = 1.0 + eps + persistence * np.eye(4)
    if constrained:
        means = symmetric_means(g, p, est_mask)
        cov = pooled_cov(g, p, means, est_mask)
        model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=500, tol=1e-4,
                            random_state=seed, init_params="s", params="st", transmat_prior=prior)
        model.means_ = means
        model.covars_ = np.tile(np.diag(cov), (4, 1))
        state_map = {k: REGIMES[k] for k in range(4)}
    else:
        q = quadrant_labels(X["g"], X["p"], theta=0.0)
        init = np.array([X.loc[m & (q == r).to_numpy()].mean().to_numpy() for r in REGIMES])
        model = GaussianHMM(n_components=4, covariance_type="full", n_iter=500, tol=1e-4,
                            random_state=seed, init_params="sc", params="stmc", transmat_prior=prior)
        model.means_ = init
        state_map = None  # filled below from fitted means
    model.transmat_ = np.full((4, 4), 0.02) + np.eye(4) * 0.92
    segs = _segments(m)
    Xfit = np.vstack([Xv[a:b] for a, b in segs])
    model.fit(Xfit, lengths=[b - a for a, b in segs])
    if not constrained:
        state_map = {k: describe_state(model.means_[k], k) for k in range(4)}
    means = np.asarray(model.means_)
    covs = np.array([np.diag(model.covars_[k]) if model.covars_.ndim == 2 else model.covars_[k] for k in range(4)])
    names = [state_map[k] for k in range(4)]
    filt, ll = forward_filter(Xv, means, covs, model.startprob_, model.transmat_)
    smooth = model.predict_proba(Xv)
    order = [n for n in REGIMES if n in names] + [n for n in names if n not in REGIMES]
    probs_f = pd.DataFrame(filt, index=X.index, columns=names)[order]
    probs_s = pd.DataFrame(smooth, index=X.index, columns=names)[order]
    emis = pd.Series([names[i] for i in ll.argmax(axis=1)], index=X.index, name="emission")
    return HMMResult(
        labels_filtered=probs_f.idxmax(axis=1).rename("hmm_filtered"),
        labels_smoothed_expost=probs_s.idxmax(axis=1).rename("hmm_smoothed_expost"),
        probs_filtered=probs_f, probs_smoothed_expost=probs_s, model=model, state_map=state_map,
        means=means, covs=covs, transmat=transition_table(model.transmat_, state_map),
        emission_labels=emis,
    )


def describe_state(mean: np.ndarray, k: int) -> str:
    """Descriptive, deterministic name for a free cluster: S<k>_<G>G_<P>Pi."""
    def band(v):
        return "Low" if v < -0.25 else ("High" if v > 0.25 else "Mid")
    return f"S{k}_{band(mean[0])}G_{band(mean[1])}Pi"


from sklearn.mixture import GaussianMixture  # noqa: E402


def quadrant_profile(cluster_labels: pd.Series, quad_labels: pd.Series,
                     clusters: list[str] | None = None) -> pd.DataFrame:
    """Empirical P(quadrant | cluster); rows sum to 1 (D7).

    `clusters` is the full list of cluster names; a cluster that never wins an
    argmax gets a uniform row so marginalisation keeps its probability mass.
    """
    df = pd.concat([cluster_labels.rename("c"), quad_labels.rename("q")], axis=1).dropna()
    prof = pd.crosstab(df["c"], df["q"]).reindex(columns=REGIMES, fill_value=0).astype(float)
    if clusters is not None:
        prof = prof.reindex(index=list(clusters), fill_value=0.0)
    empty = prof.sum(axis=1) == 0
    prof.loc[empty, :] = 1.0 / len(REGIMES)
    return prof.div(prof.sum(axis=1), axis=0)


def marginalise(cluster_probs: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """P(quadrant | t) = sum_k P(cluster k | t) P(quadrant | cluster k)."""
    out = cluster_probs[profile.index].to_numpy() @ profile.to_numpy()
    return pd.DataFrame(out, index=cluster_probs.index, columns=REGIMES)


@dataclass
class GMMResult:
    labels: pd.Series
    probs: pd.DataFrame
    model: object
    cluster_names: list
    quadrant_profile: pd.DataFrame
    quadrant_probs: pd.DataFrame


def fit_gmm4(g: pd.Series, p: pd.Series, est_mask: pd.Series, seed: int = 0) -> GMMResult:
    X, m = _aligned(g, p, est_mask)
    q = quadrant_labels(X["g"], X["p"], theta=0.0)
    init = np.array([X.loc[m & (q == r).to_numpy()].mean().to_numpy() for r in REGIMES])
    model = GaussianMixture(n_components=4, covariance_type="full", means_init=init,
                            random_state=seed, n_init=1, max_iter=500)
    model.fit(X.to_numpy()[m])
    names = [describe_state(model.means_[k], k) for k in range(4)]
    probs = pd.DataFrame(model.predict_proba(X.to_numpy()), index=X.index, columns=names)
    labels = probs.idxmax(axis=1).rename("gmm")
    prof = quadrant_profile(labels, q, clusters=names)
    return GMMResult(labels=labels, probs=probs, model=model, cluster_names=names,
                     quadrant_profile=prof, quadrant_probs=marginalise(probs, prof))


def fit_free_hmm4(g: pd.Series, p: pd.Series, est_mask: pd.Series, persistence: float = 10.0,
                  eps: float = 0.5, seed: int = 0) -> HMMResult:
    res = fit_hmm4(g, p, est_mask, persistence=persistence, eps=eps, seed=seed, constrained=False)
    X, _ = _aligned(g, p, est_mask)
    q = quadrant_labels(X["g"], X["p"], theta=0.0)
    res.quadrant_profile = quadrant_profile(res.labels_filtered, q, clusters=list(res.state_map.values()))
    res.quadrant_probs_filtered = marginalise(res.probs_filtered, res.quadrant_profile)
    res.labels_filtered = res.labels_filtered.rename("hmm_free")
    return res
