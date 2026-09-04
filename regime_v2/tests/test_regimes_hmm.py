import numpy as np
import pandas as pd
import pytest

from regime_v2 import data, factors, trend, regimes as R


@pytest.fixture(scope="module")
def gaps(vintage_path):
    b = data.build_blocks(vintage_path)
    m = b["estimation_mask"]
    gf, _ = factors.pca_factor_em(b["growth"], "INDPRO", m)
    pf, _ = factors.pca_factor_em(b["inflation"], "CPIAUCSL", m)
    G = trend.make_gap(gf["factor"], "smoothed_trailing", m)["gap"]
    P = trend.make_gap(pf["diffusion"], "smoothed_trailing", m)["gap"]
    return G, P, m


@pytest.fixture(scope="module")
def fit(gaps):
    G, P, m = gaps
    return R.fit_hmm4(G, P, m)


def test_means_symmetric_and_unmoved(gaps, fit):
    G, P, m = gaps
    target = R.symmetric_means(G, P, m)
    assert np.allclose(fit.means, target)
    assert np.allclose(np.abs(target[:, 0]), np.abs(target[0, 0]))
    assert np.allclose(np.abs(target[:, 1]), np.abs(target[0, 1]))
    assert np.allclose(np.sign(target), [[-1, -1], [1, -1], [1, 1], [-1, 1]])
    assert np.allclose(fit.model.means_, target)


def test_emission_only_equals_sign_quadrants(gaps, fit):
    G, P, _ = gaps
    q0 = R.quadrant_labels(G, P, theta=0.0).reindex(fit.emission_labels.index)
    nonzero = (G.reindex(q0.index) != 0) & (P.reindex(q0.index) != 0)
    assert (fit.emission_labels[nonzero] == q0[nonzero]).mean() >= 0.99


def test_probabilities_well_formed(fit):
    for pr in (fit.probs_filtered, fit.probs_smoothed_expost):
        assert list(pr.columns) == R.REGIMES
        assert np.allclose(pr.sum(axis=1), 1.0)
    assert fit.labels_filtered.index.equals(fit.probs_filtered.index)
    assert (fit.labels_filtered == fit.labels_smoothed_expost).mean() > 0.7


def test_no_hard_zero_transitions(fit):
    assert fit.transmat.to_numpy().min() >= 1e-3
    assert np.allclose(fit.transmat.sum(axis=1), 1.0)
    assert list(fit.transmat.index) == R.REGIMES


def test_seed_invariance(gaps):
    G, P, m = gaps
    a = R.fit_hmm4(G, P, m, seed=0).labels_filtered
    b = R.fit_hmm4(G, P, m, seed=1).labels_filtered
    assert a.equals(b)


def test_fit_excludes_masked_rows_but_scores_them(gaps, fit):
    G, P, m = gaps
    assert "2020-04-01" in fit.labels_filtered.index.strftime("%Y-%m-%d")
    # refitting with the mask removed changes the transition matrix
    alt = R.fit_hmm4(G, P, pd.Series(True, index=m.index))
    assert not np.allclose(alt.transmat.to_numpy(), fit.transmat.to_numpy())


def test_forward_filter_matches_bruteforce():
    rng = np.random.default_rng(0)
    means = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    covs = np.array([np.eye(2)] * 4)
    A = np.full((4, 4), 0.05) + np.eye(4) * 0.8
    X = rng.normal(size=(30, 2))
    probs, ll = R.forward_filter(X, means, covs, np.full(4, 0.25), A)
    # brute force for t=1
    e0 = np.exp(ll[0]) * 0.25; a0 = e0 / e0.sum()
    e1 = (a0 @ A) * np.exp(ll[1]); a1 = e1 / e1.sum()
    assert np.allclose(probs[1], a1)
    assert np.allclose(probs.sum(axis=1), 1.0)
