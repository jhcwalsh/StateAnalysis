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


def test_quadrant_profile_and_marginalise():
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    cl = pd.Series(["A", "A", "B", "B", "B", "A"], index=idx)
    q = pd.Series(["Goldilocks", "Goldilocks", "Contraction", "Stagflation", "Contraction", "Overheating"], index=idx)
    prof = R.quadrant_profile(cl, q)
    assert list(prof.columns) == R.REGIMES and np.allclose(prof.sum(axis=1), 1.0)
    assert prof.loc["A", "Goldilocks"] == pytest.approx(2 / 3)
    probs = pd.DataFrame({"A": [1.0, 0.5], "B": [0.0, 0.5]}, index=idx[:2])
    mq = R.marginalise(probs, prof)
    assert list(mq.columns) == R.REGIMES and np.allclose(mq.sum(axis=1), 1.0)
    assert mq.iloc[0]["Goldilocks"] == pytest.approx(2 / 3)


def test_gmm_reports_own_names_and_bridges(gaps):
    G, P, m = gaps
    res = R.fit_gmm4(G, P, m)
    assert len(set(res.cluster_names)) == 4
    assert not set(res.cluster_names) & set(R.REGIMES)
    assert list(res.probs.columns) == res.cluster_names
    assert np.allclose(res.quadrant_probs.sum(axis=1), 1.0)
    assert list(res.quadrant_probs.columns) == R.REGIMES
    assert res.labels.isin(res.cluster_names).all()


def test_gmm_fit_uses_mask(gaps):
    G, P, m = gaps
    a = R.fit_gmm4(G, P, m).model.means_
    b = R.fit_gmm4(G, P, pd.Series(True, index=m.index)).model.means_
    assert not np.allclose(np.sort(a, axis=0), np.sort(b, axis=0))


def test_free_hmm_names_and_bridge(gaps):
    G, P, m = gaps
    res = R.fit_free_hmm4(G, P, m)
    assert not set(res.state_map.values()) & set(R.REGIMES)
    assert res.quadrant_profile is not None and res.quadrant_probs_filtered is not None
    assert np.allclose(res.quadrant_probs_filtered.sum(axis=1), 1.0)
