import numpy as np
import pandas as pd
import pytest

from regime_v2 import data, factors


@pytest.fixture(scope="module")
def blocks(vintage_path):
    return data.build_blocks(vintage_path)


def test_growth_factor_tracks_indpro(blocks):
    f, load = factors.pca_factor_em(blocks["growth"], "INDPRO", blocks["estimation_mask"])
    z = blocks["growth"]["INDPRO"]
    c = pd.concat([f["factor"], z], axis=1).dropna().corr().iloc[0, 1]
    assert c > 0.7
    assert load["INDPRO"] > 0
    assert list(f.columns) == ["factor", "diffusion", "n_series"]
    assert set(load.index) == set(blocks["growth"].columns)


def test_inflation_factor_tracks_cpi(blocks):
    f, _ = factors.pca_factor_em(blocks["inflation"], "CPIAUCSL", blocks["estimation_mask"])
    z = blocks["inflation"]["CPIAUCSL"]
    assert pd.concat([f["factor"], z], axis=1).dropna().corr().iloc[0, 1] > 0.7


def test_factor_standardised_on_mask_only(blocks):
    f, _ = factors.pca_factor_em(blocks["growth"], "INDPRO", blocks["estimation_mask"])
    m = blocks["estimation_mask"].reindex(f.index).fillna(False)
    assert abs(f.loc[m, "factor"].mean()) < 1e-8
    assert abs(f.loc[m, "factor"].std() - 1.0) < 1e-8
    # masked months are still scored (not NaN)
    assert f.loc["2020-04-01", "factor"] < -3


def test_em_converges_before_iteration_cap(blocks):
    calls = []
    f, _ = factors.pca_factor_em(blocks["growth"], "INDPRO", blocks["estimation_mask"],
                                 n_iter=50, tol=1e-6, _trace=calls)
    assert 1 < len(calls) < 50, "sign-invariant check must converge, not run to cap"


def test_sign_anchor_stable_under_truncation(vintage_path):
    full = data.build_blocks(vintage_path)
    cut = data.build_blocks(vintage_path, asof="2007-12-31")
    f_full, _ = factors.pca_factor_em(full["growth"], "INDPRO", full["estimation_mask"])
    f_cut, _ = factors.pca_factor_em(cut["growth"], "INDPRO", cut["estimation_mask"])
    ov = f_cut.index
    assert np.corrcoef(f_full.loc[ov, "factor"], f_cut["factor"])[0, 1] > 0.99


def test_expanding_returns_endpoint_and_loadings(blocks):
    g = blocks["growth"].loc[:"1985-12-01"]
    m = blocks["estimation_mask"].loc[:"1985-12-01"]
    f, loads = factors.pca_factor_expanding(g, "INDPRO", m, min_obs=120)
    assert f["factor"].first_valid_index() == g.index[119]
    assert loads.shape[1] == g.shape[1]
    assert (loads["INDPRO"].dropna() > 0).all()
