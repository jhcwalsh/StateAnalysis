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


def test_factor_scaled_on_mask_only(blocks):
    f, _ = factors.pca_factor_em(blocks["growth"], "INDPRO", blocks["estimation_mask"])
    m = blocks["estimation_mask"].reindex(f.index).fillna(False)
    assert abs(f.loc[m, "factor"].std() - 1.0) < 1e-8
    # masked months are still scored (not NaN); the factor is scaled, not demeaned
    assert f.loc["2020-04-01", "factor"] - f.loc[m, "factor"].mean() < -3


def test_inflation_gap_has_no_sample_dependent_drift(vintage_path):
    from regime_v2 import trend
    full = data.build_blocks(vintage_path)
    cut = data.build_blocks(vintage_path, asof="2007-12-31")
    pf, _ = factors.pca_factor_em(full["inflation"], "CPIAUCSL", full["estimation_mask"])
    pc, _ = factors.pca_factor_em(cut["inflation"], "CPIAUCSL", cut["estimation_mask"])
    Pf = trend.make_gap(pf["diffusion"], "smoothed_trailing", full["estimation_mask"])["gap"]
    Pc = trend.make_gap(pc["diffusion"], "smoothed_trailing", cut["estimation_mask"])["gap"]
    d = (Pf.reindex(Pc.index) - Pc).dropna()
    assert d.abs().mean() < 0.05 and d.abs().max() < 0.2


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


def test_zero_variance_column_raises():
    idx = pd.date_range("2000-01-01", periods=200, freq="MS")
    rng = np.random.default_rng(0)
    block = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "flat": 1.0}, index=idx)
    with pytest.raises(ValueError, match="flat"):
        factors.pca_factor_em(block, "a", pd.Series(True, index=idx))


def test_row_with_no_informative_cells_scores_nan():
    idx = pd.date_range("2000-01-01", periods=200, freq="MS")
    rng = np.random.default_rng(1)
    common = rng.normal(size=200)
    block = pd.DataFrame({"a": common + 0.1 * rng.normal(size=200),
                          "b": common + 0.1 * rng.normal(size=200),
                          "noise": rng.normal(size=200)}, index=idx)
    block.loc[idx[50], ["a", "b"]] = np.nan          # only the near-zero-loading column observed
    f, load = factors.pca_factor_em(block, "a", pd.Series(True, index=idx))
    assert abs(load["noise"]) < 0.2
    assert np.isfinite(f["factor"].drop(idx[50])).all()
    assert f.loc[idx[50], "n_series"] == 1
