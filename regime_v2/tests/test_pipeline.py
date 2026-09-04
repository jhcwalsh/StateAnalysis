import numpy as np
import pandas as pd
import pytest

from regime_v2 import data, factors, trend, regimes as R, nber
from regime_v2.pipeline import run_pipeline, labels_frame, DEFAULTS


def test_nber_flag_marks_gfc_and_covid():
    idx = pd.date_range("2007-01-01", "2021-12-01", freq="MS")
    f = nber.nber_flag(idx)
    assert f.loc["2008-06-01"] and f.loc["2009-06-01"] and not f.loc["2009-07-01"]
    assert f.loc["2020-03-01"] and not f.loc["2020-05-01"]
    assert f.dtype == bool and len(nber.NBER) == 9


@pytest.fixture(scope="module")
def res(vintage_path):
    return run_pipeline(vintage_path)


def test_pipeline_matches_manual_composition(vintage_path, res):
    b = data.build_blocks(vintage_path)
    m = b["estimation_mask"]
    gf, _ = factors.pca_factor_em(b["growth"], "INDPRO", m)
    G = trend.make_gap(gf["factor"], "smoothed_trailing", m)["gap"]
    assert np.allclose(res.G.dropna(), G.dropna())
    assert res.params == DEFAULTS


def test_pipeline_outputs_aligned(res):
    assert res.quadrant.index.equals(res.hmm.labels_filtered.index)
    assert res.quadrant.name == "quadrant" and res.quadrant0.name == "quadrant"
    assert res.est_mask.dtype == bool
    assert res.growth_loadings["INDPRO"] > 0


def test_asof_pipeline_ends_at_cut(vintage_path):
    cut = run_pipeline(vintage_path, asof="2015-12-31")
    assert cut.hmm.labels_filtered.index[-1] == pd.Timestamp("2015-12-01")


def test_labels_frame_has_publication_lag(res):
    df = labels_frame(res)
    assert df["available_at"].iloc[0] == df.index[0] + pd.DateOffset(months=1)
    for c in ["growth_gap", "inflation_gap", "quadrant", "quadrant_theta",
              "hmm_filtered", "hmm_smoothed_expost"]:
        assert c in df.columns
    assert df["quadrant_theta"].iloc[0] == DEFAULTS["theta"]
