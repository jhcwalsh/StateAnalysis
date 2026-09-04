import os

import numpy as np
import pandas as pd
import pytest

from regime_v2 import figures as F, regimes as R
from regime_v2.pipeline import run_pipeline
from regime_v2.walkforward import fit_hmm4_walkforward


@pytest.fixture(scope="module")
def ctx(vintage_path):
    res = run_pipeline(vintage_path)
    free = R.fit_free_hmm4(res.G, res.P, res.est_mask)
    gmm = R.fit_gmm4(res.G, res.P, res.est_mask)
    wf = fit_hmm4_walkforward(vintage_path, start="2008-06-01", end="2009-12-01")
    return res, free, gmm, wf


def _ok(path):
    return os.path.exists(path) and os.path.getsize(path) > 10_000


def test_all_figures_render(ctx, tmp_path):
    res, free, gmm, wf = ctx
    p = lambda n: str(tmp_path / n)
    F.fig1_factors_gaps(res, p("fig1.png")); assert _ok(p("fig1.png"))
    F.fig2_regime_timeline(res, wf, free, gmm, p("fig2.png")); assert _ok(p("fig2.png"))
    F.fig3_state_space(res, wf, p("fig3.png")); assert _ok(p("fig3.png"))
    F.fig4_hmm_probabilities(res, wf, p("fig4.png")); assert _ok(p("fig4.png"))
    rev = F.fig5_revisions(res, p("fig5.png")); assert _ok(p("fig5.png"))
    assert set(rev) == {"corr_first_final", "noise_to_signal_rmse", "sign_agreement", "n"}
    F.fig6_classifier_comparison(res, free, gmm, p("fig6.png")); assert _ok(p("fig6.png"))
    lags = F.fig7_walkforward(res, wf, p("fig7.png")); assert _ok(p("fig7.png"))
    assert list(lags.columns) == ["peak", "first_low_growth_rt", "lag_months", "censored"]
    row = lags.set_index("peak").loc["2007-12"]
    assert row["censored"]            # the fixture window opens 2008-06, after the peak


def test_figures_accept_missing_walkforward(ctx, tmp_path):
    res, free, gmm, _ = ctx
    F.fig2_regime_timeline(res, None, free, gmm, str(tmp_path / "a.png"))
    F.fig3_state_space(res, None, str(tmp_path / "b.png"))
    F.fig4_hmm_probabilities(res, None, str(tmp_path / "c.png"))
    assert _ok(str(tmp_path / "c.png"))


def test_primary_caption_and_challenger_palette(ctx):
    res, free, gmm, wf = ctx
    assert "real-time" in F._primary(res, wf)[2] and "NOT" not in F._primary(res, wf)[2]
    assert "NOT real-time" in F._primary(res, None)[2]
    assert F._primary(res, None)[0].equals(res.hmm.labels_filtered)   # never the smoothed labels
    from matplotlib.colors import to_hex
    regime_hex = {c.lower() for c in R.COLORS.values()}
    assert not {to_hex(c).lower() for c in F._palette(gmm.cluster_names).values()} & regime_hex


def test_nber_lags_uncensored_and_missing():
    idx = pd.date_range("2007-06-01", "2009-12-01", freq="MS")
    lab = pd.Series("Goldilocks", index=idx)
    lab.loc["2008-03-01":"2009-06-01"] = "Contraction"
    lags = F.nber_lags(lab).set_index("peak")
    assert lags.loc["2007-12", "lag_months"] == 3 and not lags.loc["2007-12", "censored"]
    assert lags.loc["2001-03", "first_low_growth_rt"] is None and np.isnan(lags.loc["2001-03", "lag_months"])
    assert not lags.loc["2001-03", "censored"]
