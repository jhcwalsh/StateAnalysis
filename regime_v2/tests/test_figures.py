import os

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
    assert list(lags.columns) == ["peak", "first_low_growth_rt", "lag_months"]
    assert (lags["peak"] == "2007-12").any()


def test_figures_accept_missing_walkforward(ctx, tmp_path):
    res, free, gmm, _ = ctx
    F.fig2_regime_timeline(res, None, free, gmm, str(tmp_path / "a.png"))
    F.fig3_state_space(res, None, str(tmp_path / "b.png"))
    F.fig4_hmm_probabilities(res, None, str(tmp_path / "c.png"))
    assert _ok(str(tmp_path / "c.png"))
