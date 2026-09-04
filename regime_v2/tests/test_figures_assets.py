import os

import numpy as np
import pandas as pd

from regime_v2 import assets as A, figures as F, portfolio as P
from regime_v2.regimes import REGIMES


def _ok(p):
    return os.path.exists(p) and os.path.getsize(p) > 10_000


def _synthetic():
    idx = pd.date_range("2008-01-01", periods=180, freq="MS")
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0.004, 0.03, (180, 11)), index=idx, columns=list(A.UNIVERSE.values()))
    r.index.name = "date"
    lab = pd.DataFrame(index=idx.rename("date"))
    lab["available_at"] = lab.index + pd.DateOffset(months=1)
    lab["hmm_walkforward"] = (REGIMES * 45)
    lab["hmm_smoothed_expost"] = lab["hmm_walkforward"]
    probs = pd.get_dummies(lab["hmm_walkforward"]).astype(float).reindex(columns=REGIMES)
    return r, lab, probs


def test_asset_figures_render(tmp_path):
    r, lab, probs = _synthetic()
    table = A.regime_conditional_table(r, lab, n_boot=20)
    mu, cov = A.regime_moments(r, lab, "hmm_walkforward")
    path_df = A.mixture_path(mu, cov, probs.iloc[-60:], pd.Series(A.W6040))
    bt = P.backtest(r, lab, probs, start="2010-01-01", min_regime_obs=5)
    p = lambda n: str(tmp_path / n)
    F.fig8_regime_returns(table, p("f8.png")); assert _ok(p("f8.png"))
    F.fig9_mixture_6040(path_df, p("f9.png")); assert _ok(p("f9.png"))
    F.fig10_backtest_wealth(bt.returns, p("f10.png")); assert _ok(p("f10.png"))
    F.fig11_pit_weights(bt.weights["PIT_MaxSharpe"], p("f11.png")); assert _ok(p("f11.png"))
