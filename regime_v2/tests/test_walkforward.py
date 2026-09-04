import numpy as np
import pandas as pd

from regime_v2 import regimes as R
from regime_v2.walkforward import fit_hmm4_walkforward
from regime_v2.pipeline import run_pipeline


def test_walkforward_gfc_window(vintage_path):
    wf = fit_hmm4_walkforward(vintage_path, start="2008-09-01", end="2009-06-01")
    assert wf.labels_rt.name == "hmm_walkforward"
    assert list(wf.probs_rt.columns) == R.REGIMES
    assert len(wf.labels_rt) == 10
    assert np.allclose(wf.probs_rt.sum(axis=1), 1.0)
    assert (wf.labels_rt == "Contraction").mean() >= 0.7
    assert set(wf.transmat_by_month) == set(wf.labels_rt.index)


def test_walkforward_month_equals_asof_pipeline(vintage_path):
    wf = fit_hmm4_walkforward(vintage_path, start="2015-12-01", end="2015-12-01")
    single = run_pipeline(vintage_path, asof="2015-12-31")
    t = pd.Timestamp("2015-12-01")
    assert np.allclose(wf.probs_rt.loc[t], single.hmm.probs_filtered.loc[t])
    assert wf.growth_gap_rt.loc[t] == single.G.loc[t]


def test_walkforward_step(vintage_path):
    wf = fit_hmm4_walkforward(vintage_path, start="2010-01-01", end="2010-12-01", step=6)
    assert list(wf.labels_rt.index.strftime("%Y-%m")) == ["2010-01", "2010-07"]
