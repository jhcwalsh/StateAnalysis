"""Mirror of spec §8 on the pinned vintage.

Fast path (always runs): history metrics on full-sample *filtered* labels,
walk-forward only over the GFC window. Slow path (RUN_SLOW=1): full
walk-forward from min_obs, which is what run.py reports.
"""
import os

import pytest

from regime_v2 import acceptance as A
from regime_v2.pipeline import run_pipeline
from regime_v2.walkforward import fit_hmm4_walkforward


@pytest.fixture(scope="module")
def table(vintage_path):
    res = run_pipeline(vintage_path)
    vals = {}
    vals.update(A.history_metrics(res.hmm.labels_filtered, res.quadrant, res.hmm.probs_filtered))
    vals.update(A.model_metrics(res))
    vals.update(A.truncation_metrics(vintage_path))
    vals["seed_invariance_disagreements"] = A.seed_metric(vintage_path)
    wf = fit_hmm4_walkforward(vintage_path, start="2008-09-01", end="2009-06-01")
    vals["gfc_contraction_hmm"] = A.share(wf.labels_rt, "2008-09", "2009-06", ["Contraction"])
    return A.evaluate(vals)


def test_every_threshold_passes(table):
    print(table.to_string())
    assert A.blocking_failures(table) == [], table[~table["passed"]].to_string()


def test_2007_truncation_passes_after_d3_fix(table):
    assert table.loc["trunc_2007_agreement_hmm", "passed"]


def test_known_failures_are_visible(table):
    for name in A.KNOWN_FAILURES:
        assert name in table.index and table.loc[name, "known_failure"]


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="full walk-forward, ~10 min")
def test_full_walkforward_history(vintage_path):
    res = run_pipeline(vintage_path)
    wf = fit_hmm4_walkforward(vintage_path)
    vals = A.history_metrics(wf.labels_rt, res.quadrant.reindex(wf.labels_rt.index), wf.probs_rt)
    tab = A.evaluate(vals)
    hist = [t["name"] for t in A.THRESHOLDS if t["name"] in vals]
    assert tab.loc[hist, "passed"].all(), tab.loc[hist].to_string()
