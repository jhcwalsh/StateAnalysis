import numpy as np
import pandas as pd

from regime_v2 import acceptance as A, regimes as R


def test_thresholds_cover_spec_table():
    names = {t["name"] for t in A.THRESHOLDS}
    assert names == {
        "gfc_contraction_hmm", "gfc_contraction_quadrants", "covid_contraction_hmm",
        "inflation_2021_22_high_hmm", "nber_low_growth_hmm", "non_nber_contraction_hmm",
        "share_max_prob_gt_095", "emission_only_agreement", "means_unmoved_maxabs",
        "min_transition_prob", "trend_step_realtime_maxabs", "trunc_2015_agreement_hmm",
        "trunc_2007_agreement_hmm", "trunc_2015_agreement_quad", "trunc_2007_agreement_quad",
        "seed_invariance_disagreements"}
    assert all(t["rationale"] for t in A.THRESHOLDS)


def test_share_and_history_metrics():
    idx = pd.date_range("2008-01-01", "2009-12-01", freq="MS")
    lab = pd.Series("Goldilocks", index=idx)
    lab.loc["2008-09-01":"2009-06-01"] = "Contraction"
    assert A.share(lab, "2008-09", "2009-06", ["Contraction"]) == 1.0
    probs = pd.DataFrame(0.25, index=idx, columns=R.REGIMES)
    m = A.history_metrics(lab, lab, probs)
    assert m["gfc_contraction_hmm"] == 1.0
    assert m["share_max_prob_gt_095"] == 0.0
    assert 0.0 <= m["non_nber_contraction_hmm"] <= 1.0


def test_evaluate_applies_ops_and_reports_unknown():
    vals = {t["name"]: (1.0 if t["op"] == ">=" else 0.0) for t in A.THRESHOLDS}
    tab = A.evaluate(vals)
    assert A.all_passed(tab)
    vals["gfc_contraction_hmm"] = 0.5
    tab = A.evaluate(vals)
    assert not tab.loc["gfc_contraction_hmm", "passed"] and not A.all_passed(tab)
    tab2 = A.evaluate({})
    assert tab2["value"].isna().all() and not A.all_passed(tab2)
