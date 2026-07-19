import json
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ui_data_dir(tmp_path):
    """Minimal ui_data/ satisfying the schema-1 contract."""
    d = tmp_path / "ui_data"
    d.mkdir()
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(42)

    gaps = pd.DataFrame({
        "g_gap": rng.normal(0, 1, 24), "p_gap": rng.normal(0, 1, 24),
        "pit_g": rng.normal(0, 1, 24), "pit_p": rng.normal(0, 1, 24),
        "growth_factor": rng.normal(0, 1, 24),
        "inflation_factor": rng.normal(0, 1, 24)}, index=idx)
    labels = pd.DataFrame({
        "quad": ["Q2_HighG_LowPi"] * 12 + ["Q3_HighG_HighPi"] * 12,
        "pit_quad": ["Q2_HighG_LowPi"] * 24,
        "gmm_cluster": [0] * 24}, index=idx)
    probs = pd.DataFrame(
        {k: [0.25] * 24 for k in
         ["Q1_Goldilocks", "Q2_Overheating", "Q3_Stagflation", "Q4_Recession"]},
        index=idx)
    rets = pd.DataFrame({"Equity_US": rng.normal(0.005, 0.04, 24),
                         "US_Aggregate_Bonds": rng.normal(0.002, 0.01, 24)}, index=idx)
    bt = pd.DataFrame({s: rng.normal(0.004, 0.02, 24) for s in
                       ["PIT_MaxSharpe", "PIT_MinVar", "Oracle_MaxSharpe",
                        "Static_6040", "EqualWeight"]}, index=idx)

    gaps.to_parquet(d / "gaps.parquet"); labels.to_parquet(d / "labels.parquet")
    probs.to_parquet(d / "probs.parquet"); rets.to_parquet(d / "returns.parquet")
    bt.to_parquet(d / "backtest.parquet")
    pd.DataFrame({"Regime": ["Prob-Weighted (2020-12)"], "Objective": ["Max Sharpe"],
                  "Port_Return": [0.05], "Port_Vol": [0.04], "Port_Sharpe": [1.25]}
                 ).to_excel(d / "tables.xlsx", sheet_name="Table3_Portfolios", index=False)

    meta = {"schema_version": 1, "run_timestamp": "2026-07-19T12:00:00",
            "theta": 0.5, "covid_exclude": ["2020-03-01", "2020-12-01"],
            "publication_lag_m": 1,
            "sample": {"start": "2020-01-01", "end": "2021-12-01"},
            "current": {"month": "2021-12", "quadrant": "Q3_HighG_HighPi",
                        "growth_gap": 0.7, "inflation_gap": 0.6,
                        "probs": {"Q1_Goldilocks": 0.28, "Q2_Overheating": 0.32,
                                  "Q3_Stagflation": 0.28, "Q4_Recession": 0.12}},
            "pit": {"agreement": 0.65, "n_common": 24,
                    "start": "2020-01", "end": "2021-12"},
            "backtest": {"start": "2020-01", "end": "2021-12", "n_months": 24,
                         "perf": {s: {"Ann_Return": 0.05, "Ann_Vol": 0.07,
                                      "Sharpe": 0.7, "Max_Drawdown": -0.1}
                                  for s in bt.columns}}}
    (d / "meta.json").write_text(json.dumps(meta))
    return str(d)
