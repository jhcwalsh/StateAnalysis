# regime_v2 Asset Layer and Backtest (Stages 5–6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `assets.py` and `portfolio.py` to `regime_v2` so `run.py` also produces regime-conditional return tables, the probability-weighted 60/40 path, the six-strategy achievable backtest with its look-ahead decomposition, figures 8–11, and an `assets` block in `summary.json`, all joined to labels strictly through `available_at`.

**Architecture:** `assets.py` owns the universe, the return loader with a parquet cache, the single `align_to_available` join, and the descriptive statistics; `portfolio.py` owns the optimiser and a backtest loop parameterised by label column and moment window so PIT, oracle and in-sample strategies share one code path. Tests run offline on a pinned parquet fixture; only `run.py` downloads.

**Tech Stack:** Python 3.12, pandas, numpy, yfinance 1.5 (download only), pyarrow (parquet), matplotlib Agg, pytest; existing `regime_v2` modules (`pipeline`, `walkforward`, `placebo`, `figures`, `acceptance`, `nber`).

**Spec:** `docs/SPEC.md` §5 (assets.py / portfolio.py contracts), §6 Stage 5 and Stage 6, §7 figs 8–11, D11, §10 entry dated 2026-09-04.

## Global Constraints

- All paths relative to `regime_v2/`; run tests as `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest -q`. Nothing Windows-specific (§11). Commit messages `stage5: …` / `stage6: …` with the two trailer lines.
- **D11, strictly:** every join of returns to labels goes through `align_to_available`. Descriptive tables use `strict=False` (return of month r ↔ latest label with `available_at <= r`, i.e. label r−1). The backtest uses `strict=True` (decision at the end of r−1 ↔ latest label with `available_at < r`, i.e. label r−2). No `labels.loc[r]`, `shift(1)` or contemporaneous pairing anywhere in these two modules.
- Universe and display names exactly: `SPY Equity_US, VEA Equity_DevelopedExUS, EEM Equity_EM, AGG US_Aggregate_Bonds, TLT US_Long_Treasury, LQD Corp_IG, HYG Corp_HY, VNQ REITs, GLD Gold, DBC Commodities, TIP TIPS`. 60/40 = `Equity_US` 0.60, `US_Aggregate_Bonds` 0.40.
- Returns are monthly simple total returns, index month-start (`freq MS`), annualised by ×12 (mean) and ×√12 (vol).
- Backtest defaults: start `2010-01-01`, `min_regime_obs=15`, `leverage_cap=3.0`, `cost_bp` 0 and 10 reported, `rf=0.0`. Strategies exactly `PIT_MaxSharpe, PIT_MinVar, ProbWeighted_MaxSharpe, Oracle_MaxSharpe, Static_6040, EqualWeight`, plus the ex-post comparator `InSample_MaxSharpe_expost`.
- Anything computed with full-sample moments or smoothed labels carries `_expost` in its name and is never a headline number.
- No test touches the network. Regime names never as Q1–Q4. Figures 130 dpi, Agg.

## File Structure

| Path | Responsibility |
|---|---|
| `regime_v2/regime_v2/assets.py` | universe, `load_returns` + cache, `align_to_available`, conditional tables/corr/moments, mixture moments and path, growth share, Sharpe-spread placebo |
| `regime_v2/regime_v2/portfolio.py` | `mv_weights`, `backtest`, `BacktestResult`, `lookahead_decomposition`, `backtest_placebo` |
| `regime_v2/regime_v2/figures.py` | append `fig8_regime_returns`, `fig9_mixture_6040`, `fig10_backtest_wealth`, `fig11_pit_weights` |
| `regime_v2/regime_v2/acceptance.py` | `REPORT_ONLY` gains the asset rows |
| `regime_v2/run.py` | `--assets/--no-assets`, `--returns-cache`, `--refresh-returns`; asset stage after publish; summary/acceptance update |
| `regime_v2/data/returns_fixture.parquet` | tracked test fixture, 11 assets, common history |
| `regime_v2/tests/test_assets.py`, `test_portfolio.py`, `test_figures_assets.py`, `test_run.py` (extend) | tests |

---

### Task 1: Universe, return loader with cache, pinned fixture

**Files:**
- Create: `regime_v2/regime_v2/assets.py` (first part), `regime_v2/tests/test_assets.py`, `regime_v2/data/returns_fixture.parquet`
- Modify: `regime_v2/requirements.txt` (add `yfinance>=0.2.40`, `pyarrow>=16`), `.gitignore` (add `!regime_v2/data/returns_fixture.parquet` after the `regime_v2/data/*` line), `regime_v2/tests/conftest.py`

**Interfaces:**
- Produces: `UNIVERSE: dict[str, str]`, `W6040: dict[str, float]`, `load_returns(source="yfinance", tickers=None, start="2000-01-01", cache=None, refresh=False, fetch=None) -> pd.DataFrame`, `returns_to_monthly(px: pd.DataFrame) -> pd.DataFrame`; fixture `returns_path` in conftest.

- [ ] **Step 1: Write the failing tests**

`regime_v2/tests/test_assets.py`:
```python
import numpy as np
import pandas as pd
import pytest

from regime_v2 import assets as A


def test_universe_and_6040():
    assert list(A.UNIVERSE) == ["SPY", "VEA", "EEM", "AGG", "TLT", "LQD", "HYG", "VNQ", "GLD", "DBC", "TIP"]
    assert A.UNIVERSE["DBC"] == "Commodities" and A.UNIVERSE["VEA"] == "Equity_DevelopedExUS"
    assert A.W6040 == {"Equity_US": 0.6, "US_Aggregate_Bonds": 0.4}


def test_fixture_shape_and_scale(returns_path):
    r = pd.read_parquet(returns_path)
    assert list(r.columns) == list(A.UNIVERSE.values())
    assert r.index.name == "date" and r.index.freqstr in ("MS", None) and r.index[0].day == 1
    assert r.index[0] <= pd.Timestamp("2007-08-01") and len(r) >= 200
    assert r.notna().all().all() and r.abs().max().max() < 0.6


def test_load_returns_uses_cache_without_network(returns_path, tmp_path):
    called = []
    r = A.load_returns(cache=returns_path, fetch=lambda tickers, start: called.append(1))
    assert not called and list(r.columns) == list(A.UNIVERSE.values())


def test_load_returns_converts_prices_and_writes_cache(tmp_path):
    idx = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    rng = np.random.default_rng(0)
    px = pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))) for t in A.UNIVERSE}, index=idx)
    cache = tmp_path / "r.parquet"
    r = A.load_returns(cache=cache, fetch=lambda tickers, start: px[list(tickers)])
    assert list(r.columns) == list(A.UNIVERSE.values())
    assert r.index[0] == pd.Timestamp("2020-02-01") and len(r) == 11    # first month is the base
    jan_end, feb_end = px.loc["2020-01"].iloc[-1]["SPY"], px.loc["2020-02"].iloc[-1]["SPY"]
    assert np.isclose(r.loc["2020-02-01", "Equity_US"], feb_end / jan_end - 1)
    assert cache.exists() and pd.read_parquet(cache).equals(r)


def test_load_returns_reports_missing_ticker(tmp_path):
    idx = pd.date_range("2020-01-01", periods=90, freq="D")
    px = pd.DataFrame({t: 100.0 for t in list(A.UNIVERSE)[:-1]}, index=idx)
    with pytest.raises(ValueError, match="TIP"):
        A.load_returns(fetch=lambda tickers, start: px)
```

Add to `regime_v2/tests/conftest.py`:
```python
RETURNS = Path(__file__).resolve().parents[1] / "data" / "returns_fixture.parquet"


@pytest.fixture(scope="session")
def returns_path() -> str:
    assert RETURNS.exists(), f"returns fixture missing: {RETURNS}"
    return str(RETURNS)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_assets.py -q`
Expected: FAIL with `ImportError: cannot import name 'assets'`.

- [ ] **Step 3: Write the first part of assets.py**

`regime_v2/regime_v2/assets.py`:
```python
"""Stage 5 — asset layer (spec §6 Stage 5, D11).

Returns are monthly simple total returns on the 11-ETF universe. Every join
to regime labels goes through `align_to_available`, which uses the labels'
`available_at` column and never the label date itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

UNIVERSE = {
    "SPY": "Equity_US", "VEA": "Equity_DevelopedExUS", "EEM": "Equity_EM",
    "AGG": "US_Aggregate_Bonds", "TLT": "US_Long_Treasury", "LQD": "Corp_IG", "HYG": "Corp_HY",
    "VNQ": "REITs", "GLD": "Gold", "DBC": "Commodities", "TIP": "TIPS",
}
W6040 = {"Equity_US": 0.6, "US_Aggregate_Bonds": 0.4}


def _download_yfinance(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf
    px = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)
    if isinstance(px.columns, pd.MultiIndex):
        px = px["Close"]
    return px


def returns_to_monthly(px: pd.DataFrame) -> pd.DataFrame:
    """Daily adjusted closes -> monthly simple returns indexed at month start."""
    monthly = px.sort_index().resample("ME").last()
    rets = monthly.pct_change().iloc[1:]
    rets.index = rets.index.to_period("M").to_timestamp()
    rets.index.name = "date"
    return rets


def load_returns(source: str = "yfinance", tickers: dict[str, str] | None = None, start: str = "2000-01-01",
                 cache: str | Path | None = None, refresh: bool = False, fetch=None) -> pd.DataFrame:
    """Monthly total returns for the universe; cached to parquet when `cache` is given."""
    tickers = dict(tickers or UNIVERSE)
    cache = Path(cache) if cache else None
    if cache is not None and cache.exists() and not refresh:
        return pd.read_parquet(cache)[list(tickers.values())]
    if source != "yfinance":
        raise ValueError(f"unknown return source {source!r}")
    px = (fetch or _download_yfinance)(list(tickers), start)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        raise ValueError(f"tickers missing from download: {missing}")
    rets = returns_to_monthly(px[list(tickers)].rename(columns=tickers))
    rets = rets.dropna(how="any")          # common history only (VEA binds at 2007-07)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        rets.to_parquet(cache)
    return rets
```

- [ ] **Step 4: Build the pinned fixture**

Run from `regime_v2/`:
```bash
../.venv/Scripts/python.exe -c "from regime_v2.assets import load_returns; r=load_returns(cache='data/returns_fixture.parquet', refresh=True); print(r.shape, r.index[0], r.index[-1])"
```
Expected: about `(228, 11) 2007-08-01 2026-0x-01`. **If the download fails** (no network), build the fixture from the July notebook's export instead, which is the same universe:
```bash
../.venv/Scripts/python.exe -c "import pandas as pd; from regime_v2.assets import UNIVERSE; r=pd.read_parquet('../ui_data/returns.parquet').dropna(how='any')[list(UNIVERSE.values())]; r.index=r.index.to_period('M').to_timestamp(); r.index.name='date'; r.to_parquet('data/returns_fixture.parquet'); print(r.shape, r.index[0])"
```
Either way, record in the report which source built the fixture and its first month.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_assets.py -q`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add regime_v2/regime_v2/assets.py regime_v2/tests/test_assets.py regime_v2/tests/conftest.py regime_v2/data/returns_fixture.parquet regime_v2/requirements.txt .gitignore
git commit -m "stage5: universe, return loader with parquet cache, pinned returns fixture"
```

---

### Task 2: `available_at` alignment and regime-conditional statistics

**Files:**
- Modify: `regime_v2/regime_v2/assets.py` (append), `regime_v2/tests/test_assets.py` (append)

**Interfaces:**
- Consumes: `labels_frame` output of `pipeline.labels_frame` (index `date`, columns incl. `available_at`, `hmm_walkforward`, `hmm_smoothed_expost`) and `placebo.block_bootstrap`.
- Produces: `align_to_available(returns, labels, col, strict=False) -> pd.DataFrame` (index = return date; columns `label`, `label_date`, then the assets); `regime_conditional_table(returns, labels, col="hmm_walkforward", n_boot=1000, block=12, seed=0) -> pd.DataFrame` (MultiIndex (asset, regime); columns `n, ann_ret, ann_vol, sharpe, maxdd, hit, se_ann_ret, se_sharpe`); `conditional_corr(returns, labels, col="hmm_walkforward") -> dict[str, pd.DataFrame]`; `regime_moments(returns, labels, col, strict=False, min_obs=1) -> (dict[str, pd.Series], dict[str, pd.DataFrame])`; `_stats(sub: DataFrame) -> DataFrame` helper.

- [ ] **Step 1: Write the failing tests**

Append to `regime_v2/tests/test_assets.py`:
```python
def _labels(idx, seq):
    lab = pd.DataFrame(index=pd.DatetimeIndex(idx, name="date"))
    lab["available_at"] = lab.index + pd.DateOffset(months=1)
    lab["hmm_walkforward"] = seq
    lab["hmm_smoothed_expost"] = seq
    return lab


def _panel(n=60, seed=0):
    idx = pd.date_range("2015-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    r = pd.DataFrame(rng.normal(0.005, 0.03, (n, 11)), index=idx, columns=list(A.UNIVERSE.values()))
    r.index.name = "date"
    lab = _labels(idx, (["Goldilocks"] * 30 + ["Contraction"] * 30))
    return r, lab


def test_align_uses_available_at_not_date():
    r, lab = _panel()
    al = A.align_to_available(r, lab, "hmm_walkforward")
    assert list(al.columns[:2]) == ["label", "label_date"]
    # return of month r meets the label of r-1 (available on the first day of r)
    assert (al["label_date"] == al.index - pd.DateOffset(months=1)).all()
    assert al.index[0] == pd.Timestamp("2015-02-01")               # January has no available label yet
    strict = A.align_to_available(r, lab, "hmm_walkforward", strict=True)
    assert (strict["label_date"] == strict.index - pd.DateOffset(months=2)).all()
    assert strict.index[0] == pd.Timestamp("2015-03-01")


def test_align_never_sees_the_future():
    r, lab = _panel()
    lab2 = lab.copy(); lab2["available_at"] = lab2.index + pd.DateOffset(months=3)
    al = A.align_to_available(r, lab2, "hmm_walkforward")
    assert ((al.index - al["label_date"]) >= pd.Timedelta(days=85)).all()


def test_conditional_table_counts_and_ses():
    r, lab = _panel()
    r["Equity_US"] += np.where(lab["hmm_walkforward"].shift(1).reindex(r.index) == "Goldilocks", 0.02, -0.02)
    t = A.regime_conditional_table(r, lab, n_boot=50)
    assert t.index.names == ["asset", "regime"]
    assert list(t.columns) == ["n", "ann_ret", "ann_vol", "sharpe", "maxdd", "hit", "se_ann_ret", "se_sharpe"]
    n = t.xs("Equity_US", level="asset")["n"]
    assert n.sum() == 59 and set(n.index) <= {"Goldilocks", "Contraction"}
    assert t.loc[("Equity_US", "Goldilocks"), "ann_ret"] > t.loc[("Equity_US", "Contraction"), "ann_ret"]
    assert (t["se_ann_ret"] > 0).all() and (t["se_sharpe"] > 0).all() and (t["maxdd"] <= 0).all()
    assert t.equals(A.regime_conditional_table(r, lab, n_boot=50))     # seed-stable


def test_conditional_corr_and_moments():
    r, lab = _panel()
    corr = A.conditional_corr(r, lab)
    assert set(corr) == {"Goldilocks", "Contraction"}
    assert np.allclose(np.diag(corr["Goldilocks"]), 1.0) and corr["Goldilocks"].shape == (11, 11)
    mu, cov = A.regime_moments(r, lab, "hmm_walkforward")
    assert set(mu) == {"Goldilocks", "Contraction"} and cov["Contraction"].shape == (11, 11)
    al = A.align_to_available(r, lab, "hmm_walkforward")
    sub = al[al["label"] == "Goldilocks"].drop(columns=["label", "label_date"])
    assert np.allclose(mu["Goldilocks"], sub.mean() * 12) and np.allclose(cov["Goldilocks"], sub.cov() * 12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_assets.py -q`
Expected: 4 new FAIL with `AttributeError ... 'align_to_available'`.

- [ ] **Step 3: Append to assets.py**

```python
from .placebo import block_bootstrap  # noqa: E402
from .regimes import REGIMES  # noqa: E402


def align_to_available(returns: pd.DataFrame, labels: pd.DataFrame, col: str, strict: bool = False) -> pd.DataFrame:
    """Join each return month r to the latest label whose `available_at` <= r
    (`strict=True`: < r, i.e. the decision was made before r began).  D11.

    Returns a frame indexed by the return date with columns
    [label, label_date, *assets]; return months with no available label are dropped.
    """
    lab = labels[[col, "available_at"]].dropna(subset=[col]).copy()
    lab["label_date"] = lab.index
    lab = lab.sort_values("available_at").rename(columns={col: "label"})
    ret = returns.sort_index().reset_index().rename(columns={returns.index.name or "index": "date"})
    merged = pd.merge_asof(ret, lab[["available_at", "label", "label_date"]], left_on="date",
                           right_on="available_at", direction="backward", allow_exact_matches=not strict)
    merged = merged.dropna(subset=["label"]).set_index("date")
    cols = ["label", "label_date"] + list(returns.columns)
    return merged[cols]


def _stats(sub: pd.DataFrame) -> pd.DataFrame:
    """Per-asset n, ann_ret, ann_vol, sharpe, maxdd, hit for one regime's months."""
    wealth = (1 + sub).cumprod()
    maxdd = (wealth / wealth.cummax() - 1).min()
    ann_ret, ann_vol = sub.mean() * 12, sub.std() * np.sqrt(12)
    return pd.DataFrame({"n": len(sub), "ann_ret": ann_ret, "ann_vol": ann_vol,
                         "sharpe": ann_ret / ann_vol, "maxdd": maxdd, "hit": (sub > 0).mean()})


def _table_from_aligned(al: pd.DataFrame) -> pd.DataFrame:
    parts = {}
    for reg, grp in al.groupby("label"):
        parts[reg] = _stats(grp.drop(columns=["label", "label_date"]))
    out = pd.concat(parts, names=["regime", "asset"]).swaplevel().sort_index()
    return out


def regime_conditional_table(returns: pd.DataFrame, labels: pd.DataFrame, col: str = "hmm_walkforward",
                             n_boot: int = 1000, block: int = 12, seed: int = 0) -> pd.DataFrame:
    al = align_to_available(returns, labels, col)
    base = _table_from_aligned(al)
    boot = block_bootstrap(al, lambda d: _table_from_aligned(d)[["ann_ret", "sharpe"]].stack(), block=block, n=n_boot, seed=seed)
    se = boot.std(ddof=1)
    base["se_ann_ret"] = se.xs("ann_ret", level=-1).reindex(base.index)
    base["se_sharpe"] = se.xs("sharpe", level=-1).reindex(base.index)
    return base[["n", "ann_ret", "ann_vol", "sharpe", "maxdd", "hit", "se_ann_ret", "se_sharpe"]]


def conditional_corr(returns: pd.DataFrame, labels: pd.DataFrame, col: str = "hmm_walkforward") -> dict:
    al = align_to_available(returns, labels, col)
    return {reg: grp.drop(columns=["label", "label_date"]).corr() for reg, grp in al.groupby("label")}


def regime_moments(returns: pd.DataFrame, labels: pd.DataFrame, col: str, strict: bool = False,
                   min_obs: int = 1) -> tuple[dict, dict]:
    """Annualised mean vector and covariance per regime from the aligned panel."""
    al = align_to_available(returns, labels, col, strict=strict)
    mu, cov = {}, {}
    for reg, grp in al.groupby("label"):
        sub = grp.drop(columns=["label", "label_date"])
        if len(sub) >= min_obs:
            mu[reg], cov[reg] = sub.mean() * 12, sub.cov() * 12
    return mu, cov
```

Note: `block_bootstrap` builds an unnamed-index frame per draw whose `stat_fn` result is a Series indexed by (asset, regime, stat) after `.stack()`; `boot.std` is over draws.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_assets.py -q`
Expected: 9 passed. If `pd.merge_asof` complains that keys must be sorted, sort `ret` by `date` (it is) and `lab` by `available_at` (it is); if it complains about dtype, cast both to `datetime64[ns]`.

- [ ] **Step 5: Commit**

```bash
git add regime_v2/regime_v2/assets.py regime_v2/tests/test_assets.py
git commit -m "stage5: available_at alignment, regime-conditional tables, correlations and moments"
```

---

### Task 3: Mixture moments, 60/40 mixture path, growth share, Sharpe-spread placebo

**Files:**
- Modify: `regime_v2/regime_v2/assets.py` (append), `regime_v2/tests/test_assets.py` (append)

**Interfaces:**
- Produces: `mixture_moments(mu_by_regime, cov_by_regime, probs_t) -> (pd.Series, pd.DataFrame)`; `mixture_path(mu_by_regime, cov_by_regime, probs, weights) -> pd.DataFrame` (index = probs index, columns `mu, sigma`); `growth_share_6040(aligned: pd.DataFrame) -> dict` where `aligned` has columns `r6040, growth_gap, inflation_gap` (returns `r2, growth_share, inflation_share, n`); `sharpe_spread_placebo(aligned: pd.DataFrame, n=1000, seed=0) -> dict` where `aligned` has columns `label, r6040`; `portfolio_returns(returns, weights: dict) -> pd.Series`.

- [ ] **Step 1: Write the failing tests**

Append to `regime_v2/tests/test_assets.py`:
```python
def test_mixture_moments_one_hot_and_total_variance():
    r, lab = _panel()
    mu, cov = A.regime_moments(r, lab, "hmm_walkforward")
    one_hot = pd.Series({"Goldilocks": 1.0, "Contraction": 0.0})
    m, S = A.mixture_moments(mu, cov, one_hot)
    assert np.allclose(m, mu["Goldilocks"]) and np.allclose(S, cov["Goldilocks"])
    half = pd.Series({"Goldilocks": 0.5, "Contraction": 0.5})
    m2, S2 = A.mixture_moments(mu, cov, half)
    assert np.allclose(m2, 0.5 * (mu["Goldilocks"] + mu["Contraction"]))
    within = 0.5 * (cov["Goldilocks"] + cov["Contraction"])
    assert (np.diag(S2) >= np.diag(within) - 1e-12).all()      # between-regime term is PSD
    # regimes absent from the moments are dropped and probabilities renormalised
    m3, _ = A.mixture_moments(mu, cov, pd.Series({"Goldilocks": 0.25, "Stagflation": 0.75}))
    assert np.allclose(m3, mu["Goldilocks"])


def test_mixture_path_and_portfolio_returns():
    r, lab = _panel()
    mu, cov = A.regime_moments(r, lab, "hmm_walkforward")
    probs = pd.DataFrame({"Goldilocks": [1.0, 0.0], "Contraction": [0.0, 1.0]},
                         index=pd.DatetimeIndex(["2016-01-01", "2016-02-01"], name="date"))
    w = pd.Series(A.W6040).reindex(r.columns).fillna(0.0)
    path = A.mixture_path(mu, cov, probs, w)
    assert list(path.columns) == ["mu", "sigma"] and len(path) == 2
    assert np.isclose(path.iloc[0]["mu"], w @ mu["Goldilocks"])
    assert np.isclose(path.iloc[0]["sigma"], np.sqrt(w @ cov["Goldilocks"] @ w))
    pr = A.portfolio_returns(r, A.W6040)
    assert np.isclose(pr.iloc[0], 0.6 * r.iloc[0]["Equity_US"] + 0.4 * r.iloc[0]["US_Aggregate_Bonds"])


def test_growth_share_bounds_and_planted_signal():
    idx = pd.date_range("2000-01-01", periods=300, freq="MS")
    rng = np.random.default_rng(2)
    g, p = pd.Series(rng.normal(size=300), index=idx), pd.Series(rng.normal(size=300), index=idx)
    r = 0.02 * g + 0.002 * p + rng.normal(0, 0.01, 300)
    out = A.growth_share_6040(pd.DataFrame({"r6040": r, "growth_gap": g, "inflation_gap": p}))
    assert set(out) == {"r2", "growth_share", "inflation_share", "n"}
    assert 0.0 <= out["growth_share"] <= 1.0 and out["growth_share"] > 0.9
    assert np.isclose(out["growth_share"] + out["inflation_share"], 1.0) and out["n"] == 300


def test_sharpe_spread_placebo_detects_planted_premium():
    idx = pd.date_range("2000-01-01", periods=240, freq="MS")
    rng = np.random.default_rng(3)
    lab = pd.Series((["Goldilocks"] * 24 + ["Contraction"] * 24) * 5, index=idx)
    r = pd.Series(rng.normal(0.004, 0.02, 240), index=idx) + np.where(lab == "Contraction", -0.02, 0.01)
    out = A.sharpe_spread_placebo(pd.DataFrame({"label": lab, "r6040": r}), n=200, seed=0)
    assert set(out) >= {"real", "percentile", "null"} and out["percentile"] >= 95.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_assets.py -q`
Expected: 4 new FAIL with `AttributeError`.

- [ ] **Step 3: Append to assets.py**

```python
from .placebo import placebo  # noqa: E402


def mixture_moments(mu_by_regime: dict, cov_by_regime: dict, probs_t: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Law of total variance: mu = sum p_k mu_k; Sigma = sum p_k [Sigma_k + (mu_k - mu)(mu_k - mu)'].

    Regimes without moments are dropped and the remaining probabilities renormalised.
    """
    p = probs_t[[k for k in probs_t.index if k in mu_by_regime]].astype(float)
    if p.sum() <= 0:
        raise ValueError("no probability mass on regimes with estimated moments")
    p = p / p.sum()
    assets = next(iter(mu_by_regime.values())).index
    mu = sum(p[k] * mu_by_regime[k].reindex(assets) for k in p.index)
    Sigma = pd.DataFrame(0.0, index=assets, columns=assets)
    for k in p.index:
        d = (mu_by_regime[k].reindex(assets) - mu).to_numpy().reshape(-1, 1)
        Sigma += p[k] * (cov_by_regime[k].reindex(index=assets, columns=assets) + d @ d.T)
    return mu, Sigma


def mixture_path(mu_by_regime: dict, cov_by_regime: dict, probs: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Expected return and vol of a fixed-weight portfolio under the mixture, month by month."""
    rows = []
    for t, row in probs.iterrows():
        mu, S = mixture_moments(mu_by_regime, cov_by_regime, row)
        w = weights.reindex(mu.index).fillna(0.0)
        rows.append((t, float(w @ mu), float(np.sqrt(max(w @ S @ w, 0.0)))))
    out = pd.DataFrame(rows, columns=["date", "mu", "sigma"]).set_index("date")
    return out


def portfolio_returns(returns: pd.DataFrame, weights: dict) -> pd.Series:
    w = pd.Series(weights).reindex(returns.columns).fillna(0.0)
    return (returns @ w).rename("r_port")


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    return float(1.0 - resid.var() / y.var())


def growth_share_6040(aligned: pd.DataFrame) -> dict:
    """OLS of the 60/40 return on both gaps; LMG (Shapley) split of R^2 between growth and inflation."""
    df = aligned[["r6040", "growth_gap", "inflation_gap"]].dropna()
    y, g, p = df["r6040"].to_numpy(), df["growth_gap"].to_numpy(), df["inflation_gap"].to_numpy()
    r2_full, r2_g, r2_p = _r2(y, np.column_stack([g, p])), _r2(y, g[:, None]), _r2(y, p[:, None])
    lmg_g = 0.5 * (r2_g + (r2_full - r2_p))
    lmg_p = 0.5 * (r2_p + (r2_full - r2_g))
    share_g = lmg_g / r2_full if r2_full > 0 else float("nan")
    return {"r2": r2_full, "growth_share": share_g, "inflation_share": 1.0 - share_g if r2_full > 0 else float("nan"),
            "n": int(len(df))}


def sharpe_spread_placebo(aligned: pd.DataFrame, n: int = 1000, seed: int = 0) -> dict:
    """Max-minus-min annualised Sharpe of r6040 across regimes vs. run-preserving label shuffles."""
    r = aligned["r6040"]

    def spread(labels: pd.Series) -> float:
        s = r.groupby(labels).agg(["mean", "std"])
        sh = (s["mean"] * 12) / (s["std"] * np.sqrt(12))
        return float(sh.max() - sh.min())

    return placebo(aligned["label"], spread, n=n, seed=seed)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_assets.py -q`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add regime_v2/regime_v2/assets.py regime_v2/tests/test_assets.py
git commit -m "stage5: mixture moments and path, growth share of 60/40, Sharpe-spread placebo"
```

---

### Task 4: Optimiser and the achievable backtest

**Files:**
- Create: `regime_v2/regime_v2/portfolio.py`, `regime_v2/tests/test_portfolio.py`

**Interfaces:**
- Consumes: `assets.align_to_available`, `assets.mixture_moments`, `assets.W6040`; a `labels_frame` (index `date`, columns `available_at`, `hmm_walkforward`, `hmm_smoothed_expost`); `probs_rt` (index = label date, columns `REGIMES`).
- Produces: `STRATEGIES`, `EXPOST`, `mv_weights(mu, Sigma, objective="max_sharpe", rf=0.0, leverage_cap=3.0) -> (pd.Series, dict)`, `@dataclass BacktestResult(returns, weights, turnover, perf, counters, params)`, `backtest(returns, labels_frame, probs_rt, start="2010-01-01", min_regime_obs=15, cost_bp=0.0, leverage_cap=3.0, strategies=None, include_expost=True) -> BacktestResult`, `perf_table(rets, turnover) -> pd.DataFrame`.

Timing inside `backtest`, fixed by D11: for the month r whose return is earned, the current regime is the label strictly available before r (`align_to_available(..., strict=True)`, i.e. label r−2 under the one-month lag); the moment window is every return month ≤ r−1, each paired with its own strictly-available label; the probability row for `ProbWeighted_MaxSharpe` is `probs_rt` at that same label date.

- [ ] **Step 1: Write the failing tests**

`regime_v2/tests/test_portfolio.py`:
```python
import numpy as np
import pandas as pd
import pytest

from regime_v2 import assets as A, portfolio as P
from regime_v2.regimes import REGIMES


def _frame(idx, seq):
    lab = pd.DataFrame(index=pd.DatetimeIndex(idx, name="date"))
    lab["available_at"] = lab.index + pd.DateOffset(months=1)
    lab["hmm_walkforward"] = seq
    lab["hmm_smoothed_expost"] = seq
    return lab


def _planted(n=240, seed=0, premium=0.04):
    """Asset 0 pays +premium in Goldilocks and -premium otherwise, seen through the strict lag."""
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    seq = (["Goldilocks"] * 24 + ["Contraction"] * 24) * (n // 48)
    lab = _frame(idx, seq)
    r = pd.DataFrame(rng.normal(0.003, 0.02, (n, 11)), index=idx, columns=list(A.UNIVERSE.values()))
    r.index.name = "date"
    known = lab["hmm_walkforward"].shift(2).reindex(idx)          # label r-2 is what a trader knows at r
    r["Equity_US"] += np.where(known == "Goldilocks", premium, -premium)
    r["US_Long_Treasury"] -= np.where(known == "Goldilocks", premium, -premium)
    probs = pd.DataFrame(0.0, index=idx, columns=REGIMES)
    for reg in ("Goldilocks", "Contraction"):
        probs.loc[lab["hmm_walkforward"] == reg, reg] = 1.0
    return r, lab, probs


def test_mv_weights_diagonal_cases():
    mu = pd.Series([0.10, 0.05], index=["a", "b"])
    S = pd.DataFrame(np.diag([0.04, 0.01]), index=mu.index, columns=mu.index)
    w, flags = P.mv_weights(mu, S, "max_sharpe")
    assert np.isclose(w.sum(), 1.0) and np.isclose(w["a"] / w["b"], (0.10 / 0.04) / (0.05 / 0.01))
    assert not flags["negsum"] and not flags["rank_deficient"]
    w2, _ = P.mv_weights(mu, S, "min_var")
    assert np.isclose(w2["a"] / w2["b"], 0.25)
    w3, f3 = P.mv_weights(-mu, S, "max_sharpe")
    assert f3["negsum"] and np.isclose(w3.sum(), -1.0)
    big = pd.Series([0.5, -0.5], index=["a", "b"]); S2 = pd.DataFrame(np.diag([0.001, 0.001]), index=big.index, columns=big.index)
    w4, _ = P.mv_weights(big, S2, "max_sharpe", leverage_cap=3.0)
    assert np.isclose(w4.abs().sum(), 3.0)
    with pytest.raises(ValueError):
        P.mv_weights(mu, S, "sortino")


def test_backtest_shapes_and_weight_sums():
    r, lab, probs = _planted()
    bt = P.backtest(r, lab, probs, start="2005-01-01", cost_bp=0.0)
    assert list(bt.returns.columns) == P.STRATEGIES + P.EXPOST
    assert bt.returns.index[0] >= pd.Timestamp("2005-01-01") and len(bt.returns) > 150
    assert list(bt.perf.columns) == ["ann_ret", "ann_vol", "sharpe", "maxdd", "turnover"]
    for s in ("PIT_MaxSharpe", "PIT_MinVar", "Oracle_MaxSharpe", "ProbWeighted_MaxSharpe"):
        sums = bt.weights[s].sum(axis=1)
        assert np.allclose(sums.abs(), 1.0)
    assert np.allclose(bt.weights["Static_6040"].sum(axis=1), 1.0) and np.allclose(bt.weights["EqualWeight"].sum(axis=1), 1.0)
    assert bt.turnover.loc[bt.turnover.index[1:], "Static_6040"].eq(0).all()


def test_backtest_uses_only_strictly_available_labels():
    r, lab, probs = _planted()
    bt = P.backtest(r, lab, probs, start="2005-01-01")
    # the planted premium is keyed to label r-2, so PIT beats 60/40 clearly
    assert bt.perf.loc["PIT_MaxSharpe", "sharpe"] > bt.perf.loc["Static_6040", "sharpe"] + 0.5
    # oracle uses the same (synthetic) labels with the same timing -> identical path
    assert np.allclose(bt.returns["Oracle_MaxSharpe"], bt.returns["PIT_MaxSharpe"])
    # a label that only becomes available three months late must not help
    late = lab.copy(); late["available_at"] = late.index + pd.DateOffset(months=3)
    bt2 = P.backtest(r, late, probs, start="2005-01-01")
    assert bt2.perf.loc["PIT_MaxSharpe", "sharpe"] < bt.perf.loc["PIT_MaxSharpe", "sharpe"] - 0.5


def test_backtest_fallback_and_costs():
    r, lab, probs = _planted()
    bt = P.backtest(r, lab, probs, start="2005-01-01", min_regime_obs=10_000)
    assert bt.counters["pit_fallback"] > 0
    assert np.allclose(bt.returns["PIT_MaxSharpe"].iloc[1:], bt.returns["Static_6040"].iloc[1:])
    b0 = P.backtest(r, lab, probs, start="2005-01-01", cost_bp=0.0)
    b10 = P.backtest(r, lab, probs, start="2005-01-01", cost_bp=10.0)
    diff = b0.returns - b10.returns
    assert np.allclose(diff, 0.001 * b0.turnover)


def test_probweighted_reduces_to_pit_under_one_hot():
    r, lab, probs = _planted()
    bt = P.backtest(r, lab, probs, start="2005-01-01")
    assert np.allclose(bt.returns["ProbWeighted_MaxSharpe"], bt.returns["PIT_MaxSharpe"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_portfolio.py -q`
Expected: FAIL with `ImportError: cannot import name 'portfolio'`.

- [ ] **Step 3: Write portfolio.py**

`regime_v2/regime_v2/portfolio.py`:
```python
"""Stage 6 — portfolios and the achievable backtest (spec §6 Stage 6, D11).

One backtest loop serves every strategy. The label a strategy acts on is the
one strictly available before the month whose return it earns; the moments
it uses are estimated on returns up to the decision date, each paired with
its own strictly-available label. `InSample_MaxSharpe_expost` is the only
strategy allowed full-sample moments and smoothed labels, and it exists only
to measure look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .assets import W6040, align_to_available, mixture_moments

STRATEGIES = ["PIT_MaxSharpe", "PIT_MinVar", "ProbWeighted_MaxSharpe", "Oracle_MaxSharpe", "Static_6040", "EqualWeight"]
EXPOST = ["InSample_MaxSharpe_expost"]


def mv_weights(mu: pd.Series, Sigma: pd.DataFrame, objective: str = "max_sharpe", rf: float = 0.0,
               leverage_cap: float = 3.0) -> tuple[pd.Series, dict]:
    """Unconstrained mean-variance weights (shorts allowed), scaled to sum to 1.

    Guards (from the notebook's mv_opt): a raw weight sum near zero or negative
    means no fully-invested max-Sharpe portfolio exists; the sign of the risk
    premium is preserved and `negsum` is flagged (weights then sum to -1).
    `rank_deficient` flags a singular covariance (pinv is used regardless).
    """
    if objective not in ("max_sharpe", "min_var"):
        raise ValueError(f"unknown objective {objective!r}")
    Sig = Sigma.reindex(index=mu.index, columns=mu.index).to_numpy()
    flags = {"negsum": False, "rank_deficient": bool(np.linalg.matrix_rank(Sig) < len(mu))}
    vec = (mu - rf).to_numpy() if objective == "max_sharpe" else np.ones(len(mu))
    w = np.linalg.pinv(Sig) @ vec
    s = w.sum()
    if abs(s) < 1e-12:
        flags["negsum"] = True
        w = w / max(np.abs(w).sum(), 1e-12)
    elif s < 0:
        flags["negsum"] = True
        w = w / abs(s)
    else:
        w = w / s
    gross = np.abs(w).sum()
    if gross > leverage_cap:
        w = w * (leverage_cap / gross)
    return pd.Series(w, index=mu.index), flags


@dataclass
class BacktestResult:
    returns: pd.DataFrame
    weights: dict
    turnover: pd.DataFrame
    perf: pd.DataFrame
    counters: dict
    params: dict


def perf_table(rets: pd.DataFrame, turnover: pd.DataFrame) -> pd.DataFrame:
    wealth = (1 + rets).cumprod()
    ann, vol = rets.mean() * 12, rets.std() * np.sqrt(12)
    return pd.DataFrame({"ann_ret": ann, "ann_vol": vol, "sharpe": ann / vol,
                         "maxdd": (wealth / wealth.cummax() - 1).min(), "turnover": turnover.mean()})


def _moments(hist: pd.DataFrame, min_obs: int) -> tuple[dict, dict]:
    mu, cov = {}, {}
    for reg, grp in hist.groupby("label"):
        sub = grp.drop(columns=["label", "label_date"])
        if len(sub) >= min_obs:
            mu[reg], cov[reg] = sub.mean() * 12, sub.cov() * 12
    return mu, cov


def backtest(returns: pd.DataFrame, labels_frame: pd.DataFrame, probs_rt: pd.DataFrame, start: str = "2010-01-01",
             min_regime_obs: int = 15, cost_bp: float = 0.0, leverage_cap: float = 3.0,
             strategies: list[str] | None = None, include_expost: bool = True) -> BacktestResult:
    strategies = list(strategies or STRATEGIES) + (list(EXPOST) if include_expost else [])
    assets = list(returns.columns)
    pit = align_to_available(returns, labels_frame, "hmm_walkforward", strict=True)
    orc = align_to_available(returns, labels_frame, "hmm_smoothed_expost", strict=True)
    full_mu, full_cov = _moments(orc, min_regime_obs) if include_expost else ({}, {})
    w6040 = pd.Series(W6040).reindex(assets).fillna(0.0)
    weq = pd.Series(1.0 / len(assets), index=assets)
    months = [m for m in pit.index if m >= pd.Timestamp(start) and m in orc.index]
    counters = {"pit_fallback": 0, "oracle_fallback": 0, "pw_fallback": 0, "insample_fallback": 0,
                "negsum": 0, "rank_deficient": 0}
    rets = {s: {} for s in strategies}
    wts = {s: {} for s in strategies}
    turn = {s: {} for s in strategies}
    prev = {s: pd.Series(0.0, index=assets) for s in strategies}

    def regime_weights(hist, current, objective, key):
        sub = hist[hist["label"] == current].drop(columns=["label", "label_date"])
        if len(sub) < min_regime_obs:
            counters[key] += 1
            return None
        w, flags = mv_weights(sub.mean() * 12, sub.cov() * 12, objective, leverage_cap=leverage_cap)
        counters["negsum"] += int(flags["negsum"]); counters["rank_deficient"] += int(flags["rank_deficient"])
        return w

    for r in months:
        d = r - pd.DateOffset(months=1)                 # decision at the end of d
        hist_pit, hist_orc = pit.loc[:d], orc.loc[:d]   # returns <= d, each with its strictly-available label
        cur_pit, cur_orc = pit.loc[r, "label"], orc.loc[r, "label"]
        label_date = pit.loc[r, "label_date"]
        row = returns.loc[r].reindex(assets)
        for s in strategies:
            w = None
            if s == "PIT_MaxSharpe":
                w = regime_weights(hist_pit, cur_pit, "max_sharpe", "pit_fallback")
            elif s == "PIT_MinVar":
                w = regime_weights(hist_pit, cur_pit, "min_var", "pit_fallback")
            elif s == "Oracle_MaxSharpe":
                w = regime_weights(hist_orc, cur_orc, "max_sharpe", "oracle_fallback")
            elif s == "ProbWeighted_MaxSharpe":
                mu, cov = _moments(hist_pit, min_regime_obs)
                p = probs_rt.loc[label_date] if label_date in probs_rt.index else None
                if p is None or not any(k in mu for k in p.index[p > 0]):
                    counters["pw_fallback"] += 1
                else:
                    m, S = mixture_moments(mu, cov, p)
                    w, flags = mv_weights(m, S, "max_sharpe", leverage_cap=leverage_cap)
                    counters["negsum"] += int(flags["negsum"]); counters["rank_deficient"] += int(flags["rank_deficient"])
            elif s == "InSample_MaxSharpe_expost":
                if cur_orc in full_mu:
                    w, _ = mv_weights(full_mu[cur_orc], full_cov[cur_orc], "max_sharpe", leverage_cap=leverage_cap)
                else:
                    counters["insample_fallback"] += 1
            elif s == "Static_6040":
                w = w6040
            elif s == "EqualWeight":
                w = weq
            else:
                raise ValueError(f"unknown strategy {s}")
            if w is None:
                w = w6040
            w = w.reindex(assets).fillna(0.0)
            to = float((w - prev[s]).abs().sum())
            rets[s][r] = float(w @ row) - cost_bp / 1e4 * to
            wts[s][r] = w
            turn[s][r] = to
            prev[s] = w
    R = pd.DataFrame(rets); R.index.name = "date"
    T = pd.DataFrame(turn); T.index.name = "date"
    W = {s: pd.DataFrame(wts[s]).T.rename_axis("date") for s in strategies}
    return BacktestResult(returns=R, weights=W, turnover=T, perf=perf_table(R, T), counters=counters,
                          params=dict(start=str(start), min_regime_obs=min_regime_obs, cost_bp=cost_bp,
                                      leverage_cap=leverage_cap))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_portfolio.py -q`
Expected: 5 passed. If `test_backtest_uses_only_strictly_available_labels` fails on the Sharpe margins, print both Sharpes; do not change the margins, report them.

- [ ] **Step 5: Commit**

```bash
git add regime_v2/regime_v2/portfolio.py regime_v2/tests/test_portfolio.py
git commit -m "stage6: mean-variance weights and the strictly-timed six-strategy backtest"
```

---

### Task 5: Look-ahead decomposition and backtest placebo

**Files:**
- Modify: `regime_v2/regime_v2/portfolio.py` (append), `regime_v2/tests/test_portfolio.py` (append)

**Interfaces:**
- Produces: `lookahead_decomposition(perf: pd.DataFrame) -> dict` (keys `insample_sharpe, oracle_sharpe, pit_sharpe, moment_lookahead, label_lookahead, total`); `backtest_placebo(returns, labels_frame, probs_rt, n=200, seed=0, **kw) -> dict` (`real, percentile, null`).

- [ ] **Step 1: Write the failing tests**

Append to `regime_v2/tests/test_portfolio.py`:
```python
def test_lookahead_decomposition_sums():
    perf = pd.DataFrame({"sharpe": {"InSample_MaxSharpe_expost": 2.0, "Oracle_MaxSharpe": 1.2, "PIT_MaxSharpe": 0.7}})
    d = P.lookahead_decomposition(perf)
    assert d["moment_lookahead"] == pytest.approx(0.8) and d["label_lookahead"] == pytest.approx(0.5)
    assert d["total"] == pytest.approx(d["moment_lookahead"] + d["label_lookahead"]) == pytest.approx(1.3)
    assert d["insample_sharpe"] == 2.0 and d["pit_sharpe"] == 0.7


def test_backtest_placebo_ranks_real_labels_high():
    r, lab, probs = _planted()
    out = P.backtest_placebo(r, lab, probs, n=30, seed=0, start="2005-01-01")
    assert set(out) >= {"real", "percentile", "null"} and len(out["null"]) == 30
    assert out["percentile"] >= 90.0
    again = P.backtest_placebo(r, lab, probs, n=30, seed=0, start="2005-01-01")
    assert np.allclose(out["null"], again["null"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_portfolio.py -q`
Expected: 2 new FAIL with `AttributeError`.

- [ ] **Step 3: Append to portfolio.py**

```python
from .placebo import block_shuffle  # noqa: E402


def lookahead_decomposition(perf: pd.DataFrame) -> dict:
    """Sharpe by information set: in-sample (full moments, smoothed labels) -> oracle -> PIT."""
    ins, orc, pit = (float(perf.loc[k, "sharpe"]) for k in
                     ("InSample_MaxSharpe_expost", "Oracle_MaxSharpe", "PIT_MaxSharpe"))
    return {"insample_sharpe": ins, "oracle_sharpe": orc, "pit_sharpe": pit,
            "moment_lookahead": ins - orc, "label_lookahead": orc - pit, "total": ins - pit}


def backtest_placebo(returns: pd.DataFrame, labels_frame: pd.DataFrame, probs_rt: pd.DataFrame,
                     n: int = 200, seed: int = 0, **kw) -> dict:
    """PIT max-Sharpe Sharpe of the real labels vs. run-preserving shuffles of the walk-forward label."""
    rng = np.random.default_rng(seed)
    base = labels_frame.dropna(subset=["hmm_walkforward"])

    def sharpe_for(lab_series: pd.Series) -> float:
        lf = base.copy()
        lf["hmm_walkforward"] = lab_series.reindex(lf.index)
        bt = backtest(returns, lf, probs_rt, strategies=["PIT_MaxSharpe"], include_expost=False, **kw)
        return float(bt.perf.loc["PIT_MaxSharpe", "sharpe"])

    real = sharpe_for(base["hmm_walkforward"])
    null = np.array([sharpe_for(block_shuffle(base["hmm_walkforward"], rng)) for _ in range(n)])
    return {"real": real, "null": null, "percentile": float((null <= real).mean() * 100.0), "n": n}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_portfolio.py -q`
Expected: 7 passed (the placebo test runs 31 backtests; allow ~1 minute).

- [ ] **Step 5: Commit**

```bash
git add regime_v2/regime_v2/portfolio.py regime_v2/tests/test_portfolio.py
git commit -m "stage6: look-ahead decomposition and backtest placebo"
```

---

### Task 6: Figures 8–11

**Files:**
- Modify: `regime_v2/regime_v2/figures.py` (append)
- Create: `regime_v2/tests/test_figures_assets.py`

**Interfaces:**
- Produces: `fig8_regime_returns(table, path)`, `fig9_mixture_6040(path_df, path)`, `fig10_backtest_wealth(bt_returns, path)`, `fig11_pit_weights(weights, path)`. Inputs: `table` from `regime_conditional_table`; `path_df` from `mixture_path` (columns `mu, sigma`); `bt_returns` = `BacktestResult.returns`; `weights` = `BacktestResult.weights["PIT_MaxSharpe"]`.

- [ ] **Step 1: Write the failing tests**

`regime_v2/tests/test_figures_assets.py`:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_figures_assets.py -q`
Expected: FAIL with `AttributeError ... 'fig8_regime_returns'`.

- [ ] **Step 3: Append to figures.py**

```python
def fig8_regime_returns(table: pd.DataFrame, path: str) -> None:
    """Annualised return per asset and regime with 1.96 x bootstrap SE bars."""
    assets = list(table.index.get_level_values("asset").unique())
    regs = [r for r in REGIMES if r in table.index.get_level_values("regime")]
    fig, ax = plt.subplots(figsize=(13, 5))
    width = 0.8 / max(len(regs), 1)
    x = np.arange(len(assets))
    for i, reg in enumerate(regs):
        sub = table.xs(reg, level="regime").reindex(assets)
        ax.bar(x + i * width, sub["ann_ret"], width, yerr=1.96 * sub["se_ann_ret"], color=COLORS[reg],
               label=f"{reg} (n={int(sub['n'].dropna().iloc[0]) if sub['n'].notna().any() else 0})", capsize=2, lw=0)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x + width * (len(regs) - 1) / 2); ax.set_xticklabels(assets, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("annualised return"); ax.set_title("Regime-conditional returns (labels via available_at), 95% block-bootstrap bars")
    ax.legend(fontsize=8, ncol=len(regs))
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def fig9_mixture_6040(path_df: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(path_df.index, path_df["mu"], color="#1c7ed6", lw=1.2); axes[0].set_ylabel("expected return (ann.)")
    axes[1].plot(path_df.index, path_df["sigma"], color="#e8590c", lw=1.2); axes[1].set_ylabel("volatility (ann.)")
    for ax in axes:
        _shade(ax); ax.grid(True, alpha=0.3)
    axes[0].set_title("Probability-weighted 60/40 moments from walk-forward regime probabilities")
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def fig10_backtest_wealth(bt_returns: pd.DataFrame, path: str) -> None:
    wealth = (1 + bt_returns).cumprod()
    dd = wealth / wealth.cummax() - 1
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for col in wealth.columns:
        style = dict(lw=1.0, ls="--", alpha=0.7) if col.endswith("_expost") else dict(lw=2.0 if col.startswith("PIT") else 1.2)
        axes[0].plot(wealth.index, wealth[col], label=col, **style)
        axes[1].plot(dd.index, dd[col], lw=0.8)
    axes[0].set_yscale("log"); axes[0].set_ylabel("wealth (log)"); axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_title("Achievable backtest: decision at month end on strictly available labels; dashed = ex-post comparator")
    axes[1].set_ylabel("drawdown")
    for ax in axes:
        _shade(ax); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def fig11_pit_weights(weights: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in weights.columns:
        ax.plot(weights.index, weights[col], lw=1.0, label=col)
    ax.axhline(0, color="grey", lw=0.8); _shade(ax); ax.grid(True, alpha=0.3)
    ax.set_title("PIT max-Sharpe weights by month (long-short, gross cap applied)"); ax.legend(fontsize=7, ncol=4)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_figures_assets.py -q`
Expected: 1 passed, no matplotlib warnings.

- [ ] **Step 5: Commit**

```bash
git add regime_v2/regime_v2/figures.py regime_v2/tests/test_figures_assets.py
git commit -m "stage5+6: figures 8-11 for conditional returns, mixture path, backtest and weights"
```

---

### Task 7: `run.py` asset stage, acceptance report rows, docs sync, end-to-end run

**Files:**
- Modify: `regime_v2/run.py`, `regime_v2/regime_v2/acceptance.py` (`REPORT_ONLY`), `regime_v2/tests/test_run.py` (append), `regime_v2/README.md`, `docs/SPEC.md` (§10 entry + §12 published surface), `.gitignore` (no change needed: `regime_v2/data/*` already ignores the yfinance cache)

**Interfaces:**
- Produces in `run.py`: CLI flags `--no-assets`, `--returns-cache PATH` (default `HERE/data/returns_yfinance.parquet`), `--refresh-returns`, `--placebo-n N` (default 200), `--skip-placebo`; `run_assets(labels_df, probs_rt, res, out_dir, figs_dir, returns_cache, refresh, placebo_n, skip_placebo) -> dict` returning the `summary.json["assets"]` block; asset report-only metric names `pit_sharpe, oracle_sharpe, insample_sharpe, label_lookahead, moment_lookahead, growth_share_6040, backtest_placebo_pct`.

- [ ] **Step 1: Write the failing tests**

Append to `regime_v2/tests/test_run.py`:
```python
def test_assets_stage_offline(vintage_path, returns_path, tmp_path):
    out, figs = tmp_path / "out", tmp_path / "figs"
    rc = runmod.main([vintage_path, "--no-walkforward", "--skip-robustness", "--skip-expanding", "--skip-placebo",
                      "--returns-cache", returns_path, "--out-dir", str(out), "--figs-dir", str(figs),
                      "--data-sheet", str(tmp_path / "README.md")])
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    a = s["assets"]
    assert a["skipped"] is None and a["window"]["n_months"] >= 200
    assert set(a["n_per_regime"]) <= set(R.REGIMES) and sum(a["n_per_regime"].values()) == a["window"]["n_months"] - 0
    assert 0.0 <= a["growth_share_6040"]["growth_share"] <= 1.0
    assert set(a["backtest"]) == {"cost_bp_0", "cost_bp_10"} and "PIT_MaxSharpe" in a["backtest"]["cost_bp_0"]["perf"]
    assert set(a["lookahead"]) >= {"moment_lookahead", "label_lookahead", "total"}
    assert a["backtest_placebo"] is None                      # skipped in this run
    for f in ["regime_returns.csv", "regime_corr_Goldilocks.csv", "backtest_returns.csv", "portfolio_weights.csv"]:
        assert (out / f).exists(), f
    for n in ["fig8_regime_returns", "fig9_mixture_6040", "fig10_backtest_wealth", "fig11_pit_weights"]:
        assert (figs / f"{n}.png").exists(), n
    acc = pd.read_csv(out / "acceptance.csv", index_col=0)
    for n in ["pit_sharpe", "oracle_sharpe", "insample_sharpe", "label_lookahead", "moment_lookahead", "growth_share_6040"]:
        assert n in acc.index and acc.loc[n, "op"] == "report"
    assert s["acceptance_all_passed"] is True


def test_assets_stage_skips_cleanly_on_failure(vintage_path, tmp_path, monkeypatch):
    out, figs = tmp_path / "out", tmp_path / "figs"
    def boom(**kw): raise OSError("no network")
    monkeypatch.setattr(runmod.assets, "load_returns", boom)
    rc = runmod.main([vintage_path, "--no-walkforward", "--skip-robustness", "--skip-expanding", "--skip-placebo",
                      "--returns-cache", str(tmp_path / "missing.parquet"), "--out-dir", str(out), "--figs-dir", str(figs),
                      "--data-sheet", str(tmp_path / "README.md")])
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    assert "no network" in s["assets"]["skipped"]
    assert not (out / "backtest_returns.csv").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_run.py -q`
Expected: 2 new FAIL (`unrecognized arguments: --skip-placebo …`).

- [ ] **Step 3: Extend acceptance.py**

Replace `REPORT_ONLY = ["filtered_vs_smoothed_agreement"]` with:
```python
REPORT_ONLY = ["filtered_vs_smoothed_agreement",
               # Stage 5-6 asset-layer diagnostics: reported, never thresholds (spec §6 Stage 6)
               "pit_sharpe", "oracle_sharpe", "insample_sharpe", "label_lookahead", "moment_lookahead",
               "growth_share_6040", "backtest_placebo_pct"]
```
`evaluate` already renders REPORT_ONLY rows with `op="report"`; a missing value renders NaN. Update `test_thresholds_cover_spec_table` only if it asserts on `REPORT_ONLY` (it does not).

- [ ] **Step 4: Extend run.py**

Add imports: `from regime_v2 import assets, portfolio` (module imports, so tests can monkeypatch `runmod.assets.load_returns`).

Add the function (place it before `main`):
```python
def run_assets(labels_df: pd.DataFrame, probs_rt: pd.DataFrame, res, out_dir: Path, figs_dir: Path,
               returns_cache: Path, refresh: bool, placebo_n: int, skip_placebo: bool) -> dict:
    """Stage 5-6 after the engine has published. Returns the summary['assets'] block."""
    try:
        rets = assets.load_returns(cache=returns_cache, refresh=refresh)
    except Exception as e:  # network or cache failure must not change the engine's exit code
        return {"skipped": f"{type(e).__name__}: {e}"}
    col = "hmm_walkforward" if labels_df["hmm_walkforward"].notna().any() else "hmm_filtered"
    lab = labels_df.copy()
    lab["hmm_walkforward"] = lab[col]                      # portfolio.py keys on this column name
    table = assets.regime_conditional_table(rets, lab, "hmm_walkforward")
    table.to_csv(out_dir / "regime_returns.csv")
    corr = assets.conditional_corr(rets, lab, "hmm_walkforward")
    for reg, c in corr.items():
        c.to_csv(out_dir / f"regime_corr_{reg}.csv")
    mu, cov = assets.regime_moments(rets, lab, "hmm_walkforward")
    aligned = assets.align_to_available(rets, lab, "hmm_walkforward")
    r6040 = assets.portfolio_returns(aligned.drop(columns=["label", "label_date"]), assets.W6040)
    gaps = pd.concat([assets.align_to_available(rets, lab, "growth_gap")["label"].rename("growth_gap"),
                      assets.align_to_available(rets, lab, "inflation_gap")["label"].rename("inflation_gap")], axis=1)
    growth = assets.growth_share_6040(pd.concat([r6040.rename("r6040"), gaps], axis=1))
    spread = assets.sharpe_spread_placebo(pd.DataFrame({"label": aligned["label"], "r6040": r6040}), n=1000)
    probs_avail = probs_rt.reindex(lab.index).dropna(how="all")
    path_df = assets.mixture_path(mu, cov, probs_avail.loc[probs_avail.index >= rets.index[0]], pd.Series(assets.W6040))
    bts = {f"cost_bp_{c}": portfolio.backtest(rets, lab, probs_rt, cost_bp=float(c)) for c in (0, 10)}
    bt0 = bts["cost_bp_0"]
    bt0.returns.to_csv(out_dir / "backtest_returns.csv")
    pd.concat({s: bt0.weights[s] for s in ("PIT_MaxSharpe", "ProbWeighted_MaxSharpe")}, axis=1).to_csv(out_dir / "portfolio_weights.csv")
    look = portfolio.lookahead_decomposition(bt0.perf)
    plc = None if skip_placebo else portfolio.backtest_placebo(rets, lab, probs_rt, n=placebo_n)
    figures.fig8_regime_returns(table, str(figs_dir / "fig8_regime_returns.png"))
    figures.fig9_mixture_6040(path_df, str(figs_dir / "fig9_mixture_6040.png"))
    figures.fig10_backtest_wealth(bt0.returns, str(figs_dir / "fig10_backtest_wealth.png"))
    figures.fig11_pit_weights(bt0.weights["PIT_MaxSharpe"], str(figs_dir / "fig11_pit_weights.png"))
    n_per = table.xs(rets.columns[0], level="asset")["n"].astype(int).to_dict()
    return {
        "skipped": None, "label_column": col, "universe": assets.UNIVERSE,
        "window": {"start": str(aligned.index[0].date()), "end": str(aligned.index[-1].date()), "n_months": int(len(aligned))},
        "n_per_regime": n_per,
        "growth_share_6040": growth,
        "sharpe_spread_placebo": {"real": spread["real"], "percentile": spread["percentile"]},
        "backtest": {k: {"perf": v.perf.round(4).to_dict(orient="index"), "counters": v.counters, "params": v.params}
                     for k, v in bts.items()},
        "lookahead": look,
        "backtest_placebo": None if plc is None else {"real": plc["real"], "percentile": plc["percentile"], "n": plc["n"]},
    }
```

In `main`: add the arguments
```python
    ap.add_argument("--no-assets", action="store_true")
    ap.add_argument("--returns-cache", default=str(HERE / "data" / "returns_yfinance.parquet"))
    ap.add_argument("--refresh-returns", action="store_true")
    ap.add_argument("--placebo-n", type=int, default=200)
    ap.add_argument("--skip-placebo", action="store_true")
```
Keep a reference to the labels frame you write (`labels_df = labels_frame(res, extra=parts)`; write it from that variable). After `publish(staging, out_dir, figs_dir)` and its print, add:
```python
    if not a.no_assets:
        probs_src = wf.probs_rt if wf is not None else res.hmm.probs_filtered
        block = run_assets(labels_df, probs_src, res, out_dir, figs_dir, Path(a.returns_cache), a.refresh_returns,
                           a.placebo_n, a.skip_placebo)
        summary["assets"] = block
        if block.get("skipped") is None:
            vals.update({"pit_sharpe": block["lookahead"]["pit_sharpe"], "oracle_sharpe": block["lookahead"]["oracle_sharpe"],
                         "insample_sharpe": block["lookahead"]["insample_sharpe"],
                         "label_lookahead": block["lookahead"]["label_lookahead"],
                         "moment_lookahead": block["lookahead"]["moment_lookahead"],
                         "growth_share_6040": block["growth_share_6040"]["growth_share"],
                         "backtest_placebo_pct": (block["backtest_placebo"] or {}).get("percentile", float("nan"))})
            table = acceptance.evaluate(vals)
            table.to_csv(out_dir / "acceptance.csv")
            summary["acceptance_tests"] = table.reset_index().to_dict(orient="records")
        else:
            print(f"asset stage skipped: {block['skipped']}", file=sys.stderr)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print("asset stage " + ("skipped" if block.get("skipped") else "published"))
```
Because `run_assets` writes into `out_dir` after publish, its files are additive; the engine's publish gate is unchanged.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_run.py tests/test_acceptance_unit.py -q`
Expected: all pass (the offline asset test runs two backtests plus a 1000-draw Sharpe-spread placebo; allow ~1 minute).

- [ ] **Step 6: Full suite and the real run**

```bash
cd regime_v2 && ../.venv/Scripts/python.exe -m pytest -q
cd regime_v2 && ../.venv/Scripts/python.exe run.py data/fredmd_2026-07.csv
```
The real run downloads the ETF returns into `data/returns_yfinance.parquet` on first use (network). If the download fails on the dev machine, run once with `--returns-cache data/returns_fixture.parquet` and say so in the report. Paste the `assets` block's `lookahead`, `growth_share_6040`, `sharpe_spread_placebo` and both `perf` tables into the report and the commit body.

- [ ] **Step 7: Docs**

`regime_v2/README.md`: add the asset-stage flags and outputs (one paragraph). `docs/SPEC.md`: §12 "Published surface" adds the four asset figures and the four CSVs; §10 entry dated with the day of the run: "Stages 5–6 implemented per docs/superpowers/plans/2026-09-04-regime-v2-assets.md; measured: PIT Sharpe …, oracle …, in-sample …, growth share …, placebo percentile …" with the real numbers.

- [ ] **Step 8: Commit**

```bash
git add regime_v2/run.py regime_v2/regime_v2/acceptance.py regime_v2/tests/test_run.py regime_v2/README.md docs/SPEC.md
git commit -m "stage6: run.py asset stage, report-only asset metrics, docs sync"
```

---

## Self-review against the spec

- **§6 Stage 5:** universe and names (T1), loader + cache + fixture with no network in tests (T1), strict `available_at` timing (T2, T4), conditional table/corr/moments (T2), mixture moments and path (T3), growth share (T3), Sharpe-spread placebo (T3), outputs and `summary.json["assets"]` (T7), figs 8–9 (T6), the listed tests (T1–T3).
- **§6 Stage 6:** `mv_weights` with guards and cap (T4), the six strategies with fallback and costs (T4), decomposition as report-only rows (T5, T7), backtest placebo 200 (T5, T7), outputs and figs 10–11 (T6, T7), `--assets` stage that never changes the engine exit code (T7), the listed tests (T4, T5, T7).
- **D11:** `align_to_available` is the only join; `strict=False` for descriptive, `strict=True` in the backtest; `test_align_uses_available_at_not_date` and `test_backtest_uses_only_strictly_available_labels` pin both.
- **Type consistency:** `align_to_available` returns `[label, label_date, *assets]` and every consumer drops those two columns; `probs_rt` is indexed by label date and looked up at `label_date` (T4); `perf` columns `ann_ret, ann_vol, sharpe, maxdd, turnover` used by T5 and T7; `BacktestResult.weights` is a dict of frames (T6 fig 11, T7 CSV).
- **Known risks:** `pd.merge_asof` dtype strictness (T2 note); the placebo test margins on synthetic data (T3, T5 say report, do not loosen); yfinance download on the dev machine (T1 and T7 give the offline fallback).
