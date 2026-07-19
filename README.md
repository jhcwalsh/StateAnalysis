# StateAnalysis — Macro Regime Engine

`Macro_Regime_Analysis.ipynb` builds growth/inflation factors from FRED,
classifies four macro regimes (deterministic quadrants with causal
hysteresis; GMM as a robustness check), and constructs regime-conditional
portfolios — including a point-in-time track that quantifies look-ahead
bias. See `docs/superpowers/specs/` for design docs.

## Setup

    .venv/Scripts/python.exe -m pip install -r requirements.txt

Optional: set `FRED_API_KEY` in the environment.

## Dashboard

    .venv/Scripts/python.exe -m streamlit run app.py

Opens a local dashboard: current regime, live hysteresis-theta slider,
paper tables, and the OOS backtest. Press **Refresh** (≈5 min) on first
launch to populate `ui_data/`; after that it opens instantly from cache.
The notebook is the single source of truth — the app only re-labels
regimes when you move the theta slider (via the shared `regime_core.py`).

## Tests

    .venv/Scripts/python.exe -m pytest tests/ -v
