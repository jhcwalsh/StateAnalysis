#!/bin/sh
# First start: publish the pinned vintage (offline-capable) so the site is never empty.
set -e
mkdir -p /app/var
if [ ! -f /app/var/output/summary.json ]; then
  echo "no published run on the volume; publishing the pinned vintage"
  (cd /app/regime_v2 && python run.py data/fredmd_2026-07.csv --out-dir /app/var/output --figs-dir /app/var/figs \
      --returns-cache /app/var/returns_yfinance.parquet) || echo "pinned run failed; the app will show the empty state"
fi
exec streamlit run app.py --server.address=0.0.0.0 --server.headless=true --server.port="${STREAMLIT_SERVER_PORT:-8505}"
