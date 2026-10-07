#!/bin/sh
# First start: publish the pinned vintage (offline-capable) so the site is never empty.
# Later starts: redraw the documentation figures from the published run on the volume, so a
# deploy that changed only how they are drawn shows them at once (the engine figures wait for
# the next refresh). Neither step may stop the app from starting.
set -e
mkdir -p /app/var
if [ ! -f /app/var/output/summary.json ]; then
  echo "no published run on the volume; publishing the pinned vintage"
  (cd /app/regime_v2 && python run.py data/fredmd_2026-07.csv --out-dir /app/var/output --figs-dir /app/var/figs \
      --returns-cache /app/var/returns_yfinance.parquet) || echo "pinned run failed; the app will show the empty state"
else
  python scripts/redraw_doc_figures.py || echo "doc figures not regenerated; serving the ones already published"
fi
exec streamlit run app.py --server.address=0.0.0.0 --server.headless=true --server.port="${STREAMLIT_SERVER_PORT:-8505}"
