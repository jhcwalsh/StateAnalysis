FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x docker/entrypoint.sh
# Published outputs and caches live on a volume so a rebuild keeps the last good run.
ENV REGIME_OUTPUT_DIR=/app/var/output REGIME_FIGS_DIR=/app/var/figs REGIME_RETURNS_CACHE=/app/var/returns_yfinance.parquet
VOLUME ["/app/var"]
ENTRYPOINT ["docker/entrypoint.sh"]
