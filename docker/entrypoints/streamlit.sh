#!/usr/bin/env bash
set -euo pipefail

echo "Starting HippoView Streamlit UI..."
streamlit run streamlit/app.py \
  --server.port="${STREAMLIT_PORT:-8501}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false \
  --server.fileWatcherType=none
