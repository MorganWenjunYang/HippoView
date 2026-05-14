#!/usr/bin/env bash
set -euo pipefail

echo "Starting HippoView knowledge graph job..."
python kn/create_kn.py "$@"
