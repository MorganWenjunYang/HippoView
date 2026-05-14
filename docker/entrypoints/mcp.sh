#!/usr/bin/env bash
set -euo pipefail

echo "Starting HippoView MCP server..."
python RAG/rag_mcp_server.py
