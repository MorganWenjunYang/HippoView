#!/bin/bash

# HippoView Streamlit Frontend Launcher
# This script starts the Streamlit web interface for the Clinical Trials AI Assistant

echo "* Starting HippoView - Clinical Trials AI Assistant"
echo "=================================================="
echo ""

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "- ERROR: Streamlit is not installed. Installing..."
    pip install streamlit==1.41.1
fi

# Set default environment variables if not set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check for required environment variables
echo "* Checking environment..."

# MongoDB connection (optional check)
if [ -z "$MONGODB_URI" ]; then
    echo "- WARNING: MONGODB_URI not set. Using default: mongodb://localhost:27017"
    export MONGODB_URI="mongodb://localhost:27017"
fi

# Elasticsearch connection (optional check)
if [ -z "$ELASTICSEARCH_URL" ]; then
    echo "- WARNING: ELASTICSEARCH_URL not set. Using default: http://localhost:9200"
    export ELASTICSEARCH_URL="http://localhost:9200"
fi

echo "* Environment setup complete"
echo ""

# Start Streamlit
echo "* Launching Streamlit app..."
echo "* The app will open in your browser at: http://localhost:8501"
echo ""
echo "To stop the app, press Ctrl+C"
echo ""

# Run Streamlit with custom configuration
streamlit run streamlit/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=false \
    --browser.gatherUsageStats=false \
    --server.fileWatcherType=poll 