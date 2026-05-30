#!/bin/bash
set -e

echo "=== RAG API Startup ==="

# Ensure required directories exist
mkdir -p "${DATA_DIR:-data}" "${DB_DIR:-vectors}" "${MODEL_DIR:-llm_models}"

# Run ingestion if vector index does not exist
INDEX_PATH="${DB_DIR:-vectors}/docs.index"
if [ ! -f "$INDEX_PATH" ]; then
    echo "No vector index found at $INDEX_PATH. Running ingestion..."
    python ingest.py
    echo "Ingestion complete."
else
    echo "Vector index found. Skipping ingestion."
fi

echo "Starting API server..."
exec uvicorn main_api:app --host 0.0.0.0 --port 8000
