#!/bin/bash
# Simple entrypoint for Railway deployment

echo "🚀 Starting Multi-Omics Platform..."

# Set environment variables
export PYTHONPATH=/app
export PYTHONUNBUFFERED=1

# Start the application
echo "Starting FastAPI server..."
python /app/backend/main.py
