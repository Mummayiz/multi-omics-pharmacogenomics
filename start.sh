#!/bin/bash
# Railway startup script for Multi-Omics Platform

echo "🚀 Starting Multi-Omics Platform Backend..."

# Set environment variables
export PYTHONPATH=/app
export PYTHONUNBUFFERED=1

# Change to backend directory
cd /app/backend

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found in /app/backend"
    ls -la /app/backend/
    exit 1
fi

# Check Python installation
echo "Python version:"
python --version

# Install any missing dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the FastAPI application
echo "Starting FastAPI server on port 8000..."
python main.py
