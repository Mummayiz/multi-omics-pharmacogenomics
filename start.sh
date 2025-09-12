#!/bin/bash
# Railway startup script for Multi-Omics Platform

echo "🚀 Starting Multi-Omics Platform Backend..."

# Change to backend directory
cd /app/backend

# Start the FastAPI application
echo "Starting FastAPI server on port 8000..."
python main.py
