# Multi-Omics Pharmacogenomics Platform - Docker Configuration
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/uploads logs models/saved

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 3000

# Create startup script
RUN echo '#!/bin/bash\n\
# Start backend in background\n\
cd /app/backend && python main.py &\n\
# Start frontend server\n\
cd /app/frontend && python -m http.server 3000 &\n\
# Wait for any process to exit\n\
wait' > /app/start.sh && chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start the application
CMD ["/app/start.sh"]