# Multi-Omics Pharmacogenomics Platform - Local Deployment Script
# PowerShell script for Windows deployment

param(
    [string]$Environment = "development",
    [switch]$Build = $false,
    [switch]$CleanUp = $false
)

Write-Host "🧬 Multi-Omics Pharmacogenomics Platform - Local Deployment" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

# Set error action preference
$ErrorActionPreference = "Stop"

# Check if Docker is running
Write-Host "🐳 Checking Docker status..." -ForegroundColor Blue
try {
    docker version | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Clean up existing containers if requested
if ($CleanUp) {
    Write-Host "🧹 Cleaning up existing containers..." -ForegroundColor Yellow
    docker-compose down -v --remove-orphans
    docker system prune -f
}

# Create necessary directories
Write-Host "📁 Creating necessary directories..." -ForegroundColor Blue
$directories = @("logs", "data/uploads", "data/processed", "models/saved", "nginx")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir
        Write-Host "   Created: $dir" -ForegroundColor Gray
    }
}

# Copy environment file
Write-Host "⚙️  Setting up environment configuration..." -ForegroundColor Blue
$envSource = "backend/.env.$Environment"
$envTarget = "backend/.env"

if (Test-Path $envSource) {
    Copy-Item $envSource $envTarget -Force
    Write-Host "   Environment: $Environment" -ForegroundColor Gray
} else {
    Write-Host "❌ Environment file not found: $envSource" -ForegroundColor Red
    exit 1
}

# Build Docker images if requested
if ($Build) {
    Write-Host "🔨 Building Docker images..." -ForegroundColor Blue
    docker-compose build --no-cache
}

# Create Nginx configuration
Write-Host "🌐 Setting up Nginx configuration..." -ForegroundColor Blue
$nginxConfig = @"
events {
    worker_connections 1024;
}

http {
    upstream api {
        server multi-omics-api:8000;
    }

    server {
        listen 80;
        server_name localhost;

        # Serve frontend
        location / {
            root /usr/share/nginx/html;
            index index.html;
            try_files `$uri `$uri/ /index.html;
        }

        # Proxy API requests
        location /api/ {
            proxy_pass http://api;
            proxy_set_header Host `$host;
            proxy_set_header X-Real-IP `$remote_addr;
            proxy_set_header X-Forwarded-For `$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto `$scheme;
        }

        # Proxy docs and health endpoints
        location ~ ^/(docs|redoc|health) {
            proxy_pass http://api;
            proxy_set_header Host `$host;
            proxy_set_header X-Real-IP `$remote_addr;
            proxy_set_header X-Forwarded-For `$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto `$scheme;
        }
    }
}
"@

$nginxConfig | Out-File -FilePath "nginx/nginx.conf" -Encoding UTF8

# Start services
Write-Host "🚀 Starting services..." -ForegroundColor Blue
docker-compose up -d

# Wait for services to be ready
Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Blue
Start-Sleep -Seconds 10

# Check service health
Write-Host "🏥 Checking service health..." -ForegroundColor Blue
$maxRetries = 30
$retryCount = 0

do {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
        if ($response.status -eq "healthy") {
            Write-Host "✅ API service is healthy" -ForegroundColor Green
            break
        }
    } catch {
        $retryCount++
        if ($retryCount -ge $maxRetries) {
            Write-Host "❌ API service health check failed after $maxRetries attempts" -ForegroundColor Red
            docker-compose logs multi-omics-api
            exit 1
        }
        Write-Host "   Waiting for API service... ($retryCount/$maxRetries)" -ForegroundColor Gray
        Start-Sleep -Seconds 2
    }
} while ($retryCount -lt $maxRetries)

# Display service URLs
Write-Host ""
Write-Host "🎉 Deployment completed successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "🌐 Frontend:        http://localhost" -ForegroundColor Cyan
Write-Host "🔧 API:             http://localhost:8000" -ForegroundColor Cyan  
Write-Host "📚 API Docs:        http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "🔍 Health Check:    http://localhost:8000/health" -ForegroundColor Cyan
Write-Host "🐘 PostgreSQL:      localhost:5432" -ForegroundColor Cyan
Write-Host "📊 Redis:           localhost:6379" -ForegroundColor Cyan
Write-Host "📓 Jupyter:         http://localhost:8888" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 To stop services: docker-compose down" -ForegroundColor Yellow
Write-Host "📋 To view logs:     docker-compose logs -f" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Green
