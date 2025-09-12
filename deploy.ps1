# Multi-Omics Platform Deployment Script (PowerShell)
Write-Host "🚀 Multi-Omics Platform Deployment Script" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Function to check if command exists
function Test-Command {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

# Function to deploy with Docker
function Deploy-Docker {
    Write-Host "🐳 Deploying with Docker..." -ForegroundColor Blue
    
    if (-not (Test-Command "docker")) {
        Write-Host "❌ Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
        Write-Host "Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
        exit 1
    }
    
    if (-not (Test-Command "docker-compose")) {
        Write-Host "❌ Docker Compose is not installed. Please install Docker Compose first." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Building and starting containers..." -ForegroundColor Yellow
    docker-compose up --build -d
    
    Write-Host "✅ Deployment complete!" -ForegroundColor Green
    Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
    Write-Host "Backend API: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "Health Check: http://localhost:8000/api/v1/health" -ForegroundColor Cyan
}

# Function to deploy to Heroku
function Deploy-Heroku {
    Write-Host "☁️ Deploying to Heroku..." -ForegroundColor Blue
    
    if (-not (Test-Command "heroku")) {
        Write-Host "❌ Heroku CLI is not installed. Please install it first." -ForegroundColor Red
        Write-Host "Download from: https://devcenter.heroku.com/articles/heroku-cli" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "Creating Heroku app..." -ForegroundColor Yellow
    $timestamp = Get-Date -Format "yyyyMMddHHmmss"
    heroku create "multi-omics-platform-$timestamp"
    
    Write-Host "Setting environment variables..." -ForegroundColor Yellow
    heroku config:set PYTHONPATH=/app
    heroku config:set PYTHONUNBUFFERED=1
    
    Write-Host "Deploying to Heroku..." -ForegroundColor Yellow
    git push heroku main
    
    Write-Host "Opening application..." -ForegroundColor Yellow
    heroku open
    
    Write-Host "✅ Heroku deployment complete!" -ForegroundColor Green
}

# Function to deploy to Railway
function Deploy-Railway {
    Write-Host "🚂 Deploying to Railway..." -ForegroundColor Blue
    
    if (-not (Test-Command "railway")) {
        Write-Host "❌ Railway CLI is not installed. Please install it first." -ForegroundColor Red
        Write-Host "Run: npm install -g @railway/cli" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "Initializing Railway project..." -ForegroundColor Yellow
    railway init
    
    Write-Host "Deploying to Railway..." -ForegroundColor Yellow
    railway up
    
    Write-Host "Getting deployment URL..." -ForegroundColor Yellow
    railway domain
    
    Write-Host "✅ Railway deployment complete!" -ForegroundColor Green
}

# Function to show help
function Show-Help {
    Write-Host "Usage: .\deploy.ps1 [OPTION]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor White
    Write-Host "  docker    Deploy using Docker (local)" -ForegroundColor Cyan
    Write-Host "  heroku    Deploy to Heroku" -ForegroundColor Cyan
    Write-Host "  railway   Deploy to Railway" -ForegroundColor Cyan
    Write-Host "  help      Show this help message" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor White
    Write-Host "  .\deploy.ps1 docker" -ForegroundColor Gray
    Write-Host "  .\deploy.ps1 heroku" -ForegroundColor Gray
    Write-Host "  .\deploy.ps1 railway" -ForegroundColor Gray
}

# Main script logic
$option = $args[0]
if (-not $option) { $option = "help" }

switch ($option) {
    "docker" {
        Deploy-Docker
    }
    "heroku" {
        Deploy-Heroku
    }
    "railway" {
        Deploy-Railway
    }
    "help" {
        Show-Help
    }
    default {
        Write-Host "❌ Unknown option: $option" -ForegroundColor Red
        Show-Help
        exit 1
    }
}
