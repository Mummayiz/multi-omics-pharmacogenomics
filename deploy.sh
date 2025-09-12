#!/bin/bash

# Multi-Omics Platform Deployment Script
echo "🚀 Multi-Omics Platform Deployment Script"
echo "========================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to deploy with Docker
deploy_docker() {
    echo "🐳 Deploying with Docker..."
    
    if ! command_exists docker; then
        echo "❌ Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        echo "❌ Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    echo "Building and starting containers..."
    docker-compose up --build -d
    
    echo "✅ Deployment complete!"
    echo "Frontend: http://localhost:3000"
    echo "Backend API: http://localhost:8000"
    echo "Health Check: http://localhost:8000/api/v1/health"
}

# Function to deploy to Heroku
deploy_heroku() {
    echo "☁️ Deploying to Heroku..."
    
    if ! command_exists heroku; then
        echo "❌ Heroku CLI is not installed. Please install it first."
        echo "Download from: https://devcenter.heroku.com/articles/heroku-cli"
        exit 1
    fi
    
    echo "Creating Heroku app..."
    heroku create multi-omics-platform-$(date +%s)
    
    echo "Setting environment variables..."
    heroku config:set PYTHONPATH=/app
    heroku config:set PYTHONUNBUFFERED=1
    
    echo "Deploying to Heroku..."
    git push heroku main
    
    echo "Opening application..."
    heroku open
    
    echo "✅ Heroku deployment complete!"
}

# Function to deploy to Railway
deploy_railway() {
    echo "🚂 Deploying to Railway..."
    
    if ! command_exists railway; then
        echo "❌ Railway CLI is not installed. Please install it first."
        echo "Run: npm install -g @railway/cli"
        exit 1
    fi
    
    echo "Initializing Railway project..."
    railway init
    
    echo "Deploying to Railway..."
    railway up
    
    echo "Getting deployment URL..."
    railway domain
    
    echo "✅ Railway deployment complete!"
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  docker    Deploy using Docker (local)"
    echo "  heroku    Deploy to Heroku"
    echo "  railway   Deploy to Railway"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 docker"
    echo "  $0 heroku"
    echo "  $0 railway"
}

# Main script logic
case "${1:-help}" in
    docker)
        deploy_docker
        ;;
    heroku)
        deploy_heroku
        ;;
    railway)
        deploy_railway
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "❌ Unknown option: $1"
        show_help
        exit 1
        ;;
esac
