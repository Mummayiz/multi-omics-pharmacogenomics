# Multi-Omics Pharmacogenomics Platform - Deployment Guide

## 🚀 Quick Start Deployment

This guide provides step-by-step instructions to deploy your multi-omics pharmacogenomics platform in different environments.

## Prerequisites

### 📋 Required Software

1. **Docker Desktop** (for Windows)
   - Download: https://www.docker.com/products/docker-desktop
   - Install and start Docker Desktop
   - Verify: `docker --version` and `docker-compose --version`

2. **Git** (already installed)
   - Verify: `git --version`

3. **PowerShell** (already available on Windows)

### 💾 System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB free space
- **CPU**: 4 cores recommended
- **Network**: Internet connection for initial setup

## 🔧 Local Development Deployment

### Step 1: Install Docker Desktop

1. Download Docker Desktop from https://www.docker.com/products/docker-desktop
2. Install and start Docker Desktop
3. Wait for Docker to be ready (whale icon in system tray)

### Step 2: Prepare Environment

```powershell
# Copy environment configuration
Copy-Item "backend/.env.development" "backend/.env"

# Create required directories
New-Item -ItemType Directory -Force -Path "logs", "data/uploads", "data/processed", "models/saved", "nginx"
```

### Step 3: Build and Deploy

```powershell
# Build and start all services
docker-compose up --build -d

# Wait for services to be ready
Start-Sleep -Seconds 30

# Check service status
docker-compose ps
```

### Step 4: Access Your Platform

- 🌐 **Frontend**: http://localhost
- 🔧 **API**: http://localhost:8000
- 📚 **API Documentation**: http://localhost:8000/docs
- 🔍 **Health Check**: http://localhost:8000/health
- 📓 **Jupyter Notebook**: http://localhost:8888

## 🏗️ Production Deployment

### Option 1: Docker Compose (Recommended)

1. **Prepare Production Environment**:
   ```powershell
   Copy-Item "backend/.env.production" "backend/.env"
   
   # Update database password and other sensitive information in .env
   # Set POSTGRES_PASSWORD to a secure value
   # Set SECRET_KEY to a secure random string
   ```

2. **Deploy Services**:
   ```powershell
   docker-compose -f docker-compose.yml up -d
   ```

3. **Monitor Deployment**:
   ```powershell
   # Run the monitoring script
   .\monitor-deployment.ps1 -Mode health
   ```

### Option 2: Cloud Deployment (AWS/GCP)

1. **AWS Deployment**:
   ```powershell
   .\deploy-cloud.ps1 -CloudProvider aws -CreateInfrastructure -Region us-east-1
   ```

2. **GCP Deployment**:
   ```powershell
   .\deploy-cloud.ps1 -CloudProvider gcp -CreateInfrastructure -Region us-central1
   ```

## 🔍 Monitoring and Maintenance

### Health Monitoring

```powershell
# One-time health check
.\monitor-deployment.ps1 -Mode health

# Continuous monitoring
.\monitor-deployment.ps1 -Continuous -Interval 60 -Alert
```

### View Logs

```powershell
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f multi-omics-api
```

### Update Deployment

```powershell
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose up --build -d
```

## 🐛 Troubleshooting

### Common Issues

1. **Docker not found**:
   - Install Docker Desktop
   - Restart PowerShell after installation
   - Check Docker is running

2. **Port conflicts**:
   - Stop other services using ports 80, 8000, 5432, 6379
   - Or modify ports in docker-compose.yml

3. **Out of memory**:
   - Increase Docker memory limit in Docker Desktop settings
   - Close other applications

4. **Services not starting**:
   ```powershell
   # Check container status
   docker-compose ps
   
   # View logs for errors
   docker-compose logs multi-omics-api
   ```

### Service Recovery

```powershell
# Restart specific service
docker-compose restart multi-omics-api

# Full restart
docker-compose down && docker-compose up -d

# Clean rebuild
docker-compose down -v
docker system prune -f
docker-compose up --build -d
```

## 📊 Performance Tuning

### For Development

- Use SQLite database (already configured)
- Smaller batch sizes
- Limited logging

### For Production

- PostgreSQL database with connection pooling
- Redis caching
- Nginx reverse proxy
- Resource monitoring

## 🔒 Security Considerations

### Development

- Default passwords (change for production)
- Permissive CORS settings
- Debug mode enabled

### Production

- Strong passwords and secrets
- HTTPS configuration
- Restricted CORS
- Security headers
- Regular updates

## 📈 Scaling Options

### Vertical Scaling

- Increase container resources
- Upgrade server specifications

### Horizontal Scaling

- Multiple API instances
- Load balancer configuration
- Database clustering

## 🆘 Support

### Quick Commands Reference

```powershell
# Start platform
docker-compose up -d

# Stop platform
docker-compose down

# View status
docker-compose ps

# Monitor health
.\monitor-deployment.ps1

# View logs
docker-compose logs -f

# Update platform
git pull && docker-compose up --build -d
```

### Getting Help

1. Check logs: `docker-compose logs servicename`
2. Review configuration files
3. Verify system requirements
4. Check Docker Desktop status

---

## 🎯 Next Steps

After successful deployment:

1. **Upload sample data** through the web interface
2. **Train your first model** using the API
3. **Make predictions** for drug responses
4. **Explore results** in Jupyter notebooks
5. **Set up monitoring** for production

Your Multi-Omics Pharmacogenomics Platform is ready for precision medicine research! 🧬✨
