# 🚀 Multi-Omics Platform Deployment Guide

This guide covers multiple deployment options for your Multi-Omics Pharmacogenomics Platform.

## 📋 Prerequisites

- Git installed
- Docker installed (for containerized deployment)
- Heroku CLI installed (for Heroku deployment)
- Railway CLI installed (for Railway deployment)

## 🐳 Option 1: Docker Deployment (Recommended)

### Local Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   # Clone your repository
   git clone https://github.com/Mummayiz/multi-omics-pharmacogenomics.git
   cd multi-omics-pharmacogenomics

   # Build and start the application
   docker-compose up --build
   ```

2. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Health: http://localhost:8000/api/v1/health

### Docker Hub Deployment

1. **Build and push to Docker Hub:**
   ```bash
   # Build the image
   docker build -t yourusername/multi-omics-platform .

   # Push to Docker Hub
   docker push yourusername/multi-omics-platform
   ```

2. **Deploy on any cloud provider:**
   - AWS ECS
   - Google Cloud Run
   - Azure Container Instances
   - DigitalOcean App Platform

## ☁️ Option 2: Heroku Deployment

### Step 1: Prepare for Heroku

1. **Install Heroku CLI:**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login to Heroku:**
   ```bash
   heroku login
   ```

### Step 2: Deploy to Heroku

1. **Create Heroku app:**
   ```bash
   heroku create multi-omics-platform
   ```

2. **Set environment variables:**
   ```bash
   heroku config:set PYTHONPATH=/app
   heroku config:set PYTHONUNBUFFERED=1
   ```

3. **Deploy:**
   ```bash
   git push heroku main
   ```

4. **Open the application:**
   ```bash
   heroku open
   ```

### Step 3: Add Database (Optional)

```bash
# Add PostgreSQL addon
heroku addons:create heroku-postgresql:mini

# Check database URL
heroku config:get DATABASE_URL
```

## 🚂 Option 3: Railway Deployment

### Step 1: Prepare for Railway

1. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway:**
   ```bash
   railway login
   ```

### Step 2: Deploy to Railway

1. **Initialize Railway project:**
   ```bash
   railway init
   ```

2. **Deploy:**
   ```bash
   railway up
   ```

3. **Get deployment URL:**
   ```bash
   railway domain
   ```

## 🌐 Option 4: Vercel (Frontend Only)

### Deploy Frontend to Vercel

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Deploy frontend:**
   ```bash
   cd frontend
   vercel --prod
   ```

3. **Update API URLs in frontend code to point to your backend deployment**

## 🔧 Option 5: Manual Server Deployment

### Step 1: Server Setup

1. **Choose a cloud provider:**
   - AWS EC2
   - Google Cloud Compute
   - DigitalOcean Droplet
   - Linode

2. **Install dependencies:**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y

   # Install Python 3.9
   sudo apt install python3.9 python3.9-pip python3.9-venv

   # Install Node.js (for frontend)
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt-get install -y nodejs
   ```

### Step 2: Deploy Application

1. **Clone repository:**
   ```bash
   git clone https://github.com/Mummayiz/multi-omics-pharmacogenomics.git
   cd multi-omics-pharmacogenomics
   ```

2. **Setup Python environment:**
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   pip install -r backend/requirements.txt
   ```

3. **Start backend:**
   ```bash
   cd backend
   python main.py &
   ```

4. **Start frontend:**
   ```bash
   cd frontend
   python -m http.server 3000 &
   ```

5. **Setup reverse proxy (Nginx):**
   ```bash
   sudo apt install nginx
   
   # Create Nginx configuration
   sudo nano /etc/nginx/sites-available/multi-omics
   ```

   **Nginx configuration:**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:3000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }

       location /api/ {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

   ```bash
   # Enable site
   sudo ln -s /etc/nginx/sites-available/multi-omics /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

## 🔒 Option 6: SSL/HTTPS Setup

### Using Let's Encrypt (Certbot)

1. **Install Certbot:**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   ```

2. **Get SSL certificate:**
   ```bash
   sudo certbot --nginx -d your-domain.com
   ```

3. **Auto-renewal:**
   ```bash
   sudo crontab -e
   # Add: 0 12 * * * /usr/bin/certbot renew --quiet
   ```

## 📊 Monitoring and Maintenance

### Health Checks

- **Backend Health:** `GET /api/v1/health`
- **System Status:** `GET /api/v1/omics/status`

### Logs

- **Application logs:** `logs/multi_omics_*.log`
- **Docker logs:** `docker-compose logs -f`
- **Heroku logs:** `heroku logs --tail`
- **Railway logs:** `railway logs`

### Database Backup

```bash
# SQLite backup
cp multi_omics.db backup_$(date +%Y%m%d_%H%M%S).db

# PostgreSQL backup (if using)
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql
```

## 🚨 Troubleshooting

### Common Issues

1. **Port conflicts:**
   - Change ports in docker-compose.yml
   - Update frontend API URLs

2. **Memory issues:**
   - Increase server memory
   - Optimize data processing

3. **Database errors:**
   - Check database connection
   - Verify file permissions

4. **CORS errors:**
   - Update CORS settings in backend/main.py
   - Check frontend API URLs

### Performance Optimization

1. **Enable caching:**
   - Add Redis for session storage
   - Implement API response caching

2. **Database optimization:**
   - Add database indexes
   - Optimize queries

3. **CDN setup:**
   - Use CloudFlare for static assets
   - Implement image optimization

## 📈 Scaling

### Horizontal Scaling

1. **Load balancer setup**
2. **Multiple backend instances**
3. **Database clustering**
4. **CDN implementation**

### Vertical Scaling

1. **Increase server resources**
2. **Optimize application code**
3. **Database tuning**
4. **Caching strategies**

## 🔐 Security

### Security Checklist

- [ ] Enable HTTPS/SSL
- [ ] Set up firewall rules
- [ ] Regular security updates
- [ ] Database encryption
- [ ] API rate limiting
- [ ] Input validation
- [ ] Error handling

## 📞 Support

For deployment issues:

1. Check the logs
2. Verify environment variables
3. Test locally first
4. Check cloud provider documentation
5. Open an issue on GitHub

---

**Choose the deployment option that best fits your needs and budget!** 🚀