# Frontend Deployment Guide

## 🚀 Deploy Frontend to Vercel (Recommended)

### Option 1: Deploy via Vercel Dashboard (Easiest)

1. **Go to [vercel.com](https://vercel.com)**
2. **Sign up/Login** with GitHub
3. **Click "New Project"**
4. **Import your GitHub repository**: `Mummayiz/multi-omics-pharmacogenomics`
5. **Configure settings**:
   - **Framework Preset**: Other
   - **Root Directory**: `frontend`
   - **Build Command**: (leave empty)
   - **Output Directory**: (leave empty)
6. **Click "Deploy"**

### Option 2: Deploy via Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy from frontend directory
cd frontend
vercel

# Follow the prompts
```

### Option 3: Deploy to Netlify

1. **Go to [netlify.com](https://netlify.com)**
2. **Sign up/Login** with GitHub
3. **Click "New site from Git"**
4. **Select your repository**
5. **Configure**:
   - **Base directory**: `frontend`
   - **Build command**: (leave empty)
   - **Publish directory**: `frontend`
6. **Click "Deploy site"**

## 🔗 After Deployment

Your frontend will be available at:
- **Vercel**: `https://your-app-name.vercel.app`
- **Netlify**: `https://your-app-name.netlify.app`

The frontend will automatically connect to your Railway API at:
`https://web-production-0deaa.up.railway.app/api/v1`

## ✅ Test Your Complete Platform

1. **Open your deployed frontend URL**
2. **Test data upload** - Upload sample CSV files
3. **Test model training** - Train multi-omics models
4. **Test predictions** - Get drug response predictions
5. **Test all features** - Everything should work seamlessly

## 🎉 You're Done!

Your complete multi-omics pharmacogenomics platform is now live with:
- ✅ **Backend API** on Railway
- ✅ **Frontend UI** on Vercel/Netlify
- ✅ **Full functionality** - Upload, train, predict, analyze
