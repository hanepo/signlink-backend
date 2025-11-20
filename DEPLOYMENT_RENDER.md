# Deploy SignLink Backend to Render

## Step 1: Prepare Your Code

1. Make sure all files are committed to Git:
```bash
cd C:\Users\hanep\Documents\Sign_language_BIM\server
git init
git add .
git commit -m "Initial commit for Render deployment"
```

2. Create a GitHub repository (if you don't have one):
   - Go to https://github.com/new
   - Name it: `signlink-backend`
   - Don't initialize with README (you already have files)
   
3. Push your code to GitHub:
```bash
git remote add origin https://github.com/YOUR_USERNAME/signlink-backend.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy on Render

1. **Sign up for Render**
   - Go to https://render.com
   - Sign up with your GitHub account (free)

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `signlink-backend`
   - Configure settings:
     - **Name**: `signlink-api`
     - **Region**: Singapore (closest to Malaysia)
     - **Branch**: `main`
     - **Root Directory**: Leave empty (or set to `server` if you push the whole project)
     - **Runtime**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
     - **Instance Type**: Free

3. **Environment Variables** (Optional)
   - Add if needed: `PYTHON_VERSION = 3.11.0`

4. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for build to complete
   - You'll get a URL like: `https://signlink-api.onrender.com`

## Step 3: Update Your Expo App

Once deployed, update the API URL in your React Native app:

### File: `SignLinkExpo/src/config.js`

Change:
```javascript
export const API_BASE_URL = 'http://192.168.100.6:8000';
```

To:
```javascript
export const API_BASE_URL = 'https://signlink-api.onrender.com';
```

## Step 4: Test Your Deployment

1. **Test API health**:
   - Visit: `https://signlink-api.onrender.com/`
   - Should return: `{"service":"SignLink Gesture Recognition API","status":"running",...}`

2. **Test in your app**:
   - Open Expo Go
   - Go to Camera/Translate screen
   - Should see "AI Ready" status
   - Test ASL detection

## Important Notes

### Free Tier Limitations
- ✅ **Always online** (no need to run server manually)
- ✅ **Free forever** for this project size
- ⚠️ **Spins down after 15 minutes of inactivity** (first request takes ~30 seconds to wake up)
- ⚠️ **750 hours/month free** (plenty for development)

### Model Files
Your model files (`bim_decision_tree.pkl`, etc.) will be uploaded with your code. They're small enough (2MB) for free tier.

### Auto-Deploy
Any push to GitHub `main` branch will automatically redeploy to Render!

## Troubleshooting

### Build Failed
- Check `requirements.txt` is in root directory
- Make sure all dependencies are listed
- Check build logs on Render dashboard

### "Service Unavailable"
- Free tier sleeps after inactivity
- First request wakes it up (~30 seconds)
- Add loading state in your app

### Wrong API URL
- Render provides URL after deployment
- Update `API_BASE_URL` in `config.js`
- Don't forget `https://` (not `http://`)

## Alternative: Keep It Awake (Optional)

To prevent sleep, use a free uptime monitor:
- **UptimeRobot** (https://uptimerobot.com)
- Ping your API every 5 minutes
- Free tier allows 50 monitors

---

**Done!** Your backend is now deployed and accessible from anywhere. No need to run `main.py` locally anymore! 🎉
