# Material Dashboard PRO Integration Guide

## Step-by-Step Instructions

### BEFORE YOU START
1. Purchase Material Dashboard PRO: https://www.creative-tim.com/product/material-dashboard-pro-react
2. Download the ZIP file from your Creative Tim dashboard
3. Extract to: `~/Downloads/material-dashboard-pro-react/`

---

## Quick Start After Purchase

### 1. Create New Project Structure
```bash
# Navigate to your workspace
cd ~/coding_directory/workspace/reinforcement-learning

# Create new v2 directory
mkdir atari-rl-dashboard-v2
cd atari-rl-dashboard-v2

# Copy your backend
cp -r ../atari/* .

# Rename current frontend to backup
mv frontend frontend-backup

# Copy Material Dashboard PRO
cp -r ~/Downloads/material-dashboard-pro-react/ frontend/
```

### 2. Install Dependencies
```bash
cd frontend/
npm install
npm install socket.io-client
```

### 3. Test Template
```bash
npm start
# Should open http://localhost:3000 with Material Dashboard demo
```

### 4. Configure for Your Backend
The configuration files have been created for you in:
- `frontend/src/config.js` - API endpoints
- `frontend/.env.development` - Development settings
- `frontend/.env.production` - Production settings

### 5. Start Both Servers

**Terminal 1 - Backend:**
```bash
cd ~/coding_directory/workspace/reinforcement-learning/atari-rl-dashboard-v2/
python run_server.py
```

**Terminal 2 - Frontend:**
```bash
cd ~/coding_directory/workspace/reinforcement-learning/atari-rl-dashboard-v2/frontend/
npm start
```

---

## File Locations

### Files You'll Create
- `frontend/src/config.js` ✓ (created for you)
- `frontend/src/services/socket.service.js` ✓ (created for you)
- `frontend/src/layouts/AtariDashboard.js` (you'll create this)
- `frontend/src/components/GameCanvas/` (migrate your canvas logic)

### Files You'll Modify
- `frontend/src/App.js` - Add your Atari dashboard route
- `frontend/src/assets/jss/material-dashboard-pro-react.js` - Customize colors
- `frontend/package.json` - Already configured

---

## Backend is Ready

Your backend has been configured with:
- ✓ CORS settings for localhost:3000
- ✓ Support for both development and production builds
- ✓ All existing functionality preserved

---

## Next Steps

1. Purchase and download template
2. Follow "Quick Start After Purchase" above
3. Start migrating components one by one
4. Test thoroughly
5. Build for production: `npm run build`
6. Deploy to Hetzner VPS

---

## Need Help?

If you encounter issues:
1. Check Material Dashboard PRO documentation
2. Review troubleshooting section in plan
3. Contact Creative Tim support
4. Ask me for specific code help

## Timeline
- Purchase & setup: 30 min
- Component migration: 3-4 hours
- Testing: 1 hour
- Deployment: 30 min
**Total: ~5-6 hours**
