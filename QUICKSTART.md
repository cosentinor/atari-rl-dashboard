# Quick Start Guide - Enhanced Dashboard

## 🚀 Get Started in 3 Steps

### 1. Start the Server Locally
```bash
cd /Users/riccardocosentino/coding_directory/workspace/reinforcement-learning/atari
python run_server.py
```

### 2. Open Your Browser
```
http://localhost:5001
```

### 3. Experience the Flow
1. **Hero Section** - See the new landing page
2. **Click "Get Started"** - Try the email modal (you can skip)
3. **Choose Mode** - Select Watch or Train
4. **Explore Features**:
   - Hover over technical terms to see tooltips
   - Click the feedback button (bottom-right)
   - Try the comparison view
   - Check out the enhanced leaderboard

## 🎯 Key Features to Test

### Email Collection
- ✅ Shows on first visit
- ✅ Can be skipped
- ✅ Stores visitor UUID in localStorage
- ✅ Remembers returning visitors

### Mode Selection
- ✅ Watch Mode: View pre-trained agents
- ✅ Train Mode: Full interactive experience
- ✅ Mode badge shows in header

### Educational Tooltips
- ✅ Hover over "Rainbow DQN" in header
- ✅ Hover over "Episode", "Reward", "Q-Value"
- ✅ Click for detailed explanations

### Feedback System
- ✅ Floating button bottom-right
- ✅ Quick 5-star rating
- ✅ Detailed feedback form
- ✅ Stores in database

### Analytics
- ✅ Automatic page view tracking
- ✅ Event tracking (check browser console)
- ✅ Batch upload every 5 seconds

## 📊 Check the Data

### View Visitor Stats
```python
from db_manager import TrainingDatabase
db = TrainingDatabase()

# Visitor statistics
print(db.get_visitor_stats())

# Conversion funnel
print(db.get_conversion_funnel())

# Feedback stats
print(db.get_feedback_stats())
```

### View Database Tables
```bash
sqlite3 data/rl_training.db

# List all tables
.tables

# View visitors
SELECT * FROM visitors;

# View analytics events
SELECT event_type, COUNT(*) FROM analytics_events GROUP BY event_type;

# View feedback
SELECT * FROM feedback;

# Exit
.quit
```

## 🎮 Create Sample Challenges

```python
from db_manager import TrainingDatabase
from datetime import date, timedelta

db = TrainingDatabase()

# Daily challenge
db.create_challenge(
    game_id='ALE/MsPacman-v5',
    challenge_type='daily',
    target_value=1000,
    description='Score 1000 points in Ms. Pac-Man',
    start_date=date.today().isoformat(),
    end_date=(date.today() + timedelta(days=1)).isoformat()
)

print("✅ Challenge created!")
```

## 🚀 Deploy to Production

### Option 1: Quick Deploy
```bash
# SSH to VPS
ssh riccardo@46.224.26.78

# Pull changes
cd /home/riccardo/atari-rl-dashboard
git pull origin main

# Restart
sudo systemctl restart atari
```

### Option 2: Full Deploy with Checks
```bash
# SSH to VPS
ssh riccardo@46.224.26.78

# Navigate to project
cd /home/riccardo/atari-rl-dashboard

# Backup database
cp data/rl_training.db data/rl_training.db.$(date +%Y%m%d)

# Pull changes
git pull origin main

# Install dependencies
./.venv/bin/pip install -r requirements.txt

# Restart service
sudo systemctl restart atari

# Check status
sudo systemctl status atari

# View logs
tail -f /var/log/atari/app.log
```

## 🔍 Verify Deployment

### Check Endpoints
```bash
# Public stats
curl http://atari.riccardocosentino.com/api/stats/public

# Queue status
curl http://atari.riccardocosentino.com/api/queue/status

# Games list
curl http://atari.riccardocosentino.com/api/games
```

### Check Website
1. Visit: https://atari.riccardocosentino.com
2. Should see hero section
3. Click "Get Started"
4. Complete onboarding flow

## 📈 Monitor Performance

### Real-time Logs
```bash
ssh riccardo@46.224.26.78
tail -f /var/log/atari/app.log
```

### Check Metrics
```bash
# Visitor stats
curl http://localhost:5001/api/stats/public | jq

# Feedback stats
curl http://localhost:5001/api/feedback/stats | jq

# Queue status
curl http://localhost:5001/api/queue/status | jq
```

## 🐛 Troubleshooting

### Issue: Can't see new features
**Solution**: Hard refresh browser (Cmd+Shift+R or Ctrl+Shift+R)

### Issue: Database errors
**Solution**: 
```bash
# Check database exists
ls -lh data/rl_training.db

# Check permissions
chmod 644 data/rl_training.db

# Restart server
```

### Issue: Analytics not tracking
**Solution**: 
1. Open browser console (F12)
2. Check for JavaScript errors
3. Verify analytics.js is loaded
4. Check Network tab for /api/analytics/batch calls

### Issue: Email modal not showing
**Solution**:
```javascript
// Clear localStorage and refresh
localStorage.clear();
location.reload();
```

## 📚 Documentation

- **FEATURES.md** - Complete feature documentation
- **DEPLOYMENT.md** - Deployment guide
- **IMPLEMENTATION_SUMMARY.md** - Technical overview

## 🎉 You're All Set!

The enhanced dashboard is ready to:
- ✅ Collect visitor emails (optional)
- ✅ Track comprehensive analytics
- ✅ Gather user feedback
- ✅ Display daily challenges
- ✅ Enable social sharing
- ✅ Compare model performance
- ✅ Handle multiple concurrent users

**Next Steps**:
1. Test locally
2. Deploy to production
3. Create sample challenges
4. Monitor analytics
5. Gather feedback
6. Iterate and improve!

---

**Questions?** Check the documentation or contact support.

