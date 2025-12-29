# Deployment Guide - Enhanced Atari RL Dashboard

## Quick Start

### 1. Database Migration
The database schema has been extended with new tables. The changes are backward-compatible and will be created automatically on first run.

```bash
# Backup existing database (recommended)
cp data/rl_training.db data/rl_training.db.backup

# The new tables will be created automatically when you start the server
python run_server.py
```

### 2. Local Testing

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Start the server
python run_server.py

# Open browser
# Navigate to http://localhost:5001
```

### 3. Remote Deployment (Hetzner VPS)

```bash
# SSH into your VPS
ssh riccardo@46.224.26.78

# Navigate to project directory
cd /home/riccardo/atari-rl-dashboard

# Pull latest changes
git pull origin main

# Install any new dependencies
./.venv/bin/pip install -r requirements.txt

# Restart the service
sudo systemctl restart atari

# Check status
sudo systemctl status atari

# View logs
tail -f /var/log/atari/app.log
```

## Configuration

### Environment Variables (Optional)
Create a `.env` file for custom configuration:

```bash
# Server Configuration
HOST=0.0.0.0
PORT=5001

# Database
DB_PATH=data/rl_training.db

# Performance
MAX_CONCURRENT_TRAINING=3

# Analytics (if using external service)
# ANALYTICS_KEY=your_key_here
```

### Performance Tuning

Edit `server.py` to adjust:
```python
MAX_CONCURRENT_TRAINING = 3  # Max simultaneous training sessions
```

## Database Management

### View Statistics
```python
from db_manager import TrainingDatabase

db = TrainingDatabase()

# Visitor stats
print(db.get_visitor_stats())

# Feedback stats
print(db.get_feedback_stats())

# Database stats
print(db.get_database_stats())

# Conversion funnel
print(db.get_conversion_funnel())
```

### Create Sample Challenges
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

# Weekly challenge
db.create_challenge(
    game_id='ALE/SpaceInvaders-v5',
    challenge_type='weekly',
    target_value=3,
    description='Train 3 different games this week',
    start_date=date.today().isoformat(),
    end_date=(date.today() + timedelta(days=7)).isoformat()
)
```

### Cleanup Old Data
```python
# Remove analytics and feedback older than 90 days
db.cleanup_old_data(days=90)
```

## Monitoring

### Key Metrics to Track

1. **Visitor Metrics**
   - Total visitors
   - Email collection rate
   - Returning visitors

2. **Engagement Metrics**
   - Average session duration
   - Mode selection (Watch vs Train)
   - Training sessions started

3. **Feedback Metrics**
   - Total feedback submissions
   - Average rating
   - Category distribution

4. **Performance Metrics**
   - Active training sessions
   - Queue length
   - Server response time

### Access Metrics via API

```bash
# Public stats
curl http://localhost:5001/api/stats/public

# Visitor stats (requires admin access)
curl http://localhost:5001/api/visitor/stats

# Feedback stats
curl http://localhost:5001/api/feedback/stats

# Queue status
curl http://localhost:5001/api/queue/status

# Conversion funnel
curl http://localhost:5001/api/analytics/funnel
```

## Troubleshooting

### Issue: Database locked
```bash
# Check for zombie processes
ps aux | grep python

# Kill if necessary
kill -9 <PID>

# Restart service
sudo systemctl restart atari
```

### Issue: High memory usage
```bash
# Check memory
free -h

# Reduce MAX_CONCURRENT_TRAINING in server.py
# Restart service
```

### Issue: Slow page load
```bash
# Check database size
ls -lh data/rl_training.db

# Cleanup old data
python -c "from db_manager import TrainingDatabase; TrainingDatabase().cleanup_old_data(30)"

# Optimize database
sqlite3 data/rl_training.db "VACUUM;"
```

### Issue: Analytics not tracking
```bash
# Check browser console for errors
# Verify analytics.js is loaded
# Check /api/analytics/batch endpoint
```

## Security Considerations

### 1. Rate Limiting (Recommended)
Add Flask-Limiter to prevent abuse:

```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.headers.get('X-Forwarded-For', request.remote_addr)
)

@limiter.limit("100 per hour")
@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    # ...
```

### 2. Input Validation
All user inputs are sanitized via SQLite parameterized queries. Additional validation can be added:

```python
from flask import request
from werkzeug.exceptions import BadRequest

def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise BadRequest('Invalid email format')
```

### 3. CORS Configuration
Update CORS settings for production:

```python
from flask_cors import CORS

CORS(app, origins=['https://atari.riccardocosentino.com'])
```

## Backup Strategy

### Automated Backups
Add to crontab:

```bash
# Daily database backup at 2 AM
0 2 * * * /home/riccardo/atari-rl-dashboard/deployment/backup_models.sh

# Weekly full backup
0 3 * * 0 tar -czf /home/riccardo/backups/atari-$(date +\%Y\%m\%d).tar.gz /home/riccardo/atari-rl-dashboard
```

### Manual Backup
```bash
# Backup database
cp data/rl_training.db data/rl_training.db.$(date +%Y%m%d)

# Backup models
tar -czf saved_models_backup.tar.gz saved_models/
```

## Scaling Considerations

### Horizontal Scaling
For high traffic, consider:

1. **Load Balancer**: Nginx reverse proxy
2. **Database**: PostgreSQL instead of SQLite
3. **Redis**: For session management and caching
4. **CDN**: CloudFlare for static assets

### Vertical Scaling
Current setup handles:
- 100+ concurrent viewers (watch mode)
- 3 concurrent trainers
- 1000+ visitors/day

To scale up:
- Increase `MAX_CONCURRENT_TRAINING`
- Add more RAM (4GB → 8GB)
- Use GPU for faster training

## Support

For issues or questions:
1. Check logs: `/var/log/atari/app.log`
2. Review FEATURES.md for feature documentation
3. Check GitHub issues
4. Contact: riccardo@riccardocosentino.com

