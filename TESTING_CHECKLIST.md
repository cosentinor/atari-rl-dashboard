# Testing Checklist - Enhanced Dashboard

## Automated Tests
Run the automated test suite first:
```bash
# Terminal 1: Start server
python run_server.py

# Terminal 2: Run tests
python test_deployment.py
```

## Manual Testing Checklist

### 1. First-Time Visitor Flow
- [ ] Visit http://localhost:5001
- [ ] Verify hero section appears
- [ ] Check live stats are displayed
- [ ] Click "Get Started Free"
- [ ] Email modal appears
- [ ] Enter email and submit
- [ ] Verify email is stored (check browser console or localStorage)
- [ ] Mode selector appears
- [ ] Select "Watch Mode" or "Train Mode"
- [ ] Verify dashboard loads

### 2. Returning Visitor Flow
- [ ] Refresh page or visit again
- [ ] Should skip hero and email modal
- [ ] Should go directly to dashboard
- [ ] Check if mode badge shows in header

### 3. Educational Tooltips
- [ ] Hover over "Rainbow DQN" in header
- [ ] Tooltip appears with explanation
- [ ] Click tooltip to see detailed view
- [ ] Test other tooltips: "Episode", "Reward", "Q-Value", "Loss"
- [ ] Verify "Learn More" links work (if applicable)

### 4. Training Functionality
- [ ] Select a game from dropdown
- [ ] Click "Start Fresh" or "Resume" button
- [ ] Verify training starts
- [ ] Check game canvas shows frames
- [ ] Verify stats update (Episode, Reward, FPS)
- [ ] Check charts populate with data
- [ ] Click "Stop" button
- [ ] Verify training stops cleanly

### 5. Feedback System
- [ ] Click floating feedback button (bottom-right)
- [ ] Test quick rating (click stars)
- [ ] Verify success message appears
- [ ] Click feedback button again
- [ ] Click "Send Detailed Feedback"
- [ ] Fill out form (category, rating, message)
- [ ] Submit feedback
- [ ] Verify success message

### 6. Analytics Tracking
- [ ] Open browser Developer Tools
- [ ] Go to Network tab
- [ ] Filter by "analytics"
- [ ] Perform actions (navigate, click buttons)
- [ ] Verify POST requests to /api/analytics/batch
- [ ] Check request payload contains events

### 7. Leaderboard
- [ ] Verify leaderboard shows entries
- [ ] Test "All Games" filter
- [ ] Test per-game filter
- [ ] Test timeframe filters (Today, Week, Month, All Time)
- [ ] Check badges appear for top entries (🥇🥈🥉)

### 8. Challenges Panel
- [ ] Verify challenges panel appears (if challenges exist)
- [ ] Check progress bars display correctly
- [ ] Verify countdown timers work
- [ ] Check completed challenges show checkmark

### 9. Model Comparison
- [ ] Select a game with multiple checkpoints
- [ ] Click "Compare Models" button
- [ ] Select 2-3 checkpoints
- [ ] Click "Compare" button
- [ ] Verify comparison view displays
- [ ] Check stats and improvement percentage

### 10. Social Sharing
- [ ] Start a training session
- [ ] After some episodes, click "Share" button
- [ ] Test Twitter share link
- [ ] Test "Copy Link" button
- [ ] Verify link is copied to clipboard
- [ ] Test "Download Card" button
- [ ] Verify image downloads

### 11. Queue System
- [ ] Open 3 browser tabs
- [ ] Start training in all 3 tabs simultaneously
- [ ] First 3 should start immediately
- [ ] 4th tab should show queue message
- [ ] Stop one training session
- [ ] Verify queued session starts automatically

### 12. Performance
- [ ] Monitor CPU and memory usage during training
- [ ] Check for memory leaks (training for extended period)
- [ ] Verify no console errors in browser
- [ ] Check server logs for errors

### 13. Responsive Design
- [ ] Resize browser window to mobile size
- [ ] Verify layout adjusts correctly
- [ ] Test on actual mobile device if possible
- [ ] Check all buttons are tappable
- [ ] Verify modals display correctly on mobile

### 14. Browser Compatibility
- [ ] Test in Chrome
- [ ] Test in Firefox
- [ ] Test in Safari
- [ ] Test in Edge
- [ ] Verify all features work in each browser

### 15. Database Verification
```bash
# Connect to database
sqlite3 data/rl_training.db

# Check new tables exist
.tables

# Check visitors table has data
SELECT COUNT(*) FROM visitors;

# Check analytics events
SELECT event_type, COUNT(*) FROM analytics_events GROUP BY event_type;

# Check feedback
SELECT COUNT(*) FROM feedback;

# Check challenges
SELECT * FROM challenges;
```

### 16. API Endpoint Testing
```bash
# Test visitor registration
curl -X POST http://localhost:5001/api/visitor/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","opt_in_marketing":true}'

# Test public stats
curl http://localhost:5001/api/stats/public

# Test queue status
curl http://localhost:5001/api/queue/status

# Test challenges
curl http://localhost:5001/api/challenges

# Test feedback stats
curl http://localhost:5001/api/feedback/stats
```

## Known Issues to Watch For

1. **Email Modal Re-appearing**: If you clear localStorage, modal will show again
2. **Watch Mode**: Not fully implemented - shows UI but may not have pre-trained models
3. **Queue System**: Needs testing with actual concurrent users
4. **Mobile Safari**: May have issues with tooltips
5. **Analytics**: Check if events are batched correctly

## Debug Logs Location

If tests fail, check these logs:
- **Backend logs**: `/var/log/atari/app.log` (production) or console (local)
- **Frontend logs**: Browser Developer Console
- **Debug logs**: `.cursor/debug.log` (if instrumentation is active)

## Success Criteria

All checkboxes above should be checked. If any fail:
1. Note the failure
2. Check debug logs
3. Check browser console for errors
4. Check server logs
5. Report issue with details

## Post-Testing Cleanup

After successful testing:
```bash
# Clean up test data
sqlite3 data/rl_training.db "DELETE FROM visitors WHERE email LIKE 'test%';"
sqlite3 data/rl_training.db "DELETE FROM feedback WHERE message LIKE 'Test%';"
sqlite3 data/rl_training.db "DELETE FROM challenges WHERE challenge_type = 'test';"
```

