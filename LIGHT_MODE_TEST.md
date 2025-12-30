# Light Training Mode - Stability Test

**Started**: December 30, 2025, 18:51 UTC  
**Purpose**: Test if heavy parallel load was causing reboots

## Test Configuration

### Previous (Heavy) Load:
- **Games**: 8 (all games)
- **Parallel Envs**: 256 per optimized game
- **Batch Size**: 2048
- **Total Processes**: ~365
- **GPU Memory**: 35+ GB
- **Result**: Rebooted every 15-20 minutes ❌

### Current (Light) Load:
- **Games**: 2 (MsPacman, Pong)
- **Parallel Envs**: 32 (MsPacman), 1 (Pong)
- **Batch Size**: 512 (MsPacman), 32 (Pong)
- **Total Processes**: ~18-20
- **GPU Memory**: ~2 GB
- **Status**: Testing...

## Monitoring Schedule

### Check Points:
- ✅ **5 min**: Stable (18:51-18:56)
- ⏳ **15 min**: Check at 19:06
- ⏳ **30 min**: Check at 19:21
- ⏳ **60 min**: Check at 19:51

### Success Criteria:
- **30 min stable**: Load likely wasn't the issue
- **60 min stable**: Definitely not load-related
- **Reboot before 30 min**: Confirms load wasn't the problem

## Hypothesis Testing

### If Stable for 30+ Minutes:
**Conclusion**: Heavy parallel load was overwhelming the system

**Next Steps:**
1. Keep light mode for safety
2. Gradually increase (add 1 game at a time)
3. Find maximum stable configuration
4. Optimize within stable limits

### If Reboots Continue:
**Conclusion**: Instance hardware is faulty, not load-related

**Next Steps:**
1. Contact Thunder Compute immediately
2. Request instance replacement
3. Consider switching providers
4. This instance is defective

## Gradual Scale-Up Plan (If Stable)

### Step 1: Add One More Game (After 1 hour stable)
- Add Asteroids with light settings
- Monitor for 30 minutes
- 3 games total

### Step 2: Increase Parallel Envs (After stable)
- Increase MsPacman to 64 envs
- Monitor for 30 minutes

### Step 3: Add More Games (After stable)
- Add one game at a time
- 30-minute stability check each
- Target: 4-6 games maximum

### Step 4: Fine-Tune Settings
- Optimize batch sizes
- Find sweet spot for this instance
- Balance speed vs stability

## Current Status Log

| Time  | Uptime | Processes | GPU Mem | Status |
|-------|--------|-----------|---------|--------|
| 18:51 | 4 min  | 18        | 1.8 GB  | ✅ Started |
| 18:56 | 5 min  | 18        | 1.8 GB  | ✅ Stable |
| 19:06 | ?      | ?         | ?       | Checking... |

## Recommendations Based on Results

### Scenario A: Stable for 60+ Minutes
**Action**: This instance can't handle heavy loads
- Keep 2-4 games max
- Use lower parallel settings
- Accept slower training
- OR switch to more powerful instance

### Scenario B: Reboots Continue
**Action**: Instance is broken
- Not your code
- Not your settings
- Hardware/system issue
- MUST switch providers

### Scenario C: Stable 30-60 Minutes
**Action**: Cautiously scale up
- Very gradual increases
- Extensive testing at each level
- Find maximum stable config
- Document limits

## Cost Impact

### With Light Load (2 games):
- Training time: 3-4x slower than heavy load
- But 99% uptime vs 83% uptime
- Net: Still faster than rebooting every 15 min

### Trade-off:
- Heavy load: 50x speedup with 17% downtime = 41x effective
- Light load: 15x speedup with 1% downtime = 14.8x effective
- **Still 15x faster than original!**

## Next Check-In

**In 10 minutes (19:01 UTC)**:
```bash
ssh tnr-0 "uptime -p && ps aux | grep train | wc -l"
```

Expected: up 14 minutes, ~18 processes

**If uptime resets**: Test failed, instance is broken  
**If uptime continues**: Test promising, continue monitoring

---

**Status**: TESTING IN PROGRESS  
**Next Update**: 19:01 UTC (10 min check)
