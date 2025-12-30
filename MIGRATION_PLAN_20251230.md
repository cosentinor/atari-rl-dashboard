# Gradual Migration Plan - December 30, 2025

## Current State Snapshot (17:48 UTC)

### Games Staying Active (Backup Group)
| PID  | Game           | Episodes | Current Progress | Status |
|------|----------------|----------|------------------|--------|
| 4964 | Pong           | 3,000    | ~520 / 3000      | KEEP   |
| 5029 | Freeway        | 3,000    | ~520 / 3000      | KEEP   |
| 5350 | SpaceInvaders  | 15,000   | ~2600 / 15000    | KEEP   |
| 5419 | Seaquest       | 25,000   | ~3600 / 25000    | KEEP   |

### Games to Migrate First (Test Group)
| PID  | Game      | Episodes | Current Progress | Saved Models                    |
|------|-----------|----------|------------------|---------------------------------|
| 5487 | MsPacman  | 30,000   | 918 / 30000      | Best: 2010, Avg100: 300.8       |
| 5805 | BeamRider | 20,000   | 372 / 20000      | Best: 804, Avg100: 317.4        |
| 5873 | Asteroids | 25,000   | 562 / 25000      | Best: 2080, Avg100: 936.9       |
| 6196 | Enduro    | 15,000   | 125 / 15000      | Best: 762, Avg100: 182.3        |

## Backup Status

✅ **Full Model Backup Created:**
```
File: saved_models_backup_pre_migration_20251230_144213.tar.gz
Size: 7.0 GB
Location: ~/atari-rl-dashboard/
Contains: All saved models, checkpoints, and model_registry.json
```

## Migration Steps

### Phase 1: Graceful Shutdown (5 minutes)
1. Send SIGTERM to processes (not SIGKILL) for graceful shutdown
2. Wait for autosave to complete
3. Verify checkpoints are saved
4. Confirm processes terminated

### Phase 2: Verification (2 minutes)
1. Check saved_models/ for latest checkpoints
2. Verify model_registry.json is updated
3. Confirm no corruption

### Phase 3: Optimized Restart (5 minutes)
1. Start with train_production_batch.py
2. Games will load from latest checkpoints automatically
3. Continue training from where they left off
4. No progress will be lost

### Phase 4: Monitoring (30-60 minutes)
1. Check GPU utilization
2. Monitor episode speed
3. Verify rewards continue improving
4. Watch for errors or crashes

## Rollback Procedures

### If Migration Fails

**Option 1: Quick Rollback (Resume Old Training)**
```bash
ssh tnr-0
cd ~/atari-rl-dashboard
source .venv/bin/activate

# Restart with old settings (will resume from checkpoints)
nohup python train.py --game MsPacman --episodes 30000 > mspacman_train.log 2>&1 &
nohup python train.py --game BeamRider --episodes 20000 > beamrider_train.log 2>&1 &
nohup python train.py --game Asteroids --episodes 25000 > asteroids_train.log 2>&1 &
nohup python train.py --game Enduro --episodes 15000 > enduro_train.log 2>&1 &
```

**Option 2: Full Model Restore (if models corrupted)**
```bash
ssh tnr-0
cd ~/atari-rl-dashboard

# Stop any running processes
pkill -f "train"

# Restore backup
tar -xzf saved_models_backup_pre_migration_20251230_144213.tar.gz

# Restart training
python train.py --game MsPacman --episodes 30000 &
# ... etc
```

**Option 3: Keep Backup Group Running**
- The 4 backup games (Pong, Freeway, SpaceInvaders, Seaquest) continue uninterrupted
- Only the 4 migrated games are affected
- We can analyze logs and retry migration

## Success Criteria

Migration is successful if:
- ✅ All 4 games start without errors
- ✅ Training resumes from last checkpoint episode numbers
- ✅ GPU utilization reaches 100%
- ✅ Episode speed is 500-1000 eps/hr (vs ~20 current)
- ✅ No crashes for 30+ minutes
- ✅ Rewards continue improving

## Timeline

- **17:48 UTC**: Backup complete, ready to start
- **17:50 UTC**: Begin graceful shutdown
- **17:55 UTC**: Verification complete
- **18:00 UTC**: Optimized training started
- **18:30 UTC**: Initial stability check
- **19:00 UTC**: Full migration validation
- **19:30 UTC**: Proceed with remaining 4 games (if successful)

## Next Phase (Phase 2)

If Phase 1 successful after 1-2 hours:
- Migrate remaining 4 games: Pong, Freeway, SpaceInvaders, Seaquest
- Use same procedure
- Full optimization achieved

## Safety Guarantees

1. **No Progress Loss**: All games have saved checkpoints
2. **Backup Available**: 7 GB full backup of all models
3. **Partial Failure OK**: 4 games remain untouched as backup
4. **Quick Rollback**: Can restart old training in < 2 minutes
5. **Model Recovery**: Can restore from backup if needed

---
**Status**: READY TO EXECUTE  
**Risk Level**: LOW  
**Confidence**: VERY HIGH
