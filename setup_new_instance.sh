#!/bin/bash
# ============================================================
# Quick Reference: Setup New Thunder Compute Instance
# ============================================================
# This script just prints instructions - follow them step by step
# ============================================================

cat << 'EOF'

⚡ THUNDER COMPUTE SETUP - QUICK REFERENCE
============================================================

🎯 YOU ARE HERE: About to set up a new Thunder Compute instance

📋 CHECKLIST:

  Step 1: Create Thunder Compute Instance (Browser)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  → Go to: https://www.thundercompute.com/
  → Click: "Create Instance"
  → Select: "Production Mode" (NOT Prototyping!)
  → GPU: A100 80GB
  → Note down: IP address, SSH port, SSH key path
  
  Step 2: Configure SSH (Local Machine - THIS MACHINE)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Run this command:
  
    bash add_production_instance.sh
  
  → Enter the details from Step 1
  → Choose an alias (e.g., "tnr-prod", "tnr-2")
  
  Step 3: Setup Instance (Copy Script)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Replace <alias> with your chosen alias:
  
    scp setup_production.sh <alias>:~/
  
  Step 4: Run Setup (On Instance)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  
    ssh <alias>
    bash setup_production.sh
  
  Wait 5-10 minutes while it installs everything.
  
  Step 5: Launch Training (On Instance)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  
    cd ~/atari-rl-dashboard
    bash launch_production_training.sh
  
  This starts 6 games training in parallel!
  You can disconnect - it runs in tmux.
  
  Step 6: Monitor Progress (Local Machine)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  
    python monitor_production.py --host <alias> --watch
  
  Press Ctrl+C to stop monitoring (training keeps running)

============================================================

📖 Full Guide: THUNDER_COMPUTE_SETUP.md
🔧 Need help? Check PRODUCTION_SETUP.md

⏱️  Total time: ~10 minutes
💰 Expected cost: ~$2-3/day for 2-3 days = $5-9 total

✅ You're all set! Start with Step 2 above.

============================================================

EOF

