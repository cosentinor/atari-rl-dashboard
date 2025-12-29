#!/bin/bash
# ============================================================
# Add Production Instance to SSH Config (LOCAL SCRIPT)
# ============================================================
# Run this on your LOCAL machine to add new Thunder Compute
# Production instance to SSH config
# Usage: bash add_production_instance.sh
# ============================================================

set -e

echo "============================================================"
echo "🔧 Add Thunder Compute Production Instance to SSH Config"
echo "============================================================"
echo ""
echo "This script will add the new Production instance to your"
echo "SSH config file (~/.ssh/config)"
echo ""

# Get instance details
echo "Please provide the Thunder Compute Production instance details:"
echo ""

read -p "Host alias (e.g., tnr-prod): " HOST_ALIAS
if [ -z "$HOST_ALIAS" ]; then
    echo "❌ Error: Host alias is required"
    exit 1
fi

read -p "IP Address (e.g., 185.216.20.179): " IP_ADDRESS
if [ -z "$IP_ADDRESS" ]; then
    echo "❌ Error: IP address is required"
    exit 1
fi

read -p "SSH Port (e.g., 31897): " PORT
if [ -z "$PORT" ]; then
    echo "❌ Error: Port is required"
    exit 1
fi

read -p "SSH Key Path (e.g., ~/.thunder/keys/abc123): " KEY_PATH
if [ -z "$KEY_PATH" ]; then
    echo "❌ Error: Key path is required"
    exit 1
fi

# Expand tilde in key path
KEY_PATH="${KEY_PATH/#\~/$HOME}"

# Verify key exists
if [ ! -f "$KEY_PATH" ]; then
    echo "⚠️  Warning: SSH key not found at $KEY_PATH"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 1
    fi
fi

# Verify SSH config exists
SSH_CONFIG="$HOME/.ssh/config"
if [ ! -f "$SSH_CONFIG" ]; then
    echo "📝 Creating SSH config file..."
    mkdir -p "$HOME/.ssh"
    touch "$SSH_CONFIG"
    chmod 600 "$SSH_CONFIG"
fi

# Check if host already exists
if grep -q "^Host $HOST_ALIAS$" "$SSH_CONFIG"; then
    echo ""
    echo "⚠️  Host '$HOST_ALIAS' already exists in SSH config!"
    echo ""
    read -p "Overwrite? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 1
    fi
    
    # Remove existing entry
    echo "Removing existing entry..."
    # This is a bit tricky - we need to remove the Host block
    # For now, we'll just append and let the user clean up duplicates manually
    echo ""
    echo "⚠️  Please manually remove the old entry from $SSH_CONFIG"
    echo "    and re-run this script, or edit the file directly."
    exit 1
fi

# Add to SSH config
echo ""
echo "✍️  Adding to SSH config..."

cat >> "$SSH_CONFIG" << EOF

# Thunder Compute Production Instance - Added $(date +%Y-%m-%d)
Host $HOST_ALIAS
    HostName $IP_ADDRESS
    Port $PORT
    User ubuntu
    IdentityFile $KEY_PATH
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF

echo "✅ Added to $SSH_CONFIG"
echo ""

# Test connection
echo "🔌 Testing SSH connection..."
if ssh -o ConnectTimeout=10 "$HOST_ALIAS" "echo '✅ Connection successful!'" 2>/dev/null; then
    echo ""
    echo "============================================================"
    echo "✅ SUCCESS! Production instance configured"
    echo "============================================================"
    echo ""
    echo "You can now connect with:"
    echo "  ssh $HOST_ALIAS"
    echo ""
    echo "Or use in scripts:"
    echo "  python monitor_production.py --host $HOST_ALIAS"
    echo "  scp $HOST_ALIAS:~/atari-rl-dashboard/saved_models/*.pt ."
    echo ""
    echo "Next steps:"
    echo "  1. Copy setup script to instance:"
    echo "     scp setup_production.sh $HOST_ALIAS:~/"
    echo ""
    echo "  2. SSH into instance and run setup:"
    echo "     ssh $HOST_ALIAS"
    echo "     bash setup_production.sh"
    echo ""
    echo "  3. Launch training:"
    echo "     bash launch_production_training.sh"
    echo ""
    echo "  4. Monitor from your local machine:"
    echo "     python monitor_production.py --host $HOST_ALIAS --watch"
    echo ""
else
    echo ""
    echo "⚠️  Warning: Could not connect to $HOST_ALIAS"
    echo ""
    echo "This might be normal if:"
    echo "  - Instance is still starting up"
    echo "  - Firewall rules need to be configured"
    echo "  - SSH key permissions are wrong"
    echo ""
    echo "Config has been added to: $SSH_CONFIG"
    echo ""
    echo "Try connecting manually:"
    echo "  ssh $HOST_ALIAS"
    echo ""
    echo "If connection fails, verify:"
    echo "  - Instance is running"
    echo "  - IP and port are correct"
    echo "  - SSH key has correct permissions: chmod 600 $KEY_PATH"
    echo ""
fi

echo "============================================================"

