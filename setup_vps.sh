#!/bin/bash
# ============================================================
# Hetzner VPS Master Setup Script - Atari RL Dashboard
# ============================================================
# This script automates the complete setup of your server.
# Run as root or with sudo.
# ============================================================

set -e

echo "🚀 Starting Atari RL VPS Automation Setup..."

# 1. Update and install base dependencies
echo "📦 Installing system dependencies..."
apt-get update -qq
# libgl1 is the modern replacement for the deprecated libgl1-mesa-glx
apt-get install -y -qq python3-pip python3-venv git wget libgl1 libglib2.0-0 unattended-upgrades apt-listchanges curl mailutils

# 2. Configure Unattended-Upgrades (Security)
echo "🔒 Configuring automatic security updates..."
dpkg-reconfigure -plow unattended-upgrades # Non-interactive in some envs, but good to have

cat <<EOF > /etc/apt/apt.conf.d/50unattended-upgrades-custom
Unattended-Upgrade::Automatic-Reboot "true";
Unattended-Upgrade::Automatic-Reboot-Time "03:00";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
EOF

# 3. Setup User 'riccardo' if not exists
if ! id "riccardo" &>/dev/null; then
    echo "👤 Creating user 'riccardo'..."
    useradd -m -s /bin/bash riccardo
    usermod -aG sudo riccardo
    echo "riccardo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
fi

# 4. Clone repo and setup venv as 'riccardo'
echo "📥 Setting up application directory..."
sudo -u riccardo -i bash <<'EOF'
cd ~
if [ ! -d "atari-rl-dashboard" ]; then
    git clone https://github.com/cosentinor/atari-rl-dashboard.git
fi
cd atari-rl-dashboard
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
mkdir -p saved_models data
EOF

# 5. Install Systemd Service
echo "⚙️ Installing systemd service..."
cp /home/riccardo/atari-rl-dashboard/deployment/atari.service /etc/systemd/system/
mkdir -p /var/log/atari
chown riccardo:riccardo /var/log/atari
systemctl daemon-reload
systemctl enable atari
systemctl start atari

# 6. Install Logrotate
echo "📝 Configuring log rotation..."
cp /home/riccardo/atari-rl-dashboard/deployment/atari.logrotate /etc/logrotate.d/atari

# 7. Setup Cron Jobs
echo "⏰ Setting up cron jobs..."
CRON_FILE="/tmp/atari_cron"
echo "*/5 * * * * /home/riccardo/atari-rl-dashboard/deployment/health_check.sh >> /var/log/atari/health.log 2>&1" > $CRON_FILE
echo "0 4 * * * /home/riccardo/atari-rl-dashboard/deployment/cleanup.sh >> /var/log/atari/cleanup.log 2>&1" >> $CRON_FILE
echo "0 5 * * * /home/riccardo/atari-rl-dashboard/deployment/backup_models.sh >> /var/log/atari/backup.log 2>&1" >> $CRON_FILE
crontab -u riccardo $CRON_FILE
rm $CRON_FILE

# 8. Set permissions for scripts
chmod +x /home/riccardo/atari-rl-dashboard/deployment/*.sh

echo "============================================================"
echo "✅ SETUP COMPLETE!"
echo "Dashboard should be running at: http://$(curl -s ifconfig.me):5001"
echo "Logs available in: /var/log/atari/"
echo "============================================================"

