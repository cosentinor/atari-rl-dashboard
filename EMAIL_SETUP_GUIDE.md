# Email Notification Setup Guide

## 🚨 Current Status

**Email notifications are NOT working** due to Hetzner blocking port 25 (standard anti-spam measure for cloud providers).

Emails are queued but cannot be delivered:
```
Connection timed out to riccardocosentino-com.mail.protection.outlook.com:25
```

## ✅ Solution Options

### Option 1: Gmail SMTP Relay (Recommended)

**Pros:**
- Free
- Reliable
- Easy setup (~5 minutes)
- Works from anywhere

**Setup Steps:**

1. **Create App-Specific Password:**
   - Go to https://myaccount.google.com/apppasswords
   - Sign in to your Gmail account
   - Create new app password (name it "Atari VPS Monitoring")
   - Copy the 16-character password

2. **Run this command** (replace with your details):
```bash
# On your local machine
ssh atari "sudo bash -c 'cat > /etc/postfix/sasl_passwd_gmail << EOF
[smtp.gmail.com]:587 your-email@gmail.com:YOUR-APP-PASSWORD
EOF
'"

# Secure and activate
ssh atari "sudo chmod 600 /etc/postfix/sasl_passwd_gmail && \
  sudo postmap /etc/postfix/sasl_passwd_gmail && \
  sudo postconf -e 'relayhost = [smtp.gmail.com]:587' && \
  sudo postconf -e 'smtp_sasl_auth_enable = yes' && \
  sudo postconf -e 'smtp_sasl_password_maps = hash:/etc/postfix/sasl_passwd_gmail' && \
  sudo postconf -e 'smtp_sasl_security_options = noanonymous' && \
  sudo postconf -e 'smtp_tls_security_level = encrypt' && \
  sudo systemctl restart postfix"

# Test
ssh atari 'echo "Test from Atari VPS" | mail -s "Gmail SMTP Test" riccardo@riccardocosentino.com'
```

3. **Check if email arrived** (should be instant)

---

### Option 2: Office365 SMTP

**Pros:**
- Use existing Office365 account
- Professional sender address
- Already partially configured

**Setup Steps:**

1. **Get Office365 SMTP Credentials:**
   - Email: Your full Office365 email
   - Password: Your Office365 password OR app password
   - Server: smtp.office365.com
   - Port: 587

2. **Configure:**
```bash
# On your local machine
ssh atari "sudo bash -c 'cat > /etc/postfix/sasl_passwd_o365 << EOF
[smtp.office365.com]:587 your-email@riccardocosentino.com:YOUR-PASSWORD
EOF
'"

# Secure and activate
ssh atari "sudo chmod 600 /etc/postfix/sasl_passwd_o365 && \
  sudo postmap /etc/postfix/sasl_passwd_o365 && \
  sudo postconf -e 'relayhost = [smtp.office365.com]:587' && \
  sudo postconf -e 'smtp_sasl_auth_enable = yes' && \
  sudo postconf -e 'smtp_sasl_password_maps = hash:/etc/postfix/sasl_passwd_o365' && \
  sudo postconf -e 'smtp_sasl_security_options = noanonymous' && \
  sudo postconf -e 'smtp_tls_security_level = encrypt' && \
  sudo systemctl restart postfix"

# Test
ssh atari 'echo "Test from Atari VPS" | mail -s "O365 SMTP Test" riccardo@riccardocosentino.com'
```

---

### Option 3: SendGrid (Professional)

**Pros:**
- Professional email delivery service
- 100 emails/day free
- Best deliverability
- Analytics

**Setup:**

1. Sign up at https://sendgrid.com
2. Create API key
3. Configure postfix:

```bash
ssh atari "sudo bash -c 'cat > /etc/postfix/sasl_passwd_sendgrid << EOF
[smtp.sendgrid.net]:587 apikey:YOUR-SENDGRID-API-KEY
EOF
'"

ssh atari "sudo chmod 600 /etc/postfix/sasl_passwd_sendgrid && \
  sudo postmap /etc/postfix/sasl_passwd_sendgrid && \
  sudo postconf -e 'relayhost = [smtp.sendgrid.net]:587' && \
  sudo postconf -e 'smtp_sasl_auth_enable = yes' && \
  sudo postconf -e 'smtp_sasl_password_maps = hash:/etc/postfix/sasl_passwd_sendgrid' && \
  sudo postconf -e 'smtp_sasl_security_options = noanonymous' && \
  sudo postconf -e 'smtp_tls_security_level = encrypt' && \
  sudo systemctl restart postfix"
```

---

## 🧪 Testing

After configuring any option:

```bash
# Send test email
ssh atari 'echo "Test email from Atari VPS monitoring system

If you receive this, email notifications are working!

Time: $(date)
Server: $(hostname)" | mail -s "TEST: Email Working" riccardo@riccardocosentino.com'

# Check queue (should be empty if delivered)
ssh atari "mailq"

# Check logs
ssh atari "sudo tail -30 /var/log/mail.log"
```

**Success indicators:**
- `status=sent` in mail.log
- Email arrives in inbox
- Queue is empty

---

## 🔧 Troubleshooting

### Emails stuck in queue?
```bash
# View queue
ssh atari "mailq"

# Force delivery attempt
ssh atari "sudo postqueue -f"

# Check why
ssh atari "sudo tail -50 /var/log/mail.log | grep 'status='"
```

### Authentication failing?
- Double-check username and password
- For Gmail: Use app-specific password, not regular password
- For O365: May need app password if 2FA enabled

### Still timing out?
- Verify relay host: `ssh atari "sudo postconf relayhost"`
- Check port 587 is open: `ssh atari "telnet smtp.gmail.com 587"`

---

## 📊 What Happens After Setup

Once configured, you'll receive:

**Daily (3 emails):**
- 4:00 AM UTC - Cleanup summary
- 5:00 AM UTC - Backup summary
- 8:00 AM UTC - Status report

**On Issues:**
- Service crashes/restarts
- HTTP endpoint failures  
- High resource usage
- Critical failures

---

## 🎯 Recommended Next Steps

1. **Choose Option 1 (Gmail)** - Fastest and most reliable
2. Create Gmail app password
3. Run the configuration commands
4. Test with the test email command
5. Wait for tomorrow's 8 AM status report

**Estimated time:** 5 minutes

---

## 💡 Alternative: Log-Only Mode

If you don't want to set up email right now, the monitoring still works! All events are logged to:

```bash
/var/log/atari/health.log    # Health checks
/var/log/atari/cleanup.log   # Cleanup activities  
/var/log/atari/backup.log    # Backup status
/var/log/atari/status.log    # Status reports
```

You can check them anytime:
```bash
ssh atari "tail -f /var/log/atari/health.log"
```

The auto-restart functionality works regardless of email!

---

**Need help?** Let me know which option you prefer and I can help configure it.
