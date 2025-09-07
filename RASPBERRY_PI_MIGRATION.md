# 🍓 T8 Delays Monitor - Raspberry Pi Migration Guide

This guide will help you migrate your T8 Delays Monitor from Windows to Raspberry Pi for 24/7 operation.

## 📋 Prerequisites

- **Raspberry Pi 4** (recommended) or Pi 3B+ with at least 2GB RAM
- **MicroSD card** (32GB+ recommended)
- **Raspberry Pi OS** (latest version)
- **Internet connection**
- Your existing **API keys and tokens**

## 🚀 Quick Setup (Automated)

### Option 1: Automated Setup Script

1. **Download and run the setup script:**
   ```bash
   curl -fsSL https://raw.githubusercontent.com/yourusername/T8DelayApp/main/raspberry_pi_setup.sh -o setup.sh
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Copy your Python files:**
   ```bash
   cd ~/t8-monitor
   # Copy your files here (see manual steps below for details)
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   nano .env  # Add your API keys
   ```

4. **Start the service:**
   ```bash
   sudo systemctl enable t8-monitor
   sudo systemctl start t8-monitor
   ```

## 🔧 Manual Setup (Step by Step)

### Step 1: Prepare Raspberry Pi

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3 python3-pip python3-venv git curl build-essential \
    python3-dev libxml2-dev libxslt1-dev zlib1g-dev libffi-dev libssl-dev
```

### Step 2: Create Project Directory

```bash
# Create project directory
mkdir -p ~/t8-monitor
cd ~/t8-monitor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
# Create requirements.txt
cat > requirements.txt << EOF
tweepy>=4.14.0
python-telegram-bot>=20.0
python-dateutil>=2.8.0
ollama
aiohttp>=3.8.0
requests>=2.25.0
holidays>=0.34
icalendar>=5.0.0
beautifulsoup4>=4.9.0
EOF

# Install Python packages
pip install -r requirements.txt
```

### Step 4: Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
sudo systemctl enable ollama
sudo systemctl start ollama

# Download the AI model
ollama pull llama3.2:3b
```

### Step 5: Copy Your Files

**From your Windows machine, copy these files to the Pi:**

```bash
# On Raspberry Pi, in ~/t8-monitor directory:
# Copy these files from your Windows machine:
# - monitor_t8_delays_polling.py
# - sydney_school_day_checker.py

# You can use SCP, SFTP, or USB drive to transfer files
```

**Using SCP from Windows (if you have WSL or Git Bash):**
```bash
scp monitor_t8_delays_polling.py pi@your-pi-ip:~/t8-monitor/
scp sydney_school_day_checker.py pi@your-pi-ip:~/t8-monitor/
```

### Step 6: Configure Environment Variables

```bash
# Create .env file
cat > .env << 'EOF'
# Choose your API backend
USE_TWITTERAPI_IO=true

# TwitterAPI.io (if USE_TWITTERAPI_IO=true)
TWITTERAPI_IO_KEY=your_key_here

# X API (if USE_TWITTERAPI_IO=false)
X_BEARER_TOKEN=your_token_here

# Telegram (required)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Ollama settings
OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434

# Polling interval (minutes)
POLLING_INTERVAL_MINUTES=60

# Debug (optional)
DEBUG=false
EOF

# Edit with your actual values
nano .env
```

### Step 7: Test the Setup

```bash
# Activate virtual environment
cd ~/t8-monitor
source venv/bin/activate

# Test run
python monitor_t8_delays_polling.py
```

**Expected output:**
```
🚀 Starting T8 Delays Monitor with Ollama AI Analysis...
🔍 Starting T8 Monitor Startup Validation...
✅ SchoolDayChecker initialized successfully
✅ All required environment variables present for TwitterAPI.io
🔗 Testing API connections...
✅ TwitterAPI.io connected, found @T8SydneyTrains
All connections tested. Starting monitoring...
```

### Step 8: Create System Service

```bash
# Create systemd service file
sudo tee /etc/systemd/system/t8-monitor.service > /dev/null << EOF
[Unit]
Description=T8 Delays Monitor
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/t8-monitor
Environment=PATH=/home/pi/t8-monitor/venv/bin
ExecStart=/home/pi/t8-monitor/venv/bin/python monitor_t8_delays_polling.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable t8-monitor
sudo systemctl start t8-monitor
```

### Step 9: Setup Log Rotation

```bash
# Create log rotation config
sudo tee /etc/logrotate.d/t8-monitor > /dev/null << EOF
/home/pi/t8-monitor/t8_monitor.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 pi pi
}
EOF
```

## 📊 Monitoring and Management

### Check Service Status
```bash
# Check if service is running
sudo systemctl status t8-monitor

# View live logs
journalctl -u t8-monitor -f

# View recent logs
journalctl -u t8-monitor -n 50
```

### Service Management
```bash
# Start service
sudo systemctl start t8-monitor

# Stop service
sudo systemctl stop t8-monitor

# Restart service
sudo systemctl restart t8-monitor

# Disable auto-start
sudo systemctl disable t8-monitor
```

### Log Files
```bash
# View application logs
tail -f ~/t8-monitor/t8_monitor.log

# View system logs
journalctl -u t8-monitor -f
```

## 🔧 Troubleshooting

### Common Issues

1. **Permission errors:**
   ```bash
   # Fix file permissions
   chmod +x ~/t8-monitor/monitor_t8_delays_polling.py
   chown -R pi:pi ~/t8-monitor
   ```

2. **Virtual environment issues:**
   ```bash
   # Recreate virtual environment
   rm -rf ~/t8-monitor/venv
   cd ~/t8-monitor
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Ollama not responding:**
   ```bash
   # Restart Ollama
   sudo systemctl restart ollama
   
   # Check Ollama status
   sudo systemctl status ollama
   
   # Test Ollama directly
   ollama list
   ```

4. **Network connectivity issues:**
   ```bash
   # Test internet connection
   ping -c 4 google.com
   
   # Test specific APIs
   curl -I https://api.twitterapi.io
   ```

### Performance Optimization

1. **For Pi 3B+ or limited RAM:**
   ```bash
   # Use smaller Ollama model
   ollama pull llama3.2:1b
   
   # Update .env file
   echo "OLLAMA_MODEL=llama3.2:1b" >> .env
   ```

2. **Reduce polling frequency:**
   ```bash
   # Edit .env file
   nano .env
   # Set POLLING_INTERVAL_MINUTES=120  # Check every 2 hours
   ```

## 🔄 Updates and Maintenance

### Updating the Monitor
```bash
# Stop service
sudo systemctl stop t8-monitor

# Backup current version
cp ~/t8-monitor/monitor_t8_delays_polling.py ~/t8-monitor/monitor_t8_delays_polling.py.backup

# Copy new files
# ... copy updated files ...

# Restart service
sudo systemctl start t8-monitor
```

### System Maintenance
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python packages
cd ~/t8-monitor
source venv/bin/activate
pip install --upgrade -r requirements.txt

# Restart service after updates
sudo systemctl restart t8-monitor
```

## 📈 Performance Monitoring

### Resource Usage
```bash
# Check CPU and memory usage
htop

# Check disk space
df -h

# Check service resource usage
systemctl status t8-monitor
```

### Log Analysis
```bash
# Check for errors in logs
journalctl -u t8-monitor | grep -i error

# Monitor API call frequency
grep "API calls per month" ~/t8-monitor/t8_monitor.log
```

## 🎯 Benefits of Raspberry Pi Deployment

- **24/7 Operation:** Always-on monitoring
- **Low Power:** ~3-5W power consumption
- **Reliable:** Automatic restarts and error recovery
- **Cost-effective:** One-time hardware cost
- **Remote Access:** SSH access for maintenance
- **Automatic Updates:** Can be configured for automatic updates

## 🆘 Support

If you encounter issues:

1. Check the logs: `journalctl -u t8-monitor -f`
2. Verify configuration: `cat ~/t8-monitor/.env`
3. Test connectivity: Run the monitor manually
4. Check system resources: `htop`, `df -h`

Your T8 monitor is now running 24/7 on Raspberry Pi! 🎉
