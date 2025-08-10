# T8 Airport Line Delay Monitor

A Python script that monitors the Sydney Trains T8 Airport Line Twitter account for delay and service disruption announcements, sending real-time alerts to Telegram during school days and peak travel times.

## Features

- 🚆 **Smart Monitoring**: Only monitors during school days and peak travel times (7:00-8:45 AM and 1:00-4:00 PM AEST)
- 📱 **Telegram Alerts**: Sends formatted notifications to your Telegram chat
- 🔒 **Secure**: Uses environment variables for API credentials
- 📊 **Logging**: Comprehensive logging to file and console
- ⚡ **Real-time**: Uses Twitter polling API for reliable notifications
- 🛠️ **Easy Setup**: Interactive setup and testing scripts
- 🍓 **Raspberry Pi Ready**: Complete deployment guide for 24/7 monitoring
- 🚨 **Health Monitoring**: Automatic Telegram alerts when the monitor itself has issues
- 💓 **Self-Healing**: Auto-restart failed services with recovery notifications
- 📈 **Quota Optimized**: Configurable polling intervals to manage API usage within limits

## Prerequisites

1. **Twitter API Access** (Choose one):
   - **TwitterAPI.io** (Recommended): Simple signup at [twitterapi.io](https://twitterapi.io/) - $0.15 per 1,000 tweets
   - **X API**: Traditional developer account from [developer.twitter.com](https://developer.twitter.com/)
2. **Telegram Bot**: Create a bot using [@BotFather](https://t.me/BotFather) on Telegram
3. **Python 3.7+**: Required for async/await support

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Calculate optimal polling interval**:
   ```bash
   python quota_calculator.py
   ```

3. **Set up credentials** (interactive):
   ```bash
   python setup_env_simple.py
   ```

4. **Test your setup**:
   ```bash
   python test_setup_polling.py
   ```

5. **Test health monitoring**:
   ```bash
   python test_health_monitor.py
   ```

6. **Start monitoring**:
   ```bash
   python monitor_t8_delays_polling.py
   ```

## 📊 Quota Management

The script now includes **intelligent quota management** to work within Twitter API limits.

### **Default Settings (100 calls/month)**
- **Polling interval**: 60 minutes
- **Monthly usage**: ~95 API calls
- **Perfect for free Twitter API accounts** ✅

### **Quota Calculator**
Use the built-in calculator to find your optimal interval:

```bash
python quota_calculator.py
```

**Sample output:**
```
📊 T8 Monitor API Quota Calculator
============================================================

Quota Usage by Polling Interval:
------------------------------------------------------------
Interval     Calls/Day    Calls/Month     Status
------------------------------------------------------------
 2 minutes      142.5          2850      ❌ Exceeds quota
15 minutes       19.0           380      🔶 High usage
30 minutes        9.5           190      ⚠️  Moderate usage
60 minutes        4.8            95      ✅ Within quota
90 minutes        3.2            63      ✅ Within quota
```

### **Custom Polling Intervals**

**Option 1: Environment Variable**
```bash
export POLLING_INTERVAL_MINUTES=30
python monitor_t8_delays_polling.py
```

**Option 2: .env File**
Add to your `.env` file:
```
POLLING_INTERVAL_MINUTES=60
```

**Option 3: Interactive Setup**
The setup script will ask for your preferred interval.

### **Recommended Intervals by Quota**

| API Quota | Recommended Interval | Monthly Usage | Status |
|-----------|---------------------|---------------|---------|
| 100 calls | 60 minutes | 95 calls | ✅ Safe |
| 500 calls | 15 minutes | 380 calls | ✅ Good |
| 1,000 calls | 8 minutes | 713 calls | ✅ Optimal |
| 10,000+ calls | 2 minutes | 2,850 calls | ⚡ Real-time |

## Detailed Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Credentials

**Option A: Interactive Setup (Recommended)**
```bash
python setup_env_simple.py
```

**Option B: Manual Environment Variables**

For **TwitterAPI.io** (Recommended):
```bash
# TwitterAPI.io Configuration
export TWITTERAPI_IO_KEY="your_twitterapi_io_key_here"
export USE_TWITTERAPI_IO=true

# Telegram Bot Credentials
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token_here"
export TELEGRAM_CHAT_ID="your_telegram_chat_id_here"

# Optional: Custom polling interval (default: 60 minutes)
export POLLING_INTERVAL_MINUTES=60
```

For **X API** (Traditional):
```bash
# X API Configuration
export X_BEARER_TOKEN="your_twitter_bearer_token_here"
export USE_TWITTERAPI_IO=false

# Telegram Bot Credentials
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token_here"
export TELEGRAM_CHAT_ID="your_telegram_chat_id_here"

# Optional: Custom polling interval (default: 60 minutes)
export POLLING_INTERVAL_MINUTES=60
```

**Option C: .env File**

For **TwitterAPI.io** (Recommended):
```
# TwitterAPI.io Configuration
TWITTERAPI_IO_KEY=your_twitterapi_io_key_here
USE_TWITTERAPI_IO=true

# Telegram Bot Credentials
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Optional Configuration
POLLING_INTERVAL_MINUTES=60
OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434
```

For **X API** (Traditional):
```
# X API Configuration
X_BEARER_TOKEN=your_twitter_bearer_token_here
USE_TWITTERAPI_IO=false

# Telegram Bot Credentials
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Optional Configuration
POLLING_INTERVAL_MINUTES=60
OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434
```

## Getting API Credentials

### TwitterAPI.io Setup (Recommended)
1. Go to [twitterapi.io](https://twitterapi.io/)
2. Sign up for an account (no approval process required)
3. Navigate to your dashboard
4. Copy your **API Key** (starts with your account info)
5. **Cost**: $0.15 per 1,000 tweets (very affordable for monitoring)

### X API Setup (Traditional)
1. Go to [developer.twitter.com](https://developer.twitter.com/)
2. Apply for developer account (approval required)
3. Create a new app
4. Generate API keys and tokens
5. Copy the **Bearer Token** (this is all you need for the polling version)
6. **Note**: May have usage limits and higher costs

### Telegram Bot Setup
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Create a new bot with `/newbot`
3. Get your bot token
4. Get your chat ID by messaging [@userinfobot](https://t.me/userinfobot)

## Environment Variables Reference

### Required Variables

**For TwitterAPI.io (Recommended):**
- `TWITTERAPI_IO_KEY`: Your API key from twitterapi.io dashboard
- `USE_TWITTERAPI_IO`: Set to `true` to enable TwitterAPI.io
- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token from @BotFather
- `TELEGRAM_CHAT_ID`: Your Telegram chat ID from @userinfobot

**For X API (Traditional):**
- `X_BEARER_TOKEN`: Your Twitter Bearer Token from developer.twitter.com
- `USE_TWITTERAPI_IO`: Set to `false` to use X API
- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token from @BotFather
- `TELEGRAM_CHAT_ID`: Your Telegram chat ID from @userinfobot

### Optional Variables

- `POLLING_INTERVAL_MINUTES`: Polling interval in minutes (default: 60)
- `OLLAMA_MODEL`: AI model for tweet analysis (default: llama3.2:3b)
- `OLLAMA_HOST`: Ollama server URL (default: http://localhost:11434)

### Migration Variables

During the migration from X API to TwitterAPI.io, you can have both sets of credentials:
```bash
# Current X API (will be phased out)
X_BEARER_TOKEN=your_x_token_here

# New TwitterAPI.io (migration target)
TWITTERAPI_IO_KEY=your_twitterapi_io_key_here
USE_TWITTERAPI_IO=true  # Set to true to switch to TwitterAPI.io
```

## Testing Your Setup

Before running the monitor, test your configuration:

```bash
# Calculate optimal polling interval
python quota_calculator.py

# Test basic setup
python test_setup_polling.py

# Test health monitoring system
python test_health_monitor.py
```

This will:
- ✅ Calculate optimal polling intervals for your quota
- ✅ Check all environment variables are set
- ✅ Test Twitter API connection
- ✅ Test Telegram bot connection and send a test message
- ✅ Verify time/date logic
- ✅ Test health monitoring alerts

## Usage

Run the monitoring script:

```bash
python monitor_t8_delays_polling.py
```

**Sample startup output:**
```
2025-05-28 18:32:16 - INFO - 📊 Polling interval: 60 minutes (3600 seconds)
2025-05-28 18:32:16 - INFO - 💚 Quota-friendly mode: ~95 API calls per month (within 100 limit)
2025-05-28 18:32:16 - INFO - Starting T8 Delays Monitor (Quota-Optimized Polling Mode)...
2025-05-28 18:32:16 - INFO - Telegram bot connected: YourBotName
2025-05-28 18:32:16 - INFO - Twitter API connected, found @T8SydneyTrains (ID: 123456789)
2025-05-28 18:32:16 - INFO - All connections successful. Starting monitoring...
2025-05-28 18:32:16 - INFO - Monitoring mode: Polling every 60 minutes during school days and peak hours
```

The script will:
1. Test the Telegram connection
2. Connect to Twitter API using Bearer Token
3. Poll for new tweets at your configured interval during monitoring hours
4. Send alerts for T8/airport-related delays
5. Log heartbeat messages periodically
6. Send critical error alerts if the script crashes

## 🍓 Raspberry Pi Deployment

For 24/7 monitoring, deploy on a Raspberry Pi with automatic startup and monitoring.

### Initial Pi Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install python3 python3-pip python3-venv git build-essential -y

# Create project directory
mkdir ~/t8-monitor
cd ~/t8-monitor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install tweepy python-telegram-bot python-dateutil
```

### Transfer Files to Pi

**Option A: Using SCP (from your computer):**
```bash
# Replace PI_IP with your Pi's IP address
scp monitor_t8_delays_polling.py pi@PI_IP:~/t8-monitor/
scp health_monitor.py pi@PI_IP:~/t8-monitor/
scp quota_calculator.py pi@PI_IP:~/t8-monitor/
scp manage.sh pi@PI_IP:~/t8-monitor/
scp .env pi@PI_IP:~/t8-monitor/
scp requirements.txt pi@PI_IP:~/t8-monitor/
```

**Option B: Using Git:**
```bash
git clone https://github.com/yourusername/T8DelayApp.git
cd T8DelayApp
```

**Option C: Manual Creation:**
Use `nano` to create files directly on the Pi.

### Create Systemd Service

**Exit the virtual environment first:**
```bash
deactivate
```

**Create the service file:**
```bash
sudo nano /etc/systemd/system/t8-monitor.service
```

**Add this content:**
```ini
[Unit]
Description=T8 Airport Line Delay Monitor
After=network.target
Wants=network-online.target

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
```

**Enable and start the service:**
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable t8-monitor.service

# Start the service
sudo systemctl start t8-monitor.service

# Check status
sudo systemctl status t8-monitor.service
```

### Enhanced Management & Health Monitoring

**Make management script executable:**
```bash
chmod +x manage.sh
```

**Install enhanced health monitoring:**
```bash
./manage.sh install-health
```

This will:
- Set up automated health checks every 15 minutes
- Configure Telegram alerts for system issues
- Test the health monitoring system

### Management Commands

```bash
# Service Management
./manage.sh start          # Start the service
./manage.sh stop           # Stop the service
./manage.sh restart        # Restart the service
./manage.sh status         # Show service status

# Monitoring & Logs
./manage.sh logs           # View live service logs
./manage.sh app-logs       # View live application logs
./manage.sh health         # Run health check manually
./manage.sh health-logs    # View health monitor logs
./manage.sh summary        # Show system summary

# Testing & Alerts
./manage.sh test-alert     # Send test Telegram alert
```

### Health Monitoring Features

The enhanced health monitoring system automatically sends Telegram alerts for:

- 🔴 **Service Failures**: When the T8 monitor service stops running
- 📝 **Log Inactivity**: When no log activity for 30+ minutes during monitoring hours
- 💾 **Disk Space Issues**: When disk usage exceeds 90%
- 🌐 **Network Problems**: When internet connectivity is lost
- ✅ **Service Recovery**: When issues are resolved
- 🚨 **Critical Errors**: When the main script crashes

**Smart Features:**
- **Anti-spam**: Won't send alerts more than once every 30 minutes
- **Auto-restart**: Attempts to restart failed services automatically
- **Context-aware**: Different alert thresholds for monitoring vs non-monitoring hours
- **Status tracking**: Remembers previous state to detect changes

### Log Management

**Set up log rotation:**
```bash
sudo nano /etc/logrotate.d/t8-monitor
```

```
/home/pi/t8-monitor/t8_monitor.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 pi pi
}

/home/pi/t8-monitor/health_monitor.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 pi pi
}
```

### Remote Access

**Enable SSH (if not already enabled):**
```bash
sudo systemctl enable ssh
sudo systemctl start ssh
```

**Access remotely:**
```bash
ssh pi@YOUR_PI_IP_ADDRESS
```

## 🚨 Health Monitoring Alerts

You'll receive Telegram alerts for various system issues:

### Alert Types

1. **🔴 Service Down Alert**:
   ```
   🚨 T8 Monitor ALERT

   T8 Monitor has 1 issue(s):
   • 🔴 Service not running
   • 🔧 Service restart attempted
   • ✅ Service restart successful

   Consecutive failures: 1
   ```

2. **✅ Recovery Alert**:
   ```
   ✅ T8 Monitor RECOVERY

   T8 Monitor has recovered and is now 
   running normally. All systems are healthy.
   ```

3. **🚨 Critical Error Alert**:
   ```
   🚨 T8 Monitor CRITICAL ERROR

   The monitoring script has crashed:
   Connection timeout error

   The service will attempt to restart automatically.
   ```

## Monitoring Schedule

The script only sends alerts during:
- **School days**: Monday-Friday during NSW school terms
- **Peak hours**: 7:00-8:45 AM and 1:00-4:00 PM AEST
- **Excludes**: Weekends, public holidays, and school holidays

## Keywords Detected

**T8/Airport keywords**: t8, airport
**Delay keywords**: delay, disruption, cancelled, issue, suspended, stopped, problem

## Logging

Logs are written to:
- `t8_monitor.log` (application logs)
- `health_monitor.log` (health monitoring logs)
- Console output (captured by systemd on Pi)

Log levels:
- INFO: Normal operations and alerts sent
- DEBUG: Detailed processing information
- ERROR: Issues and failures

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `monitor_t8_delays_polling.py` | Main monitoring script with quota optimization |
| `quota_calculator.py` | Interactive quota calculator |
| `health_monitor.py` | Enhanced health monitoring with Telegram alerts |
| `setup_env_simple.py` | Simplified credential setup |
| `test_setup_polling.py` | Configuration testing for polling version |
| `test_health_monitor.py` | Health monitoring system testing |
| `manage.sh` | Enhanced Pi service management script |

## Troubleshooting

### Common Issues

1. **Missing environment variables**:
   ```
   ERROR - Missing required environment variables: X_BEARER_TOKEN, TELEGRAM_BOT_TOKEN
   ```
   **Solution**: Run `python setup_env_simple.py` to configure credentials

2. **Twitter API errors**:
   ```
   ERROR - Error fetching user ID for @T8SydneyTrains: 401 Unauthorized
   ```
   **Solution**: Check your Twitter Bearer Token

3. **Telegram connection failed**:
   ```
   ERROR - Telegram connection test failed: Unauthorized
   ```
   **Solution**: Verify your Telegram bot token

4. **Service not starting on Pi**:
   ```bash
   sudo systemctl status t8-monitor.service
   sudo journalctl -u t8-monitor.service
   ```

5. **Health alerts not working**:
   ```bash
   ./manage.sh test-alert    # Test Telegram alerts
   ./manage.sh health        # Run health check manually
   ```

6. **Quota exceeded**:
   ```bash
   python quota_calculator.py  # Calculate optimal interval
   # Then update POLLING_INTERVAL_MINUTES in .env
   ```

### Testing Commands

```bash
# Calculate optimal polling interval
python quota_calculator.py

# Test credentials (simplified version)
python setup_env_simple.py test

# Full setup test (polling version)
python test_setup_polling.py

# Test health monitoring system
python test_health_monitor.py

# Test Telegram alerts (on Pi)
./manage.sh test-alert

# Check system status (on Pi)
./manage.sh summary

# Check if monitoring would be active now
python -c "from monitor_t8_delays_polling import *; import datetime, dateutil.tz; now = datetime.datetime.now(dateutil.tz.gettz('Australia/Sydney')); print(f'Would monitor: {is_sydney_school_day(now) and is_within_time_window(now)}')"
```

## File Structure

```
├── monitor_t8_delays_polling.py    # Main monitoring script (quota-optimized)
├── quota_calculator.py             # Interactive quota calculator
├── health_monitor.py               # Enhanced health monitoring with alerts
├── setup_env_simple.py             # Simplified credential setup
├── test_setup_polling.py           # Configuration testing (polling version)
├── test_health_monitor.py          # Health monitoring system testing
├── manage.sh                       # Enhanced Pi service management script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .env                            # Environment variables (create this)
├── t8_monitor.log                  # Application log file (created when running)
├── health_monitor.log              # Health monitoring log file
├── monitor_status.json             # Health monitoring status tracking
└── last_tweet_id.txt              # Tracks last processed tweet
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for educational and personal use. 