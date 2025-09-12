# Raspberry Pi Migration Guide for T8 Delay App

This guide provides step-by-step instructions for migrating the T8 Delay App with the new school calendar system to a Raspberry Pi.

## 📋 Prerequisites

### Hardware Requirements
- **Raspberry Pi 4** (recommended) or **Raspberry Pi 3B+**
- **MicroSD Card**: 32GB+ (Class 10 or better)
- **Power Supply**: 5V 3A USB-C (for Pi 4) or 5V 2.5A micro-USB (for Pi 3B+)
- **Network**: Ethernet cable or WiFi connection
- **Storage**: Optional external USB drive for database backups

### Software Requirements
- **Raspberry Pi OS** (64-bit recommended)
- **Python 3.8+**
- **PostgreSQL 12+**
- **Git**

## 🚀 Step 1: Initial Raspberry Pi Setup

### 1.1 Flash Raspberry Pi OS
```bash
# Download Raspberry Pi Imager from https://www.raspberrypi.org/downloads/
# Flash Raspberry Pi OS Lite (64-bit) to microSD card
# Enable SSH and set hostname during imaging process
```

### 1.2 Initial System Setup
```bash
# SSH into your Raspberry Pi
ssh pi@<raspberry-pi-ip>

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y git curl wget vim htop

# Set timezone
sudo timedatectl set-timezone Australia/Sydney

# Enable SSH (if not already enabled)
sudo systemctl enable ssh
sudo systemctl start ssh
```

## 🐍 Step 2: Python Environment Setup

### 2.1 Install Python and pip
```bash
# Install Python 3.9+ and pip
sudo apt install -y python3 python3-pip python3-venv python3-dev

# Install build tools for compiling packages
sudo apt install -y build-essential libffi-dev libssl-dev

# Verify Python version
python3 --version  # Should be 3.8+
```

### 2.2 Create Virtual Environment
```bash
# Create project directory
mkdir -p ~/t8-delay-app
cd ~/t8-delay-app

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## 🐘 Step 3: PostgreSQL Installation and Setup

### 3.1 Install PostgreSQL
```bash
# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib postgresql-client

# Start and enable PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Check PostgreSQL status
sudo systemctl status postgresql
```

### 3.2 Configure PostgreSQL
```bash
# Switch to postgres user
sudo -u postgres psql

# Create database and user
CREATE DATABASE aidb;
CREATE USER aiuser WITH PASSWORD 'aipass';
GRANT ALL PRIVILEGES ON DATABASE aidb TO aiuser;

# Grant additional permissions
\c aidb
GRANT ALL ON SCHEMA public TO aiuser;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aiuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aiuser;

# Exit PostgreSQL
\q
```

### 3.3 Configure PostgreSQL for Remote Access (Optional)
```bash
# Edit PostgreSQL configuration
sudo nano /etc/postgresql/*/main/postgresql.conf

# Uncomment and modify:
listen_addresses = '*'

# Edit pg_hba.conf for authentication
sudo nano /etc/postgresql/*/main/pg_hba.conf

# Add line for your network (replace with actual IP range):
host    all             all             192.168.1.0/24         md5

# Restart PostgreSQL
sudo systemctl restart postgresql
```

## 📦 Step 4: Application Deployment

### 4.1 Clone Repository
```bash
# Navigate to project directory
cd ~/t8-delay-app

# Clone the repository
git clone https://github.com/yourusername/T8DelayApp.git .

# Or if you have the files locally, copy them:
# scp -r /path/to/T8DelayApp/* pi@<raspberry-pi-ip>:~/t8-delay-app/
```

### 4.2 Install Python Dependencies
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install system dependencies for psycopg2
sudo apt install -y libpq-dev

# Install Python packages
pip install -r requirements.txt

# Verify installation
pip list
```

### 4.3 Database Schema Setup
```bash
# Run database migration/setup
python3 school_calendar_generator.py --setup-database

# Verify database tables
python3 -c "
from database.operations import SchoolCalendarOperations
ops = SchoolCalendarOperations()
print('Database connection test:', ops.test_connection())
"
```

## ⚙️ Step 5: Configuration and Environment

### 5.1 Create Environment Configuration
```bash
# Create environment file
nano ~/t8-delay-app/.env

# Add the following content:
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=aidb
DATABASE_USER=aiuser
DATABASE_PASSWORD=aipass
LOG_LEVEL=INFO
CACHE_SIZE=1000
FALLBACK_STRATEGY=hybrid
```

### 5.2 Configure Logging
```bash
# Create logs directory
mkdir -p ~/t8-delay-app/logs

# Set permissions
chmod 755 ~/t8-delay-app/logs
```

### 5.3 Test Database Connection
```bash
# Test database connectivity
python3 -c "
import sys
sys.path.append('.')
from database.connection import DatabaseConnectionManager

# Test connection
conn_mgr = DatabaseConnectionManager()
print('Database connection test:', conn_mgr.test_connection())
"
```

## 📅 Step 6: Calendar Data Population

### 6.1 Generate Calendar Data
```bash
# Generate calendar data for current year
python3 school_calendar_generator.py --year 2025 --dry-run

# If dry run looks good, generate actual data
python3 school_calendar_generator.py --year 2025

# Verify data was created
python3 school_calendar_admin.py stats
```

### 6.2 Validate System
```bash
# Run comprehensive health check
python3 school_calendar_admin.py health-monitor check

# Test school day lookup
python3 school_calendar_admin.py test-lookup --date 2025-01-15

# Check performance metrics
python3 school_calendar_admin.py performance stats
```

## 🔄 Step 7: Service Configuration

### 7.1 Create Systemd Service
```bash
# Create service file
sudo nano /etc/systemd/system/t8-delay-monitor.service

# Add the following content:
[Unit]
Description=T8 Delay Monitor Service
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/pi/t8-delay-app
Environment=PATH=/home/pi/t8-delay-app/venv/bin
ExecStart=/home/pi/t8-delay-app/venv/bin/python monitor_t8_delays_polling.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 7.2 Enable and Start Service
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable t8-delay-monitor.service

# Start service
sudo systemctl start t8-delay-monitor.service

# Check service status
sudo systemctl status t8-delay-monitor.service

# View logs
sudo journalctl -u t8-delay-monitor.service -f
```

## 🔧 Step 8: Monitoring and Maintenance

### 8.1 Health Monitoring Setup
```bash
# Create health monitoring script
nano ~/t8-delay-app/health_check.sh

# Add content:
#!/bin/bash
cd /home/pi/t8-delay-app
source venv/bin/activate
python3 school_calendar_admin.py health-monitor check
```

```bash
# Make executable
chmod +x ~/t8-delay-app/health_check.sh

# Add to crontab for regular health checks
crontab -e

# Add line for hourly health checks:
0 * * * * /home/pi/t8-delay-app/health_check.sh >> /home/pi/t8-delay-app/logs/health_check.log 2>&1
```

### 8.2 Log Rotation Setup
```bash
# Create logrotate configuration
sudo nano /etc/logrotate.d/t8-delay-app

# Add content:
/home/pi/t8-delay-app/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 pi pi
    postrotate
        systemctl reload t8-delay-monitor.service
    endscript
}
```

## 🔒 Step 9: Security Configuration

### 9.1 Firewall Setup
```bash
# Install UFW
sudo apt install -y ufw

# Configure firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 5432/tcp  # PostgreSQL (if needed for remote access)

# Enable firewall
sudo ufw enable
```

### 9.2 SSH Security
```bash
# Edit SSH configuration
sudo nano /etc/ssh/sshd_config

# Recommended settings:
PermitRootLogin no
PasswordAuthentication yes  # Change to no if using key-based auth
PubkeyAuthentication yes
```

### 9.3 Database Security
```bash
# Change default postgres password
sudo -u postgres psql
ALTER USER postgres PASSWORD 'your_secure_password';
\q
```

## 📊 Step 10: Performance Optimization

### 10.1 Raspberry Pi Specific Optimizations
```bash
# Edit boot configuration
sudo nano /boot/config.txt

# Add performance optimizations:
# GPU memory split (adjust based on needs)
gpu_mem=16

# Overclock settings (optional, use with caution)
# arm_freq=1800
# over_voltage=2

# Enable I2C/SPI if needed
dtparam=i2c_arm=on
dtparam=spi=on
```

### 10.2 Database Optimization
```bash
# Edit PostgreSQL configuration
sudo nano /etc/postgresql/*/main/postgresql.conf

# Optimize for Raspberry Pi:
shared_buffers = 128MB
effective_cache_size = 256MB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 4MB
default_statistics_target = 100

# Restart PostgreSQL
sudo systemctl restart postgresql
```

## 🚨 Step 11: Troubleshooting

### 11.1 Common Issues

**Database Connection Issues:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check database connectivity
sudo -u postgres psql -c "SELECT version();"

# Test application database connection
cd ~/t8-delay-app
source venv/bin/activate
python3 -c "from database.connection import DatabaseConnectionManager; print(DatabaseConnectionManager().test_connection())"
```

**Service Issues:**
```bash
# Check service logs
sudo journalctl -u t8-delay-monitor.service -n 50

# Restart service
sudo systemctl restart t8-delay-monitor.service

# Check service status
sudo systemctl status t8-delay-monitor.service
```

**Memory Issues:**
```bash
# Check memory usage
free -h
htop

# Check swap usage
swapon -s

# Add swap if needed
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### 11.2 Log Analysis
```bash
# View application logs
tail -f ~/t8-delay-app/logs/t8_monitor.log

# View system logs
sudo journalctl -u t8-delay-monitor.service -f

# Check database logs
sudo tail -f /var/log/postgresql/postgresql-*.log
```

## 🔄 Step 12: Backup and Recovery

### 12.1 Database Backup
```bash
# Create backup script
nano ~/t8-delay-app/backup_db.sh

# Add content:
#!/bin/bash
BACKUP_DIR="/home/pi/t8-delay-app/backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

# Backup database
pg_dump -h localhost -U aiuser -d aidb > $BACKUP_DIR/t8_delay_app_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/t8_delay_app_$DATE.sql

# Keep only last 7 days of backups
find $BACKUP_DIR -name "t8_delay_app_*.sql.gz" -mtime +7 -delete

echo "Database backup completed: t8_delay_app_$DATE.sql.gz"
```

```bash
# Make executable
chmod +x ~/t8-delay-app/backup_db.sh

# Add to crontab for daily backups
crontab -e
# Add line:
0 2 * * * /home/pi/t8-delay-app/backup_db.sh >> /home/pi/t8-delay-app/logs/backup.log 2>&1
```

### 12.2 Application Backup
```bash
# Create application backup script
nano ~/t8-delay-app/backup_app.sh

# Add content:
#!/bin/bash
BACKUP_DIR="/home/pi/t8-delay-app/backups"
DATE=$(date +%Y%m%d_%H%M%S)
APP_DIR="/home/pi/t8-delay-app"

# Create application backup (excluding logs and cache)
tar -czf $BACKUP_DIR/t8_delay_app_files_$DATE.tar.gz \
    --exclude='logs' \
    --exclude='__pycache__' \
    --exclude='venv' \
    --exclude='backups' \
    -C $APP_DIR .

echo "Application backup completed: t8_delay_app_files_$DATE.tar.gz"
```

## 📈 Step 13: Monitoring and Alerts

### 13.1 System Monitoring
```bash
# Install monitoring tools
sudo apt install -y htop iotop nethogs

# Create system monitoring script
nano ~/t8-delay-app/system_monitor.sh

# Add content:
#!/bin/bash
echo "=== System Status $(date) ===" >> ~/t8-delay-app/logs/system_monitor.log
echo "CPU Usage:" >> ~/t8-delay-app/logs/system_monitor.log
top -bn1 | grep "Cpu(s)" >> ~/t8-delay-app/logs/system_monitor.log
echo "Memory Usage:" >> ~/t8-delay-app/logs/system_monitor.log
free -h >> ~/t8-delay-app/logs/system_monitor.log
echo "Disk Usage:" >> ~/t8-delay-app/logs/system_monitor.log
df -h >> ~/t8-delay-app/logs/system_monitor.log
echo "Service Status:" >> ~/t8-delay-app/logs/system_monitor.log
systemctl is-active t8-delay-monitor.service >> ~/t8-delay-app/logs/system_monitor.log
echo "================================" >> ~/t8-delay-app/logs/system_monitor.log
```

### 13.2 Email Alerts (Optional)
```bash
# Install mail utilities
sudo apt install -y mailutils

# Configure email (if needed)
# Edit /etc/postfix/main.cf for email configuration
```

## ✅ Step 14: Verification and Testing

### 14.1 Complete System Test
```bash
# Run comprehensive system test
cd ~/t8-delay-app
source venv/bin/activate

# Test database connectivity
python3 school_calendar_admin.py stats

# Test health monitoring
python3 school_calendar_admin.py health-monitor check

# Test performance monitoring
python3 school_calendar_admin.py performance stats

# Test error recovery
python3 school_calendar_admin.py error-recovery stats

# Test school day lookup
python3 school_calendar_admin.py test-lookup --date 2025-01-15

# Test automation system
python3 school_calendar_admin.py automation status
```

### 14.2 Service Verification
```bash
# Check all services
sudo systemctl status t8-delay-monitor.service
sudo systemctl status postgresql

# Verify logs
tail -f ~/t8-delay-app/logs/t8_monitor.log

# Test service restart
sudo systemctl restart t8-delay-monitor.service
sudo systemctl status t8-delay-monitor.service
```

## 🎯 Step 15: Final Configuration

### 15.1 Set Correct Permissions
```bash
# Set ownership
sudo chown -R pi:pi ~/t8-delay-app

# Set permissions
chmod +x ~/t8-delay-app/*.py
chmod +x ~/t8-delay-app/*.sh
chmod 755 ~/t8-delay-app/logs
```

### 15.2 Create Management Scripts
```bash
# Create start script
nano ~/t8-delay-app/start.sh

# Add content:
#!/bin/bash
cd /home/pi/t8-delay-app
source venv/bin/activate
sudo systemctl start t8-delay-monitor.service
echo "T8 Delay Monitor started"

# Create stop script
nano ~/t8-delay-app/stop.sh

# Add content:
#!/bin/bash
sudo systemctl stop t8-delay-monitor.service
echo "T8 Delay Monitor stopped"

# Create restart script
nano ~/t8-delay-app/restart.sh

# Add content:
#!/bin/bash
sudo systemctl restart t8-delay-monitor.service
echo "T8 Delay Monitor restarted"

# Make executable
chmod +x ~/t8-delay-app/*.sh
```

## 📋 Migration Checklist

- [ ] Raspberry Pi OS installed and updated
- [ ] Python 3.8+ installed with virtual environment
- [ ] PostgreSQL installed and configured
- [ ] Application code deployed
- [ ] Dependencies installed
- [ ] Database schema created
- [ ] Calendar data populated
- [ ] Systemd service configured
- [ ] Health monitoring setup
- [ ] Backup scripts configured
- [ ] Security settings applied
- [ ] Performance optimizations applied
- [ ] System tested and verified
- [ ] Monitoring and alerts configured

## 🆘 Support and Maintenance

### Regular Maintenance Tasks
1. **Daily**: Check service status and logs
2. **Weekly**: Review health monitoring reports
3. **Monthly**: Update system packages
4. **Quarterly**: Review and test backup procedures

### Emergency Procedures
1. **Service Down**: `sudo systemctl restart t8-delay-monitor.service`
2. **Database Issues**: Check PostgreSQL status and logs
3. **Memory Issues**: Check swap usage and system resources
4. **Network Issues**: Verify connectivity and firewall settings

### Contact Information
- **System Logs**: `~/t8-delay-app/logs/`
- **Service Logs**: `sudo journalctl -u t8-delay-monitor.service`
- **Database Logs**: `/var/log/postgresql/`

---

## 🎉 Migration Complete!

Your T8 Delay App with the advanced school calendar system is now successfully running on Raspberry Pi with:

- ✅ **High-Performance Database**: PostgreSQL with optimized settings
- ✅ **Intelligent Caching**: Sub-1ms lookup performance
- ✅ **Automatic Recovery**: Circuit breakers and error handling
- ✅ **Health Monitoring**: Comprehensive system health checks
- ✅ **Performance Monitoring**: Real-time metrics and optimization
- ✅ **Automated Maintenance**: Self-healing and data management
- ✅ **Production Ready**: Service management and monitoring

The system will automatically handle school day lookups, maintain calendar data, and provide reliable service for the T8 delay monitoring application.