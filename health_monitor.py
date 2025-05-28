#!/usr/bin/env python3
"""
Enhanced health monitor that sends Telegram alerts when the T8 monitor has issues.
"""

import subprocess
import sys
import os
import asyncio
from datetime import datetime, timedelta
from telegram import Bot
import json

# Load environment variables
def load_env_file():
    env_paths = ['.env', '/home/pi/t8-monitor/.env']
    for env_path in env_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key] = value
                return True
            except Exception as e:
                print(f"Warning: Error loading .env file {env_path}: {e}")
    return False

load_env_file()

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
STATUS_FILE = 'monitor_status.json'
LOG_FILE = 't8_monitor.log'
HEALTH_LOG = 'health_monitor.log'

class HealthMonitor:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
        self.status_data = self.load_status()
    
    def load_status(self):
        """Load previous status data."""
        try:
            if os.path.exists(STATUS_FILE):
                with open(STATUS_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.log(f"Error loading status file: {e}")
        
        return {
            'last_alert_time': None,
            'service_was_running': True,
            'last_log_activity': None,
            'consecutive_failures': 0,
            'last_recovery_alert': None
        }
    
    def save_status(self):
        """Save current status data."""
        try:
            with open(STATUS_FILE, 'w') as f:
                json.dump(self.status_data, f, indent=2, default=str)
        except Exception as e:
            self.log(f"Error saving status file: {e}")
    
    def log(self, message):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"{timestamp} - {message}"
        print(log_message)
        
        try:
            with open(HEALTH_LOG, 'a') as f:
                f.write(log_message + '\n')
        except Exception as e:
            print(f"Error writing to health log: {e}")
    
    def check_service_status(self):
        """Check if the systemd service is running."""
        try:
            result = subprocess.run(['systemctl', 'is-active', 't8-monitor.service'], 
                                  capture_output=True, text=True)
            return result.stdout.strip() == 'active'
        except Exception as e:
            self.log(f"Error checking service status: {e}")
            return False
    
    def check_log_activity(self):
        """Check if log file has recent activity."""
        try:
            if not os.path.exists(LOG_FILE):
                return False, "Log file doesn't exist"
            
            import time
            file_time = os.path.getmtime(LOG_FILE)
            current_time = time.time()
            minutes_since_update = (current_time - file_time) / 60
            
            # Consider stale if no activity for more than 30 minutes during monitoring hours
            # or 12 hours outside monitoring hours
            max_minutes = 30 if self.is_monitoring_time() else 720  # 12 hours
            
            is_recent = minutes_since_update < max_minutes
            status_msg = f"Last activity: {minutes_since_update:.1f} minutes ago"
            
            return is_recent, status_msg
            
        except Exception as e:
            return False, f"Error checking log: {e}"
    
    def is_monitoring_time(self):
        """Check if we're currently in monitoring hours."""
        try:
            # Import the time checking function from the main script
            import sys
            sys.path.append('.')
            from monitor_t8_delays_polling import is_sydney_school_day, is_within_time_window
            import dateutil.tz
            
            now = datetime.now(dateutil.tz.gettz('Australia/Sydney'))
            return is_sydney_school_day(now) and is_within_time_window(now)
        except Exception as e:
            self.log(f"Error checking monitoring time: {e}")
            return False
    
    def check_disk_space(self):
        """Check available disk space."""
        try:
            result = subprocess.run(['df', '.'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 5:
                    used_percent = int(parts[4].rstrip('%'))
                    return used_percent < 90, f"Disk usage: {used_percent}%"
        except Exception as e:
            return False, f"Error checking disk space: {e}"
        
        return True, "Disk space OK"
    
    def check_network_connectivity(self):
        """Check internet connectivity."""
        try:
            result = subprocess.run(['ping', '-c', '1', '-W', '5', 'google.com'], 
                                  capture_output=True, text=True)
            return result.returncode == 0, "Network connectivity"
        except Exception as e:
            return False, f"Network error: {e}"
    
    async def send_telegram_alert(self, message, is_recovery=False):
        """Send alert to Telegram."""
        if not self.bot:
            self.log("No Telegram bot configured")
            return False
        
        try:
            # Add emoji and formatting
            emoji = "✅" if is_recovery else "🚨"
            alert_type = "RECOVERY" if is_recovery else "ALERT"
            
            full_message = (
                f"{emoji} **T8 Monitor {alert_type}**\n\n"
                f"{message}\n\n"
                f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S AEST')}\n"
                f"🖥️ Host: {os.uname().nodename if hasattr(os, 'uname') else 'Unknown'}"
            )
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=full_message,
                parse_mode='Markdown'
            )
            
            self.log(f"Telegram alert sent: {alert_type}")
            return True
            
        except Exception as e:
            self.log(f"Failed to send Telegram alert: {e}")
            return False
    
    def should_send_alert(self, issue_type):
        """Determine if we should send an alert (avoid spam)."""
        now = datetime.now()
        last_alert = self.status_data.get('last_alert_time')
        
        if last_alert:
            try:
                last_alert_time = datetime.fromisoformat(last_alert)
                # Don't send alerts more than once every 30 minutes
                if (now - last_alert_time).total_seconds() < 1800:  # 30 minutes
                    return False
            except:
                pass  # If we can't parse the date, allow the alert
        
        return True
    
    def should_send_recovery_alert(self):
        """Determine if we should send a recovery alert."""
        now = datetime.now()
        last_recovery = self.status_data.get('last_recovery_alert')
        
        if last_recovery:
            try:
                last_recovery_time = datetime.fromisoformat(last_recovery)
                # Don't send recovery alerts more than once every hour
                if (now - last_recovery_time).total_seconds() < 3600:  # 1 hour
                    return False
            except:
                pass
        
        return True
    
    def restart_service(self):
        """Attempt to restart the service."""
        try:
            result = subprocess.run(['sudo', 'systemctl', 'restart', 't8-monitor.service'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.log("Service restart successful")
                return True
            else:
                self.log(f"Service restart failed: {result.stderr}")
                return False
        except Exception as e:
            self.log(f"Error restarting service: {e}")
            return False
    
    async def run_health_check(self):
        """Run comprehensive health check."""
        self.log("Starting health check...")
        
        # Check service status
        service_running = self.check_service_status()
        log_active, log_status = self.check_log_activity()
        disk_ok, disk_status = self.check_disk_space()
        network_ok, network_status = self.check_network_connectivity()
        
        # Determine overall health
        is_healthy = service_running and log_active and disk_ok and network_ok
        
        # Log current status
        self.log(f"Service: {'✅' if service_running else '❌'} | "
                f"Logs: {'✅' if log_active else '❌'} | "
                f"Disk: {'✅' if disk_ok else '❌'} | "
                f"Network: {'✅' if network_ok else '❌'}")
        
        # Check if we need to send recovery alert
        was_running = self.status_data.get('service_was_running', True)
        had_failures = self.status_data.get('consecutive_failures', 0) > 0
        
        if is_healthy and (not was_running or had_failures) and self.should_send_recovery_alert():
            # System recovered
            await self.send_telegram_alert(
                "T8 Monitor has recovered and is now running normally. All systems are healthy.",
                is_recovery=True
            )
            self.status_data['consecutive_failures'] = 0
            self.status_data['last_recovery_alert'] = datetime.now().isoformat()
        
        # Check for new issues
        issues = []
        if not service_running:
            issues.append("🔴 Service not running")
        if not log_active:
            issues.append(f"📝 Log inactive: {log_status}")
        if not disk_ok:
            issues.append(f"💾 {disk_status}")
        if not network_ok:
            issues.append(f"🌐 {network_status}")
        
        # Handle new issues
        if issues and self.should_send_alert('health_check'):
            self.status_data['consecutive_failures'] += 1
            
            # Try to restart service if it's not running
            restart_attempted = False
            if not service_running:
                self.log("Attempting to restart service...")
                restart_attempted = True
                restart_success = self.restart_service()
                
                if restart_success:
                    issues.append("🔧 Service restart attempted")
                    # Wait a moment and check again
                    await asyncio.sleep(10)
                    if self.check_service_status():
                        issues.append("✅ Service restart successful")
                    else:
                        issues.append("❌ Service restart failed")
                else:
                    issues.append("❌ Service restart failed")
            
            # Send alert
            alert_message = (
                f"T8 Monitor has {len(issues)} issue(s):\n\n" +
                "\n".join(f"• {issue}" for issue in issues) +
                f"\n\nConsecutive failures: {self.status_data['consecutive_failures']}"
            )
            
            await self.send_telegram_alert(alert_message)
            self.status_data['last_alert_time'] = datetime.now().isoformat()
        
        # Update status
        self.status_data['service_was_running'] = service_running
        self.status_data['last_log_activity'] = log_status
        
        if is_healthy:
            self.status_data['consecutive_failures'] = 0
        
        self.save_status()
        
        return is_healthy

async def main():
    """Main function."""
    monitor = HealthMonitor()
    
    try:
        healthy = await monitor.run_health_check()
        sys.exit(0 if healthy else 1)
    except Exception as e:
        monitor.log(f"Health check failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 