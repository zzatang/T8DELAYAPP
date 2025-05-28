#!/bin/bash

# T8 Monitor Management Script
# Enhanced version with health monitoring and Telegram alerts

case "$1" in
    start)
        sudo systemctl start t8-monitor.service
        echo "✅ T8 Monitor started"
        ;;
    stop)
        sudo systemctl stop t8-monitor.service
        echo "🛑 T8 Monitor stopped"
        ;;
    restart)
        sudo systemctl restart t8-monitor.service
        echo "🔄 T8 Monitor restarted"
        ;;
    status)
        echo "📊 T8 Monitor Service Status:"
        echo "================================"
        sudo systemctl status t8-monitor.service
        ;;
    logs)
        echo "📋 Viewing T8 Monitor logs (Ctrl+C to exit):"
        echo "============================================="
        sudo journalctl -u t8-monitor.service -f
        ;;
    app-logs)
        echo "📄 Viewing application logs (Ctrl+C to exit):"
        echo "=============================================="
        tail -f t8_monitor.log
        ;;
    health)
        echo "🏥 Running health check..."
        echo "=========================="
        if command -v python3 &> /dev/null; then
            if [ -f "venv/bin/python" ]; then
                ./venv/bin/python health_monitor.py
            else
                python3 health_monitor.py
            fi
        else
            echo "❌ Python3 not found"
            exit 1
        fi
        ;;
    health-logs)
        echo "🏥 Viewing health monitor logs (Ctrl+C to exit):"
        echo "==============================================="
        if [ -f "health_monitor.log" ]; then
            tail -f health_monitor.log
        else
            echo "❌ Health monitor log file not found"
            echo "💡 Run './manage.sh health' first to generate logs"
        fi
        ;;
    test-alert)
        echo "🧪 Sending test alert to Telegram..."
        echo "===================================="
        if command -v python3 &> /dev/null; then
            if [ -f "venv/bin/python" ]; then
                PYTHON_CMD="./venv/bin/python"
            else
                PYTHON_CMD="python3"
            fi
            
            $PYTHON_CMD -c "
import asyncio
import sys
import os
sys.path.append('.')

async def test_alert():
    try:
        from health_monitor import HealthMonitor
        monitor = HealthMonitor()
        success = await monitor.send_telegram_alert('🧪 Test alert from T8 Monitor health system - all systems operational!')
        if success:
            print('✅ Test alert sent successfully!')
        else:
            print('❌ Failed to send test alert')
            sys.exit(1)
    except Exception as e:
        print(f'❌ Error sending test alert: {e}')
        sys.exit(1)

asyncio.run(test_alert())
"
        else
            echo "❌ Python3 not found"
            exit 1
        fi
        ;;
    install-health)
        echo "🚀 Installing enhanced health monitoring..."
        echo "=========================================="
        
        # Update cron job for health monitoring
        echo "📅 Setting up cron job for health checks..."
        (crontab -l 2>/dev/null | grep -v "health_monitor.py"; echo "*/15 * * * * $(pwd)/venv/bin/python $(pwd)/health_monitor.py >> $(pwd)/health_monitor.log 2>&1") | crontab -
        
        # Test health monitor
        echo "🧪 Testing health monitor..."
        ./manage.sh health
        
        echo "✅ Enhanced health monitoring installed!"
        echo ""
        echo "📱 You will now receive Telegram alerts for:"
        echo "   • Service stops running"
        echo "   • Log activity stops"
        echo "   • Disk space is low (>90%)"
        echo "   • Network connectivity fails"
        echo "   • Service recovers"
        echo ""
        echo "⏰ Health checks run every 15 minutes automatically"
        ;;
    summary)
        echo "📊 T8 Monitor System Summary"
        echo "============================"
        echo ""
        
        # Service status
        if systemctl is-active --quiet t8-monitor.service; then
            echo "🟢 Service Status: Running"
        else
            echo "🔴 Service Status: Stopped"
        fi
        
        # Log file status
        if [ -f "t8_monitor.log" ]; then
            last_log=$(stat -c %Y t8_monitor.log 2>/dev/null || echo 0)
            current_time=$(date +%s)
            minutes_ago=$(( (current_time - last_log) / 60 ))
            echo "📝 Last Log Activity: $minutes_ago minutes ago"
        else
            echo "📝 Log File: Not found"
        fi
        
        # Disk usage
        disk_usage=$(df . | tail -1 | awk '{print $5}' | sed 's/%//')
        if [ "$disk_usage" -gt 90 ]; then
            echo "💾 Disk Usage: ${disk_usage}% ⚠️"
        else
            echo "💾 Disk Usage: ${disk_usage}%"
        fi
        
        # Network connectivity
        if ping -c 1 google.com &> /dev/null; then
            echo "🌐 Network: Connected"
        else
            echo "🌐 Network: Disconnected ⚠️"
        fi
        
        # Monitoring status
        if command -v python3 &> /dev/null; then
            monitoring_status=$(python3 -c "
import sys
sys.path.append('.')
try:
    from monitor_t8_delays_polling import is_sydney_school_day, is_within_time_window
    import datetime, dateutil.tz
    now = datetime.datetime.now(dateutil.tz.gettz('Australia/Sydney'))
    if is_sydney_school_day(now) and is_within_time_window(now):
        print('🟢 Currently monitoring')
    else:
        print('🟡 Outside monitoring hours')
except:
    print('❓ Unknown')
" 2>/dev/null)
            echo "⏰ Status: $monitoring_status"
        fi
        ;;
    *)
        echo "🚆 T8 Monitor Management Script"
        echo "=============================="
        echo ""
        echo "Usage: $0 {command}"
        echo ""
        echo "Service Management:"
        echo "  start          Start the T8 monitor service"
        echo "  stop           Stop the T8 monitor service"
        echo "  restart        Restart the T8 monitor service"
        echo "  status         Show service status"
        echo ""
        echo "Monitoring & Logs:"
        echo "  logs           View live service logs"
        echo "  app-logs       View live application logs"
        echo "  health         Run health check manually"
        echo "  health-logs    View health monitor logs"
        echo "  summary        Show system summary"
        echo ""
        echo "Testing & Setup:"
        echo "  test-alert     Send test Telegram alert"
        echo "  install-health Install enhanced health monitoring"
        echo ""
        echo "Examples:"
        echo "  ./manage.sh status        # Check if service is running"
        echo "  ./manage.sh health        # Run health check"
        echo "  ./manage.sh test-alert    # Test Telegram alerts"
        echo ""
        exit 1
        ;;
esac 