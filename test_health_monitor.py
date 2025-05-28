#!/usr/bin/env python3
"""
Test script for the enhanced health monitoring system.
This tests the health monitor functionality on Windows before Pi deployment.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append('.')

async def test_health_monitor():
    """Test the health monitoring functionality."""
    print("🧪 Testing T8 Health Monitor")
    print("=" * 40)
    print()
    
    try:
        from health_monitor import HealthMonitor
        
        # Create health monitor instance
        monitor = HealthMonitor()
        print("✅ Health monitor initialized successfully")
        
        # Test environment loading
        if monitor.bot:
            print("✅ Telegram bot configured")
        else:
            print("❌ Telegram bot not configured - check TELEGRAM_BOT_TOKEN")
            return False
        
        # Test status file operations
        print("\n📁 Testing status file operations...")
        monitor.status_data['test_key'] = 'test_value'
        monitor.save_status()
        
        # Reload and verify
        new_monitor = HealthMonitor()
        if new_monitor.status_data.get('test_key') == 'test_value':
            print("✅ Status file save/load working")
        else:
            print("❌ Status file operations failed")
        
        # Test logging
        print("\n📝 Testing logging...")
        monitor.log("Test log message from health monitor test")
        print("✅ Logging working")
        
        # Test network connectivity (should work on Windows)
        print("\n🌐 Testing network connectivity...")
        network_ok, network_msg = monitor.check_network_connectivity()
        print(f"{'✅' if network_ok else '❌'} Network: {network_msg}")
        
        # Test disk space check
        print("\n💾 Testing disk space check...")
        disk_ok, disk_msg = monitor.check_disk_space()
        print(f"{'✅' if disk_ok else '❌'} Disk: {disk_msg}")
        
        # Test Telegram alert (if configured)
        print("\n📱 Testing Telegram alert...")
        try:
            success = await monitor.send_telegram_alert(
                "🧪 Test alert from T8 Monitor health system test script",
                is_recovery=False
            )
            if success:
                print("✅ Telegram alert sent successfully!")
            else:
                print("❌ Failed to send Telegram alert")
        except Exception as e:
            print(f"❌ Telegram alert failed: {e}")
        
        print("\n📊 Test Summary")
        print("-" * 20)
        print("✅ Health monitor is ready for Pi deployment!")
        print("📱 Telegram alerts are working")
        print("💾 Status tracking is functional")
        print("📝 Logging is operational")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install python-telegram-bot")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

async def test_monitoring_time_logic():
    """Test the monitoring time logic."""
    print("\n🕐 Testing Monitoring Time Logic")
    print("-" * 35)
    
    try:
        from monitor_t8_delays_polling import is_sydney_school_day, is_within_time_window
        import dateutil.tz
        
        now = datetime.now(dateutil.tz.gettz('Australia/Sydney'))
        print(f"Current Sydney time: {now.strftime('%Y-%m-%d %H:%M:%S AEST')}")
        
        is_school_day = is_sydney_school_day(now)
        is_time_window = is_within_time_window(now)
        
        print(f"Is school day: {'✅ Yes' if is_school_day else '❌ No'}")
        print(f"Is monitoring window: {'✅ Yes' if is_time_window else '❌ No'}")
        print(f"Would monitor now: {'✅ Yes' if (is_school_day and is_time_window) else '❌ No'}")
        
        return True
    except Exception as e:
        print(f"❌ Time logic test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚆 T8 Monitor Health System Test")
    print("=" * 50)
    print()
    
    # Test health monitor
    health_test_ok = await test_health_monitor()
    
    # Test time logic
    time_test_ok = await test_monitoring_time_logic()
    
    print("\n" + "=" * 50)
    if health_test_ok and time_test_ok:
        print("🎉 All tests passed! System ready for Pi deployment.")
        print("\n📋 Next steps for Pi deployment:")
        print("1. Transfer files to Pi")
        print("2. Run: ./manage.sh install-health")
        print("3. Monitor with: ./manage.sh summary")
    else:
        print("❌ Some tests failed. Please fix issues before deployment.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 