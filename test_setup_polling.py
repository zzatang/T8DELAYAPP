#!/usr/bin/env python3
"""
Test script for T8 Delays Monitor (Polling Mode).
Tests the simplified credentials needed for polling approach.
"""

import asyncio
import os
import sys
from datetime import datetime
import dateutil.tz

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file."""
    if os.path.exists('.env'):
        try:
            with open('.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
            return True
        except Exception as e:
            print(f"Error loading .env file: {e}")
            return False
    return False

async def test_twitter_connection():
    """Test Twitter API connection."""
    try:
        import tweepy
        
        client = tweepy.Client(bearer_token=os.getenv('X_BEARER_TOKEN'))
        
        # Test by getting user info
        user = client.get_user(username="T8SydneyTrains")
        print(f"✅ Twitter API: Connected successfully")
        print(f"   Found @T8SydneyTrains (ID: {user.data.id})")
        return True
        
    except Exception as e:
        print(f"❌ Twitter API: Connection failed - {e}")
        return False

async def test_telegram_connection():
    """Test Telegram bot connection."""
    try:
        from telegram import Bot
        
        bot = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
        bot_info = await bot.get_me()
        
        print(f"✅ Telegram Bot: Connected successfully")
        print(f"   Bot: @{bot_info.username} ({bot_info.first_name})")
        
        # Test sending a message
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        test_message = f"🧪 T8 Monitor Test (Polling Mode) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        await bot.send_message(chat_id=chat_id, text=test_message)
        print(f"✅ Telegram Message: Test message sent to chat {chat_id}")
        return True
        
    except Exception as e:
        print(f"❌ Telegram Bot: Connection failed - {e}")
        return False

def test_time_logic():
    """Test the time and date logic."""
    print("🕐 Testing Time Logic")
    print("-" * 30)
    
    now = datetime.now(dateutil.tz.gettz('Australia/Sydney'))
    print(f"Current time (AEST): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Import the functions from the polling script
    sys.path.append('.')
    try:
        from monitor_t8_delays_polling import is_sydney_school_day, is_within_time_window
        
        is_school_day = is_sydney_school_day(now)
        is_time_window = is_within_time_window(now)
        
        print(f"Is school day: {'✅ Yes' if is_school_day else '❌ No'}")
        print(f"Is monitoring window: {'✅ Yes' if is_time_window else '❌ No'}")
        print(f"Would monitor now: {'✅ Yes' if (is_school_day and is_time_window) else '❌ No'}")
        
        return True
    except Exception as e:
        print(f"❌ Time logic test failed: {e}")
        print("   Make sure monitor_t8_delays_polling.py exists")
        return False

async def main():
    """Run all tests."""
    print("🚆 T8 Delays Monitor - Polling Mode Test")
    print("=" * 50)
    print()
    
    # Load .env file
    if load_env_file():
        print("✅ .env file loaded")
    else:
        print("⚠️  No .env file found, using system environment variables")
    print()
    
    # Check required environment variables (only the ones we need for polling)
    print("🔑 Checking Environment Variables")
    print("-" * 30)
    
    required_vars = ['X_BEARER_TOKEN', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    
    missing_vars = []
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}: Set")
        else:
            print(f"❌ {var}: Missing")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n❌ Missing variables: {', '.join(missing_vars)}")
        print("Run 'python setup_env_simple.py' to configure credentials.")
        return False
    
    print()
    
    # Test connections
    print("🌐 Testing API Connections")
    print("-" * 30)
    
    twitter_ok = await test_twitter_connection()
    telegram_ok = await test_telegram_connection()
    
    print()
    
    # Test time logic
    time_ok = test_time_logic()
    
    print()
    print("📋 Test Summary")
    print("-" * 30)
    
    all_tests_passed = twitter_ok and telegram_ok and time_ok
    
    if all_tests_passed:
        print("✅ All tests passed! Your polling setup is ready.")
        print("🚀 Run 'python monitor_t8_delays_polling.py' to start monitoring.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("💡 Run 'python setup_env_simple.py' if you need to reconfigure credentials.")
    
    return all_tests_passed

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest cancelled by user.")
    except Exception as e:
        print(f"Test failed with error: {e}") 