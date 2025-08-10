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
import aiohttp

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

async def test_twitterapi_connection():
    """Test TwitterAPI.io connection."""
    try:
        api_key = os.getenv('TWITTERAPI_IO_KEY')
        if not api_key:
            print("❌ TwitterAPI.io: API key not found")
            return False
        
        # Test with T8SydneyTrains user profile
        url = 'https://api.twitterapi.io/twitter/user/profile'
        headers = {'x-api-key': api_key}
        params = {'userName': 'T8SydneyTrains'}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    user_name = data.get('userName', 'unknown')
                    followers = data.get('followersCount', 'N/A')
                    print(f"✅ TwitterAPI.io: Connected successfully")
                    print(f"   Found @{user_name} (Followers: {followers})")
                    return True
                elif response.status == 401:
                    print("❌ TwitterAPI.io: Authentication failed - check API key")
                    return False
                elif response.status == 429:
                    print("⚠️  TwitterAPI.io: Rate limit hit - API key works but throttled")
                    return True  # API key is valid
                else:
                    error_text = await response.text()
                    print(f"❌ TwitterAPI.io: Connection failed - HTTP {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ TwitterAPI.io: Connection failed - {e}")
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
    
    # Check which API backend to use
    use_twitterapi_io = os.getenv('USE_TWITTERAPI_IO', 'false').lower() == 'true'
    
    # Check required environment variables based on API choice
    print("🔑 Checking Environment Variables")
    print("-" * 30)
    
    if use_twitterapi_io:
        print("🔧 API Backend: TwitterAPI.io (Cost-effective)")
        required_vars = ['TWITTERAPI_IO_KEY', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    else:
        print("🔧 API Backend: X API (Traditional)")
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
    
    # Test the appropriate API backend
    if use_twitterapi_io:
        twitter_ok = await test_twitterapi_connection()
    else:
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