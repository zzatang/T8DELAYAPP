#!/usr/bin/env python3
"""
Simplified setup script for T8 Delays Monitor (Polling Mode).
Supports both X API (Bearer Token) and TwitterAPI.io integration.
"""

import os
import sys
import requests

def create_env_file():
    """Create a .env file with user-provided credentials."""
    print("🚆 T8 Delays Monitor - Simplified Setup")
    print("=" * 50)
    print()
    print("Choose your Twitter API provider:")
    print("1. TwitterAPI.io (Recommended - $0.15/1000 tweets)")
    print("2. X API (Traditional - requires Bearer Token)")
    print()
    
    api_choice = input("Select API provider (1 or 2): ").strip()
    use_twitterapi_io = api_choice == "1"
    
    if use_twitterapi_io:
        print()
        print("✅ Using TwitterAPI.io - Cost effective and easy setup!")
        print()
    else:
        print()
        print("✅ Using X API - Traditional Twitter API integration")
        print("⚠️  May require Twitter Developer Project")
        print()
    
    # Check if .env already exists
    if os.path.exists('.env'):
        response = input("⚠️  .env file already exists. Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    # Get API credentials based on choice
    x_bearer_token = ""
    twitterapi_io_key = ""
    
    if use_twitterapi_io:
        print("🔑 TwitterAPI.io API Key")
        print("-" * 30)
        print("Get this from: https://twitterapi.io/")
        print("Sign up → Dashboard → API Key")
        print()
        
        twitterapi_io_key = input("Enter your TwitterAPI.io API Key: ").strip()
    else:
        print("📱 Twitter Bearer Token")
        print("-" * 30)
        print("Get this from: https://developer.twitter.com/")
        print("Go to your app → Keys and Tokens → Bearer Token")
        print()
        
        x_bearer_token = input("Enter your Twitter Bearer Token: ").strip()
    
    print()
    print("🤖 Telegram Bot Credentials")
    print("-" * 30)
    print("Get bot token from: @BotFather on Telegram")
    print("Get chat ID from: @userinfobot on Telegram")
    print()
    
    telegram_bot_token = input("Enter your Telegram Bot Token: ").strip()
    telegram_chat_id = input("Enter your Telegram Chat ID: ").strip()
    
    # Validate inputs based on API choice
    if use_twitterapi_io:
        required_fields = [
            ("TwitterAPI.io API Key", twitterapi_io_key),
            ("Telegram Bot Token", telegram_bot_token),
            ("Telegram Chat ID", telegram_chat_id)
        ]
    else:
        required_fields = [
            ("Twitter Bearer Token", x_bearer_token),
            ("Telegram Bot Token", telegram_bot_token),
            ("Telegram Chat ID", telegram_chat_id)
        ]
    
    missing_fields = [name for name, value in required_fields if not value]
    
    if missing_fields:
        print(f"\n❌ Missing required fields: {', '.join(missing_fields)}")
        print("Please run the script again and provide all required information.")
        return
    
    # Create .env file with appropriate configuration
    if use_twitterapi_io:
        env_content = f"""# TwitterAPI.io Configuration (Recommended)
TWITTERAPI_IO_KEY={twitterapi_io_key}
USE_TWITTERAPI_IO=true

# X API Configuration (Legacy - can be removed after migration)
X_BEARER_TOKEN=

# Telegram Bot Credentials
TELEGRAM_BOT_TOKEN={telegram_bot_token}
TELEGRAM_CHAT_ID={telegram_chat_id}

# Ollama Configuration
OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434

# Polling Configuration
POLLING_INTERVAL_MINUTES=60
"""
    else:
        env_content = f"""# X API Configuration (Traditional)
X_BEARER_TOKEN={x_bearer_token}

# TwitterAPI.io Configuration (Future migration option)
TWITTERAPI_IO_KEY=
USE_TWITTERAPI_IO=false

# Telegram Bot Credentials
TELEGRAM_BOT_TOKEN={telegram_bot_token}
TELEGRAM_CHAT_ID={telegram_chat_id}

# Ollama Configuration
OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434

# Polling Configuration
POLLING_INTERVAL_MINUTES=60
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("\n✅ Environment file created successfully!")
        print("📁 Created: .env")
        print()
        print("Next steps:")
        if use_twitterapi_io:
            print("1. Test setup: python test_setup_polling.py")
            print("2. Run monitor: python monitor_t8_delays_polling.py")
            print("💡 You're using TwitterAPI.io - cost effective and reliable!")
        else:
            print("1. Test setup: python test_setup_polling.py")
            print("2. Run monitor: python monitor_t8_delays_polling.py")
            print("💡 Consider migrating to TwitterAPI.io for lower costs")
        print()
        print("⚠️  Keep your .env file secure and don't share it publicly!")
        
    except Exception as e:
        print(f"\n❌ Error creating .env file: {e}")

def test_credentials():
    """Test if credentials are properly configured."""
    print("🧪 Testing Credentials")
    print("-" * 30)
    
    # Load .env file if it exists
    if os.path.exists('.env'):
        try:
            with open('.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
            print("✅ .env file loaded")
        except Exception as e:
            print(f"Error loading .env file: {e}")
    
    # Check environment variables (support both API types)
    use_twitterapi_io = os.getenv('USE_TWITTERAPI_IO', 'false').lower() == 'true'
    
    if use_twitterapi_io:
        required_vars = ['TWITTERAPI_IO_KEY', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    else:
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
        print("Run setup to configure missing credentials.")
        return False
    else:
        print("\n✅ All credentials configured!")
        
        # Test API connectivity
        if use_twitterapi_io:
            return test_twitterapi_io_connection()
        else:
            print("💡 X API connection testing not implemented in this script")
            print("   Use test_setup_polling.py for full connection testing")
            return True

def test_twitterapi_io_connection():
    """Test TwitterAPI.io connection with a simple API call."""
    print("\n🧪 Testing TwitterAPI.io Connection")
    print("-" * 40)
    
    api_key = os.getenv('TWITTERAPI_IO_KEY')
    if not api_key:
        print("❌ TWITTERAPI_IO_KEY not found")
        return False
    
    try:
        # Test with a simple user profile request
        url = 'https://api.twitterapi.io/twitter/user/profile'
        headers = {'x-api-key': api_key}
        params = {'userName': 'T8SydneyTrains'}
        
        print("🔗 Testing API endpoint...")
        print(f"   URL: {url}")
        print(f"   Target: @T8SydneyTrains")
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'userName' in data:
                print(f"✅ TwitterAPI.io connection successful!")
                print(f"   Found user: @{data.get('userName', 'unknown')}")
                print(f"   Followers: {data.get('followersCount', 'N/A')}")
                return True
            else:
                print(f"⚠️  API responded but data format unexpected")
                print(f"   Response: {response.text[:200]}...")
                return False
        elif response.status_code == 401:
            print("❌ Authentication failed - check your API key")
            return False
        elif response.status_code == 429:
            print("⚠️  Rate limit exceeded - API key works but too many requests")
            return True  # API key is valid
        else:
            print(f"❌ API request failed: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out - check internet connection")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_credentials()
    else:
        create_env_file()

if __name__ == "__main__":
    main()