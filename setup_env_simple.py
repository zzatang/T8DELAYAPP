#!/usr/bin/env python3
"""
Simplified setup script for T8 Delays Monitor (Polling Mode).
This version only requires Bearer Token, not full API credentials.
"""

import os
import sys

def create_env_file():
    """Create a .env file with user-provided credentials."""
    print("🚆 T8 Delays Monitor - Simplified Setup")
    print("=" * 50)
    print()
    print("This polling version only requires:")
    print("1. Twitter Bearer Token (easier to get)")
    print("2. Telegram Bot token and chat ID")
    print()
    print("✅ No Twitter Developer Project required!")
    print()
    
    # Check if .env already exists
    if os.path.exists('.env'):
        response = input("⚠️  .env file already exists. Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
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
    
    # Validate inputs
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
    
    # Create .env file
    env_content = f"""# Twitter Bearer Token (only this is needed for polling mode)
X_BEARER_TOKEN={x_bearer_token}

# Telegram Bot Credentials
TELEGRAM_BOT_TOKEN={telegram_bot_token}
TELEGRAM_CHAT_ID={telegram_chat_id}
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("\n✅ Environment file created successfully!")
        print("📁 Created: .env")
        print()
        print("Next steps:")
        print("1. Test setup: python test_setup.py")
        print("2. Run monitor: python monitor_t8_delays_polling.py")
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
    
    # Check environment variables
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
        return True

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_credentials()
    else:
        create_env_file()

if __name__ == "__main__":
    main()