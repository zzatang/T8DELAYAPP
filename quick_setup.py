#!/usr/bin/env python3
"""
Quick setup script for T8 Monitor - simplified version
Creates .env file with required credentials
"""

import os
import sys

def create_env_file():
    """Create .env file with user input"""
    print("🚀 T8 Monitor Quick Setup")
    print("=" * 50)
    
    # Check if .env already exists
    if os.path.exists('.env'):
        response = input("⚠️  .env file already exists. Overwrite? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Setup cancelled.")
            return False
    
    print("\nChoose your Twitter API:")
    print("1. TwitterAPI.io (Recommended - $0.15/1000 tweets, no approval needed)")
    print("2. X API (Traditional - requires approval, rate limits)")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter 1 or 2")
    
    use_twitterapi_io = choice == '1'
    
    # Collect credentials
    env_vars = {}
    
    if use_twitterapi_io:
        print("\n📝 TwitterAPI.io Setup:")
        print("1. Go to https://twitterapi.io/")
        print("2. Sign up for an account")
        print("3. Get your API key from the dashboard")
        
        while True:
            api_key = input("\nEnter your TwitterAPI.io API key: ").strip()
            if api_key:
                env_vars['TWITTERAPI_IO_KEY'] = api_key
                env_vars['USE_TWITTERAPI_IO'] = 'true'
                break
            print("API key cannot be empty")
    else:
        print("\n📝 X API Setup:")
        print("1. Go to https://developer.twitter.com/")
        print("2. Apply for developer account")
        print("3. Create an app and get your Bearer Token")
        
        while True:
            bearer_token = input("\nEnter your X API Bearer Token: ").strip()
            if bearer_token:
                env_vars['X_BEARER_TOKEN'] = bearer_token
                env_vars['USE_TWITTERAPI_IO'] = 'false'
                break
            print("Bearer token cannot be empty")
    
    # Telegram setup
    print("\n📱 Telegram Bot Setup:")
    print("1. Message @BotFather on Telegram")
    print("2. Create a new bot with /newbot")
    print("3. Get your bot token")
    print("4. Message @userinfobot to get your chat ID")
    
    while True:
        bot_token = input("\nEnter your Telegram Bot Token: ").strip()
        if bot_token:
            env_vars['TELEGRAM_BOT_TOKEN'] = bot_token
            break
        print("Bot token cannot be empty")
    
    while True:
        chat_id = input("Enter your Telegram Chat ID: ").strip()
        if chat_id:
            env_vars['TELEGRAM_CHAT_ID'] = chat_id
            break
        print("Chat ID cannot be empty")
    
    # Optional settings
    print("\n⚙️  Optional Settings (press Enter for defaults):")
    
    polling_interval = input("Polling interval in minutes (default: 60): ").strip()
    if polling_interval and polling_interval.isdigit():
        env_vars['POLLING_INTERVAL_MINUTES'] = polling_interval
    else:
        env_vars['POLLING_INTERVAL_MINUTES'] = '60'
    
    # Add debug flag
    debug_mode = input("Enable debug logging? (y/N): ").strip().lower()
    if debug_mode in ['y', 'yes']:
        env_vars['DEBUG'] = 'true'
    
    # Write .env file
    try:
        with open('.env', 'w') as f:
            f.write("# T8 Monitor Configuration\n")
            f.write(f"# Generated on {os.uname().sysname if hasattr(os, 'uname') else 'Windows'} at {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if use_twitterapi_io:
                f.write("# TwitterAPI.io Configuration\n")
                f.write(f"TWITTERAPI_IO_KEY={env_vars['TWITTERAPI_IO_KEY']}\n")
                f.write(f"USE_TWITTERAPI_IO={env_vars['USE_TWITTERAPI_IO']}\n\n")
            else:
                f.write("# X API Configuration\n")
                f.write(f"X_BEARER_TOKEN={env_vars['X_BEARER_TOKEN']}\n")
                f.write(f"USE_TWITTERAPI_IO={env_vars['USE_TWITTERAPI_IO']}\n\n")
            
            f.write("# Telegram Configuration\n")
            f.write(f"TELEGRAM_BOT_TOKEN={env_vars['TELEGRAM_BOT_TOKEN']}\n")
            f.write(f"TELEGRAM_CHAT_ID={env_vars['TELEGRAM_CHAT_ID']}\n\n")
            
            f.write("# Optional Configuration\n")
            f.write(f"POLLING_INTERVAL_MINUTES={env_vars['POLLING_INTERVAL_MINUTES']}\n")
            f.write("OLLAMA_MODEL=llama3.2:3b\n")
            f.write("OLLAMA_HOST=http://localhost:11434\n")
            
            if env_vars.get('DEBUG'):
                f.write("DEBUG=true\n")
        
        print(f"\n✅ Configuration saved to .env file!")
        
        # Show summary
        api_name = "TwitterAPI.io" if use_twitterapi_io else "X API"
        print(f"\n📊 Summary:")
        print(f"   API: {api_name}")
        print(f"   Polling interval: {env_vars['POLLING_INTERVAL_MINUTES']} minutes")
        print(f"   Debug logging: {'Yes' if env_vars.get('DEBUG') else 'No'}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error creating .env file: {e}")
        return False

def main():
    """Main setup function"""
    if create_env_file():
        print("\n🎉 Setup complete!")
        print("\nNext steps:")
        print("1. Test your setup: python debug_tweet_retrieval.py")
        print("2. Run the monitor: python monitor_t8_delays_polling.py")
        print("\nFor detailed debugging, set DEBUG=true in your .env file")
    else:
        print("\n❌ Setup failed")
        sys.exit(1)

if __name__ == '__main__':
    main()

