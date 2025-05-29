import tweepy
import asyncio
import logging
import os
from datetime import datetime, timedelta
from telegram import Bot
import dateutil.tz

# Load environment variables from .env file if it exists
def load_env_file():
    if os.path.exists('.env'):
        try:
            with open('.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
        except Exception as e:
            print(f'Warning: Error loading .env file: {e}')

load_env_file()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('t8_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
X_BEARER_TOKEN = os.getenv('X_BEARER_TOKEN')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Polling interval configuration (in minutes)
# Default: 60 minutes to stay within 100 API calls per month
# For 100 calls/month: 60 minutes = ~95 calls/month
# For unlimited: 2 minutes = ~2850 calls/month
# For balanced: 30 minutes = ~190 calls/month
POLLING_INTERVAL_MINUTES = int(os.getenv('POLLING_INTERVAL_MINUTES', '60'))
POLLING_INTERVAL_SECONDS = POLLING_INTERVAL_MINUTES * 60

required_vars = ['X_BEARER_TOKEN', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f'Missing required environment variables: {", ".join(missing_vars)}')
    logger.info('Note: For this polling approach, you only need X_BEARER_TOKEN (not the full API keys)')
    exit(1)

telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Log polling configuration
logger.info(f'📊 Polling interval: {POLLING_INTERVAL_MINUTES} minutes ({POLLING_INTERVAL_SECONDS} seconds)')
if POLLING_INTERVAL_MINUTES >= 60:
    estimated_calls = int(285 * 20 / POLLING_INTERVAL_MINUTES)
    logger.info(f'💚 Quota-friendly mode: ~{estimated_calls} API calls per month (within 100 limit)')
elif POLLING_INTERVAL_MINUTES <= 5:
    estimated_calls = int(285 * 20 / POLLING_INTERVAL_MINUTES)
    logger.info(f'⚡ High-frequency mode: ~{estimated_calls} API calls per month (may exceed quota)')
else:
    estimated_calls = int(285 * 20 / POLLING_INTERVAL_MINUTES)
    logger.info(f'🔄 Custom interval: ~{estimated_calls} API calls per month')

school_terms = [
    (datetime(2025, 2, 4), datetime(2025, 4, 11)),
    (datetime(2025, 4, 29), datetime(2025, 7, 4)),
    (datetime(2025, 7, 22), datetime(2025, 9, 26)),
    (datetime(2025, 10, 13), datetime(2025, 12, 19))
]

public_holidays = [
    datetime(2025, 1, 1), datetime(2025, 1, 27), datetime(2025, 4, 18),
    datetime(2025, 4, 21), datetime(2025, 4, 25), datetime(2025, 6, 9),
    datetime(2025, 10, 6), datetime(2025, 12, 25), datetime(2025, 12, 26)
]

LAST_TWEET_FILE = 'last_tweet_id.txt'

def is_sydney_school_day(check_date):
    if check_date.weekday() >= 5:
        return False
    if check_date.date() in [h.date() for h in public_holidays]:
        return False
    for start, end in school_terms:
        if start.date() <= check_date.date() <= end.date():
            return True
    return False

def is_within_time_window(check_time):
    aest = dateutil.tz.gettz('Australia/Sydney')
    check_time = check_time.astimezone(aest)
    hour = check_time.hour
    minute = check_time.minute
    morning_window = (hour == 7 or (hour == 8 and minute <= 45))
    afternoon_window = (13 <= hour < 16 or (hour == 16 and minute == 0))
    return morning_window or afternoon_window

def load_last_tweet_id():
    try:
        if os.path.exists(LAST_TWEET_FILE):
            with open(LAST_TWEET_FILE, 'r') as f:
                return f.read().strip()
    except Exception as e:
        logger.warning(f'Error loading last tweet ID: {e}')
    return None

def save_last_tweet_id(tweet_id):
    try:
        with open(LAST_TWEET_FILE, 'w') as f:
            f.write(str(tweet_id))
    except Exception as e:
        logger.warning(f'Error saving last tweet ID: {e}')

async def process_tweet(tweet):
    try:
        now = datetime.now(dateutil.tz.gettz('Australia/Sydney'))
        if not (is_sydney_school_day(now) and is_within_time_window(now)):
            logger.debug(f'Outside monitoring window, skipping tweet: {tweet.text[:50]}...')
            return False
        
        text = tweet.text.lower()
        t8_keywords = ['t8', 'airport']
        delay_keywords = [
            'delay', 'disruption', 'cancelled', 'issue', 'suspended', 'stopped', 'problem',
            'extra travel time', 'allow extra', 'not running', 'service alert', 'altered',
            'incident', 'emergency', 'flooding', 'power supply', 'signal repairs', 
            'shuttle', 'reduced service', 'timetable order', 'longer journey', 'wait times',
            'repairs', 'urgent', 'limited', 'diverted', 'gaps', 'less frequent', 'late'
        ]
        
        has_t8_content = any(keyword in text for keyword in t8_keywords)
        has_delay_content = any(keyword in text for keyword in delay_keywords)
        
        if has_t8_content and has_delay_content:
            tweet_time = tweet.created_at.astimezone(dateutil.tz.gettz('Australia/Sydney'))
            message = (
                f'🚆 T8 Airport Line Alert:\n\n'
                f'{tweet.text}\n\n'
                f'📅 Tweet: {tweet_time.strftime("%Y-%m-%d %H:%M:%S AEST")}\n'
                f'⏰ Alert: {now.strftime("%Y-%m-%d %H:%M:%S AEST")}'
            )
            await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            logger.info(f'Telegram notification sent for tweet: {tweet.text[:100]}...')
            return True
        else:
            logger.debug(f'Tweet does not match criteria: {tweet.text[:50]}...')
            return False
    except Exception as e:
        logger.error(f'Error processing tweet: {e}')
        return False

async def fetch_and_process_tweets():
    try:
        client = tweepy.Client(bearer_token=X_BEARER_TOKEN)
        try:
            user = client.get_user(username='T8SydneyTrains')
            user_id = user.data.id
            logger.debug(f'Found user ID for @T8SydneyTrains: {user_id}')
        except Exception as e:
            logger.error(f'Error fetching user ID for @T8SydneyTrains: {e}')
            return False
        
        last_tweet_id = load_last_tweet_id()
        try:
            # Increase lookback window for longer polling intervals
            lookback_hours = max(2, POLLING_INTERVAL_MINUTES // 30 + 1)
            since_time = datetime.now(dateutil.tz.UTC) - timedelta(hours=lookback_hours)
            
            kwargs = {
                'max_results': 10,
                'tweet_fields': ['created_at', 'public_metrics'],
                'start_time': since_time
            }
            if last_tweet_id:
                kwargs['since_id'] = last_tweet_id
            
            tweets = client.get_users_tweets(user_id, **kwargs)
            if not tweets.data:
                logger.debug('No new tweets found')
                return True
            
            tweets_list = list(tweets.data)
            tweets_list.reverse()
            alerts_sent = 0
            latest_tweet_id = last_tweet_id
            
            for tweet in tweets_list:
                if await process_tweet(tweet):
                    alerts_sent += 1
                latest_tweet_id = tweet.id
            
            if latest_tweet_id:
                save_last_tweet_id(latest_tweet_id)
            
            if alerts_sent > 0:
                logger.info(f'Processed {len(tweets_list)} tweets, sent {alerts_sent} alerts')
            else:
                logger.debug(f'Processed {len(tweets_list)} tweets, no alerts sent')
            return True
        except Exception as e:
            logger.error(f'Error fetching tweets: {e}')
            return False
    except Exception as e:
        logger.error(f'Error in fetch_and_process_tweets: {e}')
        return False

async def test_telegram_connection():
    try:
        bot_info = await telegram_bot.get_me()
        logger.info(f'Telegram bot connected: {bot_info.username}')
        return True
    except Exception as e:
        logger.error(f'Telegram connection test failed: {e}')
        return False

async def test_twitter_connection():
    try:
        client = tweepy.Client(bearer_token=X_BEARER_TOKEN)
        user = client.get_user(username='T8SydneyTrains')
        logger.info(f'Twitter API connected, found @T8SydneyTrains (ID: {user.data.id})')
        return True
    except Exception as e:
        logger.error(f'Twitter API connection test failed: {e}')
        return False

async def log_heartbeat():
    """Log a heartbeat message to show the script is alive."""
    logger.info("💓 T8 Monitor heartbeat - system running normally")

async def send_critical_error_alert(error_message):
    """Send critical error alert to Telegram."""
    try:
        message = (
            f"🚨 **T8 Monitor CRITICAL ERROR**\n\n"
            f"The monitoring script has crashed:\n{str(error_message)}\n\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S AEST')}\n"
            f"The service will attempt to restart automatically."
        )
        await telegram_bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message,
            parse_mode='Markdown'
        )
        logger.info("Critical error alert sent to Telegram")
    except Exception as e:
        logger.error(f"Failed to send critical error alert: {e}")

async def main():
    logger.info('Starting T8 Delays Monitor (Quota-Optimized Polling Mode)...')
    
    if not await test_telegram_connection():
        logger.error('Failed to connect to Telegram. Exiting.')
        return
    
    if not await test_twitter_connection():
        logger.error('Failed to connect to Twitter API. Exiting.')
        return
    
    logger.info('All connections successful. Starting monitoring...')
    logger.info(f'Monitoring mode: Polling every {POLLING_INTERVAL_MINUTES} minutes during school days and peak hours')
    
    heartbeat_counter = 0
    
    try:
        while True:
            now = datetime.now(dateutil.tz.gettz('Australia/Sydney'))
            
            if is_sydney_school_day(now) and is_within_time_window(now):
                logger.debug(f'Checking for new tweets at {now.strftime("%H:%M:%S")}')
                await fetch_and_process_tweets()
                await asyncio.sleep(POLLING_INTERVAL_SECONDS)
                heartbeat_counter += 1
            else:
                logger.debug(f'Outside monitoring window, next check in {POLLING_INTERVAL_MINUTES} minutes')
                await asyncio.sleep(POLLING_INTERVAL_SECONDS)
                heartbeat_counter += 1
            
            # Log heartbeat every 2 hours (adjust based on polling interval)
            cycles_for_heartbeat = max(1, int(120 / POLLING_INTERVAL_MINUTES))
            if heartbeat_counter >= cycles_for_heartbeat:
                await log_heartbeat()
                heartbeat_counter = 0
                
    except KeyboardInterrupt:
        logger.info('Monitoring stopped by user')
    except Exception as e:
        logger.error(f'Error in main monitoring loop: {e}')
        # Send critical error alert
        await send_critical_error_alert(e)
    finally:
        logger.info('Shutting down T8 Delays Monitor...')

if __name__ == '__main__':
    asyncio.run(main()) 