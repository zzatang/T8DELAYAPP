import tweepy
import asyncio
import logging
import os
from datetime import datetime, timedelta
from telegram import Bot
import dateutil.tz
import ollama
import aiohttp

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

# TwitterAPI.io configuration
TWITTERAPI_IO_KEY = os.getenv('TWITTERAPI_IO_KEY')
USE_TWITTERAPI_IO = os.getenv('USE_TWITTERAPI_IO', 'false').lower() == 'true'

# Ollama configuration
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')  # Default to 3b model
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')  # Default Ollama port

# Polling interval configuration (in minutes)
# Default: 60 minutes to stay within 100 API calls per month
# For 100 calls/month: 60 minutes = ~95 calls/month
# For unlimited: 2 minutes = ~2850 calls/month
# For balanced: 30 minutes = ~190 calls/month
POLLING_INTERVAL_MINUTES = int(os.getenv('POLLING_INTERVAL_MINUTES', '60'))
POLLING_INTERVAL_SECONDS = POLLING_INTERVAL_MINUTES * 60

# Validate required environment variables based on API choice
if USE_TWITTERAPI_IO:
    required_vars = ['TWITTERAPI_IO_KEY', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    api_note = 'Note: Using TwitterAPI.io - you only need TWITTERAPI_IO_KEY'
else:
    required_vars = ['X_BEARER_TOKEN', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    api_note = 'Note: Using X API - you only need X_BEARER_TOKEN (not the full API keys)'

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f'Missing required environment variables: {", ".join(missing_vars)}')
    logger.info(api_note)
    exit(1)

telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Configure Ollama client for local Docker instance
ollama_client = ollama.Client(host=OLLAMA_HOST)

def api_backend_selector():
    """
    Select the appropriate API backend based on feature flag
    Returns the correct fetch function for the active API
    """
    if USE_TWITTERAPI_IO:
        return fetch_tweets_twitterapi
    else:
        return fetch_and_process_tweets

# Log API backend and polling configuration
if USE_TWITTERAPI_IO:
    logger.info(f'🔧 API Backend: TwitterAPI.io (Cost-effective)')
else:
    logger.info(f'🔧 API Backend: X API (Traditional)')

logger.info(f'📊 Polling interval: {POLLING_INTERVAL_MINUTES} minutes ({POLLING_INTERVAL_SECONDS} seconds)')
if POLLING_INTERVAL_MINUTES >= 60:
    estimated_calls = int(285 * 20 / POLLING_INTERVAL_MINUTES)
    if USE_TWITTERAPI_IO:
        estimated_cost = estimated_calls * 0.15 / 1000
        logger.info(f'💚 Quota-friendly mode: ~{estimated_calls} API calls per month (~${estimated_cost:.3f} cost)')
    else:
        logger.info(f'💚 Quota-friendly mode: ~{estimated_calls} API calls per month (within 100 limit)')
elif POLLING_INTERVAL_MINUTES <= 5:
    estimated_calls = int(285 * 20 / POLLING_INTERVAL_MINUTES)
    if USE_TWITTERAPI_IO:
        estimated_cost = estimated_calls * 0.15 / 1000
        logger.info(f'⚡ High-frequency mode: ~{estimated_calls} API calls per month (~${estimated_cost:.2f} cost)')
    else:
        logger.info(f'⚡ High-frequency mode: ~{estimated_calls} API calls per month (may exceed quota)')
else:
    estimated_calls = int(285 * 20 / POLLING_INTERVAL_MINUTES)
    if USE_TWITTERAPI_IO:
        estimated_cost = estimated_calls * 0.15 / 1000
        logger.info(f'🔄 Custom interval: ~{estimated_calls} API calls per month (~${estimated_cost:.2f} cost)')
    else:
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

async def analyze_tweet_with_ollama(tweet_text):
    """
    Use local Ollama LLM to analyze if a tweet indicates service disruption
    Returns: (should_alert: bool, confidence: str, reasoning: str)
    """
    try:
        prompt = f"""You are analyzing a tweet from T8 Sydney Trains to determine if passengers should be alerted about service disruptions.

Tweet: "{tweet_text}"

Analyze this tweet and determine if it indicates:
- Service delays, disruptions, or cancellations
- Track/signal/power issues affecting services  
- Reduced services or altered timetables
- Emergency situations affecting trains
- Platform changes or shuttle bus replacements
- Any situation requiring "extra travel time"

ALERT-WORTHY examples:
- "Allow extra travel time due to..."
- "Services suspended/cancelled/delayed"
- "Trains not running between..."
- "Reduced service due to..."
- "Platform changes" 
- "Shuttle buses replacing trains"

NOT alert-worthy:
- "Services restored to normal"
- General information without service impact
- Routine announcements
- Announcements about future trackwork
- Future changes to services

Respond EXACTLY in this format:
ALERT: YES or NO
CONFIDENCE: HIGH or MEDIUM or LOW  
REASONING: Brief explanation"""

        # Log the tweet being analyzed
        logger.info(f'🔍 OLLAMA ANALYSIS START')
        logger.info(f'📝 Tweet Text: "{tweet_text}"')
        logger.info(f'🤖 Model: {OLLAMA_MODEL}')
        
        # Call local Ollama instance
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    'role': 'system', 
                    'content': 'You are an expert at analyzing public transport service announcements. Be concise and accurate.'
                },
                {
                    'role': 'user', 
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.1,  # Low temperature for consistent responses
                'top_p': 0.9,
                'num_predict': 150   # Limit response length
            }
        )
        
        analysis = response['message']['content'].strip()
        
        # Log the raw AI response
        logger.info(f'🤖 OLLAMA RAW RESPONSE:')
        logger.info(f'"{analysis}"')
        
        # Parse the structured response
        alert = "NO"
        confidence = "LOW"
        reasoning = "Unable to parse AI response"
        
        lines = analysis.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('ALERT:'):
                alert = line.split(':', 1)[1].strip().upper()
            elif line.startswith('CONFIDENCE:'):
                confidence = line.split(':', 1)[1].strip().upper()
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
        
        should_alert = alert == "YES"
        
        # Log the parsed results
        logger.info(f'📊 OLLAMA PARSED RESULTS:')
        logger.info(f'   Alert Decision: {alert}')
        logger.info(f'   Confidence Level: {confidence}')
        logger.info(f'   Reasoning: {reasoning}')
        logger.info(f'   Will Send Telegram Alert: {should_alert}')
        logger.info(f'🔍 OLLAMA ANALYSIS END')
        logger.info(f'{"="*60}')
        
        return should_alert, confidence, reasoning
        
    except Exception as e:
        logger.error(f'❌ OLLAMA ANALYSIS ERROR: {e}')
        logger.info('🔄 Falling back to keyword analysis')
        logger.info(f'{"="*60}')
        return await fallback_keyword_analysis(tweet_text)

async def fallback_keyword_analysis(tweet_text):
    """Fallback keyword analysis if Ollama fails"""
    try:
        text = tweet_text.lower()
        delay_keywords = [
            'delay', 'disruption', 'cancelled', 'issue', 'suspended', 'stopped', 'problem',
            'extra travel time', 'allow extra', 'not running', 'service alert', 'altered',
            'incident', 'emergency', 'flooding', 'power supply', 'signal repairs', 
            'shuttle', 'reduced service', 'timetable order', 'longer journey', 'wait times',
            'repairs', 'urgent', 'limited', 'diverted', 'gaps', 'less frequent', 'late'
        ]
        
        has_delay_content = any(keyword in text for keyword in delay_keywords)
        found_keywords = [k for k in delay_keywords if k in text]
        reasoning = f"Keyword fallback - Found: {found_keywords}" if found_keywords else "No delay keywords found"
        
        # Log fallback analysis
        logger.info(f'🔄 FALLBACK KEYWORD ANALYSIS:')
        logger.info(f'📝 Tweet Text: "{tweet_text}"')
        logger.info(f'🔍 Keywords Found: {found_keywords}')
        logger.info(f'📊 Alert Decision: {"YES" if has_delay_content else "NO"}')
        logger.info(f'💭 Reasoning: {reasoning}')
        logger.info(f'🔄 FALLBACK ANALYSIS END')
        logger.info(f'{"="*60}')
        
        return has_delay_content, "MEDIUM", reasoning
    except Exception as e:
        logger.error(f'❌ Error in fallback analysis: {e}')
        return False, "LOW", "Analysis failed"

async def test_ollama_connection():
    """Test connection to local Ollama instance"""
    try:
        # Test with a simple prompt
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': 'Say "Ollama connected" if you can read this.'}],
            options={'num_predict': 10}
        )
        
        result = response['message']['content'].strip()
        logger.info(f'🤖 Ollama connection test successful: {result}')
        return True
        
    except Exception as e:
        logger.error(f'❌ Ollama connection test failed: {e}')
        logger.info('Will fall back to keyword analysis if needed')
        return False

async def process_tweet(tweet):
    try:
        now = datetime.now(dateutil.tz.gettz('Australia/Sydney'))
        tweet_time = tweet.created_at.astimezone(dateutil.tz.gettz('Australia/Sydney'))
        
        # Check if current time is within monitoring window
        if not (is_sydney_school_day(now) and is_within_time_window(now)):
            logger.debug(f'Outside monitoring window, skipping tweet: {tweet.text[:50]}...')
            return False
        
        # Check if tweet was posted within 2 hours of current check time
        time_diff = now - tweet_time
        if time_diff > timedelta(hours=2):
            logger.debug(f'Tweet too old ({time_diff}), skipping: {tweet.text[:50]}...')
            return False
        
        # Log tweet processing start
        logger.info(f'🚆 PROCESSING TWEET:')
        logger.info(f'📅 Tweet Time: {tweet_time.strftime("%Y-%m-%d %H:%M:%S AEST")}')
        logger.info(f'⏰ Current Time: {now.strftime("%Y-%m-%d %H:%M:%S AEST")}')
        logger.info(f'⏱️ Tweet Age: {str(time_diff).split(".")[0]}')
        
        # Use Ollama AI to analyze the tweet
        should_alert, confidence, reasoning = await analyze_tweet_with_ollama(tweet.text)
        
        if should_alert:
            message = (
                f'🚆 T8 Airport Line Alert:\n\n'
                f'{tweet.text}\n\n'
                f'📅 Tweet: {tweet_time.strftime("%Y-%m-%d %H:%M:%S AEST")}\n'
                f'⏰ Alert: {now.strftime("%Y-%m-%d %H:%M:%S AEST")}\n'
                f'⏱️ Age: {str(time_diff).split(".")[0]} ago\n'
                f'🤖 AI Confidence: {confidence}\n'
                f'💭 Reasoning: {reasoning}'
            )
            await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            logger.info(f'✅ TELEGRAM ALERT SENT ({confidence} confidence)')
            return True
        else:
            logger.info(f'❌ NO TELEGRAM ALERT SENT ({confidence} confidence)')
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

def convert_twitterapi_response(tweet_data):
    """
    Convert TwitterAPI.io tweet format to internal tweet object format
    
    TwitterAPI.io format example:
    {
        "id": "1234567890123456789",
        "text": "Tweet content here",
        "created_at": "2024-01-15T10:30:00.000Z",
        "public_metrics": {
            "retweet_count": 5,
            "like_count": 12,
            "reply_count": 3,
            "quote_count": 1
        },
        "author_id": "987654321",
        ...
    }
    
    Internal format needed:
    - tweet.id (int)
    - tweet.text (str)  
    - tweet.created_at (datetime with timezone)
    - tweet.public_metrics (dict, optional)
    """
    try:
        if not tweet_data or 'id' not in tweet_data or 'text' not in tweet_data:
            logger.debug(f'⚠️  Invalid tweet data: missing id or text')
            return None
        
        # Create a simple object to match the expected interface
        class TweetObject:
            def __init__(self, tweet_id, text, created_at, public_metrics=None):
                self.id = tweet_id
                self.text = text
                self.created_at = created_at
                self.public_metrics = public_metrics or {}
        
        # Extract and convert tweet ID
        try:
            tweet_id = int(tweet_data['id'])
        except (ValueError, KeyError):
            logger.debug(f'⚠️  Invalid tweet ID: {tweet_data.get("id")}')
            return None
        
        # Extract tweet text
        text = tweet_data.get('text', '').strip()
        if not text:
            logger.debug(f'⚠️  Empty tweet text for ID: {tweet_id}')
            return None
        
        # Parse created_at timestamp
        created_at_str = tweet_data.get('created_at', '')
        try:
            if created_at_str:
                # TwitterAPI.io typically returns ISO format: "2024-01-15T10:30:00.000Z"
                # Parse and convert to timezone-aware datetime
                if created_at_str.endswith('Z'):
                    created_at_str = created_at_str[:-1] + '+00:00'
                elif not created_at_str.endswith(('+', '-')):
                    created_at_str += '+00:00'
                
                created_at = datetime.fromisoformat(created_at_str)
            else:
                # Fallback to current time if no timestamp provided
                logger.debug(f'⚠️  No created_at timestamp for tweet {tweet_id}, using current time')
                created_at = datetime.now(dateutil.tz.UTC)
        except (ValueError, TypeError) as e:
            logger.debug(f'⚠️  Invalid created_at format for tweet {tweet_id}: {created_at_str} - {e}')
            created_at = datetime.now(dateutil.tz.UTC)
        
        # Extract public metrics (optional)
        public_metrics = tweet_data.get('public_metrics', {})
        
        # Create and return the tweet object
        tweet_obj = TweetObject(
            tweet_id=tweet_id,
            text=text,
            created_at=created_at,
            public_metrics=public_metrics
        )
        
        logger.debug(f'✅ Converted tweet {tweet_id}: "{text[:50]}..." ({created_at})')
        return tweet_obj
        
    except Exception as e:
        logger.error(f'❌ Error converting TwitterAPI.io tweet data: {e}')
        logger.debug(f'   Raw data: {tweet_data}')
        return None

async def fetch_tweets_twitterapi():
    """
    TwitterAPI.io implementation to fetch and process tweets
    Replaces fetch_and_process_tweets() when USE_TWITTERAPI_IO is True
    """
    try:
        # TwitterAPI.io endpoint for T8SydneyTrains tweets
        url = 'https://api.twitterapi.io/twitter/user/last_tweets'
        headers = {
            'x-api-key': TWITTERAPI_IO_KEY,
            'Content-Type': 'application/json'
        }
        params = {
            'userName': 'T8SydneyTrains',
            'count': 10  # Fetch last 10 tweets (equivalent to max_results)
        }
        
        logger.debug(f'🔗 TwitterAPI.io request: {url}?userName=T8SydneyTrains&count=10')
        
        # Use aiohttp for async HTTP requests
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                # Log response details
                logger.debug(f'📊 TwitterAPI.io response: {response.status}')
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f'❌ TwitterAPI.io request failed: HTTP {response.status}')
                    logger.error(f'   Response: {error_text[:200]}...')
                    
                    # Handle specific error codes
                    if response.status == 401:
                        logger.error('🔑 Authentication failed - check TWITTERAPI_IO_KEY')
                    elif response.status == 429:
                        logger.error('⏰ Rate limit exceeded - reduce polling frequency')
                    elif response.status >= 500:
                        logger.error('🔧 TwitterAPI.io server error - try again later')
                    
                    return False
                
                data = await response.json()
                
                # Check if we have tweets in the response
                if not data or 'tweets' not in data or not data['tweets']:
                    logger.debug('📭 No tweets found in TwitterAPI.io response')
                    return True
                
                # Process tweets using the same logic as X API
                tweets_data = data['tweets']
                logger.debug(f'📥 Retrieved {len(tweets_data)} tweets from TwitterAPI.io')
                
                # Convert TwitterAPI.io format to our internal format and process
                last_tweet_id = load_last_tweet_id()
                alerts_sent = 0
                latest_tweet_id = last_tweet_id
                processed_tweets = []
                
                for tweet_data in tweets_data:
                    # Convert TwitterAPI.io tweet to our internal format
                    converted_tweet = convert_twitterapi_response(tweet_data)
                    if not converted_tweet:
                        continue
                    
                    # Skip if we've already processed this tweet (fresh start approach)
                    tweet_id_str = str(converted_tweet.id)
                    if last_tweet_id and tweet_id_str <= str(last_tweet_id):
                        logger.debug(f'⏭️  Skipping already processed tweet: {tweet_id_str}')
                        continue
                    
                    processed_tweets.append(converted_tweet)
                    latest_tweet_id = max(int(latest_tweet_id or 0), converted_tweet.id)
                
                # Process tweets in chronological order (oldest first)
                processed_tweets.reverse()
                
                for tweet in processed_tweets:
                    if await process_tweet(tweet):
                        alerts_sent += 1
                
                # Save the latest tweet ID for next run
                if latest_tweet_id and latest_tweet_id != last_tweet_id:
                    save_last_tweet_id(str(latest_tweet_id))
                
                # Log results
                if alerts_sent > 0:
                    logger.info(f'✅ TwitterAPI.io: Processed {len(processed_tweets)} tweets, sent {alerts_sent} alerts')
                else:
                    logger.debug(f'📋 TwitterAPI.io: Processed {len(processed_tweets)} tweets, no alerts sent')
                
                return True
                
    except aiohttp.ClientError as e:
        logger.error(f'❌ TwitterAPI.io network error: {e}')
        return False
    except asyncio.TimeoutError:
        logger.error(f'⏰ TwitterAPI.io request timeout')
        return False
    except Exception as e:
        logger.error(f'❌ Error in fetch_tweets_twitterapi: {e}')
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

async def test_twitterapi_connection():
    """Test TwitterAPI.io connection for startup validation"""
    try:
        if not TWITTERAPI_IO_KEY:
            logger.error('TwitterAPI.io API key not found in environment variables')
            return False
        
        # Test with T8SydneyTrains user profile
        url = 'https://api.twitterapi.io/twitter/user/profile'
        headers = {'x-api-key': TWITTERAPI_IO_KEY}
        params = {'userName': 'T8SydneyTrains'}
        
        logger.debug(f'🧪 Testing TwitterAPI.io connection to {url}')
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        user_name = data.get('userName', 'unknown')
                        followers = data.get('followersCount', 'N/A')
                        logger.info(f'✅ TwitterAPI.io connected, found @{user_name} (Followers: {followers})')
                        return True
                    except Exception as e:
                        logger.error(f'❌ TwitterAPI.io response parsing failed: {e}')
                        return False
                elif response.status == 401:
                    logger.error('❌ TwitterAPI.io authentication failed - check TWITTERAPI_IO_KEY')
                    return False
                elif response.status == 429:
                    logger.warning('⚠️  TwitterAPI.io rate limit hit - API key is valid but throttled')
                    return True  # API key works, just rate limited
                elif response.status == 404:
                    logger.error('❌ TwitterAPI.io user not found or endpoint unavailable')
                    return False
                else:
                    error_text = await response.text()
                    logger.error(f'❌ TwitterAPI.io connection failed: HTTP {response.status}')
                    logger.debug(f'   Response: {error_text[:200]}...')
                    return False
                    
    except aiohttp.ClientError as e:
        logger.error(f'❌ TwitterAPI.io network error: {e}')
        return False
    except asyncio.TimeoutError:
        logger.error(f'❌ TwitterAPI.io connection timeout')
        return False
    except Exception as e:
        logger.error(f'❌ TwitterAPI.io connection test failed: {e}')
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
    logger.info('Starting T8 Delays Monitor with Ollama AI Analysis...')
    
    if not await test_telegram_connection():
        logger.error('Failed to connect to Telegram. Exiting.')
        return
    
    # Test the appropriate API backend based on feature flag
    if USE_TWITTERAPI_IO:
        if not await test_twitterapi_connection():
            logger.error('Failed to connect to TwitterAPI.io. Exiting.')
            return
    else:
        if not await test_twitter_connection():
            logger.error('Failed to connect to Twitter API. Exiting.')
            return
    
    # Test Ollama connection (non-blocking)
    await test_ollama_connection()
    
    logger.info('All connections tested. Starting monitoring...')
    logger.info(f'🤖 Using Ollama model: {OLLAMA_MODEL} at {OLLAMA_HOST}')
    if USE_TWITTERAPI_IO:
        logger.info(f'🔧 Active API: TwitterAPI.io (Cost-effective, pay-per-use)')
    else:
        logger.info(f'🔧 Active API: X API (Traditional Twitter API)')
    logger.info(f'Monitoring mode: Polling every {POLLING_INTERVAL_MINUTES} minutes during school days and peak hours')
    
    heartbeat_counter = 0
    
    try:
        while True:
            now = datetime.now(dateutil.tz.gettz('Australia/Sydney'))
            
            if is_sydney_school_day(now) and is_within_time_window(now):
                logger.debug(f'Checking for new tweets at {now.strftime("%H:%M:%S")}')
                # Use feature flag to select API backend
                fetch_function = api_backend_selector()
                await fetch_function()
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