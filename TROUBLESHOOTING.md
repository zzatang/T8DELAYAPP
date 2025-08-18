# T8 Monitor Troubleshooting Guide

## Issue: No Tweets Retrieved

If your T8 monitor is not retrieving any tweets from @T8SydneyTrains, follow this step-by-step troubleshooting guide.

## Quick Diagnosis

Run the debug script to identify the issue:

```bash
python debug_tweet_retrieval.py
```

This will check:
- ✅ Environment variables configuration
- ✅ API connections (Twitter/TwitterAPI.io and Telegram)
- ✅ Current time windows
- ✅ Actual tweet retrieval

## Common Issues and Solutions

### 1. Missing Environment Variables

**Symptoms:**
- Error: `Missing required environment variables`
- Script exits immediately

**Solution:**
```bash
# Quick setup
python quick_setup.py

# Or manual setup
python setup_env_simple.py
```

**Required variables:**
- For TwitterAPI.io: `TWITTERAPI_IO_KEY`, `USE_TWITTERAPI_IO=true`
- For X API: `X_BEARER_TOKEN`, `USE_TWITTERAPI_IO=false`
- For both: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

### 2. Rate Limiting (429 Errors)

**Symptoms:**
- Error: `429 Too Many Requests`
- Error: `Rate limit exceeded`

**Solutions:**
- **Increase polling interval:** Set `POLLING_INTERVAL_MINUTES=120` (2 hours)
- **Switch to TwitterAPI.io:** More generous rate limits than X API
- **Check your usage:** X API free tier has very low limits

### 3. Authentication Issues

**Symptoms:**
- Error: `403 Forbidden`
- Error: `Authentication failed`
- Error: `401 Unauthorized`

**Solutions:**

**For X API:**
- Verify your Bearer Token is correct
- Ensure your Twitter developer account is approved
- Check if your app has the right permissions

**For TwitterAPI.io:**
- Verify your API key is correct
- Check your account balance (pay-per-use service)
- Ensure you're using the correct endpoint

### 4. Outside Monitoring Window

**Symptoms:**
- Script runs but no tweets processed
- Message: `Outside monitoring window`

**Current monitoring windows:**
- **Morning:** 7:00 AM - 8:45 AM AEST
- **Afternoon:** 1:00 PM - 4:00 PM AEST  
- **Days:** Weekdays during NSW school terms only

**Solutions:**
- Wait for monitoring window
- For testing, enable debug mode: `DEBUG=true` in .env
- Check current time with: `python -c "from datetime import datetime; import dateutil.tz; print(datetime.now(dateutil.tz.gettz('Australia/Sydney')))"`

### 5. No Recent Tweets

**Symptoms:**
- API connection works
- Message: `No new tweets found`

**Possible causes:**
- @T8SydneyTrains hasn't posted recently
- All recent tweets already processed
- Tweet age filter (only processes tweets < 2 hours old)

**Solutions:**
- Check @T8SydneyTrains manually on Twitter/X
- Delete `last_tweet_id.txt` to reprocess recent tweets
- Enable debug mode to see all retrieved tweets

### 6. Ollama AI Analysis Issues

**Symptoms:**
- Error: `Ollama connection test failed`
- Falling back to keyword analysis

**Solutions:**
- Check if Ollama is running: `ollama list`
- Start Ollama: `ollama serve`
- Install required model: `ollama pull llama3.2:3b`
- Check Ollama host: Default is `http://localhost:11434`

## Debug Mode

Enable detailed logging by adding to your `.env` file:

```bash
DEBUG=true
```

This will show:
- 🔍 Detailed tweet processing
- 📡 API request/response details
- 🤖 AI analysis reasoning
- ⏰ Time window calculations

## Manual Testing

### Test API Connections

```bash
# Test TwitterAPI.io
python -c "
import asyncio
import aiohttp
import os

async def test():
    headers = {'x-api-key': os.getenv('TWITTERAPI_IO_KEY')}
    params = {'userName': 'T8SydneyTrains', 'count': 5}
    async with aiohttp.ClientSession() as session:
        async with session.get('https://api.twitterapi.io/twitter/user/last_tweets', 
                              headers=headers, params=params) as response:
            print(f'Status: {response.status}')
            data = await response.json()
            print(f'Tweets: {len(data.get(\"tweets\", []))}')

asyncio.run(test())
"

# Test X API
python -c "
import tweepy
import os
client = tweepy.Client(bearer_token=os.getenv('X_BEARER_TOKEN'))
user = client.get_user(username='T8SydneyTrains')
print(f'User: {user.data.username} (ID: {user.data.id})')
tweets = client.get_users_tweets(user.data.id, max_results=5)
print(f'Recent tweets: {len(tweets.data) if tweets.data else 0}')
"
```

### Test Telegram

```bash
python -c "
import asyncio
from telegram import Bot
import os

async def test():
    bot = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
    info = await bot.get_me()
    print(f'Bot: @{info.username}')
    await bot.send_message(chat_id=os.getenv('TELEGRAM_CHAT_ID'), 
                          text='🧪 Test message from T8 Monitor')
    print('Test message sent!')

asyncio.run(test())
"
```

## Configuration Files

### .env File Example

```bash
# TwitterAPI.io (Recommended)
TWITTERAPI_IO_KEY=your_api_key_here
USE_TWITTERAPI_IO=true

# OR X API (Traditional)
# X_BEARER_TOKEN=your_bearer_token_here
# USE_TWITTERAPI_IO=false

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Optional
POLLING_INTERVAL_MINUTES=60
DEBUG=true
OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434
```

## Log Analysis

Check the log file for detailed information:

```bash
# View recent logs
tail -f t8_monitor.log

# Search for errors
grep -i error t8_monitor.log

# Search for tweet processing
grep "PROCESSING TWEET" t8_monitor.log

# Check API responses
grep -E "(Retrieved|📥)" t8_monitor.log
```

## Common Log Messages

### Good Signs ✅
- `✅ All required environment variables present`
- `✅ Retrieved X tweets from [API]`
- `🚆 PROCESSING TWEET:`
- `✅ TELEGRAM ALERT SENT`

### Warning Signs ⚠️
- `⚠️ Currently outside monitoring window`
- `📭 No new tweets found`
- `🔇 ALERT SUPPRESSED`

### Error Signs ❌
- `❌ Missing environment variables`
- `❌ Failed to connect to [API]`
- `429 Too Many Requests`
- `403 Forbidden`

## Getting Help

1. **Run the debug script:** `python debug_tweet_retrieval.py`
2. **Enable debug mode:** Add `DEBUG=true` to .env
3. **Check logs:** Look at `t8_monitor.log` and `debug_tweets.log`
4. **Test manually:** Use the manual testing commands above
5. **Verify credentials:** Double-check your API keys and tokens

## API Comparison

| Feature | TwitterAPI.io | X API |
|---------|---------------|-------|
| **Setup** | No approval needed | Requires approval |
| **Cost** | $0.15/1000 tweets | Free tier limited |
| **Rate Limits** | Generous | Very restrictive |
| **Reliability** | High | Can be unstable |
| **Recommended** | ✅ Yes | Only if you already have access |

## Emergency Reset

If everything is broken:

```bash
# 1. Delete state files
rm -f last_tweet_id.txt
rm -f t8_monitor.log
rm -f debug_tweets.log

# 2. Reconfigure
python quick_setup.py

# 3. Test
python debug_tweet_retrieval.py

# 4. Run with debug
echo "DEBUG=true" >> .env
python monitor_t8_delays_polling.py
```

This should get you back to a working state.

