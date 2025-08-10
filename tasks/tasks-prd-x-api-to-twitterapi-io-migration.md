# Task List: X API to TwitterAPI.io Migration

## Relevant Files

- `monitor_t8_delays_polling.py` - Main monitoring script that needs complete API migration from Tweepy to TwitterAPI.io
- `requirements.txt` - Dependencies file that needs updating to remove tweepy and add aiohttp
- `test_setup_polling.py` - Connection testing script that needs TwitterAPI.io test functions
- `.env` - Environment configuration file that needs new TWITTERAPI_IO_KEY variable
- `last_tweet_id.txt` - Tweet ID tracking file (will be reset for fresh start with TwitterAPI.io)
- `setup_env_simple.py` - Environment setup script that may need updating for new variables
- `docker-compose.yml` - Docker configuration that may need environment variable updates

### Notes

- The migration involves completely replacing the Tweepy-based Twitter API integration with direct HTTP requests to TwitterAPI.io
- All existing functionality must be preserved including Ollama AI analysis, Telegram notifications, and time-based filtering
- A feature flag approach will be used during testing to allow switching between APIs
- No unit tests currently exist in the codebase, but comprehensive manual testing will be required

## Tasks

- [x] 1.0 Setup TwitterAPI.io Account and Environment Configuration
  - [x] 1.1 Create TwitterAPI.io account at https://twitterapi.io/
  - [x] 1.2 Obtain API key from TwitterAPI.io dashboard
  - [x] 1.3 Add TWITTERAPI_IO_KEY environment variable to .env file
  - [x] 1.4 Add USE_TWITTERAPI_IO feature flag environment variable to .env file
  - [x] 1.5 Update setup_env_simple.py to include new environment variables
  - [x] 1.6 Test API key connectivity with a simple HTTP request to TwitterAPI.io
  - [x] 1.7 Document new environment variables in README.md

- [ ] 2.0 Implement TwitterAPI.io HTTP Client Functions
  - [ ] 2.1 Add aiohttp import to monitor_t8_delays_polling.py
  - [ ] 2.2 Create fetch_tweets_twitterapi() function to replace fetch_and_process_tweets()
  - [ ] 2.3 Implement convert_twitterapi_response() helper function for response format conversion
  - [ ] 2.4 Create test_twitterapi_connection() function for startup validation
  - [ ] 2.5 Add comprehensive error handling for HTTP status codes (401, 403, 429, 5xx)
  - [ ] 2.6 Implement proper async HTTP session management with aiohttp
  - [ ] 2.7 Add TwitterAPI.io specific logging with request/response details
  - [ ] 2.8 Update process_tweet() function to handle new tweet data structure
  - [ ] 2.9 Reset last_tweet_id.txt file for fresh start with TwitterAPI.io

- [ ] 3.0 Add Feature Flag System for API Switching
  - [ ] 3.1 Update configuration section to read USE_TWITTERAPI_IO environment variable
  - [ ] 3.2 Create api_backend_selector() function to choose between X API and TwitterAPI.io
  - [ ] 3.3 Modify fetch_and_process_tweets() to use feature flag for API selection
  - [ ] 3.4 Add startup logging to show which API backend is active
  - [ ] 3.5 Ensure both API paths maintain identical function signatures for seamless switching
  - [ ] 3.6 Add feature flag validation and default fallback behavior

- [ ] 4.0 Update Dependencies and Remove Tweepy Integration
  - [ ] 4.1 Add aiohttp>=3.8.0 to requirements.txt
  - [ ] 4.2 Update required_vars list to include TWITTERAPI_IO_KEY instead of X_BEARER_TOKEN
  - [ ] 4.3 Remove tweepy import and all tweepy-related code when feature flag is disabled
  - [ ] 4.4 Update test_setup_polling.py to include TwitterAPI.io connection test
  - [ ] 4.5 Update docker-compose.yml environment variables section if needed
  - [ ] 4.6 Clean up unused X API configuration variables after migration

- [ ] 5.0 Testing and Production Migration
  - [ ] 5.1 Test TwitterAPI.io integration with feature flag enabled alongside existing X API
  - [ ] 5.2 Verify tweet data format conversion maintains compatibility with Ollama analysis
  - [ ] 5.3 Test error handling scenarios (network failures, rate limits, authentication errors)
  - [ ] 5.4 Validate time-based filtering still works correctly with TwitterAPI.io timestamps
  - [ ] 5.5 Compare alert frequency and accuracy between both APIs during parallel testing
  - [ ] 5.6 Test Telegram alert functionality with TwitterAPI.io backend
  - [ ] 5.7 Monitor API response times and log performance metrics
  - [ ] 5.8 Enable TwitterAPI.io in production by setting USE_TWITTERAPI_IO=true
  - [ ] 5.9 Monitor production system for 24 hours to ensure stability
  - [ ] 5.10 Remove X API code, tweepy dependency, and X_BEARER_TOKEN after successful migration
  - [ ] 5.11 Update documentation and deployment scripts to reflect new API integration

