import tweepy
import asyncio
import logging
import os
from datetime import datetime, timedelta
from telegram import Bot
import dateutil.tz
import ollama
import aiohttp
import requests

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

# School day checking functionality has been moved to school_day_lookup.py
# This provides database-backed, high-performance school day lookups

# Removed SchoolDayChecker class - replaced with SchoolDayLookup system
# CRITICAL: Set up logging BEFORE importing modules that use logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('t8_monitor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Now import modules that use logging - they will inherit the configuration
from school_day_lookup import SchoolDayLookup

# Enable debug logging if DEBUG environment variable is set
if os.getenv('DEBUG', '').lower() in ['true', '1', 'yes']:
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger('urllib3').setLevel(logging.INFO)  # Reduce HTTP noise
    logging.getLogger('aiohttp').setLevel(logging.INFO)  # Reduce HTTP noise
logger = logging.getLogger(__name__)

# Force unbuffered output for logging to prevent log file not updating
import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    # reconfigure is not available in older Python versions
    pass

# Ensure file handler flushes immediately
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.FileHandler):
        handler.flush()

# Add periodic flushing function to call after log writes
def flush_logs():
    """Force flush all log handlers to ensure immediate write to disk"""
    for handler in logging.getLogger().handlers:
        if hasattr(handler, 'flush'):
            handler.flush()

# Initialize the school day lookup system with comprehensive validation
def initialize_school_day_system():
    """
    Initialize the SchoolDayLookup system with comprehensive validation.
    
    Returns:
        tuple: (school_day_lookup_instance, initialization_success)
    """
    logger.info("🚀 Initializing School Day Lookup System...")
    
    try:
        # Initialize the lookup system
        lookup_system = SchoolDayLookup()
        logger.info("✅ SchoolDayLookup system initialized successfully")
        
        # Validate database connectivity
        db_status = lookup_system.get_database_status()
        logger.info(f"📊 Database status: {db_status}")
        
        # Extract the current status from the dictionary
        current_status = db_status.get('current_status', 'unknown') if isinstance(db_status, dict) else str(db_status)
        
        if current_status.lower() != 'healthy':
            logger.warning(f"⚠️ Database status is not optimal: {db_status}")
            logger.info("🛡️ System will use fallback mechanisms if needed")
        
        # Get performance statistics
        perf_stats = lookup_system.get_performance_stats()
        cache_size = perf_stats.get('cache_size', 0)
        hit_rate = perf_stats.get('hit_rate_percent', 0)
        
        logger.info(f"💾 Cache status: {cache_size} entries loaded, {hit_rate:.1f}% hit rate")
        
        # Test with today's date to ensure system is working
        from datetime import datetime
        import dateutil.tz
        
        today = datetime.now(dateutil.tz.gettz('Australia/Sydney'))
        test_result = lookup_system.lookup_date(today)
        
        if test_result:
            status = "school day" if test_result.is_school_day else "non-school day"
            logger.info(f"🧪 System test: Today ({today.strftime('%Y-%m-%d')}) is a {status}")
            logger.info(f"📋 Reason: {test_result.reason}")
            
            # Log lookup performance
            lookup_time = test_result.lookup_time_ms or 0
            if lookup_time < 1.0:
                logger.info(f"⚡ Lookup performance: {lookup_time:.3f}ms (sub-1ms achieved!)")
            else:
                logger.info(f"⏱️ Lookup performance: {lookup_time:.2f}ms")
        else:
            logger.warning("⚠️ System test failed - no result returned for today's date")
        
        # Check if current year data is available
        current_year = today.year
        year_stats = perf_stats.get('years_cached', [])
        
        if current_year in year_stats:
            logger.info(f"📅 Calendar data for {current_year} is available and cached")
            
            # Quick data completeness check
            try:
                test_result_today = lookup_system.lookup_date(today)
                if test_result_today:
                    logger.info(f"✅ Data completeness verified - today's lookup successful")
                else:
                    logger.warning(f"⚠️ Data completeness issue - today's lookup failed")
            except Exception as e:
                logger.warning(f"⚠️ Data completeness check failed: {e}")
        else:
            logger.warning(f"⚠️ No calendar data found for current year {current_year}")
            logger.error(f"🚨 CRITICAL: Calendar data missing for {current_year}")
            logger.info(f"💡 URGENT: Run 'python school_calendar_generator.py {current_year}' to generate data")
            logger.info("🛡️ System will use fallback mechanisms until data is available")
        
        # Initialize calendar automation system
        try:
            from calendar_automation import get_automation_system
            
            logger.info("🤖 Initializing calendar automation system...")
            automation = get_automation_system(school_day_lookup=lookup_system)
            
            # Perform initial automatic checks during startup
            logger.info("🔍 Performing startup calendar validation...")
            startup_results = automation.perform_automatic_checks()
            
            issues_count = len(startup_results.get('issues_found', []))
            if issues_count > 0:
                logger.warning(f"⚠️ Startup validation found {issues_count} calendar issues")
                for issue in startup_results['issues_found']:
                    logger.warning(f"   • {issue}")
                
                # Process any critical tasks immediately
                task_results = automation.process_pending_tasks()
                if task_results['completed'] > 0:
                    logger.info(f"✅ Automatically resolved {task_results['completed']} calendar issues")
            else:
                logger.info("✅ Startup calendar validation passed - all data up to date")
            
            # Start background monitoring if enabled
            if automation.enable_background_monitoring:
                automation.start_background_monitoring()
                logger.info("🚀 Calendar automation background monitoring started")
            else:
                logger.info("📋 Calendar automation initialized (background monitoring disabled)")
                
        except ImportError:
            logger.info("📋 Calendar automation system not available (optional feature)")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize calendar automation: {e}")
            logger.info("🛡️ Manual calendar management will be required")
        
        logger.info("✅ School Day Lookup System fully initialized and validated")
        flush_logs()  # Ensure initialization logs are written
        return lookup_system, True
        
    except ImportError as e:
        logger.error(f"❌ Import error initializing SchoolDayLookup: {e}")
        logger.error("💡 Ensure school_day_lookup.py and database modules are available")
        return None, False
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize SchoolDayLookup system: {e}")
        logger.error("🛡️ System will fall back to basic weekend checking")
        
        # Log additional debugging information
        if "database" in str(e).lower() or "connection" in str(e).lower():
            logger.error("🔧 Database connection issue detected")
            logger.error("💡 Check PostgreSQL service and credentials:")
            logger.error("   - Database server running on Raspberry Pi")
            logger.error("   - Correct DB_HOST, DB_USER, DB_PASSWORD environment variables")
            logger.error("   - Network connectivity to database server")
        
        return None, False

# Initialize the school day lookup system
school_day_lookup, initialization_success = initialize_school_day_system()

if not initialization_success:
    logger.warning("⚠️ SchoolDayLookup initialization failed - using fallback mode")
    logger.info("🔄 The monitor will continue with basic weekend checking")

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

# Hardcoded arrays removed - now using database-backed SchoolDayLookup system

LAST_TWEET_FILE = 'last_tweet_id.txt'
LAST_CHECK_FILE = 'last_check_time.txt'

def is_sydney_school_day(check_date):
    """
    Check if a given date is a Sydney school day using the high-performance SchoolDayLookup system.
    Implements comprehensive fallback mechanisms for graceful degradation.
    """
    return _school_day_with_fallbacks(check_date)

def _school_day_with_fallbacks(check_date):
    """
    Comprehensive school day checking with multiple fallback layers.
    
    Fallback hierarchy:
    1. SchoolDayLookup system (database-backed, sub-1ms)
    2. SchoolDayLookup fallback mechanisms (heuristic, cached, local file)
    3. Legacy heuristic calculation (NSW school calendar patterns)
    4. Basic weekend check (conservative fallback)
    
    Args:
        check_date: Date to check (datetime or date object)
        
    Returns:
        bool: True if it's a school day, False otherwise
    """
    from datetime import datetime, date
    
    # Convert datetime to date if necessary
    if isinstance(check_date, datetime):
        check_date = check_date.date()
    
    # Layer 1: Try SchoolDayLookup system (primary database + cache)
    try:
        if school_day_lookup:
            result = school_day_lookup.lookup_date(check_date)
            if result:
                # Log performance and source information
                lookup_time = getattr(result, 'lookup_time_ms', None)
                cache_hit = getattr(result, 'cache_hit', False)
                reason = getattr(result, 'reason', 'unknown')
                
                if lookup_time is not None:
                    source = "cache" if cache_hit else "database"
                    if lookup_time < 1.0:
                        logger.debug(f"✅ SchoolDayLookup ({source}): {check_date} -> {result.is_school_day} ({lookup_time:.3f}ms, {reason})")
                    else:
                        logger.debug(f"✅ SchoolDayLookup ({source}): {check_date} -> {result.is_school_day} ({lookup_time:.2f}ms, {reason})")
                else:
                    logger.debug(f"✅ SchoolDayLookup: {check_date} -> {result.is_school_day} ({reason})")
                
                return result.is_school_day
            else:
                logger.warning(f"⚠️ SchoolDayLookup returned no result for {check_date}")
                logger.info("💡 This may indicate missing calendar data or date outside loaded range")
        else:
            logger.debug("⚠️ SchoolDayLookup system not available - initialization may have failed")
    
    except ImportError as e:
        logger.warning(f"❌ SchoolDayLookup import error for {check_date}: {e}")
        logger.info("💡 Database modules may not be properly installed")
    except ConnectionError as e:
        logger.warning(f"❌ SchoolDayLookup connection error for {check_date}: {e}")
        logger.info("💡 Database server may be unavailable - check PostgreSQL service")
    except Exception as e:
        error_type = type(e).__name__
        logger.warning(f"❌ SchoolDayLookup error ({error_type}) for {check_date}: {e}")
        
        # Provide specific guidance based on error type
        error_str = str(e).lower()
        if "timeout" in error_str:
            logger.info("💡 Database timeout - server may be overloaded")
        elif "permission" in error_str or "access" in error_str:
            logger.info("💡 Database permission issue - check user privileges")
        elif "table" in error_str or "relation" in error_str:
            logger.info("💡 Database schema issue - check if tables exist")
    
    # Layer 2: Try SchoolDayLookup internal fallbacks (if system is available)
    try:
        if school_day_lookup:
            # The SchoolDayLookup system has its own comprehensive fallbacks
            # Try to use them directly
            fallback_result = school_day_lookup._comprehensive_fallback_lookup(check_date, "system_fallback")
            if fallback_result:
                logger.info(f"🛡️ SchoolDayLookup fallback: {check_date} is {'school day' if fallback_result.is_school_day else 'not school day'}")
                return fallback_result.is_school_day
    
    except Exception as e:
        logger.debug(f"SchoolDayLookup fallback failed: {e}")
    
    # Layer 3: Legacy heuristic calculation (NSW school calendar patterns)
    try:
        result = _legacy_school_day_heuristic(check_date)
        logger.info(f"🔄 Legacy heuristic fallback: {check_date} -> {result}")
        logger.info("💡 Using approximate NSW school calendar patterns and public holidays")
        return result
        
    except ImportError as e:
        logger.warning(f"❌ Legacy heuristic import error for {check_date}: {e}")
        logger.info("💡 holidays library may not be installed: pip install holidays")
    except Exception as e:
        error_type = type(e).__name__
        logger.warning(f"❌ Legacy heuristic error ({error_type}) for {check_date}: {e}")
        logger.info("💡 Fallback calculation failed - using most basic check")
    
    # Layer 4: Basic weekend check (most conservative fallback)
    try:
        is_weekday = check_date.weekday() < 5  # Monday = 0, Friday = 4
        
        logger.warning(f"🚨 Using basic weekend check: {check_date} -> {'weekday' if is_weekday else 'weekend'}")
        logger.error("💡 All advanced school day checking methods failed!")
        logger.info("   Recommendations:")
        logger.info("   - Check database connectivity and PostgreSQL service status")
        logger.info("   - Verify calendar data exists for the current year")
        logger.info("   - Ensure all required Python packages are installed")
        logger.info("   - Review system logs for specific error details")
        
        # Log fallback usage for monitoring purposes
        logger.info(f"📊 Final fallback layer used: basic_weekend_check")
        
        return is_weekday
        
    except Exception as e:
        # This should never happen, but just in case
        logger.critical(f"❌ CRITICAL: Even basic weekend check failed for {check_date}: {e}")
        logger.critical("🚨 System is in an unexpected state - defaulting to 'not school day'")
        logger.critical("💡 This indicates a serious system issue - immediate investigation required")
        return False

def _legacy_school_day_heuristic(check_date):
    """
    Legacy school day calculation using NSW school calendar patterns.
    This provides a more sophisticated fallback than just weekend checking.
    
    Args:
        check_date: Date to check
        
    Returns:
        bool: True if likely a school day based on heuristics
    """
    from datetime import date
    import calendar
    
    # First check if it's a weekend
    if check_date.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    # Check for obvious public holidays (approximate dates)
    month = check_date.month
    day = check_date.day
    
    # New Year's Day
    if month == 1 and day == 1:
        return False
    
    # Australia Day (January 26)
    if month == 1 and day == 26:
        return False
    
    # Christmas period (December 25-31, January 1-31)
    if (month == 12 and day >= 25) or (month == 1):
        return False
    
    # Easter period (approximate - first two weeks of April)
    if month == 4 and day <= 14:
        return False
    
    # ANZAC Day (April 25)
    if month == 4 and day == 25:
        return False
    
    # Queen's Birthday (second Monday in June - approximate)
    if month == 6 and 8 <= day <= 14:
        return False
    
    # Labour Day (first Monday in October - approximate)
    if month == 10 and 1 <= day <= 7:
        return False
    
    # NSW school holiday periods (approximate)
    # Term 1: Late January to early April
    # Term 2: Late April to early July  
    # Term 3: Mid July to late September
    # Term 4: Mid October to mid December
    
    # School holiday periods (conservative estimates)
    school_holiday_periods = [
        # Summer holidays (December - January)
        (month == 12) or (month == 1),
        
        # Autumn holidays (mid April)
        (month == 4 and 15 <= day <= 25),
        
        # Winter holidays (early July)
        (month == 7 and 1 <= day <= 15),
        
        # Spring holidays (late September - early October)
        (month == 9 and day >= 25) or (month == 10 and day <= 12)
    ]
    
    if any(school_holiday_periods):
        return False
    
    # If we get here, it's likely a school day (weekday during term time)
    return True

def _get_fallback_status():
    """
    Get the current status of fallback systems for monitoring and debugging.
    
    Returns:
        dict: Status information about fallback systems
    """
    from datetime import datetime
    
    status = {
        'timestamp': datetime.now().isoformat(),
        'school_day_lookup_available': school_day_lookup is not None,
        'initialization_success': initialization_success,
        'fallback_layers': [
            'SchoolDayLookup system (database-backed)',
            'SchoolDayLookup internal fallbacks',
            'Legacy heuristic calculation',
            'Basic weekend check'
        ]
    }
    
    if school_day_lookup:
        try:
            db_status = school_day_lookup.get_database_status()
            perf_stats = school_day_lookup.get_performance_stats()
            
            status.update({
                'database_status': db_status.get('current_status', 'unknown') if isinstance(db_status, dict) else str(db_status),
                'cache_size': perf_stats.get('cache_size', 0),
                'fallback_usage_rate': perf_stats.get('fallback_system', {}).get('fallback_rate_percent', 0)
            })
        except Exception as e:
            status['status_error'] = str(e)
    
    return status

async def test_fallback_system():
    """
    Comprehensive test of the fallback system for startup validation.
    Tests all fallback layers to ensure graceful degradation works properly.
    """
    logger.info('🧪 Testing comprehensive fallback system...')
       
    from datetime import datetime, date, timedelta
    import dateutil.tz

    test_results = {
        'timestamp': datetime.now().isoformat(),
        'tests_passed': 0,
        'total_tests': 0,
        'layer_results': {}
    }
    
    # Test dates: today, weekend, obvious holiday, school holiday period
    sydney_tz = dateutil.tz.gettz('Australia/Sydney')
    today = datetime.now(sydney_tz).date()
    
    test_dates = [
        ('today', today),
        ('weekend', today + timedelta(days=(5 - today.weekday()) % 7)),  # Next Saturday
        ('christmas', date(today.year, 12, 25)),
        ('january_holidays', date(today.year, 1, 15)),
        ('term_time_weekday', date(today.year, 3, 15)),  # Mid-March (usually term time)
    ]
    
    logger.info(f'🎯 Testing fallback system with {len(test_dates)} test dates...')
    
    for test_name, test_date in test_dates:
        test_results['total_tests'] += 1
        
        try:
            # Test the complete fallback system
            result = _school_day_with_fallbacks(test_date)
            
            # Validate result is boolean
            if isinstance(result, bool):
                test_results['tests_passed'] += 1
                test_results['layer_results'][test_name] = {
                    'date': test_date.isoformat(),
                    'result': result,
                    'status': 'passed'
                }
                logger.debug(f'✅ {test_name} ({test_date}): {"school day" if result else "not school day"}')
            else:
                test_results['layer_results'][test_name] = {
                    'date': test_date.isoformat(),
                    'result': None,
                    'status': 'failed',
                    'error': f'Invalid result type: {type(result)}'
                }
                logger.warning(f'❌ {test_name} ({test_date}): Invalid result type {type(result)}')
                
        except Exception as e:
            test_results['layer_results'][test_name] = {
                'date': test_date.isoformat(),
                'result': None,
                'status': 'failed',
                'error': str(e)
            }
            logger.warning(f'❌ {test_name} ({test_date}): {e}')
    
    # Calculate success rate
    success_rate = (test_results['tests_passed'] / test_results['total_tests']) * 100 if test_results['total_tests'] > 0 else 0
    
    if success_rate >= 100:
        logger.info(f'✅ Fallback system test: {success_rate:.0f}% success rate ({test_results["tests_passed"]}/{test_results["total_tests"]})')
    elif success_rate >= 80:
        logger.warning(f'⚠️ Fallback system test: {success_rate:.0f}% success rate ({test_results["tests_passed"]}/{test_results["total_tests"]})')
    else:
        logger.error(f'❌ Fallback system test: {success_rate:.0f}% success rate ({test_results["tests_passed"]}/{test_results["total_tests"]})')
    
    # Get system status
    fallback_status = _get_fallback_status()
    logger.info(f'🛡️ Fallback system status: {len(fallback_status["fallback_layers"])} layers available')
    
    if school_day_lookup:
        logger.info('✅ Primary SchoolDayLookup system available')
    else:
        logger.info('⚠️ Primary SchoolDayLookup system unavailable - using fallbacks')
    
    return success_rate >= 80  # Return True if at least 80% of tests passed

async def validate_calendar_data_availability():
    """
    Comprehensive validation of calendar data availability for the current year.
    Provides actionable guidance when data is missing or incomplete.
    
    Returns:
        dict: Validation results with status, recommendations, and actions
    """
    logger.info('📅 Validating calendar data availability...')
    
    from datetime import datetime, date, timedelta
    import dateutil.tz
    
    validation_result = {
        'timestamp': datetime.now().isoformat(),
        'current_year_available': False,
        'next_year_available': False,
        'data_completeness': 0.0,
        'recommendations': [],
        'critical_issues': [],
        'status': 'unknown'
    }
    
    try:
        sydney_tz = dateutil.tz.gettz('Australia/Sydney')
        now = datetime.now(sydney_tz)
        current_year = now.year
        next_year = current_year + 1
        
        validation_result['current_year'] = current_year
        validation_result['next_year'] = next_year
        
        if school_day_lookup:
            # Get performance statistics to check cached years
            perf_stats = school_day_lookup.get_performance_stats()
            years_cached = perf_stats.get('years_cached', [])
            cache_size = perf_stats.get('cache_size', 0)
            
            logger.info(f'📊 Years cached: {years_cached}')
            logger.info(f'💾 Total cache entries: {cache_size}')
            
            # Check current year availability
            if current_year in years_cached:
                validation_result['current_year_available'] = True
                logger.info(f'✅ Calendar data for {current_year} is available')
                
                # Test data completeness by checking a few key dates
                test_dates = [
                    now.date(),  # Today
                    date(current_year, 1, 1),  # New Year
                    date(current_year, 12, 25),  # Christmas
                    date(current_year, 6, 15),  # Mid-year
                ]
                
                successful_lookups = 0
                total_tests = len(test_dates)
                
                for test_date in test_dates:
                    try:
                        result = school_day_lookup.lookup_date(test_date)
                        if result:
                            successful_lookups += 1
                    except Exception as e:
                        logger.debug(f'Lookup failed for {test_date}: {e}')
                
                validation_result['data_completeness'] = (successful_lookups / total_tests) * 100
                logger.info(f'📈 Data completeness: {validation_result["data_completeness"]:.1f}% ({successful_lookups}/{total_tests} test dates)')
                
            else:
                validation_result['current_year_available'] = False
                validation_result['critical_issues'].append(f'No calendar data found for current year {current_year}')
                logger.warning(f'❌ No calendar data for current year {current_year}')
            
            # Check next year availability (important for year transitions)
            if next_year in years_cached:
                validation_result['next_year_available'] = True
                logger.info(f'✅ Calendar data for {next_year} is available (good for year transition)')
            else:
                # Only warn about next year if we're in Q4
                if now.month >= 10:  # October or later
                    validation_result['recommendations'].append(f'Consider generating calendar data for {next_year} (approaching year transition)')
                    logger.info(f'💡 Consider generating calendar data for {next_year} (year transition approaching)')
                else:
                    logger.debug(f'Next year ({next_year}) data not yet needed')
        
        else:
            validation_result['critical_issues'].append('SchoolDayLookup system not available')
            logger.warning('❌ SchoolDayLookup system not available - cannot validate calendar data')
        
        # Determine overall status
        if validation_result['current_year_available'] and validation_result['data_completeness'] >= 90:
            validation_result['status'] = 'excellent'
            logger.info('🎯 Calendar data status: EXCELLENT')
            
        elif validation_result['current_year_available'] and validation_result['data_completeness'] >= 70:
            validation_result['status'] = 'good'
            logger.info('✅ Calendar data status: GOOD')
            validation_result['recommendations'].append('Some test dates failed - consider regenerating calendar data')
            
        elif validation_result['current_year_available']:
            validation_result['status'] = 'degraded'
            logger.warning('⚠️ Calendar data status: DEGRADED')
            validation_result['recommendations'].append('Data completeness is low - regenerate calendar data recommended')
            
        else:
            validation_result['status'] = 'critical'
            logger.error('❌ Calendar data status: CRITICAL')
            validation_result['critical_issues'].append('Current year calendar data is missing')
            validation_result['recommendations'].extend([
                f'URGENT: Generate calendar data for {current_year}',
                f'Run: python school_calendar_generator.py {current_year}',
                'System will use fallback mechanisms until data is available'
            ])
        
        # Add specific recommendations based on findings
        if validation_result['critical_issues']:
            logger.error('🚨 Critical calendar data issues found:')
            for issue in validation_result['critical_issues']:
                logger.error(f'   • {issue}')
        
        if validation_result['recommendations']:
            logger.info('💡 Calendar data recommendations:')
            for rec in validation_result['recommendations']:
                logger.info(f'   • {rec}')
        
        return validation_result
        
    except Exception as e:
        logger.error(f'❌ Calendar data validation failed: {e}')
        validation_result['status'] = 'error'
        validation_result['critical_issues'].append(f'Validation error: {str(e)}')
        return validation_result

def is_within_time_window(check_time):
    aest = dateutil.tz.gettz('Australia/Sydney')
    check_time = check_time.astimezone(aest)
    hour = check_time.hour
    minute = check_time.minute
    morning_window = (hour == 7 or (hour == 8 and minute <= 45))
    afternoon_window = (12 <= hour < 16)
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

def load_last_check_time():
    """Load the last check time from file, or return 1 hour ago if first run"""
    try:
        if os.path.exists(LAST_CHECK_FILE):
            with open(LAST_CHECK_FILE, 'r') as f:
                time_str = f.read().strip()
                return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
    except Exception as e:
        logger.warning(f'Error loading last check time: {e}')
    
    # Default: check tweets from 1 hour ago
    return datetime.now(dateutil.tz.UTC) - timedelta(hours=1)

def save_last_check_time(check_time):
    """Save the last check time to file"""
    try:
        with open(LAST_CHECK_FILE, 'w') as f:
            f.write(check_time.isoformat())
    except Exception as e:
        logger.warning(f'Error saving last check time: {e}')

async def analyze_tweet_with_ollama(tweet_text, tweet_source="Unknown"):
    """
    Use local Ollama LLM to analyze if a tweet indicates service disruption
    Returns: (should_alert: bool, confidence: str, reasoning: str)
    """
    try:
        prompt = f"""You are analyzing a tweet from T8 Sydney Trains to determine if passengers should be alerted about service disruptions.

Tweet Source: {tweet_source}
Tweet: "{tweet_text}"

CRITICAL FILTERING RULES:
1. ONLY alert for T8 Airport Line service disruptions
2. IGNORE tweets from Sydney Metro (@SydneyMetro) - these are about metro services, not T8 trains
3. ONLY alert for IMMEDIATE/CURRENT disruptions, NOT future/scheduled events
4. IGNORE weekend trackwork announcements (these are planned maintenance)

Analyze this tweet and determine if it indicates IMMEDIATE/CURRENT T8 Airport Line disruptions requiring urgent passenger action:
- Service delays, disruptions, or cancellations happening NOW on T8 line
- Track/signal/power issues currently affecting T8 services  
- Reduced T8 services or altered timetables in effect NOW
- Emergency situations currently affecting T8 trains
- T8 platform changes or shuttle bus replacements active NOW
- Any situation requiring "extra travel time" on T8 line RIGHT NOW

ALERT-WORTHY examples (IMMEDIATE T8 disruptions):
- "T8 Airport Line: Allow extra travel time due to..." (happening NOW)
- "T8 services suspended/cancelled/delayed" (current disruption)
- "T8 trains not running between..." (active now)
- "Airport Line reduced service due to..." (current issue)
- "T8 platform changes" (immediate change)
- "Shuttle buses replacing T8 trains" (active replacement)

NOT alert-worthy (IGNORE these):
- Sydney Metro service announcements (different service line)
- "Services restored to normal"
- General information without T8 service impact
- Routine announcements
- "This weekend, metro services do not run..." (weekend trackwork)
- "Are you travelling on [future date]..." with trackwork
- "From [time] to [time], trains may run to a changed timetable"
- Scheduled trackwork announcements for future dates or weekends
- "Due to trackwork between [stations]" for future dates
- Weekend trackwork between stations (planned maintenance)
- Trackwork scheduled outside school hours (Monday to Friday, 7am to 4pm)
- Any announcement asking about future travel with planned disruptions
- Tweets about bus replacements for weekend trackwork
- "Buses replace services between..." for weekend maintenance

Respond EXACTLY in this format:
ALERT: YES or NO
CONFIDENCE: HIGH or MEDIUM or LOW  
REASONING: Brief explanation"""

        # Log the tweet being analyzed
        logger.info(f'🔍 OLLAMA ANALYSIS START')
        logger.info(f'🏷️  Tweet Source: {tweet_source}')
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
        return await fallback_keyword_analysis(tweet_text, tweet_source)

async def fallback_keyword_analysis(tweet_text, tweet_source="Unknown"):
    """Fallback keyword analysis if Ollama fails"""
    try:
        text = tweet_text.lower()
        
        # Check for weekend trackwork patterns (should NOT alert)
        weekend_trackwork_patterns = [
            'this weekend',
            'weekend trackwork',
            'buses replace services',
            'metro services do not run',
            'planned trackwork',
            'scheduled maintenance',
            'weekend closure',
            'weekend disruption'
        ]
        
        # Check for Sydney Metro source (should NOT alert for T8 monitor)
        sydney_metro_patterns = [
            'sydney metro',
            'sydneymetro',
            'metro services',
            'tallawong',
            'sydenham'
        ]
        
        # Check if this is weekend trackwork
        is_weekend_trackwork = any(pattern in text for pattern in weekend_trackwork_patterns)
        
        # Check if this is Sydney Metro (wrong service line)
        is_sydney_metro = any(pattern in text for pattern in sydney_metro_patterns) or 'sydneymetro' in tweet_source.lower()
        
        # Immediate disruption keywords
        delay_keywords = [
            'delay', 'disruption', 'cancelled', 'issue', 'suspended', 'stopped', 'problem',
            'extra travel time', 'allow extra', 'not running', 'service alert', 'altered',
            'incident', 'emergency', 'flooding', 'power supply', 'signal repairs', 
            'shuttle', 'reduced service', 'timetable order', 'longer journey', 'wait times',
            'repairs', 'urgent', 'limited', 'diverted', 'gaps', 'less frequent', 'late'
        ]
        
        # T8 specific keywords (higher priority)
        t8_keywords = [
            't8', 'airport line', 'airport train', 'green square', 'mascot', 'domestic airport',
            'international airport', 'wolli creek', 'tempe', 'sydenham'
        ]
        
        has_delay_content = any(keyword in text for keyword in delay_keywords)
        has_t8_content = any(keyword in text for keyword in t8_keywords)
        found_delay_keywords = [k for k in delay_keywords if k in text]
        found_t8_keywords = [k for k in t8_keywords if k in text]
        
        # Decision logic
        should_alert = False
        reasoning = ""
        
        if is_sydney_metro:
            should_alert = False
            reasoning = f"Sydney Metro tweet - not relevant for T8 Airport Line monitoring"
        elif is_weekend_trackwork:
            should_alert = False 
            reasoning = f"Weekend trackwork announcement - planned maintenance, not immediate disruption"
        elif has_delay_content and has_t8_content:
            should_alert = True
            reasoning = f"T8-specific disruption keywords found: {found_t8_keywords} + {found_delay_keywords}"
        elif has_delay_content and not has_t8_content:
            should_alert = False  # Conservative approach - need T8 context
            reasoning = f"General disruption keywords but no T8 context: {found_delay_keywords}"
        else:
            should_alert = False
            reasoning = "No relevant disruption keywords found"
        
        # Log fallback analysis
        logger.info(f'🔄 FALLBACK KEYWORD ANALYSIS:')
        logger.info(f'🏷️  Tweet Source: {tweet_source}')
        logger.info(f'📝 Tweet Text: "{tweet_text}"')
        logger.info(f'🔍 Delay Keywords: {found_delay_keywords}')
        logger.info(f'🚆 T8 Keywords: {found_t8_keywords}')
        logger.info(f'📅 Weekend Trackwork: {is_weekend_trackwork}')
        logger.info(f'🚇 Sydney Metro: {is_sydney_metro}')
        logger.info(f'📊 Alert Decision: {"YES" if should_alert else "NO"}')
        logger.info(f'💭 Reasoning: {reasoning}')
        logger.info(f'🔄 FALLBACK ANALYSIS END')
        logger.info(f'{"="*60}')
        
        return should_alert, "MEDIUM", reasoning
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
        time_diff = now - tweet_time
        
        logger.debug(f'🔍 TWEET PROCESSING DETAILS:')
        logger.debug(f'   Tweet ID: {tweet.id}')
        logger.debug(f'   Tweet Time: {tweet_time.strftime("%Y-%m-%d %H:%M:%S AEST")}')
        logger.debug(f'   Current Time: {now.strftime("%Y-%m-%d %H:%M:%S AEST")}')
        logger.debug(f'   Age: {str(time_diff).split(".")[0]}')
        logger.debug(f'   Text Preview: {tweet.text[:100]}...')
        
        # Check if current time is within monitoring window
        is_school_day = is_sydney_school_day(now)
        is_time_window = is_within_time_window(now)
        
        logger.debug(f'   School Day: {is_school_day}')
        logger.debug(f'   Time Window: {is_time_window}')
        
        # IMPORTANT: Process tweets even outside monitoring window for debugging
        # This helps identify if tweets are being retrieved but filtered out
        if not (is_school_day and is_time_window):
            logger.info(f'⏸️  Outside monitoring window, but processing for debug: {tweet.text[:50]}...')
            # Continue processing instead of returning False
        
        # Check if tweet was posted within 2 hours of current check time
        if time_diff > timedelta(hours=2):
            logger.info(f'⏰ Tweet older than 2 hours ({time_diff}), skipping: {tweet.text[:50]}...')
            return False
        
        # Log tweet processing start
        logger.info(f'🚆 PROCESSING TWEET:')
        logger.info(f'📅 Tweet Time: {tweet_time.strftime("%Y-%m-%d %H:%M:%S AEST")}')
        logger.info(f'⏰ Current Time: {now.strftime("%Y-%m-%d %H:%M:%S AEST")}')
        logger.info(f'⏱️ Tweet Age: {str(time_diff).split(".")[0]}')
        logger.info(f'📝 Full Tweet Text: "{tweet.text}"')
        
        # Determine tweet source for analysis - check for retweets
        tweet_source = "T8SydneyTrains"  # Default assumption since we're fetching from T8 account
        
        # Check if this is a retweet of Sydney Metro content
        if hasattr(tweet, 'referenced_tweets') and tweet.referenced_tweets:
            for ref in tweet.referenced_tweets:
                if ref.type == 'retweeted':
                    logger.info(f'🔄 This is a retweet - original tweet type: {ref.type}')
        
        # Check tweet text for Sydney Metro indicators
        if any(indicator in tweet.text.lower() for indicator in ['sydney metro', 'metro services', '@sydneymetro']):
            tweet_source = "SydneyMetro (retweeted by T8SydneyTrains)"
            logger.warning(f'⚠️  Detected Sydney Metro content in T8 feed: "{tweet.text[:100]}..."')
        
        # Use Ollama AI to analyze the tweet
        should_alert, confidence, reasoning = await analyze_tweet_with_ollama(tweet.text, tweet_source)
        
        # Determine if we should actually send alert based on monitoring window
        # Allow longer age limit for high confidence alerts (up to 4 hours)
        max_age = timedelta(hours=4) if confidence == "HIGH" else timedelta(hours=2)
        should_send_alert = should_alert and (is_school_day and is_time_window) and (time_diff <= max_age)
        
        if should_alert:
            if should_send_alert:
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
                logger.info(f'🔇 ALERT SUPPRESSED - Outside monitoring window or too old ({confidence} confidence)')
                logger.info(f'   Would have sent: {reasoning}')
                return False
        else:
            logger.info(f'❌ NO ALERT NEEDED ({confidence} confidence)')
            logger.info(f'   Reasoning: {reasoning}')
            return False
    except Exception as e:
        logger.error(f'Error processing tweet: {e}')
        return False

async def fetch_and_process_tweets():
    try:
        logger.debug('🔗 Creating X API client...')
        client = tweepy.Client(bearer_token=X_BEARER_TOKEN)
        
        try:
            logger.debug('👤 Looking up @T8SydneyTrains user...')
            user = client.get_user(username='T8SydneyTrains')
            user_id = user.data.id
            logger.debug(f'✅ Found user ID for @T8SydneyTrains: {user_id}')
        except Exception as e:
            logger.error(f'❌ Error fetching user ID for @T8SydneyTrains: {e}')
            if "429" in str(e):
                logger.error('   Rate limit exceeded. Try increasing POLLING_INTERVAL_MINUTES')
            elif "403" in str(e):
                logger.error('   Authentication failed. Check your X_BEARER_TOKEN')
            return False
        
        last_tweet_id = load_last_tweet_id()
        logger.debug(f'📋 Last processed tweet ID: {last_tweet_id or "None (first run)"}')
        
        try:
            # Increase lookback window for longer polling intervals
            lookback_hours = max(2, POLLING_INTERVAL_MINUTES // 30 + 1)
            since_time = datetime.now(dateutil.tz.UTC) - timedelta(hours=lookback_hours)
            logger.debug(f'🕐 Looking back {lookback_hours} hours to {since_time.strftime("%Y-%m-%d %H:%M:%S UTC")}')
            
            kwargs = {
                'max_results': 5,
                'tweet_fields': ['created_at', 'public_metrics'],
                'start_time': since_time
            }
            if last_tweet_id:
                kwargs['since_id'] = last_tweet_id
                logger.debug(f'🔄 Using since_id filter: {last_tweet_id}')
            
            logger.debug('📡 Fetching tweets from X API...')
            tweets = client.get_users_tweets(user_id, **kwargs)
            
            if not tweets.data:
                logger.debug('📭 No new tweets found in the specified time window')
                logger.debug(f'   Search criteria: user_id={user_id}, max_results=5, since={since_time}')
                if last_tweet_id:
                    logger.debug(f'   Since tweet ID: {last_tweet_id}')
                return True
            
            tweets_list = list(tweets.data)
            tweets_list.reverse()  # Process oldest first
            logger.info(f'📥 Retrieved {len(tweets_list)} new tweets from X API')
            
            alerts_sent = 0
            latest_tweet_id = last_tweet_id
            
            for i, tweet in enumerate(tweets_list):
                logger.debug(f'🔍 Processing tweet {i+1}/{len(tweets_list)}: ID {tweet.id}')
                logger.debug(f'   Text: {tweet.text[:100]}...')
                logger.debug(f'   Created: {tweet.created_at}')
                
                if await process_tweet(tweet):
                    alerts_sent += 1
                latest_tweet_id = tweet.id
            
            if latest_tweet_id and latest_tweet_id != last_tweet_id:
                save_last_tweet_id(latest_tweet_id)
                logger.debug(f'💾 Saved latest tweet ID: {latest_tweet_id}')
            
            if alerts_sent > 0:
                logger.info(f'✅ X API: Processed {len(tweets_list)} tweets, sent {alerts_sent} alerts')
            else:
                logger.debug(f'📋 X API: Processed {len(tweets_list)} tweets, no alerts sent')
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
    
    TwitterAPI.io actual format:
    {
        "type": "tweet",
        "id": "1957321619568808194",
        "url": "https://x.com/T8SydneyTrains/status/1957321619568808194",
        "text": "Tweet content here",
        "createdAt": "Mon Aug 18 06:00:14 +0000 2025",
        "retweetCount": 1,
        "replyCount": 0,
        "likeCount": 2,
        "quoteCount": 0,
        "viewCount": 417,
        "lang": "en",
        ...
    }
    
    Internal format needed:
    - tweet.id (int)
    - tweet.text (str)  
    - tweet.created_at (datetime with timezone)
    - tweet.public_metrics (dict, optional)
    """
    try:
        if not tweet_data or 'id' not in tweet_data:
            logger.debug(f'⚠️  Invalid tweet data: missing id')
            logger.debug(f'   Available keys: {list(tweet_data.keys()) if isinstance(tweet_data, dict) else "Not a dict"}')
            return None
        
        # TwitterAPI.io uses 'text' field, but let's check both 'text' and 'full_text'
        text = tweet_data.get('text') or tweet_data.get('full_text', '')
        if not text:
            logger.debug(f'⚠️  Invalid tweet data: missing text content')
            logger.debug(f'   Available keys: {list(tweet_data.keys())}')
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
        
        # Parse created_at timestamp - TwitterAPI.io uses 'createdAt' field
        created_at_str = tweet_data.get('createdAt') or tweet_data.get('created_at', '')
        try:
            if created_at_str:
                # TwitterAPI.io returns format like: "Mon Aug 18 06:00:14 +0000 2025"
                # This is Twitter's standard format, need to parse it properly
                import dateutil.parser
                created_at = dateutil.parser.parse(created_at_str)
                # Ensure it's timezone-aware
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=dateutil.tz.UTC)
            else:
                # Fallback to current time if no timestamp provided
                logger.debug(f'⚠️  No createdAt timestamp for tweet {tweet_id}, using current time')
                created_at = datetime.now(dateutil.tz.UTC)
        except (ValueError, TypeError) as e:
            logger.debug(f'⚠️  Invalid createdAt format for tweet {tweet_id}: {created_at_str} - {e}')
            created_at = datetime.now(dateutil.tz.UTC)
        
        # Extract public metrics (optional) - TwitterAPI.io uses different field names
        public_metrics = {
            'retweet_count': tweet_data.get('retweetCount', 0),
            'like_count': tweet_data.get('likeCount', 0),
            'reply_count': tweet_data.get('replyCount', 0),
            'quote_count': tweet_data.get('quoteCount', 0),
            'view_count': tweet_data.get('viewCount', 0)
        }
        
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
    TwitterAPI.io implementation using Advanced Search API (from blog post)
    Only fetches NEW tweets since last check - much more efficient
    """
    try:
        # Load last check time (blog post approach)
        raw_last_checked_time = load_last_check_time()
        current_time = datetime.now(dateutil.tz.UTC)

        # Clamp the lookback window to match X API behaviour
        lookback_hours = max(2, POLLING_INTERVAL_MINUTES // 30 + 1)
        earliest_allowed = current_time - timedelta(hours=lookback_hours)
        last_checked_time = max(raw_last_checked_time, earliest_allowed)

        if last_checked_time != raw_last_checked_time:
            logger.debug(f"Clamped TwitterAPI.io lookback start: {raw_last_checked_time.isoformat()} -> {last_checked_time.isoformat()}")

        # Format times as strings in the format Twitter's API expects (from blog)
        since_str = last_checked_time.strftime("%Y-%m-%d_%H:%M:%S_UTC")
        until_str = current_time.strftime("%Y-%m-%d_%H:%M:%S_UTC")
        
        # Construct the query (exact format from blog post) - exclude retweets to avoid Sydney Metro content
        query = f"from:T8SydneyTrains since:{since_str} until:{until_str} -is:retweet"
        
        # API endpoint (from blog post)
        url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
        
        # Request parameters (from blog post)
        params = {
            "query": query,
            "queryType": "Latest"
        }
        
        # Headers with API key (note: X-API-Key from blog, not x-api-key)
        headers = {
            "X-API-Key": TWITTERAPI_IO_KEY
        }
        
        logger.info(f'🔗 TwitterAPI.io Advanced Search (blog post method)')
        logger.info(f'📋 Query: {query}')
        logger.info(f'🕐 Time window: {since_str} to {until_str}')
        
        # Make the request and handle pagination (from blog post)
        all_tweets = []
        next_cursor = None
        
        async with aiohttp.ClientSession() as session:
            while True:
                # Add cursor to params if we have one (pagination from blog)
                current_params = params.copy()
                if next_cursor:
                    current_params["cursor"] = next_cursor
                
                async with session.get(url, headers=headers, params=current_params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    # Log response details
                    logger.debug(f'📊 TwitterAPI.io response: {response.status}')
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f'❌ TwitterAPI.io Advanced Search failed: HTTP {response.status}')
                        logger.error(f'   Response: {error_text[:200]}...')
                        
                        # Handle specific error codes
                        if response.status == 401:
                            logger.error('🔑 Authentication failed - check TWITTERAPI_IO_KEY')
                        elif response.status == 429:
                            logger.error('⏰ Rate limit exceeded - reduce polling frequency')
                        elif response.status >= 500:
                            logger.error('🔧 TwitterAPI.io server error - try again later')
                        
                        return False
                    
                    # Parse the response (blog post structure)
                    data = await response.json()
                    tweets = data.get("tweets", [])
                    
                    if tweets:
                        all_tweets.extend(tweets)
                        logger.debug(f'📥 Found {len(tweets)} tweets in this page')
                    
                    # Check if there are more pages (blog post pagination logic)
                    if data.get("has_next_page", False) and data.get("next_cursor", "") != "":
                        next_cursor = data.get("next_cursor")
                        continue
                    else:
                        break
        
        # Process all collected tweets (blog post approach)
        if not all_tweets:
            logger.info('📭 No new tweets found since last check')
            save_last_check_time(current_time)  # Update timestamp even if no tweets
            return True
        
        logger.info(f'📥 Found {len(all_tweets)} NEW tweets from T8SydneyTrains (Advanced Search)')
        
        # Convert TwitterAPI.io format to our internal format and process
        alerts_sent = 0
        processed_tweets = []
        
        for i, tweet_data in enumerate(all_tweets):
            logger.debug(f'🔄 Converting tweet {i+1}/{len(all_tweets)}')
            
            # Convert TwitterAPI.io tweet to our internal format
            converted_tweet = convert_twitterapi_response(tweet_data)
            if not converted_tweet:
                logger.debug(f'   ❌ Failed to convert tweet {i+1}')
                continue
            
            logger.debug(f'✅ New tweet {i+1}: ID {converted_tweet.id}')
            logger.debug(f'   Text: {converted_tweet.text[:100]}...')
            
            processed_tweets.append(converted_tweet)
        
        # Process tweets in chronological order (oldest first)
        processed_tweets.reverse()
        
        for tweet in processed_tweets:
            if await process_tweet(tweet):
                alerts_sent += 1
        
        # Update the last checked time (blog post approach)
        save_last_check_time(current_time)
        
        # Log results
        if alerts_sent > 0:
            logger.info(f'✅ TwitterAPI.io Advanced Search: Processed {len(processed_tweets)} tweets, sent {alerts_sent} alerts')
        else:
            logger.debug(f'📋 TwitterAPI.io Advanced Search: Processed {len(processed_tweets)} tweets, no alerts sent')
        
        return True
        
    except aiohttp.ClientError as e:
        logger.error(f'❌ TwitterAPI.io network error: {e}')
        return False
    except asyncio.TimeoutError:
        logger.error(f'⏰ TwitterAPI.io request timeout')
        return False
    except Exception as e:
        logger.error(f'❌ Error in fetch_tweets_twitterapi (Advanced Search): {e}')
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

async def test_database_connection():
    """Test SchoolDayLookup database connection for startup validation with enhanced error handling"""
    try:
        if not school_day_lookup:
            logger.error('❌ SchoolDayLookup system not initialized')
            logger.info('💡 This may indicate database connectivity issues or configuration problems')
            logger.info('   - Check PostgreSQL service on Raspberry Pi')
            logger.info('   - Verify DB_HOST, DB_USER, DB_PASSWORD environment variables')
            logger.info('   - Ensure network connectivity to database server')
            return False
        
        logger.debug('🧪 Testing SchoolDayLookup database connection...')
        
        # Test database connectivity with detailed status reporting
        db_status = school_day_lookup.get_database_status()
        current_status = db_status.get('current_status', 'unknown') if isinstance(db_status, dict) else str(db_status)
        
        if current_status.lower() == 'healthy':
            logger.info('✅ Database connection healthy')
            
            # Log additional database health metrics
            if isinstance(db_status, dict):
                if 'last_successful_query' in db_status:
                    logger.debug(f'   Last successful query: {db_status["last_successful_query"]}')
                if 'connection_pool_status' in db_status:
                    logger.debug(f'   Connection pool: {db_status["connection_pool_status"]}')
            
            # Test a lookup to ensure full functionality
            from datetime import datetime
            import dateutil.tz
            
            test_date = datetime.now(dateutil.tz.gettz('Australia/Sydney'))
            result = school_day_lookup.lookup_date(test_date)
            
            if result:
                lookup_time = result.lookup_time_ms or 0
                
                # Performance categorization with actionable insights
                if lookup_time < 1.0:
                    logger.info(f'✅ Database lookup successful - Excellent performance ({lookup_time:.3f}ms)')
                elif lookup_time < 10.0:
                    logger.info(f'✅ Database lookup successful - Good performance ({lookup_time:.2f}ms)')
                else:
                    logger.warning(f'✅ Database lookup successful - Slow performance ({lookup_time:.2f}ms)')
                    logger.info('💡 Consider cache optimization or database performance tuning')
                
                # Log cache utilization if available
                if hasattr(result, 'cache_hit'):
                    cache_status = "cache hit" if result.cache_hit else "database query"
                    logger.debug(f'📊 Result source: {cache_status}')
                
                return True
            else:
                logger.error('❌ Database lookup returned no result')
                logger.warning('💡 This may indicate missing calendar data or date outside loaded range')
                logger.info(f'   - Check if calendar data exists for {test_date.year}')
                logger.info(f'   - Run: python school_calendar_generator.py {test_date.year}')
                return False
        
        elif current_status.lower() == 'degraded':
            logger.warning(f'⚠️ Database status: DEGRADED - performance issues detected')
            logger.info('💡 System will use fallbacks and caching to maintain functionality')
            logger.info('   - Monitor database server resources (CPU, memory, disk)')
            logger.info('   - Check for slow queries or connection pool exhaustion')
            return True  # System can still work with fallbacks
            
        elif current_status.lower() == 'recovering':
            logger.info(f'🔄 Database status: RECOVERING - system is stabilizing')
            logger.info('💡 Previous issues detected, automatic recovery in progress')
            return True  # System should stabilize automatically
            
        elif current_status.lower() == 'unavailable':
            logger.error(f'❌ Database status: UNAVAILABLE - complete database failure')
            logger.error('💡 Critical database connectivity issue:')
            logger.error('   - PostgreSQL service may be down on Raspberry Pi')
            logger.error('   - Network connectivity issues to database host')
            logger.error('   - Database server resource exhaustion')
            logger.info('🛡️ System will fall back to heuristic school day checking')
            return False
            
        else:
            logger.error(f'❌ Database status: {current_status.upper()} (unknown status)')
            logger.warning('💡 Unexpected database status - this may indicate a system issue')
            return False
            
    except ImportError as e:
        logger.error(f'❌ Import error during database test: {e}')
        logger.error('💡 Missing required database modules:')
        logger.error('   - Check if psycopg2-binary is installed: pip install psycopg2-binary')
        logger.error('   - Verify school_day_lookup.py module is available')
        logger.info('🛡️ System will fall back to basic weekend checking')
        return False
        
    except ConnectionError as e:
        logger.error(f'❌ Database connection failed: {e}')
        logger.error('💡 Network or database connectivity issue:')
        logger.error('   - Verify PostgreSQL service is running on Raspberry Pi')
        logger.error('   - Check network connectivity to database host')
        logger.error('   - Verify firewall settings allow database connections')
        logger.error('   - Test database credentials (DB_USER, DB_PASSWORD)')
        logger.info('🛡️ System will fall back to heuristic school day checking')
        return False
        
    except Exception as e:
        error_type = type(e).__name__
        logger.error(f'❌ Database connection test failed ({error_type}): {e}')
        
        # Provide context-specific error guidance
        error_str = str(e).lower()
        if "timeout" in error_str:
            logger.error('💡 Timeout error: Database response too slow')
            logger.error('   - Check database server performance and load')
            logger.error('   - Verify network latency to database host')
            logger.error('   - Consider increasing connection timeout settings')
        elif any(word in error_str for word in ["authentication", "password", "login"]):
            logger.error('💡 Authentication error: Invalid database credentials')
            logger.error('   - Verify DB_USER and DB_PASSWORD environment variables')
            logger.error('   - Check database user exists and has correct password')
            logger.error('   - Ensure database user has required permissions')
        elif "permission" in error_str or "access" in error_str:
            logger.error('💡 Permission error: Database user lacks required privileges')
            logger.error('   - Check database user permissions for school_calendar table')
            logger.error('   - Verify schema access rights and SELECT permissions')
        elif "table" in error_str or "relation" in error_str:
            logger.error('💡 Schema error: Required database tables missing')
            logger.error('   - Run database schema initialization')
            logger.error('   - Check if school_calendar table exists')
        else:
            logger.error('💡 Unexpected error: This may indicate a system or configuration issue')
            logger.debug(f'   Full error details: {repr(e)}')
        
        logger.info('🛡️ System will fall back to basic weekend checking')
        return False

async def log_heartbeat():
    """Log a heartbeat message to show the script is alive with enhanced system status."""
    try:
        # Basic heartbeat
        logger.info("💓 T8 Monitor heartbeat - system running normally")
        flush_logs()  # Ensure heartbeat is written immediately
        
        # Enhanced database status reporting during heartbeat
        if school_day_lookup:
            try:
                db_status = school_day_lookup.get_database_status()
                current_status = db_status.get('current_status', 'unknown') if isinstance(db_status, dict) else str(db_status)
                
                # Only log if status is not healthy
                if current_status.lower() != 'healthy':
                    logger.info(f"📊 Database status: {current_status.upper()}")
                    
                    # Provide actionable guidance based on status
                    if current_status.lower() == 'unavailable':
                        logger.warning("💡 Database unavailable - using fallback mechanisms")
                    elif current_status.lower() == 'degraded':
                        logger.info("💡 Database degraded - performance may be affected")
                    elif current_status.lower() == 'recovering':
                        logger.info("💡 Database recovering - should stabilize soon")
                
                # Log performance metrics if available
                try:
                    perf_stats = school_day_lookup.get_performance_stats()
                    total_lookups = perf_stats.get('total_lookups', 0)
                    cache_hit_rate = perf_stats.get('cache_hit_rate', 0)
                    avg_lookup_time = perf_stats.get('average_lookup_time_ms', 0)
                    
                    # Only log performance if there have been lookups since last heartbeat
                    if total_lookups > 0:
                        logger.debug(f"📈 Performance: {total_lookups} lookups, "
                                   f"{cache_hit_rate:.1%} cache hits, "
                                   f"{avg_lookup_time:.2f}ms avg")
                        
                        # Warn if performance is degrading
                        if avg_lookup_time > 10.0:
                            logger.warning(f"⏱️ Slow lookup performance: {avg_lookup_time:.2f}ms average")
                            logger.info("💡 Consider cache optimization or database tuning")
                        
                except Exception as perf_error:
                    logger.debug(f"Performance stats unavailable: {perf_error}")
                    
            except Exception as db_error:
                logger.debug(f"Database status check during heartbeat failed: {db_error}")
                
    except Exception as e:
        logger.warning(f"⚠️ Enhanced heartbeat logging failed: {e}")
        # Fallback to basic heartbeat
        logger.info("💓 T8 Monitor heartbeat - system running (basic mode)")

async def monitor_database_health():
    """
    Monitor database health and log issues with actionable recommendations.
    This function can be called periodically or triggered by errors.
    """
    if not school_day_lookup:
        logger.warning("📊 Database health check: SchoolDayLookup system not available")
        return
    
    try:
        logger.debug("🔍 Performing database health check...")
        
        # Get comprehensive health status
        health_status = school_day_lookup.get_health_status()
        
        overall_status = health_status.get('overall_status', 'unknown')
        logger.info(f"📊 Database health: {overall_status.upper()}")
        
        # Report specific issues
        if 'issues' in health_status and health_status['issues']:
            logger.warning("⚠️ Database health issues detected:")
            for issue in health_status['issues']:
                logger.warning(f"   • {issue}")
        
        # Report database connection status
        if 'database_connection' in health_status:
            if not health_status['database_connection']:
                logger.error("❌ Database connection failed")
                logger.error("💡 Troubleshooting steps:")
                logger.error("   - Check PostgreSQL service on Raspberry Pi")
                logger.error("   - Verify network connectivity")
                logger.error("   - Test database credentials")
        
        # Report cache status
        if 'cache_status' in health_status:
            cache_status = health_status['cache_status']
            if cache_status.get('status') != 'healthy':
                logger.warning(f"⚠️ Cache status: {cache_status.get('status', 'unknown')}")
                logger.info(f"💡 Cache entries: {cache_status.get('entry_count', 0)}")
        
        # Report fallback system status
        if 'fallback_system' in health_status:
            fallback_status = health_status['fallback_system']
            if fallback_status.get('status') != 'ready':
                logger.warning(f"⚠️ Fallback system: {fallback_status.get('status', 'unknown')}")
        
        # Performance analysis
        if 'performance' in health_status:
            perf = health_status['performance']
            avg_time = perf.get('average_lookup_time_ms', 0)
            
            if avg_time > 10.0:
                logger.warning(f"⏱️ Performance concern: {avg_time:.2f}ms average lookup time")
                logger.info("💡 Consider cache preloading or database optimization")
            elif avg_time > 1.0:
                logger.debug(f"📈 Performance: {avg_time:.2f}ms average lookup time")
        
        return overall_status
        
    except Exception as e:
        logger.error(f"❌ Database health check failed: {e}")
        logger.info("💡 This may indicate a serious system issue requiring investigation")
        return 'error'

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

async def startup_validation():
    """Comprehensive startup validation and diagnostics"""
    logger.info('🔍 Starting T8 Monitor Startup Validation...')
    
    # Check environment variables
    logger.info('📋 Checking environment variables...')
    if USE_TWITTERAPI_IO:
        required_vars = ['TWITTERAPI_IO_KEY', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
        api_type = 'TwitterAPI.io'
    else:
        required_vars = ['X_BEARER_TOKEN', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
        api_type = 'X API'
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f'❌ Missing environment variables: {", ".join(missing_vars)}')
        logger.error(f'   Run "python quick_setup.py" to configure credentials')
        return False
    
    logger.info(f'✅ All required environment variables present for {api_type}')
    
    # Validate SchoolDayLookup system status
    logger.info('📊 Validating School Day Lookup System...')
    if school_day_lookup and initialization_success:
        try:
            # Get system health status
            db_status = school_day_lookup.get_database_status()
            perf_stats = school_day_lookup.get_performance_stats()
            current_status = db_status.get('current_status', 'unknown') if isinstance(db_status, dict) else str(db_status)
            
            logger.info(f'✅ SchoolDayLookup system operational')
            logger.info(f'📊 Database status: {current_status}')
            logger.info(f'💾 Cache entries: {perf_stats.get("cache_size", 0)}')
            logger.info(f'🎯 Cache hit rate: {perf_stats.get("hit_rate_percent", 0):.1f}%')
            
            # Validate current year data availability
            from datetime import datetime
            current_year = datetime.now(dateutil.tz.gettz('Australia/Sydney')).year
            years_cached = perf_stats.get('years_cached', [])
            
            if current_year in years_cached:
                logger.info(f'📅 Calendar data for {current_year} is available')
            else:
                logger.warning(f'⚠️ No calendar data for current year {current_year}')
                logger.warning('💡 System will use fallback mechanisms')
            
        except Exception as e:
            logger.error(f'❌ SchoolDayLookup system validation failed: {e}')
            logger.warning('🛡️ System will fall back to basic weekend checking')
    else:
        logger.warning('⚠️ SchoolDayLookup system not available')
        logger.info('🔄 Using comprehensive fallback system')
        
        # Test fallback system functionality
        logger.info('🧪 Testing fallback system...')
        try:
            from datetime import datetime
            test_date = datetime.now(dateutil.tz.gettz('Australia/Sydney')).date()
            fallback_result = _school_day_with_fallbacks(test_date)
            logger.info(f'✅ Fallback system test: {test_date} = {"school day" if fallback_result else "not school day"}')
            
            # Get fallback status
            fallback_status = _get_fallback_status()
            logger.info(f'🛡️ Fallback layers available: {len(fallback_status["fallback_layers"])}')
            
        except Exception as e:
            logger.error(f'❌ Fallback system test failed: {e}')
    
    # Check current time window
    now = datetime.now(dateutil.tz.gettz('Australia/Sydney'))
    is_school_day = is_sydney_school_day(now)
    is_time_window = is_within_time_window(now)
    
    logger.info(f'🕐 Current time: {now.strftime("%Y-%m-%d %H:%M:%S AEST (%A)")}')
    logger.info(f'📚 School day: {"Yes" if is_school_day else "No"}')
    logger.info(f'⏰ Monitoring window: {"Yes" if is_time_window else "No"}')
    
    if not (is_school_day and is_time_window):
        logger.warning('⚠️  Currently outside monitoring window')
        logger.info('   Monitoring windows: 7:00-8:45 AM and 12:00-4:00 PM AEST on school days')
    
    return True

async def main():
    logger.info('🚀 Starting T8 Delays Monitor with Ollama AI Analysis...')
    flush_logs()  # Ensure startup message is written
    
    # Comprehensive startup validation
    if not await startup_validation():
        logger.error('❌ Startup validation failed. Exiting.')
        return
    
    # Test connections
    logger.info('🔗 Testing API connections...')
    
    if not await test_telegram_connection():
        logger.error('❌ Failed to connect to Telegram. Exiting.')
        logger.error('   Check your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID')
        return
    
    # Test the appropriate API backend based on feature flag
    if USE_TWITTERAPI_IO:
        if not await test_twitterapi_connection():
            logger.error('❌ Failed to connect to TwitterAPI.io. Exiting.')
            logger.error('   Check your TWITTERAPI_IO_KEY')
            return
    else:
        if not await test_twitter_connection():
            logger.error('❌ Failed to connect to Twitter API. Exiting.')
            logger.error('   Check your X_BEARER_TOKEN')
            return
    
    # Test Ollama connection (non-blocking)
    await test_ollama_connection()
    
    # Test database connection (non-blocking - system can work with fallbacks)
    logger.info('🗄️ Testing database connection...')
    db_test_result = await test_database_connection()
    if not db_test_result:
        logger.warning('⚠️ Database connection test failed - continuing with fallback mode')
        logger.info('🛡️ The monitor will use comprehensive fallback system when database is unavailable')
    
    # Test fallback system (non-blocking - ensures graceful degradation works)
    logger.info('🛡️ Testing fallback system for graceful degradation...')
    fallback_test_result = await test_fallback_system()
    if fallback_test_result:
        logger.info('✅ Fallback system test passed - graceful degradation confirmed')
    else:
        logger.warning('⚠️ Fallback system test had some issues - monitor will continue but may have reduced reliability')
    
    # Comprehensive calendar data validation (critical for proper operation)
    logger.info('📅 Performing comprehensive calendar data validation...')
    calendar_validation = await validate_calendar_data_availability()
    
    if calendar_validation['status'] == 'excellent':
        logger.info('🎯 Calendar data validation: EXCELLENT - optimal performance expected')
    elif calendar_validation['status'] == 'good':
        logger.info('✅ Calendar data validation: GOOD - system ready for operation')
    elif calendar_validation['status'] == 'degraded':
        logger.warning('⚠️ Calendar data validation: DEGRADED - consider regenerating data for optimal performance')
    elif calendar_validation['status'] == 'critical':
        logger.error('❌ Calendar data validation: CRITICAL - system will rely heavily on fallback mechanisms')
        logger.error('🚨 URGENT: Generate calendar data to ensure accurate school day detection')
    else:
        logger.error('❌ Calendar data validation: ERROR - validation process failed')
    
    # Provide startup summary
    logger.info('📊 Startup Validation Summary:')
    logger.info(f'   📡 API connections: {"✅" if True else "❌"}')  # APIs tested above
    logger.info(f'   🗄️ Database: {"✅" if db_test_result else "⚠️ (fallback mode)"}')
    logger.info(f'   🛡️ Fallback system: {"✅" if fallback_test_result else "⚠️ (issues detected)"}')
    logger.info(f'   📅 Calendar data: {"✅" if calendar_validation["status"] in ["excellent", "good"] else "⚠️ (degraded/critical)"}')
    
    logger.info('All connections, systems, and data validated. Starting monitoring...')
    logger.info(f'🤖 Using Ollama model: {OLLAMA_MODEL} at {OLLAMA_HOST}')
    if USE_TWITTERAPI_IO:
        logger.info(f'🔧 Active API: TwitterAPI.io (Cost-effective, pay-per-use)')
    else:
        logger.info(f'🔧 Active API: X API (Traditional Twitter API)')
    logger.info(f'Monitoring mode: Polling every {POLLING_INTERVAL_MINUTES} minutes during school days and peak hours')
    flush_logs()  # Ensure all startup info is written
    
    heartbeat_counter = 0
    
    try:
        while True:
            try:
                now = datetime.now(dateutil.tz.gettz('Australia/Sydney'))
                
                # Enhanced school day checking with error handling
                try:
                    is_school_day = is_sydney_school_day(now)
                    is_time_window = is_within_time_window(now)
                except Exception as e:
                    logger.warning(f"⚠️ Error checking school day status: {e}")
                    logger.info("🛡️ Falling back to basic time window checking")
                    # Fallback to basic time checking - assume weekdays during business hours
                    is_school_day = now.weekday() < 5  # Monday = 0, Friday = 4
                    is_time_window = is_within_time_window(now)
                
                if is_school_day and is_time_window:
                    logger.debug(f'📅 Monitoring active: {now.strftime("%Y-%m-%d %H:%M:%S")} (school day)')
                    
                    # Enhanced API backend selection with error handling
                    try:
                        fetch_function = api_backend_selector()
                        await fetch_function()
                    except Exception as e:
                        error_type = type(e).__name__
                        logger.error(f"❌ Twitter API error ({error_type}): {e}")
                        flush_logs()  # Ensure error is written immediately
                        
                        # Provide specific guidance based on error type
                        if "timeout" in str(e).lower():
                            logger.error("💡 API timeout - network or service issues")
                        elif "authentication" in str(e).lower() or "401" in str(e):
                            logger.error("💡 Authentication failed - check API credentials")
                        elif "rate limit" in str(e).lower() or "429" in str(e):
                            logger.error("💡 Rate limit exceeded - reduce polling frequency")
                        elif "connection" in str(e).lower():
                            logger.error("💡 Network connectivity issue - check internet connection")
                        
                        # Continue monitoring despite API errors
                        logger.info("🔄 Will retry on next polling cycle")
                    
                    # Use the configured polling interval from .env
                    await asyncio.sleep(POLLING_INTERVAL_SECONDS)
                    heartbeat_counter += 1
                else:
                    if not is_school_day:
                        logger.debug(f'📅 Outside school day: {now.strftime("%Y-%m-%d %H:%M:%S")} (non-school day)')
                    else:
                        logger.debug(f'⏰ Outside monitoring window: {now.strftime("%H:%M:%S")} (school day but outside hours)')
                    
                    await asyncio.sleep(POLLING_INTERVAL_SECONDS)
                    heartbeat_counter += 1
                
                # Enhanced heartbeat logging with system status
                cycles_for_heartbeat = max(1, int(120 / POLLING_INTERVAL_MINUTES))
                if heartbeat_counter >= cycles_for_heartbeat:
                    try:
                        await log_heartbeat()
                        
                        # Periodic system health check
                        if school_day_lookup:
                            try:
                                db_status = school_day_lookup.get_database_status()
                                current_status = db_status.get('current_status', 'unknown') if isinstance(db_status, dict) else str(db_status)
                                
                                if current_status.lower() != 'healthy':
                                    logger.info(f"📊 Database status during heartbeat: {current_status.upper()}")
                                    
                                    # Log cache performance if available
                                    try:
                                        perf_stats = school_day_lookup.get_performance_stats()
                                        if perf_stats.get('cache_hit_rate', 0) < 0.8:  # Less than 80% cache hit rate
                                            logger.info(f"📈 Cache performance: {perf_stats.get('cache_hit_rate', 0):.1%} hit rate")
                                    except Exception:
                                        pass  # Performance stats are optional
                                        
                            except Exception as e:
                                logger.debug(f"Heartbeat database status check failed: {e}")
                        
                        heartbeat_counter = 0
                    except Exception as e:
                        logger.warning(f"⚠️ Heartbeat logging failed: {e}")
                        heartbeat_counter = 0  # Reset counter to avoid repeated failures
                        
            except asyncio.CancelledError:
                logger.info("🛑 Monitoring task cancelled")
                break
            except Exception as loop_error:
                error_type = type(loop_error).__name__
                logger.error(f"❌ Error in monitoring cycle ({error_type}): {loop_error}")
                logger.info("🔄 Continuing monitoring after error...")
                
                # Add a small delay to prevent rapid error loops
                await asyncio.sleep(5)
                
    except KeyboardInterrupt:
        logger.info('🛑 Monitoring stopped by user (Ctrl+C)')
    except SystemExit:
        logger.info('🛑 System exit requested')
    except Exception as e:
        error_type = type(e).__name__
        logger.critical(f'❌ CRITICAL ERROR in main monitoring loop ({error_type}): {e}')
        
        # Enhanced error context for debugging
        logger.critical('💡 Critical system failure details:')
        logger.critical(f'   - Error type: {error_type}')
        logger.critical(f'   - Error message: {str(e)}')
        logger.critical(f'   - Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Send critical error alert with enhanced information
        try:
            await send_critical_error_alert(f"{error_type}: {e}")
        except Exception as alert_error:
            logger.critical(f'❌ Failed to send critical error alert: {alert_error}')
            
    finally:
        logger.info('🔄 Shutting down T8 Delays Monitor...')
        
        # Enhanced cleanup logging
        if school_day_lookup:
            try:
                final_stats = school_day_lookup.get_performance_stats()
                logger.info(f"📊 Final performance stats: {final_stats.get('total_lookups', 0)} lookups, "
                           f"{final_stats.get('cache_hit_rate', 0):.1%} cache hit rate")
            except Exception:
                pass  # Stats are optional during shutdown
        
        logger.info('✅ T8 Delays Monitor shutdown complete')

if __name__ == '__main__':
    asyncio.run(main()) 
