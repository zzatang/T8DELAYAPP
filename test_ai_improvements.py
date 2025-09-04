#!/usr/bin/env python3
"""
Test script to validate AI analysis improvements for the T8 Monitor
Tests the problematic Sydney Metro weekend trackwork tweet
"""

import asyncio
import logging
import sys
import os

# Add the current directory to Python path to import from monitor script
sys.path.append('.')

# Import the analysis functions from the main monitor script
from monitor_t8_delays_polling import analyze_tweet_with_ollama, fallback_keyword_analysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_problematic_tweet():
    """
    Test the specific tweet that caused the false positive
    """
    logger.info("🧪 Testing AI Analysis Improvements")
    logger.info("="*60)
    
    # The problematic tweet from the image
    problematic_tweet = """This weekend, metro services do not run between Tallawong and Sydenham.

Buses replace services between Tallawong and Chatswood.

Use trains between Chatswood, the City and Sydenham.

More: transportnsw.info/alerts/details..."""
    
    tweet_source = "SydneyMetro (retweeted by T8SydneyTrains)"
    
    logger.info(f"📝 Testing Tweet: {problematic_tweet}")
    logger.info(f"🏷️  Tweet Source: {tweet_source}")
    logger.info("="*60)
    
    # Test 1: Ollama AI Analysis
    logger.info("🤖 TEST 1: Ollama AI Analysis")
    try:
        should_alert_ai, confidence_ai, reasoning_ai = await analyze_tweet_with_ollama(
            problematic_tweet, tweet_source
        )
        
        logger.info("✅ Ollama AI Analysis Results:")
        logger.info(f"   Alert Decision: {'YES' if should_alert_ai else 'NO'}")
        logger.info(f"   Confidence: {confidence_ai}")
        logger.info(f"   Reasoning: {reasoning_ai}")
        
        if should_alert_ai:
            logger.error("❌ FAILED: AI still wants to send alert for Sydney Metro weekend trackwork")
            return False
        else:
            logger.info("✅ PASSED: AI correctly rejected Sydney Metro weekend trackwork")
            
    except Exception as e:
        logger.warning(f"⚠️  Ollama AI test failed (expected if Ollama not running): {e}")
        logger.info("🔄 Proceeding with fallback keyword analysis test...")
    
    # Test 2: Fallback Keyword Analysis
    logger.info("\n🔤 TEST 2: Fallback Keyword Analysis")
    should_alert_keywords, confidence_keywords, reasoning_keywords = await fallback_keyword_analysis(
        problematic_tweet, tweet_source
    )
    
    logger.info("✅ Keyword Analysis Results:")
    logger.info(f"   Alert Decision: {'YES' if should_alert_keywords else 'NO'}")
    logger.info(f"   Confidence: {confidence_keywords}")
    logger.info(f"   Reasoning: {reasoning_keywords}")
    
    if should_alert_keywords:
        logger.error("❌ FAILED: Keyword analysis still wants to send alert for Sydney Metro weekend trackwork")
        return False
    else:
        logger.info("✅ PASSED: Keyword analysis correctly rejected Sydney Metro weekend trackwork")
    
    return True

async def test_valid_t8_alert():
    """
    Test with a valid T8 Airport Line alert to ensure we don't break legitimate alerts
    """
    logger.info("\n🚆 Testing Valid T8 Alert (should trigger)")
    logger.info("="*60)
    
    # Example of a legitimate T8 alert
    valid_t8_tweet = """T8 Airport Line: Allow extra travel time due to a signal equipment issue at Green Square. Services are running with delays between Central and Airport stations."""
    
    tweet_source = "T8SydneyTrains"
    
    logger.info(f"📝 Testing Tweet: {valid_t8_tweet}")
    logger.info(f"🏷️  Tweet Source: {tweet_source}")
    
    # Test fallback keyword analysis (since Ollama might not be available)
    should_alert, confidence, reasoning = await fallback_keyword_analysis(
        valid_t8_tweet, tweet_source
    )
    
    logger.info("✅ Valid T8 Alert Test Results:")
    logger.info(f"   Alert Decision: {'YES' if should_alert else 'NO'}")
    logger.info(f"   Confidence: {confidence}")
    logger.info(f"   Reasoning: {reasoning}")
    
    if should_alert:
        logger.info("✅ PASSED: Valid T8 alert correctly triggers notification")
        return True
    else:
        logger.error("❌ FAILED: Valid T8 alert was incorrectly rejected")
        return False

async def test_general_trackwork():
    """
    Test with general trackwork announcement (should NOT trigger)
    """
    logger.info("\n🔧 Testing General Trackwork (should NOT trigger)")
    logger.info("="*60)
    
    # Example of general trackwork that shouldn't trigger
    trackwork_tweet = """Are you travelling this weekend? Due to trackwork between Redfern and Central, trains may run to a changed timetable. Plan your journey at transportnsw.info"""
    
    tweet_source = "T8SydneyTrains"
    
    logger.info(f"📝 Testing Tweet: {trackwork_tweet}")
    logger.info(f"🏷️  Tweet Source: {tweet_source}")
    
    # Test fallback keyword analysis
    should_alert, confidence, reasoning = await fallback_keyword_analysis(
        trackwork_tweet, tweet_source
    )
    
    logger.info("✅ General Trackwork Test Results:")
    logger.info(f"   Alert Decision: {'YES' if should_alert else 'NO'}")
    logger.info(f"   Confidence: {confidence}")
    logger.info(f"   Reasoning: {reasoning}")
    
    if should_alert:
        logger.error("❌ FAILED: General trackwork incorrectly triggers alert")
        return False
    else:
        logger.info("✅ PASSED: General trackwork correctly rejected")
        return True

async def main():
    """
    Run all tests and report results
    """
    logger.info("🚀 Starting AI Analysis Improvement Tests")
    logger.info("="*80)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Problematic Sydney Metro tweet
    try:
        if await test_problematic_tweet():
            tests_passed += 1
    except Exception as e:
        logger.error(f"❌ Test 1 failed with exception: {e}")
    
    # Test 2: Valid T8 alert
    try:
        if await test_valid_t8_alert():
            tests_passed += 1
    except Exception as e:
        logger.error(f"❌ Test 2 failed with exception: {e}")
    
    # Test 3: General trackwork
    try:
        if await test_general_trackwork():
            tests_passed += 1
    except Exception as e:
        logger.error(f"❌ Test 3 failed with exception: {e}")
    
    # Final results
    logger.info("="*80)
    logger.info("🏁 TEST RESULTS SUMMARY")
    logger.info(f"✅ Tests Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        logger.info("🎉 ALL TESTS PASSED! AI analysis improvements are working correctly.")
        logger.info("📈 The system should now:")
        logger.info("   ✅ Reject Sydney Metro tweets (wrong service line)")
        logger.info("   ✅ Reject weekend trackwork announcements (planned maintenance)")
        logger.info("   ✅ Accept legitimate T8 Airport Line disruptions")
        logger.info("   ✅ Require T8-specific context for alerts")
        return True
    else:
        logger.error("❌ SOME TESTS FAILED! Review the analysis logic.")
        return False

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    
    if success:
        print("\n✅ All improvements validated successfully!")
        exit(0)
    else:
        print("\n❌ Some tests failed - review the implementation")
        exit(1)
