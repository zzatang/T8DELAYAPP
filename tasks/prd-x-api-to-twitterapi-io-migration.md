# Product Requirements Document: X API to TwitterAPI.io Migration

## 1. Introduction/Overview

This PRD outlines the complete migration of the T8 Delays Monitor (`monitor_t8_delays_polling.py`) from X's official API to TwitterAPI.io. The primary goal is to achieve significant cost reduction while maintaining the exact same monitoring functionality for T8 Sydney Trains service disruptions.

**Problem Statement:** The current X API implementation is cost-prohibitive, potentially costing up to $5,000/month for the Pro plan, while our actual usage requires only ~950 tweets/month (~$0.14/month with TwitterAPI.io).

**Solution:** Complete replacement of X API with TwitterAPI.io's cost-effective pay-as-you-go model ($0.15 per 1,000 tweets).

## 2. Goals

1. **Primary Goal:** Reduce API costs from potentially $5,000/month to ~$0.14/month
2. **Maintain Functionality:** Preserve all existing monitoring capabilities without degradation
3. **Improve Reliability:** Implement robust error handling with Telegram alerts
4. **Simplify Configuration:** Streamline environment variable management
5. **Fresh Start:** Begin with clean tweet tracking (no historical data preservation required)

## 3. User Stories

**As a T8 commuter,** I want to continue receiving real-time delay alerts via Telegram so that I can plan my journey accordingly, regardless of which API backend is used.

**As a system administrator,** I want the monitoring system to automatically alert me via Telegram when the API service fails so that I can take corrective action immediately.

**As a cost-conscious operator,** I want to minimize API costs while maintaining service quality so that the monitoring system remains financially sustainable.

**As a developer,** I want a clean, maintainable codebase that uses modern HTTP clients instead of deprecated Twitter libraries so that future maintenance is simplified.

## 4. Functional Requirements

### 4.1 Core API Migration Requirements

1. **FR-001:** The system MUST completely replace Tweepy library with direct HTTP requests to TwitterAPI.io
2. **FR-002:** The system MUST authenticate using TwitterAPI.io's `x-api-key` header method
3. **FR-003:** The system MUST fetch tweets from @T8SydneyTrains using TwitterAPI.io's `/twitter/user/last_tweets` endpoint
4. **FR-004:** The system MUST maintain the same polling interval configuration (POLLING_INTERVAL_MINUTES)
5. **FR-005:** The system MUST preserve all existing time window filtering (school days, peak hours)

### 4.2 Data Processing Requirements

6. **FR-006:** The system MUST convert TwitterAPI.io response format to match existing internal tweet structure
7. **FR-007:** The system MUST maintain compatibility with existing Ollama AI analysis pipeline
8. **FR-008:** The system MUST continue filtering tweets by age (2-hour window)
9. **FR-009:** The system MUST reset tweet ID tracking (start fresh with TwitterAPI.io)

### 4.3 Error Handling & Monitoring Requirements

10. **FR-010:** The system MUST stop monitoring and send Telegram alert when TwitterAPI.io API fails
11. **FR-011:** The system MUST log all API response codes and error details
12. **FR-012:** The system MUST maintain existing heartbeat logging functionality
13. **FR-013:** The system MUST preserve all existing logging levels and formats

### 4.4 Configuration Requirements

14. **FR-014:** The system MUST replace X_BEARER_TOKEN with TWITTERAPI_IO_KEY environment variable
15. **FR-015:** The system MUST update required_vars validation to check for new API key
16. **FR-016:** The system MUST maintain backward compatibility during feature flag testing phase

### 4.5 Feature Flag Implementation

17. **FR-017:** The system MUST implement a feature flag (USE_TWITTERAPI_IO) to switch between APIs during testing
18. **FR-018:** The system MUST allow runtime switching without code changes
19. **FR-019:** The system MUST log which API backend is currently active

## 5. Non-Goals (Out of Scope)

1. **NG-001:** Preserving historical tweet ID tracking from X API
2. **NG-002:** Maintaining dual API support in production (only for testing phase)
3. **NG-003:** Implementing TwitterAPI.io premium features beyond basic tweet retrieval
4. **NG-004:** Modifying the Ollama AI analysis logic or prompts
5. **NG-005:** Changing the Telegram notification format or content
6. **NG-006:** Implementing fallback to X API if TwitterAPI.io fails
7. **NG-007:** Adding new monitoring features or capabilities

## 6. Technical Considerations

### 6.1 Dependencies
- **Remove:** `tweepy` library (no longer needed)
- **Add:** `aiohttp` for async HTTP requests (or use existing `requests` library)
- **Keep:** All existing dependencies (telegram, ollama, dateutil, etc.)

### 6.2 API Endpoint Mapping
- **Current:** Twitter API v2 `/users/:id/tweets`
- **New:** TwitterAPI.io `/twitter/user/last_tweets?userName=T8SydneyTrains`

### 6.3 Authentication Change
- **Current:** `Authorization: Bearer {token}`
- **New:** `x-api-key: {api_key}`

### 6.4 Response Format Conversion
TwitterAPI.io response structure needs mapping to existing internal format:
```python
# TwitterAPI.io format → Internal format
{
    'id': tweet_data.get('id'),
    'text': tweet_data.get('text'),
    'created_at': parse_iso_datetime(tweet_data.get('created_at')),
    'public_metrics': tweet_data.get('public_metrics', {})
}
```

### 6.5 Error Handling Strategy
- HTTP 200: Process normally
- HTTP 429: Rate limit exceeded → Stop and alert
- HTTP 401/403: Authentication error → Stop and alert  
- HTTP 5xx: Server error → Stop and alert
- Network errors: Connection timeout → Stop and alert

## 7. Design Considerations

### 7.1 Code Structure
- Maintain existing function signatures where possible
- Create new `fetch_tweets_twitterapi()` function
- Add `convert_twitterapi_response()` helper function
- Implement `test_twitterapi_connection()` for startup validation

### 7.2 Configuration Management
```bash
# New environment variables
TWITTERAPI_IO_KEY=your_api_key_here
USE_TWITTERAPI_IO=true  # Feature flag for testing phase

# Remove after migration
# X_BEARER_TOKEN=old_token
```

### 7.3 Logging Strategy
- Log API backend selection at startup
- Log all HTTP requests/responses for debugging
- Maintain existing log levels and formatting
- Add TwitterAPI.io specific error codes

## 8. Success Metrics

### 8.1 Cost Metrics
- **Target:** Reduce monthly API costs from $5,000 to under $1
- **Measure:** Track actual TwitterAPI.io usage and costs

### 8.2 Reliability Metrics  
- **Target:** Maintain 99.9% uptime during monitoring windows
- **Measure:** Count successful tweet fetches vs. failures

### 8.3 Functionality Metrics
- **Target:** Zero missed delay alerts due to API migration
- **Measure:** Compare alert frequency before/after migration

### 8.4 Performance Metrics
- **Target:** Maintain or improve API response times
- **Measure:** Log response times for each API call

## 9. Implementation Plan

### Phase 1: Setup (Day 1)
- Create TwitterAPI.io account and obtain API key
- Add new environment variables
- Test API connectivity

### Phase 2: Code Development (Days 2-3)
- Implement TwitterAPI.io HTTP client functions
- Add response format conversion logic  
- Implement feature flag switching mechanism
- Add comprehensive error handling

### Phase 3: Testing (Days 4-5)
- Unit test individual functions
- Integration test with feature flag enabled
- Compare results between both APIs
- Test error scenarios and alerting

### Phase 4: Production Migration (Day 6)
- Enable TwitterAPI.io in production
- Monitor for 24 hours
- Remove X API code and dependencies
- Update documentation

## 10. Open Questions

1. **Q:** Does TwitterAPI.io support tweet timestamp filtering equivalent to Twitter API's `start_time` parameter?
   **Status:** Needs investigation during implementation

2. **Q:** What is the exact rate limit for TwitterAPI.io's tweet endpoint?
   **Status:** Need to test and document for monitoring frequency optimization

3. **Q:** Does TwitterAPI.io provide tweet ID consistency for tracking processed tweets?
   **Status:** Critical for avoiding duplicate alerts - needs verification

4. **Q:** How does TwitterAPI.io handle deleted or unavailable tweets?
   **Status:** Need to understand error responses for robust handling

5. **Q:** What is the maximum number of tweets returnable in a single API call?
   **Status:** Current code requests 10 tweets - verify if this is supported

## 11. Risk Assessment

### High Risk
- **API Response Format Differences:** Could break existing data processing
- **Mitigation:** Thorough testing with response format conversion

### Medium Risk  
- **Rate Limiting Differences:** Could affect monitoring frequency
- **Mitigation:** Test with actual usage patterns and adjust if needed

### Low Risk
- **Network Connectivity Issues:** Standard HTTP client risks
- **Mitigation:** Existing error handling patterns apply

---

**Document Version:** 1.0  
**Created:** 2024  
**Target Implementation:** 6 days  
**Primary Stakeholder:** T8 Commuters & System Administrator
