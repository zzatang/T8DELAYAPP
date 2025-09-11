# Product Requirements Document: School Day Calendar Generator

## Introduction/Overview

The School Day Calendar Generator is an automated system that dynamically generates comprehensive school day lookup tables for NSW, Australia. This feature solves the critical problem of manual date maintenance in the T8 monitoring system by automatically detecting new years and generating pre-computed school day data from official NSW Education Department sources.

The system eliminates hardcoded school term dates, improves lookup performance to O(1) complexity, and provides reliable offline school day checking for the T8 Airport Line delay monitoring system.

## Goals

1. **Eliminate Manual Maintenance**: Remove all hardcoded school dates and automate yearly calendar generation
2. **Improve Performance**: Achieve sub-1ms school day lookups through pre-computed data tables
3. **Ensure Reliability**: Provide 99.9% uptime for school day checking with comprehensive fallback mechanisms
4. **Future-Proof System**: Automatically handle year transitions (2025→2026→2027) without human intervention
5. **Maintain Data Accuracy**: Ensure 100% accuracy by using official NSW Education Department data sources

## User Stories

**As the T8 monitoring system**, I want to:
- Check if any date is a school day in under 1ms so that I can make real-time monitoring decisions
- Automatically have updated school calendars each year so that I don't miss monitoring school days
- Have reliable offline access to school day data so that network issues don't affect monitoring

**As a system administrator**, I want to:
- Have the calendar data regenerate automatically so that I don't need to manually update it each year
- Be able to manually trigger calendar regeneration so that I can fix data issues if needed
- Have detailed logs of calendar generation so that I can troubleshoot any problems

**As a developer maintaining the system**, I want to:
- Have the system work without any hardcoded dates so that future maintenance is minimal
- Have comprehensive error handling so that the system degrades gracefully
- Have clear data validation so that I can verify calendar accuracy

## Functional Requirements

### Core Generation Requirements
1. The system MUST automatically detect when a new year begins (e.g., 2026) and generate calendar data without human intervention
2. The system MUST parse NSW Education Department ICS calendar files to extract accurate school term dates
3. The system MUST identify weekends, NSW public holidays, school holidays, and development days for each date
4. The system MUST generate a complete calendar record for every day of the current year (365/366 days)
5. The system MUST store calendar data in PostgreSQL database with optimized schema for fast lookups

### Data Source Requirements
6. The system MUST attempt to download ICS calendar files from education.nsw.gov.au as primary data source
7. The system MUST use the NSW holidays library for accurate public holiday detection
8. The system MUST fall back to web scraping if ICS download fails
9. The system MUST maintain a hierarchy of data sources: ICS → Web Scraping → Cached Data

### Performance Requirements
10. The system MUST provide school day lookups in under 1ms (O(1) complexity)
11. The system MUST complete calendar generation within 10 minutes for a full year
12. The system MUST use less than 10MB of memory during normal operation
13. The system MUST cache frequently accessed data in memory for optimal performance

### Database Requirements
14. The system MUST store calendar data in PostgreSQL database on the Raspberry Pi
15. The system MUST create tables with proper indexing on date fields for fast queries
16. The system MUST handle database connection failures gracefully
17. The system MUST support atomic transactions for data consistency during updates

### Integration Requirements
18. The system MUST completely replace the existing hardcoded SchoolDayChecker in the T8 monitor
19. The system MUST provide a simple `is_school_day(date)` interface that maintains backward compatibility
20. The system MUST initialize automatically when the T8 monitor starts
21. The system MUST not break existing T8 monitor functionality

### Error Handling Requirements
22. The system MUST log all generation activities, errors, and performance metrics
23. The system MUST gracefully handle network failures during data source access
24. The system MUST provide detailed error messages for troubleshooting
25. The system MUST continue operating with cached data if generation fails

### Maintenance Requirements
26. The system MUST provide a manual regeneration command for administrative use
27. The system MUST validate generated data for completeness and accuracy
28. The system MUST detect and report data inconsistencies
29. The system MUST clean up old calendar data automatically

## Non-Goals (Out of Scope)

1. **Multi-State Support**: This feature will only support NSW, Australia school calendars
2. **Historical Data**: Will not generate calendars for past years (before current year)
3. **Real-time Updates**: Will not monitor for mid-year changes to school calendars
4. **User Interface**: Will not include a web dashboard or GUI for calendar management
5. **API Endpoints**: Will not expose REST APIs for external calendar access
6. **Multi-Year Bulk Generation**: Will only generate current year data, not multiple years at once
7. **Custom Holiday Support**: Will not allow users to add custom holidays or exceptions

## Design Considerations

### Database Schema
```sql
CREATE TABLE school_calendar (
    date DATE PRIMARY KEY,
    day_of_week VARCHAR(10) NOT NULL,
    school_day BOOLEAN NOT NULL,
    reason VARCHAR(100),
    term VARCHAR(20),
    week_of_term INTEGER,
    month INTEGER,
    quarter INTEGER,
    week_of_year INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_school_calendar_date ON school_calendar(date);
CREATE INDEX idx_school_calendar_school_day ON school_calendar(school_day);
CREATE INDEX idx_school_calendar_month ON school_calendar(month);
```

### Data Flow Architecture
1. **Detection Phase**: Monitor for year changes or missing data
2. **Collection Phase**: Download from NSW Education sources
3. **Processing Phase**: Parse ICS files and calculate school day status
4. **Validation Phase**: Verify data completeness and accuracy
5. **Storage Phase**: Insert into PostgreSQL with atomic transactions
6. **Activation Phase**: Update T8 monitor to use new data

## Technical Considerations

### Dependencies
- **PostgreSQL**: Already installed on Raspberry Pi
- **Python Libraries**: psycopg2-binary, icalendar, holidays, beautifulsoup4, requests
- **Existing Code**: Integration with current sydney_school_day_checker.py logic

### Performance Optimizations
- Use prepared statements for database queries
- Implement connection pooling for PostgreSQL
- Cache current year data in memory for fastest access
- Use batch inserts for calendar data generation

### Security Considerations
- Use environment variables for database credentials
- Implement proper SQL injection prevention
- Validate all external data before processing
- Use HTTPS for all external data source connections

## Success Metrics

### Performance Metrics
1. **Lookup Speed**: 100% of school day queries complete in <1ms
2. **Generation Time**: Full year calendar generation completes in <10 minutes
3. **Memory Usage**: System uses <10MB RAM during normal operation
4. **Uptime**: 99.9% availability for school day checking functionality

### Accuracy Metrics
5. **Data Accuracy**: 100% match with official NSW Education calendar
6. **Coverage**: 100% of days in current year have calendar entries
7. **Validation**: 0% data inconsistencies detected during validation

### Automation Metrics
8. **Year Transitions**: 100% successful automatic detection of new years
9. **Manual Interventions**: <1 manual regeneration required per year
10. **Error Recovery**: 100% successful fallback to cached data during failures

## Open Questions

1. **Database Backup**: Should the system automatically backup calendar data before regenerating?
2. **Monitoring Integration**: Should calendar generation status be reported to external monitoring systems?
3. **Multi-Instance Support**: How should the system behave if multiple T8 monitor instances are running?
4. **Data Retention**: How long should old calendar data be retained in the database?
5. **Update Notifications**: Should the system notify administrators when new calendar data is generated?
6. **Performance Monitoring**: Should the system track and report lookup performance metrics?
7. **Timezone Handling**: Should all dates be stored in UTC or Sydney timezone?

## Implementation Priority

### Phase 1 (Critical - Week 1)
- Core calendar generation logic
- PostgreSQL schema and connection handling
- Basic school day lookup functionality
- Integration with existing T8 monitor

### Phase 2 (High - Week 2)
- Automatic year detection and regeneration
- Comprehensive error handling and logging
- Data validation and consistency checks
- Manual regeneration command interface

### Phase 3 (Medium - Week 3)
- Performance optimizations and caching
- Advanced fallback mechanisms
- Statistics and health monitoring
- Documentation and deployment guides

This PRD provides a comprehensive roadmap for implementing a robust, automated school day calendar generation system that will eliminate manual maintenance while providing superior performance and reliability.
