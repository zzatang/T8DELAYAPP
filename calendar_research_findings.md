# Phase 1: Research & Data Sources - Findings

## Executive Summary

This document presents the research findings for implementing Dynamic Calendar Integration in the T8 Monitor system. The goal is to replace hard-coded 2025 school terms and public holidays with dynamic data sources that will work indefinitely.

## Key Findings

### 1. NSW School Term Data Sources

#### Primary Source: Data.NSW Government Portal
- **URL**: https://www.data.nsw.gov.au/data/dataset/2-school-and-public-holidays
- **Format**: CSV and PDF downloads
- **Coverage**: Historic, current, and future school term dates
- **Update Frequency**: Annually or as needed
- **Reliability**: ⭐⭐⭐⭐⭐ (Official government source)
- **API Available**: ❌ No direct API, requires file download and parsing

#### Secondary Source: NSW Department of Education
- **URL**: https://education.nsw.gov.au/schooling/calendars/future-and-past-nsw-term-and-vacation-dates
- **Format**: HTML pages with structured data
- **Coverage**: Current and future term dates
- **Update Frequency**: Annually
- **Reliability**: ⭐⭐⭐⭐⭐ (Official source)
- **API Available**: ❌ No API, requires web scraping

### 2. NSW Public Holiday Data Sources

#### Primary Source: Data.NSW Government Portal
- **URL**: Same dataset as school terms (combined data)
- **Format**: CSV and PDF downloads
- **Coverage**: NSW-specific public holidays
- **Update Frequency**: Annually
- **Reliability**: ⭐⭐⭐⭐⭐ (Official government source)
- **API Available**: ❌ No direct API

#### Secondary Source: NSW Government Public Holidays
- **URL**: https://www.nsw.gov.au/about-nsw/public-holidays
- **Format**: HTML pages
- **Coverage**: Current year public holidays
- **Reliability**: ⭐⭐⭐⭐⭐ (Official source)
- **API Available**: ❌ No API, requires web scraping

### 3. Third-Party API Options

#### API Ninjas Public Holidays API
- **URL**: https://api-ninjas.com/api/publicholidays
- **Format**: REST API returning JSON
- **Coverage**: Australia-wide public holidays (1980-2050)
- **Cost**: Free tier available, paid plans for higher usage
- **Reliability**: ⭐⭐⭐⭐ (Third-party, needs validation)
- **API Available**: ✅ Yes, requires API key
- **Limitations**: May not include NSW-specific holidays, no school term data

#### Data.gov.au Australian Holidays Dataset
- **URL**: https://data.gov.au/data/en/dataset/australian-holidays-machine-readable-dataset
- **Format**: CSV download
- **Coverage**: Federal public holidays
- **Update Frequency**: Irregular
- **Reliability**: ⭐⭐⭐ (May not be actively maintained)
- **API Available**: ❌ No direct API

### 4. Transport for NSW Open Data Hub
- **URL**: https://developer.transport.nsw.gov.au/data/dataset/school-and-public-holidays
- **Format**: CSV and PDF downloads
- **Coverage**: NSW school and public holidays
- **Access**: Requires registration and API key
- **Update Frequency**: Last updated September 24, 2024
- **Reliability**: ⭐⭐⭐⭐ (Government transport authority)
- **API Available**: ❌ No direct API, dataset downloads only

## Implementation Recommendations

### Recommended Architecture: Multi-Source Approach

#### Primary Data Source (Tier 1)
**Data.NSW Government Portal** - Official, comprehensive, reliable
- Download CSV files programmatically
- Parse both school terms and public holidays from single source
- Cache data locally with expiration dates
- Update quarterly or when cache expires

#### Fallback Data Source (Tier 2)
**Web Scraping NSW Education Website** - When primary source fails
- Scrape structured data from official education website
- Use as backup when CSV download fails
- Implement with robust error handling

#### Emergency Fallback (Tier 3)
**Calculated Holiday Algorithm** - When all external sources fail
- Calculate major holidays algorithmically (Easter, Christmas, etc.)
- Use generic NSW school term patterns
- Ensure system continues operating even with data source failures

#### Validation Layer
**Cross-Reference Multiple Sources** - Ensure data accuracy
- Compare Data.NSW with NSW Education website
- Validate against API Ninjas for public holidays
- Log discrepancies for manual review

### Technical Implementation Strategy

#### 1. Data Fetching Module
```python
class CalendarDataFetcher:
    async def fetch_data_nsw_csv(self, year: int) -> dict
    async def scrape_nsw_education(self, year: int) -> dict
    async def fetch_api_ninjas_holidays(self, year: int) -> list
    async def calculate_fallback_data(self, year: int) -> dict
```

#### 2. Data Validation & Caching
```python
class CalendarCache:
    def validate_data_integrity(self, data: dict) -> bool
    def cache_data(self, data: dict, expiry_days: int = 90)
    def is_cache_valid(self) -> bool
    def get_cached_data(self, year: int) -> dict
```

#### 3. Error Handling & Monitoring
- Implement comprehensive logging for data fetch attempts
- Monitor cache expiry and refresh cycles
- Alert administrators when data sources fail
- Graceful degradation to fallback sources

### Data Update Schedule

#### Automatic Updates
- **Quarterly checks**: March, June, September, December
- **Cache expiry**: 90 days maximum
- **Proactive fetching**: Get next year's data in October

#### Manual Override Capability
- Allow administrators to force data refresh
- Provide manual data entry interface for emergencies
- Include data validation tools

### Cost Analysis

#### Data.NSW Portal: **FREE**
- No API costs
- Minimal bandwidth for CSV downloads
- Official government data

#### API Ninjas: **$0-50/month**
- Free tier: 1000 requests/month
- Pro tier: $10/month for 100k requests
- Only needed for validation/backup

#### Infrastructure: **Minimal**
- Local caching reduces external requests
- Quarterly updates minimize bandwidth
- No additional hosting requirements

## Next Steps for Phase 2

1. **Implement CSV parser** for Data.NSW downloads
2. **Create web scraper** for NSW Education website
3. **Design caching mechanism** with expiration logic
4. **Build validation system** to cross-check data sources
5. **Develop fallback algorithms** for emergency scenarios
6. **Create monitoring dashboard** for data source health

## Risk Assessment

### Low Risk
- ✅ Data.NSW portal availability (government-maintained)
- ✅ Data accuracy (official sources)
- ✅ Implementation complexity (straightforward parsing)

### Medium Risk
- ⚠️ Website structure changes (requires scraper updates)
- ⚠️ Data format changes (need flexible parsers)
- ⚠️ Cache management (requires proper expiry logic)

### Mitigation Strategies
- Multiple data sources prevent single points of failure
- Robust error handling ensures continued operation
- Fallback algorithms provide emergency data
- Regular monitoring detects issues early

## Conclusion

The research confirms that dynamic calendar integration is feasible using official NSW government data sources. The recommended multi-tier approach ensures reliability while maintaining cost-effectiveness. Implementation can begin immediately with the Data.NSW CSV parsing approach as the primary method.

