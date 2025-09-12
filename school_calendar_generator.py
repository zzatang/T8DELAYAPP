#!/usr/bin/env python3
"""
School Calendar Generator for NSW, Australia
Generates comprehensive school day calendar data for entire years using official NSW Education sources.
"""

import datetime
from datetime import timedelta, date
import requests
import holidays
from icalendar import Calendar
import dateutil.tz
from typing import Optional, Dict, List, Tuple, Any
from urllib.parse import urljoin
import re
from bs4 import BeautifulSoup
import logging
from pathlib import Path
import json
from dateutil import parser as date_parser
import calendar
from dataclasses import dataclass

from database import insert_calendar_data, get_database_system

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CalendarRecord:
    """Data class for a single calendar day record."""
    date: date
    day_of_week: str
    school_day: bool
    reason: Optional[str] = None
    term: Optional[str] = None
    week_of_term: Optional[int] = None
    month: int = None
    quarter: int = None
    week_of_year: int = None
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.month is None:
            self.month = self.date.month
        if self.quarter is None:
            self.quarter = (self.date.month - 1) // 3 + 1
        if self.week_of_year is None:
            self.week_of_year = self.date.isocalendar()[1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            'date': self.date,
            'day_of_week': self.day_of_week,
            'school_day': self.school_day,
            'reason': self.reason,
            'term': self.term,
            'week_of_term': self.week_of_term,
            'month': self.month,
            'quarter': self.quarter,
            'week_of_year': self.week_of_year
        }


class SchoolCalendarGenerator:
    """
    Generates comprehensive school day calendar data for NSW, Australia.
    Extends the existing SchoolDayChecker logic to generate full year calendars.
    """
    
    def __init__(self, cache_dir: str = ".school_day_cache"):
        """
        Initialize the School Calendar Generator.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.sydney_tz = dateutil.tz.gettz("Australia/Sydney")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize NSW holidays
        self.nsw_holidays = holidays.Australia(state='NSW')
        
        # Cache file paths
        self.term_cache_file = self.cache_dir / "nsw_terms.json"
        self.ics_cache_file = self.cache_dir / "nsw_school_calendar.ics"
        
        # NSW Education URLs
        self.base_url = "https://education.nsw.gov.au"
        self.calendar_url = urljoin(self.base_url, "/schooling/calendars/")
        self.ics_pattern = r'/content/dam/main-education/public-schools/going-to-a-public-school/media/ics-files/\d{4}_Calendar\.ics'
        
        # Comprehensive generation statistics and performance metrics
        self.generation_stats = {
            'total_days': 0,
            'school_days': 0,
            'weekends': 0,
            'public_holidays': 0,
            'school_holidays': 0,
            'development_days': 0,
            'term_days': 0,
            'generation_time': None,
            'data_sources_used': [],
            'performance_metrics': {
                'start_time': None,
                'end_time': None,
                'data_fetch_time': 0,
                'parsing_time': 0,
                'generation_time': 0,
                'validation_time': 0,
                'database_save_time': 0,
                'records_per_second': 0,
                'memory_usage_mb': 0
            },
            'data_source_metrics': {
                'ics_download_attempts': 0,
                'ics_download_successes': 0,
                'web_scraping_attempts': 0,
                'web_scraping_successes': 0,
                'cache_hits': 0,
                'cache_misses': 0
            },
            'error_metrics': {
                'validation_errors': 0,
                'database_errors': 0,
                'network_errors': 0,
                'parsing_errors': 0
            }
        }
    
    def _get_current_year(self) -> int:
        """Get current year in Sydney timezone."""
        return datetime.datetime.now(self.sydney_tz).year
    
    def _download_ics_calendar(self, year: int) -> Optional[str]:
        """
        Download the ICS calendar file from NSW Education website with metrics tracking.
        
        Args:
            year: Year to download calendar for
            
        Returns:
            ICS calendar content as string, or None if failed
        """
        # Track download attempt
        self.generation_stats['data_source_metrics']['ics_download_attempts'] += 1
        
        try:
            # First, get the page for the specific year
            year_url = urljoin(self.calendar_url, str(year))
            response = requests.get(year_url, timeout=10)
            response.raise_for_status()
            
            # Parse the HTML to find the ICS file link
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for the ICS file link
            ics_link = None
            for link in soup.find_all('a', href=True):
                if '.ics' in link['href'] and str(year) in link['href']:
                    ics_link = link['href']
                    break
            
            # If not found, try regex pattern
            if not ics_link:
                matches = re.findall(self.ics_pattern, response.text)
                if matches:
                    ics_link = matches[0]
            
            if ics_link:
                # Download the ICS file
                if not ics_link.startswith('http'):
                    ics_link = urljoin(self.base_url, ics_link)
                
                ics_response = requests.get(ics_link, timeout=10)
                ics_response.raise_for_status()
                
                # Cache the ICS file
                with open(self.ics_cache_file, 'w') as f:
                    f.write(ics_response.text)
                
                logger.info(f"✅ Downloaded ICS calendar for {year}")
                self.generation_stats['data_sources_used'].append(f"ICS file for {year}")
                self.generation_stats['data_source_metrics']['ics_download_successes'] += 1
                return ics_response.text
                
        except Exception as e:
            logger.warning(f"❌ Failed to download ICS calendar: {e}")
            self.generation_stats['error_metrics']['network_errors'] += 1
            
        return None
    
    def _parse_ics_calendar(self, ics_content: str) -> Dict:
        """
        Parse ICS calendar content to extract term dates and holidays.
        
        Args:
            ics_content: ICS calendar content as string
            
        Returns:
            Dictionary containing term dates and holidays
        """
        cal = Calendar.from_ical(ics_content)
        term_data = {
            'terms': {},
            'holidays': [],
            'development_days': []
        }
        
        # Track term weeks to calculate start/end dates
        term_weeks = {}
        
        for component in cal.walk():
            if component.name == "VEVENT":
                summary = str(component.get('summary', '')).lower()
                dtstart = component.get('dtstart')
                dtend = component.get('dtend')
                
                if dtstart:
                    start_date = dtstart.dt
                    if isinstance(start_date, datetime.datetime):
                        start_date = start_date.date()
                    
                    end_date = None
                    if dtend:
                        end_date = dtend.dt
                        if isinstance(end_date, datetime.datetime):
                            end_date = end_date.date()
                    
                    # Parse term week information (e.g., "Term 3 Week 1 (10 Wk Term)")
                    term_week_match = re.search(r'term\s*(\d+)\s*week\s*(\d+)', summary)
                    if term_week_match:
                        term_num = int(term_week_match.group(1))
                        week_num = int(term_week_match.group(2))
                        
                        if term_num not in term_weeks:
                            term_weeks[term_num] = {}
                        
                        term_weeks[term_num][week_num] = start_date
                    
                    # Parse school development days
                    elif 'development' in summary or 'pupil free' in summary or 'staff' in summary:
                        term_data['development_days'].append(start_date)
                    
                    # Parse holidays - look for "School Holidays" or similar
                    elif 'holiday' in summary or 'vacation' in summary:
                        if end_date:
                            term_data['holidays'].append((start_date, end_date))
                        else:
                            term_data['holidays'].append((start_date, start_date))
        
        # Calculate term start and end dates from week data
        for term_num, weeks in term_weeks.items():
            if weeks:  # If we have week data for this term
                min_week = min(weeks.keys())
                max_week = max(weeks.keys())
                
                if min_week in weeks and max_week in weeks:
                    # Term starts on the first day of Week 1
                    term_start = weeks[min_week]
                    
                    # Term ends on the last day of the last week (add 6 days for full week)
                    term_end = weeks[max_week] + timedelta(days=6)
                    
                    term_data['terms'][f'term{term_num}'] = {
                        'start': term_start,
                        'end': term_end
                    }
        
        logger.info(f"✅ Parsed ICS calendar: {len(term_data['terms'])} terms, {len(term_data['holidays'])} holiday periods")
        return term_data
    
    def _scrape_term_dates(self, year: int) -> Optional[Dict]:
        """
        Scrape term dates from NSW Education website as fallback.
        
        Args:
            year: Year to scrape term dates for
            
        Returns:
            Dictionary containing term dates, or None if failed
        """
        try:
            url = urljoin(self.calendar_url, str(year))
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            term_data = {
                'terms': {},
                'holidays': [],
                'development_days': [],
                'year': year
            }
            
            # Look for term date patterns in the text
            text = soup.get_text()
            
            # Pattern for term dates (e.g., "Term 1: Thursday 6 February to Friday 11 April")
            term_pattern = r'Term\s+(\d):\s+\w+\s+(\d+\s+\w+)\s+to\s+\w+\s+(\d+\s+\w+)'
            matches = re.findall(term_pattern, text)
            
            for match in matches:
                term_num = int(match[0])
                try:
                    # Add year to date strings
                    start_date = date_parser.parse(f"{match[1]} {year}").date()
                    end_date = date_parser.parse(f"{match[2]} {year}").date()
                    
                    term_data['terms'][f'term{term_num}'] = {
                        'start': start_date,
                        'end': end_date
                    }
                except:
                    continue
            
            if term_data['terms']:
                logger.info(f"✅ Scraped term dates for {year}: {len(term_data['terms'])} terms")
                self.generation_stats['data_sources_used'].append(f"Web scraping for {year}")
                return term_data
                
        except Exception as e:
            logger.warning(f"❌ Failed to scrape term dates: {e}")
            
        return None
    
    def _get_term_dates(self, year: int) -> Dict:
        """
        Get term dates for a specific year, trying multiple methods.
        
        Args:
            year: Year to get term dates for
            
        Returns:
            Dictionary containing term dates
        """
        # Check cache first
        if self.term_cache_file.exists():
            try:
                with open(self.term_cache_file, 'r') as f:
                    cached_data = json.load(f)
                    
                # Check if cache is for the requested year
                if cached_data.get('year') == year:
                    logger.info(f"✅ Using cached term dates for {year}")
                    # Convert string dates back to date objects
                    for term_key in cached_data.get('terms', {}):
                        for date_key in ['start', 'end']:
                            if date_key in cached_data['terms'][term_key]:
                                cached_data['terms'][term_key][date_key] = datetime.date.fromisoformat(
                                    cached_data['terms'][term_key][date_key]
                                )
                    
                    # Convert holiday tuples
                    cached_data['holidays'] = [
                        (datetime.date.fromisoformat(start), datetime.date.fromisoformat(end))
                        for start, end in cached_data.get('holidays', [])
                    ]
                    
                    # Convert development days
                    cached_data['development_days'] = [
                        datetime.date.fromisoformat(day)
                        for day in cached_data.get('development_days', [])
                    ]
                    
                    self.generation_stats['data_sources_used'].append(f"Cached data for {year}")
                    return cached_data
            except Exception as e:
                logger.warning(f"❌ Failed to load cache: {e}")
        
        # Try to download and parse ICS calendar
        ics_content = self._download_ics_calendar(year)
        if ics_content:
            term_data = self._parse_ics_calendar(ics_content)
            if term_data['terms']:
                term_data['year'] = year
                self._save_cache(term_data)
                return term_data
        
        # Fallback to web scraping
        term_data = self._scrape_term_dates(year)
        if term_data:
            self._save_cache(term_data)
            return term_data
        
        # If all else fails, return empty structure
        logger.error(f"❌ Could not retrieve term dates for {year} from any source")
        return {
            'terms': {},
            'holidays': [],
            'development_days': [],
            'year': year
        }
    
    def _save_cache(self, term_data: Dict):
        """Save term data to cache file."""
        try:
            # Convert date objects to strings for JSON serialization
            cache_data = {
                'year': term_data.get('year'),
                'terms': {},
                'holidays': [],
                'development_days': []
            }
            
            # Convert term dates
            for term_key, dates in term_data.get('terms', {}).items():
                cache_data['terms'][term_key] = {}
                for date_key in ['start', 'end']:
                    if date_key in dates:
                        cache_data['terms'][term_key][date_key] = dates[date_key].isoformat()
            
            # Convert holidays
            cache_data['holidays'] = [
                (start.isoformat(), end.isoformat())
                for start, end in term_data.get('holidays', [])
            ]
            
            # Convert development days
            cache_data['development_days'] = [
                day.isoformat() for day in term_data.get('development_days', [])
            ]
            
            with open(self.term_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"✅ Cached term dates for {cache_data['year']}")
        except Exception as e:
            logger.warning(f"❌ Failed to save cache: {e}")
    
    def _is_weekend(self, check_date: date) -> bool:
        """Check if date is a weekend."""
        return check_date.weekday() >= 5  # Saturday = 5, Sunday = 6
    
    def _is_public_holiday(self, check_date: date) -> bool:
        """Check if date is a public holiday in NSW."""
        return check_date in self.nsw_holidays
    
    def _is_school_holiday(self, check_date: date, term_data: Dict) -> bool:
        """Check if date falls during school holidays."""
        for start, end in term_data.get('holidays', []):
            if start <= check_date <= end:
                return True
        return False
    
    def _is_development_day(self, check_date: date, term_data: Dict) -> bool:
        """Check if date is a school development day (pupil-free day)."""
        return check_date in term_data.get('development_days', [])
    
    def _get_term_info(self, check_date: date, term_data: Dict) -> Tuple[Optional[str], Optional[int]]:
        """
        Get term information for a date.
        
        Args:
            check_date: Date to check
            term_data: Term data dictionary
            
        Returns:
            Tuple of (term_name, week_of_term) or (None, None) if not in term
        """
        for term_key, dates in term_data.get('terms', {}).items():
            if 'start' in dates and 'end' in dates:
                if dates['start'] <= check_date <= dates['end']:
                    # Calculate week of term
                    days_since_start = (check_date - dates['start']).days
                    week_of_term = (days_since_start // 7) + 1
                    
                    # Convert term key to readable format
                    term_name = term_key.replace('term', 'Term ')
                    
                    return term_name, week_of_term
        
        return None, None
    
    def _determine_school_day_status(self, check_date: date, term_data: Dict) -> CalendarRecord:
        """
        Determine the school day status for a specific date.
        
        Args:
            check_date: Date to analyze
            term_data: Term data dictionary
            
        Returns:
            CalendarRecord with complete information
        """
        day_name = check_date.strftime('%A')
        
        # Check conditions in order of precedence
        if self._is_weekend(check_date):
            return CalendarRecord(
                date=check_date,
                day_of_week=day_name,
                school_day=False,
                reason='weekend'
            )
        
        if self._is_public_holiday(check_date):
            holiday_name = self.nsw_holidays.get(check_date)
            return CalendarRecord(
                date=check_date,
                day_of_week=day_name,
                school_day=False,
                reason=f'public holiday: {holiday_name}'
            )
        
        if self._is_school_holiday(check_date, term_data):
            return CalendarRecord(
                date=check_date,
                day_of_week=day_name,
                school_day=False,
                reason='school holidays'
            )
        
        if self._is_development_day(check_date, term_data):
            return CalendarRecord(
                date=check_date,
                day_of_week=day_name,
                school_day=False,
                reason='development day'
            )
        
        # Check if during term time
        term_name, week_of_term = self._get_term_info(check_date, term_data)
        
        if term_name and week_of_term:
            # It's a school day
            return CalendarRecord(
                date=check_date,
                day_of_week=day_name,
                school_day=True,
                reason='school day',
                term=term_name,
                week_of_term=week_of_term
            )
        else:
            # Not during term time
            return CalendarRecord(
                date=check_date,
                day_of_week=day_name,
                school_day=False,
                reason='not during term'
            )
    
    def generate_year_calendar(self, year: int) -> List[CalendarRecord]:
        """
        Generate complete calendar data for a specific year with comprehensive performance tracking.
        
        Args:
            year: Year to generate calendar for
            
        Returns:
            List of CalendarRecord objects for every day of the year
        """
        logger.info(f"🗓️ Generating school calendar for {year}...")
        
        # Initialize comprehensive performance tracking
        self._start_performance_tracking()
        
        # Reset statistics (but keep performance metrics)
        performance_backup = self.generation_stats['performance_metrics'].copy()
        data_source_backup = self.generation_stats['data_source_metrics'].copy()
        error_backup = self.generation_stats['error_metrics'].copy()
        
        self.generation_stats = {
            'total_days': 0,
            'school_days': 0,
            'weekends': 0,
            'public_holidays': 0,
            'school_holidays': 0,
            'development_days': 0,
            'term_days': 0,
            'generation_time': None,
            'data_sources_used': [],
            'performance_metrics': performance_backup,
            'data_source_metrics': data_source_backup,
            'error_metrics': error_backup
        }
        
        # Get term data for the year with timing
        data_fetch_start = datetime.datetime.now()
        term_data = self._get_term_dates(year)
        self.generation_stats['performance_metrics']['data_fetch_time'] = (datetime.datetime.now() - data_fetch_start).total_seconds()
        
        # Generate calendar records for every day of the year
        generation_start = datetime.datetime.now()
        calendar_records = []
        
        # Determine if leap year
        is_leap_year = calendar.isleap(year)
        total_days = 366 if is_leap_year else 365
        
        logger.info(f"📅 Processing {total_days} days for {year} ({'leap year' if is_leap_year else 'regular year'})")
        
        # Start from January 1st
        current_date = date(year, 1, 1)
        
        # Progress logging for large datasets
        progress_interval = max(1, total_days // 10)  # Log every 10%
        
        for day_num in range(total_days):
            record = self._determine_school_day_status(current_date, term_data)
            calendar_records.append(record)
            
            # Update statistics
            self.generation_stats['total_days'] += 1
            
            if record.school_day:
                self.generation_stats['school_days'] += 1
                if record.term:
                    self.generation_stats['term_days'] += 1
            else:
                if record.reason == 'weekend':
                    self.generation_stats['weekends'] += 1
                elif record.reason and 'public holiday' in record.reason:
                    self.generation_stats['public_holidays'] += 1
                elif record.reason == 'school holidays':
                    self.generation_stats['school_holidays'] += 1
                elif record.reason == 'development day':
                    self.generation_stats['development_days'] += 1
            
            # Progress logging
            if (day_num + 1) % progress_interval == 0:
                progress_pct = ((day_num + 1) / total_days) * 100
                logger.info(f"📈 Progress: {progress_pct:.0f}% ({day_num + 1}/{total_days} days) - Current: {current_date}")
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Record generation timing
        self.generation_stats['performance_metrics']['generation_time'] = (datetime.datetime.now() - generation_start).total_seconds()
        
        # Finalize performance tracking
        self._end_performance_tracking()
        
        # Log comprehensive performance metrics
        self._log_performance_metrics()
        
        logger.info(f"✅ Generated {len(calendar_records)} calendar records for {year}")
        logger.info(f"📊 Quick Stats: {self.generation_stats['school_days']} school days, "
                   f"{self.generation_stats['weekends']} weekends, "
                   f"{self.generation_stats['public_holidays']} public holidays")
        
        return calendar_records
    
    def save_to_database(self, calendar_records: List[CalendarRecord], batch_size: int = 1000) -> bool:
        """
        Save calendar records to the PostgreSQL database using atomic batch operations.
        
        Args:
            calendar_records: List of CalendarRecord objects
            batch_size: Number of records to insert per batch
            
        Returns:
            True if successful, False otherwise
        """
        if not calendar_records:
            logger.warning("No calendar records to save")
            return False
        
        try:
            # Use the enhanced atomic batch insert
            result = self.atomic_batch_insert(calendar_records, batch_size)
            
            if result['success']:
                logger.info(f"✅ Atomically saved {result['inserted_count']} calendar records to database")
                logger.info(f"📊 Processed {result['batches_processed']} batches in {result['total_time']:.2f} seconds")
                return True
            else:
                logger.error(f"❌ Atomic batch insert failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to save calendar records to database: {e}")
            return False
    
    def atomic_batch_insert(self, calendar_records: List[CalendarRecord], batch_size: int = 1000) -> Dict[str, Any]:
        """
        Perform atomic batch insert operations with comprehensive transaction management.
        
        Args:
            calendar_records: List of CalendarRecord objects to insert
            batch_size: Number of records per batch
            
        Returns:
            Dictionary with detailed results of the atomic batch operation
        """
        start_time = datetime.datetime.now()
        
        result = {
            'success': False,
            'inserted_count': 0,
            'batches_processed': 0,
            'total_batches': 0,
            'total_time': 0,
            'year': None,
            'error': None,
            'batch_details': []
        }
        
        if not calendar_records:
            result['error'] = "No calendar records provided"
            return result
        
        try:
            # Get year and prepare data
            year = calendar_records[0].date.year
            result['year'] = year
            result['total_batches'] = (len(calendar_records) + batch_size - 1) // batch_size
            
            logger.info(f"🔄 Starting atomic batch insert for {len(calendar_records)} records in {result['total_batches']} batches")
            
            # Convert records to dictionaries
            record_dicts = [record.to_dict() for record in calendar_records]
            
            # Get database manager for direct transaction control
            from database import get_database_manager
            db_manager = get_database_manager()
            
            # Perform atomic transaction across all batches
            with db_manager.get_connection() as conn:
                conn.autocommit = False  # Ensure we control transactions
                
                try:
                    with conn.cursor() as cur:
                        # First, clear any existing data for this year to ensure consistency
                        logger.info(f"🗑️ Clearing existing calendar data for {year}")
                        cur.execute(
                            "DELETE FROM school_calendar WHERE EXTRACT(YEAR FROM date) = %s",
                            (year,)
                        )
                        deleted_count = cur.rowcount
                        logger.info(f"📝 Deleted {deleted_count} existing records for {year}")
                        
                        # Prepare the optimized insert statement with UPSERT capability
                        insert_sql = """
                            INSERT INTO school_calendar (
                                date, day_of_week, school_day, reason, term, 
                                week_of_term, month, quarter, week_of_year
                            ) VALUES (
                                %(date)s, %(day_of_week)s, %(school_day)s, %(reason)s, %(term)s,
                                %(week_of_term)s, %(month)s, %(quarter)s, %(week_of_year)s
                            )
                        """
                        
                        # Process records in batches within the same transaction
                        total_inserted = 0
                        for batch_num in range(result['total_batches']):
                            start_idx = batch_num * batch_size
                            end_idx = min(start_idx + batch_size, len(record_dicts))
                            batch_records = record_dicts[start_idx:end_idx]
                            
                            batch_start_time = datetime.datetime.now()
                            
                            # Execute batch insert
                            cur.executemany(insert_sql, batch_records)
                            batch_inserted = cur.rowcount
                            total_inserted += batch_inserted
                            
                            batch_time = (datetime.datetime.now() - batch_start_time).total_seconds()
                            
                            # Record batch details
                            batch_detail = {
                                'batch_number': batch_num + 1,
                                'records_in_batch': len(batch_records),
                                'records_inserted': batch_inserted,
                                'batch_time_seconds': round(batch_time, 3),
                                'date_range': f"{batch_records[0]['date']} to {batch_records[-1]['date']}"
                            }
                            result['batch_details'].append(batch_detail)
                            
                            logger.info(f"📦 Batch {batch_num + 1}/{result['total_batches']}: "
                                      f"inserted {batch_inserted} records in {batch_time:.3f}s "
                                      f"({batch_records[0]['date']} to {batch_records[-1]['date']})")
                        
                        # Verify the insert was successful
                        logger.info(f"🔍 Verifying insert success for year {year}...")
                        try:
                            cur.execute(
                                "SELECT COUNT(*) as count FROM school_calendar WHERE EXTRACT(YEAR FROM date) = %s",
                                (year,)
                            )
                            result_row = cur.fetchone()
                            logger.info(f"🔍 Query result: {result_row}")
                            
                            # Handle both tuple and dictionary-like results
                            if result_row:
                                if hasattr(result_row, 'keys'):  # Dictionary-like (RealDictRow)
                                    final_count = result_row['count']
                                else:  # Tuple-like
                                    final_count = result_row[0]
                            else:
                                final_count = 0
                                
                        except Exception as verify_error:
                            logger.error(f"❌ Verification query failed: {verify_error}")
                            raise Exception(f"Database verification query failed: {verify_error}")
                        
                        logger.info(f"📊 Database verification: expected {len(calendar_records)} records, found {final_count} records")
                        logger.info(f"📊 Batch processing summary: {total_inserted} total inserted from cur.rowcount")
                        
                        # Update the total_inserted count to reflect actual database state
                        # (cur.rowcount from executemany can be unreliable)
                        if final_count > 0:
                            total_inserted = final_count
                        
                        if final_count != len(calendar_records):
                            logger.warning(f"⚠️ Insert count mismatch: expected {len(calendar_records)}, found {final_count}")
                            # Only fail if we have zero records (complete failure)
                            if final_count == 0:
                                raise Exception(f"Insert verification failed: no records found in database (expected {len(calendar_records)})")
                            elif final_count < len(calendar_records) * 0.9:  # Allow 10% tolerance
                                raise Exception(f"Insert verification failed: expected {len(calendar_records)}, found {final_count}")
                            else:
                                logger.info(f"✅ Insert verification passed with minor discrepancy: {final_count} records confirmed")
                        else:
                            logger.info(f"✅ Insert verification passed: {final_count} records confirmed in database")
                        
                        # Commit the entire transaction
                        conn.commit()
                        
                        result['success'] = True
                        result['inserted_count'] = total_inserted
                        result['batches_processed'] = result['total_batches']
                        
                        logger.info(f"✅ Atomic transaction committed successfully: {total_inserted} records")
                        
                except Exception as e:
                    # Rollback the entire transaction on any error
                    conn.rollback()
                    logger.error(f"❌ Transaction rolled back due to error: {e}")
                    result['error'] = f"Database transaction failed: {str(e)}"
                    raise
                
        except Exception as e:
            logger.error(f"❌ Atomic batch insert failed: {e}")
            result['error'] = str(e)
            result['success'] = False
        
        # Calculate total time
        end_time = datetime.datetime.now()
        result['total_time'] = (end_time - start_time).total_seconds()
        
        return result
    
    def verify_database_integrity(self, year: int) -> Dict[str, Any]:
        """
        Verify the integrity of calendar data in the database for a specific year.
        
        Args:
            year: Year to verify
            
        Returns:
            Dictionary with integrity check results
        """
        logger.info(f"🔍 Verifying database integrity for {year}")
        
        try:
            from database import get_calendar_operations
            ops = get_calendar_operations()
            
            # Get validation results from database
            validation_result = ops.validate_calendar_data(year)
            
            integrity_result = {
                'year': year,
                'is_valid': validation_result.get('is_valid', False),
                'record_count': validation_result.get('record_count', 0),
                'issues_found': validation_result.get('issues', []),
                'recommendations': validation_result.get('recommendations', [])
            }
            
            if integrity_result['is_valid']:
                logger.info(f"✅ Database integrity verified for {year}")
            else:
                logger.warning(f"⚠️ Database integrity issues found for {year}")
            
            return integrity_result
            
        except Exception as e:
            logger.error(f"❌ Database integrity verification failed: {e}")
            return {
                'year': year,
                'is_valid': False,
                'error': str(e)
            }
    
    def generate_and_save_year(self, year: int, batch_size: int = 1000, validate: bool = True) -> Dict[str, Any]:
        """
        Generate and save complete calendar data for a year with optional validation.
        
        Args:
            year: Year to generate and save
            batch_size: Batch size for database insertion
            validate: Whether to perform data validation before saving
            
        Returns:
            Dictionary with generation results, validation, and statistics
        """
        logger.info(f"🚀 Starting complete calendar generation and save for {year}")
        
        try:
            # Generate calendar records
            calendar_records = self.generate_year_calendar(year)
            
            if not calendar_records:
                return {
                    'success': False,
                    'error': 'No calendar records generated',
                    'year': year
                }
            
            result = {
                'success': False,
                'year': year,
                'total_records': len(calendar_records),
                'statistics': self.generation_stats.copy(),
                'data_sources': self.generation_stats['data_sources_used']
            }
            
            # Perform validation if requested
            if validate:
                logger.info("🔍 Validating generated calendar data...")
                validation_results = self.validate_calendar_data(calendar_records)
                result['validation'] = validation_results
                
                if not validation_results['is_valid']:
                    logger.error(f"❌ Calendar data validation failed: {len(validation_results['errors'])} errors")
                    result['error'] = f"Data validation failed: {validation_results['errors'][:3]}"
                    return result
                else:
                    logger.info("✅ Calendar data validation passed")
            
            # Save to database
            save_success = self.save_to_database(calendar_records, batch_size)
            
            if save_success:
                result['success'] = True
                logger.info(f"✅ Successfully generated, validated, and saved calendar for {year}")
            else:
                result['error'] = 'Failed to save records to database'
                logger.error(f"❌ Failed to save calendar records to database for {year}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Calendar generation and save failed for {year}: {e}")
            return {
                'success': False,
                'error': str(e),
                'year': year,
                'statistics': self.generation_stats.copy()
            }
    
    def validate_calendar_data(self, calendar_records: List[CalendarRecord]) -> Dict[str, Any]:
        """
        Perform comprehensive validation and consistency checks on generated calendar data.
        
        Args:
            calendar_records: List of CalendarRecord objects to validate
            
        Returns:
            Dictionary with validation results and detailed findings
        """
        logger.info("🔍 Performing comprehensive calendar data validation...")
        
        validation_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'total_records': len(calendar_records),
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'consistency_checks': {},
            'recommendations': []
        }
        
        if not calendar_records:
            validation_results['is_valid'] = False
            validation_results['errors'].append("No calendar records provided for validation")
            return validation_results
        
        try:
            # Get year from first record
            year = calendar_records[0].date.year
            is_leap_year = calendar.isleap(year)
            expected_days = 366 if is_leap_year else 365
            
            validation_results['year'] = year
            validation_results['is_leap_year'] = is_leap_year
            validation_results['expected_days'] = expected_days
            
            # Check 1: Total day count
            if len(calendar_records) != expected_days:
                validation_results['is_valid'] = False
                validation_results['errors'].append(
                    f"Incorrect number of days: expected {expected_days}, got {len(calendar_records)}"
                )
            
            # Check 2: Date continuity and completeness
            sorted_records = sorted(calendar_records, key=lambda r: r.date)
            expected_date = date(year, 1, 1)
            
            for i, record in enumerate(sorted_records):
                if record.date != expected_date:
                    validation_results['is_valid'] = False
                    if record.date < expected_date:
                        validation_results['errors'].append(f"Duplicate or out-of-order date: {record.date}")
                    else:
                        validation_results['errors'].append(f"Missing date: {expected_date}")
                    break
                expected_date += timedelta(days=1)
            
            # Check 3: Required field completeness
            missing_fields = {}
            for i, record in enumerate(calendar_records):
                required_fields = ['date', 'day_of_week', 'school_day', 'month', 'quarter', 'week_of_year']
                for field in required_fields:
                    value = getattr(record, field, None)
                    if value is None:
                        if field not in missing_fields:
                            missing_fields[field] = []
                        missing_fields[field].append(record.date)
                        if len(missing_fields[field]) <= 5:  # Limit examples
                            validation_results['errors'].append(f"Missing {field} for {record.date}")
            
            # Check 4: Data type and range validation
            for record in calendar_records:
                # Month validation
                if not (1 <= record.month <= 12):
                    validation_results['errors'].append(f"Invalid month {record.month} for {record.date}")
                
                # Quarter validation
                if not (1 <= record.quarter <= 4):
                    validation_results['errors'].append(f"Invalid quarter {record.quarter} for {record.date}")
                
                # Week of year validation
                if not (1 <= record.week_of_year <= 53):
                    validation_results['errors'].append(f"Invalid week_of_year {record.week_of_year} for {record.date}")
                
                # Day of week validation
                expected_day = record.date.strftime('%A')
                if record.day_of_week != expected_day:
                    validation_results['errors'].append(
                        f"Incorrect day_of_week for {record.date}: expected {expected_day}, got {record.day_of_week}"
                    )
            
            # Check 5: Logical consistency
            consistency_issues = []
            
            for record in calendar_records:
                # Weekend consistency
                is_weekend = record.date.weekday() >= 5
                if is_weekend and record.school_day:
                    consistency_issues.append(f"School day on weekend: {record.date}")
                elif is_weekend and record.reason != 'weekend':
                    consistency_issues.append(f"Weekend date with non-weekend reason: {record.date} - {record.reason}")
                
                # School day reason consistency
                if record.school_day and record.reason not in ['school day', None]:
                    consistency_issues.append(f"School day with conflicting reason: {record.date} - {record.reason}")
                
                # Non-school day reason validation
                if not record.school_day and not record.reason:
                    consistency_issues.append(f"Non-school day missing reason: {record.date}")
                
                # Term information consistency
                if record.school_day and record.term and not record.week_of_term:
                    consistency_issues.append(f"School day with term but no week_of_term: {record.date}")
                elif not record.school_day and record.term:
                    consistency_issues.append(f"Non-school day with term information: {record.date}")
            
            validation_results['consistency_checks']['logical_issues'] = len(consistency_issues)
            if consistency_issues:
                validation_results['is_valid'] = False
                validation_results['errors'].extend(consistency_issues[:10])  # Limit to first 10
            
            # Check 6: Statistical validation
            stats = {
                'school_days': sum(1 for r in calendar_records if r.school_day),
                'weekends': sum(1 for r in calendar_records if r.reason == 'weekend'),
                'public_holidays': sum(1 for r in calendar_records if r.reason and 'public holiday' in r.reason),
                'school_holidays': sum(1 for r in calendar_records if r.reason == 'school holidays'),
                'development_days': sum(1 for r in calendar_records if r.reason == 'development day'),
                'term_days': sum(1 for r in calendar_records if r.term),
                'not_during_term': sum(1 for r in calendar_records if r.reason == 'not during term')
            }
            
            validation_results['statistics'] = stats
            
            # Statistical reasonableness checks
            expected_weekends = 52 * 2 + (1 if date(year, 1, 1).weekday() >= 5 and is_leap_year else 0)
            if abs(stats['weekends'] - expected_weekends) > 2:
                validation_results['warnings'].append(
                    f"Unexpected weekend count: expected ~{expected_weekends}, got {stats['weekends']}"
                )
            
            # School days should be reasonable (typically 180-200 per year)
            if stats['school_days'] < 150 or stats['school_days'] > 220:
                validation_results['warnings'].append(
                    f"Unusual school day count: {stats['school_days']} (typically 180-200)"
                )
            
            # Check 7: Term coverage and gaps
            term_coverage = self._validate_term_coverage(calendar_records)
            validation_results['consistency_checks'].update(term_coverage)
            
            if term_coverage.get('gaps'):
                validation_results['warnings'].extend([
                    f"Term gap detected: {gap}" for gap in term_coverage['gaps'][:5]
                ])
            
            # Check 8: Holiday period validation
            holiday_validation = self._validate_holiday_periods(calendar_records)
            validation_results['consistency_checks'].update(holiday_validation)
            
            # Generate recommendations
            if stats['school_days'] < 180:
                validation_results['recommendations'].append("Consider reviewing term dates - school day count seems low")
            
            if len(validation_results['errors']) > 20:
                validation_results['recommendations'].append("Multiple validation errors detected - consider regenerating calendar data")
            
            if validation_results['warnings']:
                validation_results['recommendations'].append("Review warnings for potential data quality issues")
            
            # Final validation status
            if len(validation_results['errors']) == 0:
                validation_results['is_valid'] = True
                logger.info("✅ Calendar data validation passed")
            else:
                validation_results['is_valid'] = False
                logger.warning(f"❌ Calendar data validation failed: {len(validation_results['errors'])} errors")
            
            if validation_results['warnings']:
                logger.info(f"⚠️ {len(validation_results['warnings'])} warnings found")
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation process failed: {str(e)}")
            logger.error(f"❌ Calendar validation error: {e}")
        
        return validation_results
    
    def _validate_term_coverage(self, calendar_records: List[CalendarRecord]) -> Dict[str, Any]:
        """
        Validate term coverage and detect gaps.
        
        Args:
            calendar_records: List of calendar records
            
        Returns:
            Dictionary with term coverage validation results
        """
        term_periods = {}
        gaps = []
        
        # Group records by term
        for record in calendar_records:
            if record.term and record.school_day:
                if record.term not in term_periods:
                    term_periods[record.term] = {'start': record.date, 'end': record.date, 'days': 0}
                
                term_periods[record.term]['start'] = min(term_periods[record.term]['start'], record.date)
                term_periods[record.term]['end'] = max(term_periods[record.term]['end'], record.date)
                term_periods[record.term]['days'] += 1
        
        # Check for reasonable term lengths (typically 9-11 weeks = 45-55 school days)
        for term, info in term_periods.items():
            if info['days'] < 35 or info['days'] > 65:
                gaps.append(f"{term} has unusual length: {info['days']} school days")
        
        # Check for gaps between terms
        sorted_terms = sorted(term_periods.items(), key=lambda x: x[1]['start'])
        for i in range(len(sorted_terms) - 1):
            current_term = sorted_terms[i]
            next_term = sorted_terms[i + 1]
            
            gap_days = (next_term[1]['start'] - current_term[1]['end']).days
            if gap_days > 30:  # More than 4 weeks between terms
                gaps.append(f"Large gap between {current_term[0]} and {next_term[0]}: {gap_days} days")
        
        return {
            'term_count': len(term_periods),
            'term_periods': {k: {'start': v['start'].isoformat(), 'end': v['end'].isoformat(), 'days': v['days']} 
                           for k, v in term_periods.items()},
            'gaps': gaps
        }
    
    def _validate_holiday_periods(self, calendar_records: List[CalendarRecord]) -> Dict[str, Any]:
        """
        Validate school holiday periods for consistency.
        
        Args:
            calendar_records: List of calendar records
            
        Returns:
            Dictionary with holiday validation results
        """
        holiday_periods = []
        current_period = None
        
        # Find consecutive holiday periods
        for record in sorted(calendar_records, key=lambda r: r.date):
            if record.reason == 'school holidays':
                if current_period is None:
                    current_period = {'start': record.date, 'end': record.date, 'days': 1}
                else:
                    current_period['end'] = record.date
                    current_period['days'] += 1
            else:
                if current_period:
                    holiday_periods.append(current_period)
                    current_period = None
        
        # Don't forget the last period
        if current_period:
            holiday_periods.append(current_period)
        
        # Validate holiday period lengths (typically 1-6 weeks)
        issues = []
        for period in holiday_periods:
            if period['days'] > 45:  # More than 6 weeks
                issues.append(f"Very long holiday period: {period['start']} to {period['end']} ({period['days']} days)")
        
        return {
            'holiday_periods': len(holiday_periods),
            'holiday_details': [
                {'start': p['start'].isoformat(), 'end': p['end'].isoformat(), 'days': p['days']}
                for p in holiday_periods
            ],
            'holiday_issues': issues
        }
    
    def _log_performance_metrics(self):
        """Log comprehensive performance metrics."""
        metrics = self.generation_stats['performance_metrics']
        data_metrics = self.generation_stats['data_source_metrics']
        error_metrics = self.generation_stats['error_metrics']
        
        logger.info("📊 COMPREHENSIVE PERFORMANCE REPORT")
        logger.info("=" * 50)
        
        # Timing metrics
        if metrics['start_time'] and metrics['end_time']:
            total_time = (metrics['end_time'] - metrics['start_time']).total_seconds()
            logger.info(f"⏱️ Total Execution Time: {total_time:.3f} seconds")
            logger.info(f"   Data Fetch Time: {metrics['data_fetch_time']:.3f}s ({(metrics['data_fetch_time']/total_time*100):.1f}%)")
            logger.info(f"   Parsing Time: {metrics['parsing_time']:.3f}s ({(metrics['parsing_time']/total_time*100):.1f}%)")
            logger.info(f"   Generation Time: {metrics['generation_time']:.3f}s ({(metrics['generation_time']/total_time*100):.1f}%)")
            logger.info(f"   Validation Time: {metrics['validation_time']:.3f}s ({(metrics['validation_time']/total_time*100):.1f}%)")
            logger.info(f"   Database Save Time: {metrics['database_save_time']:.3f}s ({(metrics['database_save_time']/total_time*100):.1f}%)")
        
        # Performance metrics
        logger.info(f"🚀 Performance Metrics:")
        logger.info(f"   Records per Second: {metrics['records_per_second']:.1f}")
        if metrics['memory_usage_mb'] > 0:
            logger.info(f"   Peak Memory Usage: {metrics['memory_usage_mb']:.1f} MB")
        
        # Data source metrics
        logger.info(f"📡 Data Source Metrics:")
        logger.info(f"   ICS Downloads: {data_metrics['ics_download_successes']}/{data_metrics['ics_download_attempts']} successful")
        logger.info(f"   Web Scraping: {data_metrics['web_scraping_successes']}/{data_metrics['web_scraping_attempts']} successful")
        logger.info(f"   Cache: {data_metrics['cache_hits']} hits, {data_metrics['cache_misses']} misses")
        
        # Error metrics
        if sum(error_metrics.values()) > 0:
            logger.info(f"⚠️ Error Summary:")
            if error_metrics['validation_errors'] > 0:
                logger.info(f"   Validation Errors: {error_metrics['validation_errors']}")
            if error_metrics['database_errors'] > 0:
                logger.info(f"   Database Errors: {error_metrics['database_errors']}")
            if error_metrics['network_errors'] > 0:
                logger.info(f"   Network Errors: {error_metrics['network_errors']}")
            if error_metrics['parsing_errors'] > 0:
                logger.info(f"   Parsing Errors: {error_metrics['parsing_errors']}")
        else:
            logger.info("✅ No errors encountered during generation")
        
        # Calendar statistics
        logger.info(f"📅 Calendar Statistics:")
        logger.info(f"   Total Days: {self.generation_stats['total_days']}")
        logger.info(f"   School Days: {self.generation_stats['school_days']} ({(self.generation_stats['school_days']/self.generation_stats['total_days']*100):.1f}%)")
        logger.info(f"   Weekends: {self.generation_stats['weekends']} ({(self.generation_stats['weekends']/self.generation_stats['total_days']*100):.1f}%)")
        logger.info(f"   Public Holidays: {self.generation_stats['public_holidays']}")
        logger.info(f"   School Holidays: {self.generation_stats['school_holidays']}")
        logger.info(f"   Development Days: {self.generation_stats['development_days']}")
        
        logger.info("=" * 50)
    
    def _measure_memory_usage(self) -> float:
        """Measure current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        except ImportError:
            # psutil not available, return 0
            return 0
        except Exception:
            return 0
    
    def _start_performance_tracking(self):
        """Initialize performance tracking for the generation process."""
        self.generation_stats['performance_metrics']['start_time'] = datetime.datetime.now()
        initial_memory = self._measure_memory_usage()
        if initial_memory > 0:
            self.generation_stats['performance_metrics']['memory_usage_mb'] = initial_memory
        
        logger.info(f"🚀 Starting calendar generation with performance tracking")
        logger.info(f"   Start Time: {self.generation_stats['performance_metrics']['start_time'].isoformat()}")
        if initial_memory > 0:
            logger.info(f"   Initial Memory: {initial_memory:.1f} MB")
    
    def _end_performance_tracking(self):
        """Finalize performance tracking and calculate metrics."""
        self.generation_stats['performance_metrics']['end_time'] = datetime.datetime.now()
        
        # Update peak memory usage
        current_memory = self._measure_memory_usage()
        if current_memory > self.generation_stats['performance_metrics']['memory_usage_mb']:
            self.generation_stats['performance_metrics']['memory_usage_mb'] = current_memory
        
        # Calculate records per second
        start_time = self.generation_stats['performance_metrics']['start_time']
        end_time = self.generation_stats['performance_metrics']['end_time']
        
        if start_time and end_time:
            total_seconds = (end_time - start_time).total_seconds()
            if total_seconds > 0:
                self.generation_stats['performance_metrics']['records_per_second'] = self.generation_stats['total_days'] / total_seconds
        
        logger.info(f"✅ Calendar generation completed")
        logger.info(f"   End Time: {end_time.isoformat()}")
        if current_memory > 0:
            logger.info(f"   Final Memory: {current_memory:.1f} MB")
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get the latest generation statistics with comprehensive metrics."""
        stats = self.generation_stats.copy()
        
        # Add calculated metrics
        if stats['performance_metrics']['start_time'] and stats['performance_metrics']['end_time']:
            total_time = (stats['performance_metrics']['end_time'] - stats['performance_metrics']['start_time']).total_seconds()
            stats['performance_metrics']['total_execution_time'] = total_time
            
            # Calculate efficiency metrics
            if stats['total_days'] > 0:
                stats['performance_metrics']['ms_per_record'] = (total_time * 1000) / stats['total_days']
        
        # Add data source success rates
        if stats['data_source_metrics']['ics_download_attempts'] > 0:
            stats['data_source_metrics']['ics_success_rate'] = (
                stats['data_source_metrics']['ics_download_successes'] / 
                stats['data_source_metrics']['ics_download_attempts']
            )
        
        if stats['data_source_metrics']['web_scraping_attempts'] > 0:
            stats['data_source_metrics']['web_scraping_success_rate'] = (
                stats['data_source_metrics']['web_scraping_successes'] / 
                stats['data_source_metrics']['web_scraping_attempts']
            )
        
        return stats
    
    def export_performance_metrics(self, filepath: str = None) -> str:
        """
        Export detailed performance metrics to a JSON file.
        
        Args:
            filepath: Optional custom filepath. If None, generates timestamped filename.
            
        Returns:
            Filepath where metrics were saved
        """
        if filepath is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"calendar_generation_metrics_{timestamp}.json"
        
        try:
            metrics_data = {
                'export_timestamp': datetime.datetime.now().isoformat(),
                'generation_statistics': self.get_generation_statistics(),
                'system_info': {
                    'python_version': sys.version,
                    'platform': sys.platform
                }
            }
            
            # Convert datetime objects to strings for JSON serialization
            def convert_datetime(obj):
                if isinstance(obj, datetime.datetime):
                    return obj.isoformat()
                elif isinstance(obj, datetime.date):
                    return obj.isoformat()
                return obj
            
            # Process the data to handle datetime objects
            import json
            json_str = json.dumps(metrics_data, default=convert_datetime, indent=2)
            
            with open(filepath, 'w') as f:
                f.write(json_str)
            
            logger.info(f"📊 Performance metrics exported to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Failed to export performance metrics: {e}")
            raise


def main():
    """Command-line interface for school calendar generation."""
    import argparse
    import sys
    import os
    
    # Fix Windows console encoding issues for emoji characters
    if os.name == 'nt':  # Windows
        try:
            # Try to set console code page to UTF-8
            os.system('chcp 65001 > nul 2>&1')
            # Reconfigure stdout/stderr with UTF-8 encoding if available
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass  # Ignore errors, will fallback to safe printing
    
    # Safe print function to handle encoding issues
    def safe_print(message):
        try:
            print(message)
        except UnicodeEncodeError:
            # Fallback: replace problematic characters with safe alternatives
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            print(safe_message)
    
    parser = argparse.ArgumentParser(description='NSW School Calendar Generator')
    parser.add_argument('year', type=int, help='Year to generate calendar for')
    parser.add_argument('--batch-size', type=int, default=1000, help='Database batch size')
    parser.add_argument('--cache-dir', type=str, default='.school_day_cache', help='Cache directory')
    parser.add_argument('--generate-only', action='store_true', help='Generate only, do not save to database')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Create generator
        generator = SchoolCalendarGenerator(cache_dir=args.cache_dir)
        
        if args.generate_only:
            # Generate only
            records = generator.generate_year_calendar(args.year)
            stats = generator.get_generation_statistics()
            
            print(f"✅ Generated {len(records)} records for {args.year}")
            print(f"📊 School days: {stats['school_days']}, Weekends: {stats['weekends']}")
            print(f"⏱️ Generation time: {stats['generation_time']:.2f} seconds")
        else:
            # Generate and save
            result = generator.generate_and_save_year(args.year, args.batch_size)
            
            if result['success']:
                safe_print(f"✅ Successfully generated and saved calendar for {args.year}")
                safe_print(f"📊 Total records: {result['total_records']}")
                safe_print(f"📊 School days: {result['statistics']['school_days']}")
                generation_time = result['statistics'].get('generation_time', 0) or 0
                safe_print(f"⏱️ Generation time: {generation_time:.2f} seconds")
                sys.exit(0)
            else:
                safe_print(f"❌ Failed to generate calendar for {args.year}: {result.get('error', 'Unknown error')}")
                sys.exit(1)
                
    except Exception as e:
        safe_print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
