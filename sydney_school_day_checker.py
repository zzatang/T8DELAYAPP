#!/usr/bin/env python3
"""
Sydney School Day Checker
Dynamically determines if today is a school day in Sydney, Australia
without hardcoding school term dates.
"""

import datetime
from datetime import timedelta
import requests
import holidays
from icalendar import Calendar
import dateutil.tz
from typing import Optional, Dict, List, Tuple
from urllib.parse import urljoin
import re
from bs4 import BeautifulSoup
import logging
from pathlib import Path
import json
from dateutil import parser as date_parser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SchoolDayChecker:
    """
    Determines if a given date is a school day in Sydney, NSW, Australia.
    Dynamically fetches term dates from NSW Education Department.
    """
    
    def __init__(self, cache_dir: str = "~/.school_day_cache"):
        """
        Initialize the School Day Checker.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.sydney_tz = dateutil.tz.gettz("Australia/Sydney")
        self.cache_dir = Path(cache_dir).expanduser()
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
        
        # Load term dates
        self.term_dates = self._get_term_dates()
        
    def _get_current_year(self) -> int:
        """Get current year in Sydney timezone."""
        return datetime.datetime.now(self.sydney_tz).year
    
    def _download_ics_calendar(self, year: int) -> Optional[str]:
        """
        Download the ICS calendar file from NSW Education website.
        
        Args:
            year: Year to download calendar for
            
        Returns:
            ICS calendar content as string, or None if failed
        """
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
                
                logger.info(f"Downloaded ICS calendar for {year}")
                return ics_response.text
                
        except Exception as e:
            logger.warning(f"Failed to download ICS calendar: {e}")
            
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
            
            # Pattern for holidays (e.g., "Autumn: Monday 14 April to Thursday 24 April")
            holiday_pattern = r'(Summer|Autumn|Winter|Spring).*?:\s+\w+\s+(\d+\s+\w+(?:\s+\d{4})?)\s+to\s+\w+\s+(\d+\s+\w+(?:\s+\d{4})?)'
            holiday_matches = re.findall(holiday_pattern, text)
            
            for match in holiday_matches:
                try:
                    # Parse dates, adding year if not present
                    start_str = match[1]
                    end_str = match[2]
                    
                    if str(year) not in start_str:
                        start_str = f"{start_str} {year}"
                    if str(year) not in end_str:
                        # Check if it spans to next year
                        if 'January' in end_str or 'February' in end_str:
                            end_str = f"{end_str} {year + 1}"
                        else:
                            end_str = f"{end_str} {year}"
                    
                    start_date = date_parser.parse(start_str).date()
                    end_date = date_parser.parse(end_str).date()
                    
                    term_data['holidays'].append((start_date, end_date))
                except:
                    continue
            
            if term_data['terms']:
                logger.info(f"Scraped term dates for {year}")
                return term_data
                
        except Exception as e:
            logger.warning(f"Failed to scrape term dates: {e}")
            
        return None
    
    def _get_term_dates(self) -> Dict:
        """
        Get term dates, trying multiple methods in order of preference.
        
        Returns:
            Dictionary containing term dates
        """
        current_year = self._get_current_year()
        
        # Check cache first
        if self.term_cache_file.exists():
            try:
                with open(self.term_cache_file, 'r') as f:
                    cached_data = json.load(f)
                    
                # Check if cache is for current year
                if cached_data.get('year') == current_year:
                    logger.info("Using cached term dates")
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
                    
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Try to download and parse ICS calendar
        ics_content = self._download_ics_calendar(current_year)
        if ics_content:
            term_data = self._parse_ics_calendar(ics_content)
            if term_data['terms']:
                term_data['year'] = current_year
                self._save_cache(term_data)
                return term_data
        
        # Fallback to web scraping
        term_data = self._scrape_term_dates(current_year)
        if term_data:
            self._save_cache(term_data)
            return term_data
        
        # If all else fails, return empty structure
        logger.error("Could not retrieve term dates from any source")
        return {
            'terms': {},
            'holidays': [],
            'development_days': [],
            'year': current_year
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
            
            logger.info("Cached term dates")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def is_weekend(self, date: datetime.date) -> bool:
        """Check if date is a weekend."""
        return date.weekday() >= 5  # Saturday = 5, Sunday = 6
    
    def is_public_holiday(self, date: datetime.date) -> bool:
        """Check if date is a public holiday in NSW."""
        return date in self.nsw_holidays
    
    def is_school_holiday(self, date: datetime.date) -> bool:
        """Check if date falls during school holidays."""
        for start, end in self.term_dates.get('holidays', []):
            if start <= date <= end:
                return True
        return False
    
    def is_development_day(self, date: datetime.date) -> bool:
        """Check if date is a school development day (pupil-free day)."""
        return date in self.term_dates.get('development_days', [])
    
    def is_during_term(self, date: datetime.date) -> bool:
        """Check if date falls within any school term."""
        for term_key, dates in self.term_dates.get('terms', {}).items():
            if 'start' in dates and 'end' in dates:
                if dates['start'] <= date <= dates['end']:
                    return True
        return False
    
    def is_school_day(self, date: Optional[datetime.date] = None) -> bool:
        """
        Determine if a given date is a school day in Sydney.
        
        Args:
            date: Date to check. If None, uses today in Sydney timezone.
            
        Returns:
            True if it's a school day, False otherwise
        """
        if date is None:
            date = datetime.datetime.now(self.sydney_tz).date()
        
        # Check conditions in order
        if self.is_weekend(date):
            logger.info(f"{date} is a weekend")
            return False
        
        if self.is_public_holiday(date):
            holiday_name = self.nsw_holidays.get(date)
            logger.info(f"{date} is a public holiday: {holiday_name}")
            return False
        
        if self.is_school_holiday(date):
            logger.info(f"{date} is during school holidays")
            return False
        
        if self.is_development_day(date):
            logger.info(f"{date} is a school development day")
            return False
        
        if not self.is_during_term(date):
            logger.info(f"{date} is not during a school term")
            return False
        
        # If all checks pass, it's a school day
        logger.info(f"{date} is a school day")
        return True
    
    def get_status_details(self, date: Optional[datetime.date] = None) -> Dict:
        """
        Get detailed status information about a date.
        
        Args:
            date: Date to check. If None, uses today in Sydney timezone.
            
        Returns:
            Dictionary with detailed status information
        """
        if date is None:
            date = datetime.datetime.now(self.sydney_tz).date()
        
        status = {
            'date': date.isoformat(),
            'is_school_day': self.is_school_day(date),
            'is_weekend': self.is_weekend(date),
            'is_public_holiday': self.is_public_holiday(date),
            'is_school_holiday': self.is_school_holiday(date),
            'is_development_day': self.is_development_day(date),
            'is_during_term': self.is_during_term(date),
            'day_name': date.strftime('%A')
        }
        
        if status['is_public_holiday']:
            status['holiday_name'] = self.nsw_holidays.get(date)
        
        # Find current or next term
        for term_key, dates in self.term_dates.get('terms', {}).items():
            if 'start' in dates and 'end' in dates:
                if dates['start'] <= date <= dates['end']:
                    status['current_term'] = term_key.replace('term', 'Term ')
                    status['term_ends'] = dates['end'].isoformat()
                    break
        
        return status


def main():
    """Main function to run the school day checker."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check if today is a school day in Sydney, Australia')
    parser.add_argument('--date', type=str, help='Check specific date (YYYY-MM-DD format)')
    parser.add_argument('--details', action='store_true', help='Show detailed status information')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cached data and re-download')
    
    args = parser.parse_args()
    
    checker = SchoolDayChecker()
    
    if args.clear_cache:
        if checker.term_cache_file.exists():
            checker.term_cache_file.unlink()
        if checker.ics_cache_file.exists():
            checker.ics_cache_file.unlink()
        print("Cache cleared")
        checker = SchoolDayChecker()  # Reinitialize to download fresh data
    
    # Parse date if provided
    check_date = None
    if args.date:
        try:
            check_date = datetime.date.fromisoformat(args.date)
        except ValueError:
            print(f"Invalid date format: {args.date}. Use YYYY-MM-DD format.")
            return
    
    # Get status
    if args.details:
        status = checker.get_status_details(check_date)
        print(f"\n📅 Date: {status['date']} ({status['day_name']})")
        print(f"🏫 School Day: {'✅ Yes' if status['is_school_day'] else '❌ No'}")
        
        if not status['is_school_day']:
            print("\nReason(s):")
            if status['is_weekend']:
                print("  • Weekend")
            if status['is_public_holiday']:
                print(f"  • Public Holiday: {status.get('holiday_name', 'Unknown')}")
            if status['is_school_holiday']:
                print("  • School Holiday Period")
            if status['is_development_day']:
                print("  • School Development Day (Pupil-Free)")
            if not status['is_during_term'] and not status['is_school_holiday']:
                print("  • Outside School Term")
        
        
        if 'current_term' in status:
            print(f"\n📚 Current Term: {status['current_term']}")
            print(f"   Term ends: {status['term_ends']}")
    else:
        if checker.is_school_day(check_date):
            print("✅ Today is a SCHOOL DAY in Sydney!")
        else:
            print("❌ Today is NOT a school day in Sydney.")


if __name__ == "__main__":
    main()
