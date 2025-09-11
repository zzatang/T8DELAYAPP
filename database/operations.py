#!/usr/bin/env python3
"""
Database Operations for School Day Calendar System
Provides high-level database operations with robust error handling and connection pooling.
"""

import logging
import psycopg2
from psycopg2 import sql
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, date
from contextlib import contextmanager
import time
from functools import wraps

from .connection import get_database_manager, DatabaseConnectionManager

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseOperationError(Exception):
    """Custom exception for database operation errors."""
    pass


class RetryableError(DatabaseOperationError):
    """Exception for errors that can be retried."""
    pass


class NonRetryableError(DatabaseOperationError):
    """Exception for errors that should not be retried."""
    pass


def retry_on_database_error(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry database operations on transient errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        logger.info(f"Retrying in {current_delay:.1f} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"Database operation failed after {max_retries + 1} attempts")
                        raise RetryableError(f"Operation failed after {max_retries + 1} attempts: {e}")
                except (psycopg2.ProgrammingError, psycopg2.IntegrityError, psycopg2.DataError) as e:
                    # These errors are not retryable
                    logger.error(f"Non-retryable database error: {e}")
                    raise NonRetryableError(f"Database operation failed: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error in database operation: {e}")
                    raise DatabaseOperationError(f"Unexpected database error: {e}")
            
            # This should never be reached, but just in case
            raise RetryableError(f"Operation failed: {last_exception}")
        
        return wrapper
    return decorator


class SchoolCalendarOperations:
    """
    High-level database operations for the school calendar system.
    Provides CRUD operations with robust error handling and connection pooling.
    """
    
    def __init__(self, db_manager: Optional[DatabaseConnectionManager] = None):
        """
        Initialize school calendar operations.
        
        Args:
            db_manager: Database manager instance. If None, uses global manager.
        """
        self.db_manager = db_manager or get_database_manager()
    
    @retry_on_database_error(max_retries=3)
    def insert_calendar_data(self, calendar_records: List[Dict[str, Any]], batch_size: int = 1000) -> int:
        """
        Insert calendar data in batches with transaction safety.
        
        Args:
            calendar_records: List of calendar record dictionaries
            batch_size: Number of records to insert per batch
            
        Returns:
            Number of records successfully inserted
            
        Raises:
            DatabaseOperationError: If insertion fails
        """
        if not calendar_records:
            logger.warning("No calendar records to insert")
            return 0
        
        total_inserted = 0
        
        try:
            with self.db_manager.get_connection() as conn:
                conn.autocommit = False  # Use explicit transactions
                
                try:
                    with conn.cursor() as cur:
                        # Prepare the insert statement
                        insert_sql = sql.SQL("""
                            INSERT INTO school_calendar (
                                date, day_of_week, school_day, reason, term, 
                                week_of_term, month, quarter, week_of_year
                            ) VALUES (
                                %(date)s, %(day_of_week)s, %(school_day)s, %(reason)s, %(term)s,
                                %(week_of_term)s, %(month)s, %(quarter)s, %(week_of_year)s
                            ) ON CONFLICT (date) DO UPDATE SET
                                day_of_week = EXCLUDED.day_of_week,
                                school_day = EXCLUDED.school_day,
                                reason = EXCLUDED.reason,
                                term = EXCLUDED.term,
                                week_of_term = EXCLUDED.week_of_term,
                                month = EXCLUDED.month,
                                quarter = EXCLUDED.quarter,
                                week_of_year = EXCLUDED.week_of_year,
                                updated_at = CURRENT_TIMESTAMP
                        """)
                        
                        # Insert in batches
                        for i in range(0, len(calendar_records), batch_size):
                            batch = calendar_records[i:i + batch_size]
                            
                            # Execute batch insert
                            cur.executemany(insert_sql, batch)
                            batch_inserted = cur.rowcount
                            total_inserted += batch_inserted
                            
                            logger.info(f"Inserted batch {i//batch_size + 1}: {batch_inserted} records")
                    
                    # Commit transaction
                    conn.commit()
                    logger.info(f"✅ Successfully inserted {total_inserted} calendar records")
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"❌ Failed to insert calendar data, transaction rolled back: {e}")
                    raise
                    
        except psycopg2.Error as e:
            logger.error(f"❌ Database error during calendar data insertion: {e}")
            raise DatabaseOperationError(f"Failed to insert calendar data: {e}")
        
        return total_inserted
    
    @retry_on_database_error(max_retries=2)
    def get_school_day_status(self, check_date: Union[date, datetime, str]) -> Optional[Dict[str, Any]]:
        """
        Get school day status for a specific date.
        
        Args:
            check_date: Date to check (date object, datetime object, or ISO string)
            
        Returns:
            Dictionary with school day information, or None if not found
            
        Raises:
            DatabaseOperationError: If query fails
        """
        # Convert input to date object
        if isinstance(check_date, str):
            try:
                check_date = datetime.fromisoformat(check_date).date()
            except ValueError as e:
                raise NonRetryableError(f"Invalid date format: {check_date}")
        elif isinstance(check_date, datetime):
            check_date = check_date.date()
        
        try:
            with self.db_manager.get_cursor() as cur:
                cur.execute("""
                    SELECT date, day_of_week, school_day, reason, term, 
                           week_of_term, month, quarter, week_of_year, 
                           created_at, updated_at
                    FROM school_calendar 
                    WHERE date = %s
                """, (check_date,))
                
                result = cur.fetchone()
                
                if result:
                    # Convert to regular dictionary and handle date serialization
                    record = dict(result)
                    record['date'] = record['date'].isoformat()
                    if record['created_at']:
                        record['created_at'] = record['created_at'].isoformat()
                    if record['updated_at']:
                        record['updated_at'] = record['updated_at'].isoformat()
                    
                    logger.debug(f"Found school day record for {check_date}: school_day={record['school_day']}")
                    return record
                else:
                    logger.debug(f"No school day record found for {check_date}")
                    return None
                    
        except psycopg2.Error as e:
            logger.error(f"❌ Database error getting school day status: {e}")
            raise DatabaseOperationError(f"Failed to get school day status: {e}")
    
    @retry_on_database_error(max_retries=2)
    def is_school_day(self, check_date: Union[date, datetime, str]) -> bool:
        """
        Quick check if a date is a school day.
        
        Args:
            check_date: Date to check
            
        Returns:
            True if it's a school day, False otherwise
            
        Raises:
            DatabaseOperationError: If query fails
        """
        record = self.get_school_day_status(check_date)
        return record['school_day'] if record else False
    
    @retry_on_database_error(max_retries=2)
    def get_calendar_stats(self, year: Optional[int] = None) -> Dict[str, Any]:
        """
        Get calendar statistics for a specific year or all years.
        
        Args:
            year: Year to get stats for. If None, gets stats for all years.
            
        Returns:
            Dictionary with calendar statistics
            
        Raises:
            DatabaseOperationError: If query fails
        """
        try:
            with self.db_manager.get_cursor() as cur:
                if year:
                    cur.execute("""
                        SELECT * FROM school_calendar_stats 
                        WHERE year = %s
                    """, (year,))
                    result = cur.fetchone()
                    
                    if result:
                        stats = dict(result)
                        # Convert dates to ISO format
                        if stats['first_date']:
                            stats['first_date'] = stats['first_date'].isoformat()
                        if stats['last_date']:
                            stats['last_date'] = stats['last_date'].isoformat()
                        if stats['last_updated']:
                            stats['last_updated'] = stats['last_updated'].isoformat()
                        return stats
                    else:
                        return {'year': year, 'total_days': 0, 'error': 'No data found for year'}
                else:
                    cur.execute("SELECT * FROM school_calendar_stats ORDER BY year")
                    results = cur.fetchall()
                    
                    stats_list = []
                    for result in results:
                        stats = dict(result)
                        # Convert dates to ISO format
                        if stats['first_date']:
                            stats['first_date'] = stats['first_date'].isoformat()
                        if stats['last_date']:
                            stats['last_date'] = stats['last_date'].isoformat()
                        if stats['last_updated']:
                            stats['last_updated'] = stats['last_updated'].isoformat()
                        stats_list.append(stats)
                    
                    return {'years': stats_list, 'total_years': len(stats_list)}
                    
        except psycopg2.Error as e:
            logger.error(f"❌ Database error getting calendar stats: {e}")
            raise DatabaseOperationError(f"Failed to get calendar stats: {e}")
    
    @retry_on_database_error(max_retries=3)
    def delete_calendar_year(self, year: int) -> int:
        """
        Delete all calendar data for a specific year.
        
        Args:
            year: Year to delete
            
        Returns:
            Number of records deleted
            
        Raises:
            DatabaseOperationError: If deletion fails
        """
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM school_calendar 
                        WHERE EXTRACT(YEAR FROM date) = %s
                    """, (year,))
                    
                    deleted_count = cur.rowcount
                    logger.info(f"✅ Deleted {deleted_count} records for year {year}")
                    return deleted_count
                    
        except psycopg2.Error as e:
            logger.error(f"❌ Database error deleting calendar year {year}: {e}")
            raise DatabaseOperationError(f"Failed to delete calendar year: {e}")
    
    @retry_on_database_error(max_retries=2)
    def get_date_range_records(self, start_date: Union[date, str], end_date: Union[date, str]) -> List[Dict[str, Any]]:
        """
        Get calendar records for a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of calendar record dictionaries
            
        Raises:
            DatabaseOperationError: If query fails
        """
        # Convert string dates to date objects
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date).date()
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date).date()
        
        try:
            with self.db_manager.get_cursor() as cur:
                cur.execute("""
                    SELECT date, day_of_week, school_day, reason, term, 
                           week_of_term, month, quarter, week_of_year,
                           created_at, updated_at
                    FROM school_calendar 
                    WHERE date BETWEEN %s AND %s
                    ORDER BY date
                """, (start_date, end_date))
                
                results = cur.fetchall()
                
                records = []
                for result in results:
                    record = dict(result)
                    # Convert dates to ISO format
                    record['date'] = record['date'].isoformat()
                    if record['created_at']:
                        record['created_at'] = record['created_at'].isoformat()
                    if record['updated_at']:
                        record['updated_at'] = record['updated_at'].isoformat()
                    records.append(record)
                
                logger.debug(f"Retrieved {len(records)} records for date range {start_date} to {end_date}")
                return records
                
        except psycopg2.Error as e:
            logger.error(f"❌ Database error getting date range records: {e}")
            raise DatabaseOperationError(f"Failed to get date range records: {e}")
    
    @retry_on_database_error(max_retries=2)
    def validate_calendar_data(self, year: int) -> Dict[str, Any]:
        """
        Validate calendar data for a specific year.
        
        Args:
            year: Year to validate
            
        Returns:
            Dictionary with validation results
            
        Raises:
            DatabaseOperationError: If validation query fails
        """
        try:
            with self.db_manager.get_cursor() as cur:
                validation_results = {
                    'year': year,
                    'is_valid': True,
                    'issues': [],
                    'stats': {}
                }
                
                # Check total days (should be 365 or 366)
                cur.execute("""
                    SELECT COUNT(*) as total_days,
                           MIN(date) as first_date,
                           MAX(date) as last_date
                    FROM school_calendar 
                    WHERE EXTRACT(YEAR FROM date) = %s
                """, (year,))
                
                result = cur.fetchone()
                if result:
                    total_days = result['total_days']
                    expected_days = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
                    
                    validation_results['stats'] = {
                        'total_days': total_days,
                        'expected_days': expected_days,
                        'first_date': result['first_date'].isoformat() if result['first_date'] else None,
                        'last_date': result['last_date'].isoformat() if result['last_date'] else None
                    }
                    
                    if total_days != expected_days:
                        validation_results['is_valid'] = False
                        validation_results['issues'].append(f"Missing days: expected {expected_days}, found {total_days}")
                
                # Check for missing consecutive dates
                cur.execute("""
                    SELECT date + INTERVAL '1 day' as missing_date
                    FROM school_calendar sc1
                    WHERE EXTRACT(YEAR FROM date) = %s
                      AND NOT EXISTS (
                          SELECT 1 FROM school_calendar sc2 
                          WHERE sc2.date = sc1.date + INTERVAL '1 day'
                            AND EXTRACT(YEAR FROM sc2.date) = %s
                      )
                      AND date < %s::date
                    ORDER BY date
                    LIMIT 10
                """, (year, year, f"{year}-12-31"))
                
                missing_dates = cur.fetchall()
                if missing_dates:
                    validation_results['is_valid'] = False
                    missing_list = [d['missing_date'].isoformat() for d in missing_dates]
                    validation_results['issues'].append(f"Missing consecutive dates: {missing_list}")
                
                # Check for null required fields
                cur.execute("""
                    SELECT COUNT(*) as null_count
                    FROM school_calendar 
                    WHERE EXTRACT(YEAR FROM date) = %s
                      AND (day_of_week IS NULL OR school_day IS NULL OR month IS NULL)
                """, (year,))
                
                null_result = cur.fetchone()
                if null_result and null_result['null_count'] > 0:
                    validation_results['is_valid'] = False
                    validation_results['issues'].append(f"Records with null required fields: {null_result['null_count']}")
                
                logger.info(f"Calendar validation for {year}: {'✅ VALID' if validation_results['is_valid'] else '❌ INVALID'}")
                return validation_results
                
        except psycopg2.Error as e:
            logger.error(f"❌ Database error during calendar validation: {e}")
            raise DatabaseOperationError(f"Failed to validate calendar data: {e}")


# Global operations instance
_calendar_ops: Optional[SchoolCalendarOperations] = None


def get_calendar_operations() -> SchoolCalendarOperations:
    """
    Get the global calendar operations instance (singleton pattern).
    
    Returns:
        SchoolCalendarOperations instance
    """
    global _calendar_ops
    
    if _calendar_ops is None:
        _calendar_ops = SchoolCalendarOperations()
    
    return _calendar_ops


# Convenience functions
def insert_calendar_data(records: List[Dict[str, Any]], batch_size: int = 1000) -> int:
    """Insert calendar data using the global operations instance."""
    return get_calendar_operations().insert_calendar_data(records, batch_size)


def is_school_day(check_date: Union[date, datetime, str]) -> bool:
    """Check if a date is a school day using the global operations instance."""
    return get_calendar_operations().is_school_day(check_date)


def get_school_day_status(check_date: Union[date, datetime, str]) -> Optional[Dict[str, Any]]:
    """Get school day status using the global operations instance."""
    return get_calendar_operations().get_school_day_status(check_date)


def main():
    """Command-line interface for database operations testing."""
    import argparse
    import json
    from datetime import date
    
    parser = argparse.ArgumentParser(description='Database Operations Testing Tool')
    parser.add_argument('--check-date', type=str, help='Check if specific date is school day (YYYY-MM-DD)')
    parser.add_argument('--stats', type=int, help='Get calendar stats for specific year')
    parser.add_argument('--validate', type=int, help='Validate calendar data for specific year')
    
    args = parser.parse_args()
    
    try:
        ops = get_calendar_operations()
        
        if args.check_date:
            result = ops.get_school_day_status(args.check_date)
            if result:
                print(f"Date: {args.check_date}")
                print(f"School Day: {'✅ YES' if result['school_day'] else '❌ NO'}")
                print(f"Reason: {result.get('reason', 'N/A')}")
                print(f"Term: {result.get('term', 'N/A')}")
            else:
                print(f"❌ No data found for date: {args.check_date}")
            return 0
        
        if args.stats:
            stats = ops.get_calendar_stats(args.stats)
            print(json.dumps(stats, indent=2))
            return 0
        
        if args.validate:
            validation = ops.validate_calendar_data(args.validate)
            print(json.dumps(validation, indent=2))
            return 0 if validation['is_valid'] else 1
        
        # Default: show help
        parser.print_help()
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
