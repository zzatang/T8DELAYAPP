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
    Uses prepared statements for optimal performance.
    """
    
    def __init__(self, db_manager: Optional[DatabaseConnectionManager] = None):
        """
        Initialize school calendar operations.
        
        Args:
            db_manager: Database manager instance. If None, uses global manager.
        """
        self.db_manager = db_manager or get_database_manager()
        
        # Prepared statement definitions for optimal performance
        self._prepared_statements = {
            'get_school_day_info': """
                SELECT date, day_of_week, school_day, reason, term, 
                       week_of_term, month, quarter, week_of_year
                FROM school_calendar 
                WHERE date = $1
            """,
            'get_year_data': """
                SELECT date, day_of_week, school_day, reason, term, 
                       week_of_term, month, quarter, week_of_year
                FROM school_calendar 
                WHERE EXTRACT(YEAR FROM date) = $1
                ORDER BY date
            """,
            'get_date_range_data': """
                SELECT date, day_of_week, school_day, reason, term, 
                       week_of_term, month, quarter, week_of_year
                FROM school_calendar 
                WHERE date BETWEEN $1 AND $2
                ORDER BY date
            """,
            'get_school_day_status': """
                SELECT date, day_of_week, school_day, reason, term, week_of_term
                FROM school_calendar 
                WHERE date = $1
            """,
            'get_calendar_stats': """
                SELECT * FROM school_calendar_stats 
                WHERE year = $1
            """,
            'count_school_days_range': """
                SELECT COUNT(*) as total_days,
                       COUNT(CASE WHEN school_day = true THEN 1 END) as school_days,
                       COUNT(CASE WHEN school_day = false THEN 1 END) as non_school_days
                FROM school_calendar 
                WHERE date BETWEEN $1 AND $2
            """,
            'get_term_dates': """
                SELECT DISTINCT term, MIN(date) as start_date, MAX(date) as end_date
                FROM school_calendar 
                WHERE EXTRACT(YEAR FROM date) = $1 AND term IS NOT NULL
                GROUP BY term
                ORDER BY MIN(date)
            """
        }
        
        # Track prepared statement usage for performance monitoring
        self._prepared_statement_stats = {stmt_name: {'executions': 0, 'total_time_ms': 0.0} 
                                        for stmt_name in self._prepared_statements}
        self._statements_prepared = False
    
    def _ensure_prepared_statements(self, connection) -> None:
        """
        Ensure all prepared statements are created for the given connection.
        
        Args:
            connection: Database connection to prepare statements for
        """
        if self._statements_prepared:
            return
        
        try:
            with connection.cursor() as cur:
                for stmt_name, stmt_sql in self._prepared_statements.items():
                    # PostgreSQL prepared statement syntax
                    prepare_sql = f"PREPARE {stmt_name} AS {stmt_sql}"
                    cur.execute(prepare_sql)
                    logger.debug(f"✅ Prepared statement '{stmt_name}' created")
                
                self._statements_prepared = True
                logger.info(f"✅ All {len(self._prepared_statements)} prepared statements created")
                
        except psycopg2.Error as e:
            # If statements already exist, that's okay
            if "already exists" in str(e).lower():
                self._statements_prepared = True
                logger.debug("Prepared statements already exist, continuing...")
            else:
                logger.warning(f"⚠️ Failed to prepare statements: {e}")
                # Continue without prepared statements
    
    def _execute_prepared_statement(self, connection, stmt_name: str, params: tuple = ()) -> Any:
        """
        Execute a prepared statement with performance tracking.
        
        Args:
            connection: Database connection
            stmt_name: Name of the prepared statement
            params: Parameters for the statement
            
        Returns:
            Cursor result
        """
        start_time = time.time()
        
        try:
            # Ensure prepared statements exist
            self._ensure_prepared_statements(connection)
            
            with connection.cursor() as cur:
                if self._statements_prepared:
                    # Use prepared statement
                    execute_sql = f"EXECUTE {stmt_name}"
                    if params:
                        execute_sql += f" ({', '.join(['%s'] * len(params))})"
                    cur.execute(execute_sql, params)
                else:
                    # Fall back to direct execution
                    cur.execute(self._prepared_statements[stmt_name].replace('$1', '%s').replace('$2', '%s'), params)
                
                result = cur.fetchall()
                
                # Track performance
                execution_time_ms = (time.time() - start_time) * 1000
                self._prepared_statement_stats[stmt_name]['executions'] += 1
                self._prepared_statement_stats[stmt_name]['total_time_ms'] += execution_time_ms
                
                logger.debug(f"⚡ Executed '{stmt_name}' in {execution_time_ms:.2f}ms")
                return result
                
        except psycopg2.Error as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Failed to execute prepared statement '{stmt_name}' after {execution_time_ms:.2f}ms: {e}")
            raise
    
    def get_prepared_statement_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for prepared statements.
        
        Returns:
            Dictionary containing execution statistics
        """
        stats = {
            'statements_prepared': self._statements_prepared,
            'total_executions': sum(stat['executions'] for stat in self._prepared_statement_stats.values()),
            'statements': {}
        }
        
        for stmt_name, stat in self._prepared_statement_stats.items():
            if stat['executions'] > 0:
                avg_time_ms = stat['total_time_ms'] / stat['executions']
                stats['statements'][stmt_name] = {
                    'executions': stat['executions'],
                    'total_time_ms': round(stat['total_time_ms'], 2),
                    'average_time_ms': round(avg_time_ms, 2)
                }
        
        return stats
    
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
                    # CRITICAL: Explicit commit to ensure data is actually deleted
                    conn.commit()
                    logger.info(f"✅ Deleted {deleted_count} records for year {year}")
                    return deleted_count
                    
        except psycopg2.Error as e:
            logger.error(f"❌ Database error deleting calendar year {year}: {e}")
            raise DatabaseOperationError(f"Failed to delete calendar year: {e}")
    
    @retry_on_database_error(max_retries=3)
    def cleanup_old_calendar_data(self, retention_years: int = 3, current_year: Optional[int] = None) -> Dict[str, Any]:
        """
        Clean up old calendar data beyond the retention period.
        
        Args:
            retention_years: Number of years to keep (past and future from current year)
            current_year: Reference year (defaults to current year)
            
        Returns:
            Dictionary with cleanup results
            
        Raises:
            DatabaseOperationError: If cleanup fails
        """
        if current_year is None:
            current_year = datetime.now().year
            
        cutoff_year_past = current_year - retention_years
        cutoff_year_future = current_year + retention_years
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # First, get list of years that will be deleted
                    cur.execute("""
                        SELECT EXTRACT(YEAR FROM date) as year, COUNT(*) as count
                        FROM school_calendar 
                        WHERE EXTRACT(YEAR FROM date) < %s OR EXTRACT(YEAR FROM date) > %s
                        GROUP BY EXTRACT(YEAR FROM date)
                        ORDER BY year
                    """, (cutoff_year_past, cutoff_year_future))
                    
                    years_to_delete = cur.fetchall()
                    
                    if not years_to_delete:
                        logger.info(f"✅ No old data found beyond retention period ({retention_years} years)")
                        return {
                            'success': True,
                            'years_deleted': [],
                            'total_records_deleted': 0,
                            'retention_policy': f'{retention_years} years',
                            'cutoff_range': f'{cutoff_year_past} to {cutoff_year_future}'
                        }
                    
                    # Delete old data
                    cur.execute("""
                        DELETE FROM school_calendar 
                        WHERE EXTRACT(YEAR FROM date) < %s OR EXTRACT(YEAR FROM date) > %s
                    """, (cutoff_year_past, cutoff_year_future))
                    
                    total_deleted = cur.rowcount
                    years_list = []
                    for row in years_to_delete:
                        # Handle both tuple and dictionary-like results
                        if hasattr(row, 'keys'):  # Dictionary-like (RealDictRow)
                            years_list.append({'year': int(row['year']), 'records': row['count']})
                        else:  # Tuple-like
                            years_list.append({'year': int(row[0]), 'records': row[1]})
                    
                    # CRITICAL: Explicit commit to ensure data is actually deleted
                    conn.commit()
                    logger.info(f"✅ Cleaned up {total_deleted} records from {len(years_to_delete)} years beyond retention period")
                    
                    return {
                        'success': True,
                        'years_deleted': years_list,
                        'total_records_deleted': total_deleted,
                        'retention_policy': f'{retention_years} years',
                        'cutoff_range': f'{cutoff_year_past} to {cutoff_year_future}'
                    }
                    
        except psycopg2.Error as e:
            logger.error(f"❌ Database error during cleanup: {e}")
            raise DatabaseOperationError(f"Failed to cleanup old calendar data: {e}")
    
    @retry_on_database_error(max_retries=3)
    def cleanup_invalid_calendar_data(self, min_school_days: int = 50) -> Dict[str, Any]:
        """
        Clean up years with invalid calendar data (e.g., 0 school days).
        
        Args:
            min_school_days: Minimum school days required for a year to be considered valid
            
        Returns:
            Dictionary with cleanup results
            
        Raises:
            DatabaseOperationError: If cleanup fails
        """
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Find years with invalid data
                    cur.execute("""
                        SELECT 
                            EXTRACT(YEAR FROM date) as year,
                            COUNT(*) as total_days,
                            COUNT(*) FILTER (WHERE school_day = true) as school_days
                        FROM school_calendar 
                        GROUP BY EXTRACT(YEAR FROM date)
                        HAVING COUNT(*) FILTER (WHERE school_day = true) < %s
                        ORDER BY year
                    """, (min_school_days,))
                    
                    invalid_years = cur.fetchall()
                    
                    if not invalid_years:
                        logger.info(f"✅ No invalid data found (all years have >= {min_school_days} school days)")
                        return {
                            'success': True,
                            'years_deleted': [],
                            'total_records_deleted': 0,
                            'validation_criteria': f'minimum {min_school_days} school days'
                        }
                    
                    # Delete invalid years
                    years_to_delete = []
                    years_list = []
                    total_deleted = 0
                    
                    for row in invalid_years:
                        # Handle both tuple and dictionary-like results
                        if hasattr(row, 'keys'):  # Dictionary-like (RealDictRow)
                            year = int(row['year'])
                            total_days = row['total_days']
                            school_days = row['school_days']
                        else:  # Tuple-like
                            year = int(row[0])
                            total_days = row[1]
                            school_days = row[2]
                        
                        years_to_delete.append(year)
                        years_list.append({
                            'year': year, 
                            'total_days': total_days, 
                            'school_days': school_days,
                            'reason': f'Only {school_days} school days (< {min_school_days})'
                        })
                    
                    for year in years_to_delete:
                        cur.execute("""
                            DELETE FROM school_calendar 
                            WHERE EXTRACT(YEAR FROM date) = %s
                        """, (year,))
                        total_deleted += cur.rowcount
                    
                    # CRITICAL: Explicit commit to ensure data is actually deleted
                    conn.commit()
                    logger.info(f"✅ Cleaned up {total_deleted} records from {len(invalid_years)} invalid years")
                    
                    return {
                        'success': True,
                        'years_deleted': years_list,
                        'total_records_deleted': total_deleted,
                        'validation_criteria': f'minimum {min_school_days} school days'
                    }
                    
        except psycopg2.Error as e:
            logger.error(f"❌ Database error during invalid data cleanup: {e}")
            raise DatabaseOperationError(f"Failed to cleanup invalid calendar data: {e}")
    
    @retry_on_database_error(max_retries=3)
    def cleanup_orphaned_calendar_data(self) -> Dict[str, Any]:
        """
        Clean up orphaned or incomplete calendar data.
        
        Returns:
            Dictionary with cleanup results
            
        Raises:
            DatabaseOperationError: If cleanup fails
        """
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Find years with incomplete data (< 300 days, which suggests incomplete generation)
                    cur.execute("""
                        SELECT 
                            EXTRACT(YEAR FROM date) as year,
                            COUNT(*) as total_days,
                            MIN(date) as first_date,
                            MAX(date) as last_date
                        FROM school_calendar 
                        GROUP BY EXTRACT(YEAR FROM date)
                        HAVING COUNT(*) < 300
                        ORDER BY year
                    """)
                    
                    orphaned_years = cur.fetchall()
                    
                    if not orphaned_years:
                        logger.info("✅ No orphaned data found (all years have >= 300 days)")
                        return {
                            'success': True,
                            'years_deleted': [],
                            'total_records_deleted': 0,
                            'cleanup_criteria': 'years with < 300 days (incomplete data)'
                        }
                    
                    # Delete orphaned years
                    total_deleted = 0
                    years_list = []
                    
                    for row in orphaned_years:
                        # Handle both tuple and dictionary-like results
                        if hasattr(row, 'keys'):  # Dictionary-like (RealDictRow)
                            year = int(row['year'])
                            total_days = row['total_days']
                            first_date = str(row['first_date'])
                            last_date = str(row['last_date'])
                        else:  # Tuple-like
                            year = int(row[0])
                            total_days = row[1]
                            first_date = str(row[2])
                            last_date = str(row[3])
                        
                        cur.execute("""
                            DELETE FROM school_calendar 
                            WHERE EXTRACT(YEAR FROM date) = %s
                        """, (year,))
                        deleted_count = cur.rowcount
                        total_deleted += deleted_count
                        
                        years_list.append({
                            'year': year,
                            'total_days': total_days,
                            'first_date': first_date,
                            'last_date': last_date,
                            'records_deleted': deleted_count,
                            'reason': f'Incomplete data ({total_days} days < 300)'
                        })
                    
                    # CRITICAL: Explicit commit to ensure data is actually deleted
                    conn.commit()
                    logger.info(f"✅ Cleaned up {total_deleted} orphaned records from {len(orphaned_years)} incomplete years")
                    
                    return {
                        'success': True,
                        'years_deleted': years_list,
                        'total_records_deleted': total_deleted,
                        'cleanup_criteria': 'years with < 300 days (incomplete data)'
                    }
                    
        except psycopg2.Error as e:
            logger.error(f"❌ Database error during orphaned data cleanup: {e}")
            raise DatabaseOperationError(f"Failed to cleanup orphaned calendar data: {e}")
    
    @retry_on_database_error(max_retries=3)
    def get_cleanup_candidates(self, retention_years: int = 3, min_school_days: int = 50, current_year: Optional[int] = None) -> Dict[str, Any]:
        """
        Identify calendar data that would be cleaned up without actually deleting it.
        
        Args:
            retention_years: Number of years to keep (past and future from current year)
            min_school_days: Minimum school days required for a year to be considered valid
            current_year: Reference year (defaults to current year)
            
        Returns:
            Dictionary with cleanup analysis
            
        Raises:
            DatabaseOperationError: If analysis fails
        """
        if current_year is None:
            current_year = datetime.now().year
            
        cutoff_year_past = current_year - retention_years
        cutoff_year_future = current_year + retention_years
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Get all years with their statistics
                    cur.execute("""
                        SELECT 
                            EXTRACT(YEAR FROM date) as year,
                            COUNT(*) as total_days,
                            COUNT(*) FILTER (WHERE school_day = true) as school_days,
                            MIN(date) as first_date,
                            MAX(date) as last_date,
                            MAX(updated_at) as last_updated
                        FROM school_calendar 
                        GROUP BY EXTRACT(YEAR FROM date)
                        ORDER BY year
                    """)
                    
                    all_years = cur.fetchall()
                    
                    candidates = {
                        'old_data': [],
                        'invalid_data': [],
                        'orphaned_data': [],
                        'safe_data': []
                    }
                    
                    total_records_to_delete = 0
                    
                    for row in all_years:
                        # Handle both tuple and dictionary-like results
                        if hasattr(row, 'keys'):  # Dictionary-like (RealDictRow)
                            year = int(row['year'])
                            total_days = row['total_days']
                            school_days = row['school_days']
                            first_date = str(row['first_date'])
                            last_date = str(row['last_date'])
                            last_updated = str(row['last_updated']) if row['last_updated'] else None
                        else:  # Tuple-like
                            year = int(row[0])
                            total_days = row[1]
                            school_days = row[2]
                            first_date = str(row[3])
                            last_date = str(row[4])
                            last_updated = str(row[5]) if row[5] else None
                        
                        year_info = {
                            'year': year,
                            'total_days': total_days,
                            'school_days': school_days,
                            'first_date': first_date,
                            'last_date': last_date,
                            'last_updated': last_updated
                        }
                        
                        # Categorize the year
                        if year < cutoff_year_past or year > cutoff_year_future:
                            year_info['reason'] = f'Beyond retention period ({retention_years} years)'
                            candidates['old_data'].append(year_info)
                            total_records_to_delete += total_days
                        elif school_days < min_school_days:
                            year_info['reason'] = f'Invalid data ({school_days} < {min_school_days} school days)'
                            candidates['invalid_data'].append(year_info)
                            total_records_to_delete += total_days
                        elif total_days < 300:
                            year_info['reason'] = f'Incomplete data ({total_days} < 300 days)'
                            candidates['orphaned_data'].append(year_info)
                            total_records_to_delete += total_days
                        else:
                            candidates['safe_data'].append(year_info)
                    
                    return {
                        'success': True,
                        'analysis': candidates,
                        'summary': {
                            'total_years': len(all_years),
                            'years_to_delete': len(candidates['old_data']) + len(candidates['invalid_data']) + len(candidates['orphaned_data']),
                            'years_to_keep': len(candidates['safe_data']),
                            'total_records_to_delete': total_records_to_delete
                        },
                        'retention_policy': f'{retention_years} years',
                        'validation_criteria': f'minimum {min_school_days} school days',
                        'current_year': current_year,
                        'retention_range': f'{cutoff_year_past} to {cutoff_year_future}'
                    }
                    
        except psycopg2.Error as e:
            logger.error(f"❌ Database error during cleanup analysis: {e}")
            raise DatabaseOperationError(f"Failed to analyze cleanup candidates: {e}")
    
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

    @retry_on_database_error(max_retries=3)
    def get_school_day_info(self, lookup_date: Union[date, datetime, str]) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive school day information for a specific date using prepared statements.
        
        Args:
            lookup_date: Date to look up (date, datetime, or YYYY-MM-DD string)
            
        Returns:
            Dictionary with school day information, or None if not found
        """
        # Convert input to date object
        if isinstance(lookup_date, str):
            try:
                lookup_date = datetime.strptime(lookup_date, '%Y-%m-%d').date()
            except ValueError:
                logger.error(f"Invalid date format: {lookup_date}. Expected YYYY-MM-DD")
                return None
        elif isinstance(lookup_date, datetime):
            lookup_date = lookup_date.date()
        
        try:
            with self.db_manager.get_connection() as conn:
                results = self._execute_prepared_statement(conn, 'get_school_day_info', (lookup_date,))
                
                if results:
                    result = results[0]  # Get first row
                    return {
                        'date': result['date'],
                        'day_of_week': result['day_of_week'],
                        'school_day': result['school_day'],
                        'reason': result['reason'],
                        'term': result['term'],
                        'week_of_term': result['week_of_term'],
                        'month': result['month'],
                        'quarter': result['quarter'],
                        'week_of_year': result['week_of_year']
                    }
                else:
                    logger.debug(f"No calendar data found for date: {lookup_date}")
                    return None
                        
        except psycopg2.Error as e:
            logger.error(f"❌ Database error during school day lookup: {e}")
            raise DatabaseOperationError(f"Failed to get school day info: {e}")

    @retry_on_database_error(max_retries=3)
    def get_year_data(self, year: int) -> List[Dict[str, Any]]:
        """
        Get all calendar data for a specific year using prepared statements.
        
        Args:
            year: Year to retrieve data for
            
        Returns:
            List of dictionaries containing calendar data for the year
        """
        try:
            with self.db_manager.get_connection() as conn:
                results = self._execute_prepared_statement(conn, 'get_year_data', (year,))
                
                if results:
                    logger.debug(f"Retrieved {len(results)} calendar entries for year {year}")
                    return [
                        {
                            'date': row['date'],
                            'day_of_week': row['day_of_week'],
                            'school_day': row['school_day'],
                            'reason': row['reason'],
                            'term': row['term'],
                            'week_of_term': row['week_of_term'],
                            'month': row['month'],
                            'quarter': row['quarter'],
                            'week_of_year': row['week_of_year']
                        }
                        for row in results
                    ]
                else:
                    logger.warning(f"No calendar data found for year {year}")
                    return []
                        
        except psycopg2.Error as e:
            logger.error(f"❌ Database error during year data retrieval: {e}")
            raise DatabaseOperationError(f"Failed to get year data: {e}")

    @retry_on_database_error(max_retries=3)
    def get_date_range_data(self, start_date: Union[date, str], end_date: Union[date, str]) -> List[Dict[str, Any]]:
        """
        Get calendar data for a date range using prepared statements.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of dictionaries containing calendar data for the date range
        """
        # Convert inputs to date objects
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        try:
            with self.db_manager.get_connection() as conn:
                results = self._execute_prepared_statement(conn, 'get_date_range_data', (start_date, end_date))
                
                logger.debug(f"Retrieved {len(results)} calendar entries for date range {start_date} to {end_date}")
                return [
                    {
                        'date': row['date'],
                        'day_of_week': row['day_of_week'],
                        'school_day': row['school_day'],
                        'reason': row['reason'],
                        'term': row['term'],
                        'week_of_term': row['week_of_term'],
                        'month': row['month'],
                        'quarter': row['quarter'],
                        'week_of_year': row['week_of_year']
                    }
                    for row in results
                ]
                        
        except psycopg2.Error as e:
            logger.error(f"❌ Database error during date range data retrieval: {e}")
            raise DatabaseOperationError(f"Failed to get date range data: {e}")


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
