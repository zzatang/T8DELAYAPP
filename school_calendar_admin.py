#!/usr/bin/env python3
"""
School Calendar Administration CLI

Comprehensive command-line interface for administrators to manage the school calendar system.
Provides calendar generation, database maintenance, cache management, and system monitoring.

Usage:
    python school_calendar_admin.py [command] [options]
    
Commands:
    generate        Generate calendar data for specific years
    validate        Validate calendar data integrity
    cleanup         Remove old or invalid calendar data
    cache           Manage cache operations (refresh, clear, stats)
    health          System health checks and diagnostics
    stats           Display system statistics and performance metrics
    automation      Manage calendar automation system
    
Examples:
    python school_calendar_admin.py generate 2024 2025 2026
    python school_calendar_admin.py validate --all-years
    python school_calendar_admin.py cleanup --older-than 2020
    python school_calendar_admin.py cache refresh --year 2025
    python school_calendar_admin.py health --detailed
    python school_calendar_admin.py stats --performance
"""

import sys
import os
import argparse
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import time

# Fix Windows console encoding issues for emoji characters
if os.name == 'nt':  # Windows
    try:
        os.system('chcp 65001 > nul 2>&1')
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Safe print function for cross-platform emoji support
def safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        safe_message = message.encode('ascii', errors='replace').decode('ascii')
        print(safe_message)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SchoolCalendarAdmin:
    """Administrative interface for school calendar system management."""
    
    def __init__(self):
        """Initialize the admin interface."""
        self.start_time = datetime.now()
        self.operations = None
        self.lookup_system = None
        self.automation = None
        self.RefreshStrategy = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize database operations, lookup system, and automation components."""
        try:
            # Import and initialize database operations
            from database.operations import get_calendar_operations
            self.operations = get_calendar_operations()
            safe_print("✅ Database operations initialized")
            
            # Import and initialize school day lookup system
            from school_day_lookup import SchoolDayLookup, RefreshStrategy
            self.lookup_system = SchoolDayLookup()
            self.RefreshStrategy = RefreshStrategy
            safe_print("✅ School day lookup system initialized")
            
            # Import and initialize automation system (optional)
            try:
                from calendar_automation import get_automation_system
                self.automation = get_automation_system(school_day_lookup=self.lookup_system)
                safe_print("✅ Calendar automation system initialized")
            except ImportError:
                safe_print("⚠️ Calendar automation system not available (optional)")
                self.automation = None
            except Exception as e:
                safe_print(f"⚠️ Failed to initialize automation: {e}")
                self.automation = None
            
            # Import and initialize health monitoring system (optional)
            try:
                from calendar_health_monitor import get_health_monitor
                self.health_monitor = get_health_monitor(
                    school_day_lookup=self.lookup_system,
                    operations=self.operations,
                    check_interval_minutes=15.0
                )
                safe_print("✅ Calendar health monitor initialized")
            except ImportError:
                safe_print("⚠️ Calendar health monitor not available (optional)")
                self.health_monitor = None
            except Exception as e:
                safe_print(f"⚠️ Failed to initialize health monitor: {e}")
                self.health_monitor = None
                
        except Exception as e:
            safe_print(f"❌ Failed to initialize components: {e}")
            safe_print("Some administrative functions may not be available.")
    
    def generate_calendar(self, years: List[int], force: bool = False, batch_size: int = 1000) -> Dict[str, Any]:
        """
        Generate calendar data for specified years.
        
        Args:
            years: List of years to generate
            force: Force regeneration even if data exists
            batch_size: Database batch size for insertions
            
        Returns:
            Dictionary with generation results
        """
        safe_print(f"🚀 Starting calendar generation for years: {', '.join(map(str, years))}")
        
        results = {
            'years_processed': 0,
            'years_successful': 0,
            'years_failed': 0,
            'details': {},
            'total_time': 0
        }
        
        start_time = time.time()
        
        for year in years:
            safe_print(f"\n📅 Processing year {year}...")
            
            try:
                # Check if data already exists
                if not force:
                    validation = self.operations.validate_calendar_data(year)
                    if validation['is_valid']:
                        safe_print(f"✅ Year {year} already has valid data (use --force to regenerate)")
                        results['details'][year] = {'status': 'skipped', 'reason': 'data_exists'}
                        results['years_processed'] += 1
                        continue
                
                # Generate calendar using the existing generator
                import subprocess
                script_path = Path(__file__).parent / "school_calendar_generator.py"
                command = [
                    sys.executable,
                    str(script_path),
                    str(year),
                    "--batch-size", str(batch_size)
                ]
                
                safe_print(f"⚙️ Executing: {' '.join(command)}")
                
                # Run the generator
                process = subprocess.run(command, capture_output=True, text=True, check=False)
                
                if process.returncode == 0:
                    safe_print(f"✅ Successfully generated calendar for {year}")
                    results['details'][year] = {'status': 'success', 'returncode': 0}
                    results['years_successful'] += 1
                    
                    # Refresh cache if lookup system is available
                    if self.lookup_system:
                        try:
                            self.lookup_system.refresh_cache(year=year, strategy=self.RefreshStrategy.IMMEDIATE)
                            safe_print(f"🔄 Cache refreshed for year {year}")
                        except Exception as e:
                            safe_print(f"⚠️ Cache refresh failed for {year}: {e}")
                            
                else:
                    error_output = process.stderr or process.stdout or "Unknown error"
                    safe_print(f"❌ Failed to generate calendar for {year}")
                    safe_print(f"Error: {error_output}")
                    results['details'][year] = {
                        'status': 'failed', 
                        'returncode': process.returncode,
                        'error': error_output
                    }
                    results['years_failed'] += 1
                    
                results['years_processed'] += 1
                
            except Exception as e:
                safe_print(f"❌ Exception while processing year {year}: {e}")
                results['details'][year] = {'status': 'exception', 'error': str(e)}
                results['years_failed'] += 1
                results['years_processed'] += 1
        
        results['total_time'] = time.time() - start_time
        
        # Summary
        safe_print(f"\n📊 Generation Summary:")
        safe_print(f"   Years processed: {results['years_processed']}")
        safe_print(f"   Successful: {results['years_successful']}")
        safe_print(f"   Failed: {results['years_failed']}")
        safe_print(f"   Total time: {results['total_time']:.2f} seconds")
        
        return results
    
    def validate_calendar(self, years: Optional[List[int]] = None, all_years: bool = False) -> Dict[str, Any]:
        """
        Validate calendar data for specified years or all available years.
        
        Args:
            years: Specific years to validate
            all_years: Validate all years in database
            
        Returns:
            Dictionary with validation results
        """
        safe_print("🔍 Starting calendar data validation...")
        
        if all_years:
            # Get all years from database
            try:
                # Query distinct years from the database
                with self.operations.db_manager.get_cursor() as cur:
                    cur.execute("SELECT DISTINCT EXTRACT(YEAR FROM date) as year FROM school_calendar ORDER BY year")
                    years = [int(row[0]) for row in cur.fetchall()]
                safe_print(f"📅 Found data for years: {', '.join(map(str, years))}")
            except Exception as e:
                safe_print(f"❌ Failed to get years from database: {e}")
                return {'error': str(e)}
        
        if not years:
            # Default to current year and next year
            current_year = datetime.now().year
            years = [current_year - 1, current_year, current_year + 1]
            safe_print(f"📅 Validating default years: {', '.join(map(str, years))}")
        
        results = {
            'years_validated': 0,
            'years_valid': 0,
            'years_invalid': 0,
            'details': {},
            'issues_found': []
        }
        
        for year in years:
            safe_print(f"\n🔍 Validating year {year}...")
            
            try:
                validation = self.operations.validate_calendar_data(year)
                results['details'][year] = validation
                results['years_validated'] += 1
                
                if validation['is_valid']:
                    safe_print(f"✅ Year {year}: VALID ({validation['stats']['total_days']} days)")
                    results['years_valid'] += 1
                else:
                    safe_print(f"❌ Year {year}: INVALID")
                    results['years_invalid'] += 1
                    for issue in validation['issues']:
                        safe_print(f"   • {issue}")
                        results['issues_found'].append(f"Year {year}: {issue}")
                        
            except Exception as e:
                safe_print(f"❌ Validation failed for year {year}: {e}")
                results['details'][year] = {'error': str(e)}
                results['years_invalid'] += 1
        
        # Summary
        safe_print(f"\n📊 Validation Summary:")
        safe_print(f"   Years validated: {results['years_validated']}")
        safe_print(f"   Valid: {results['years_valid']}")
        safe_print(f"   Invalid: {results['years_invalid']}")
        
        if results['issues_found']:
            safe_print(f"   Issues found: {len(results['issues_found'])}")
            
        return results
    
    def cleanup_calendar(self, older_than: Optional[int] = None, invalid_only: bool = False, dry_run: bool = False) -> Dict[str, Any]:
        """
        Clean up old or invalid calendar data.
        
        Args:
            older_than: Remove data older than this year
            invalid_only: Only remove invalid data
            dry_run: Show what would be deleted without actually deleting
            
        Returns:
            Dictionary with cleanup results
        """
        safe_print("🧹 Starting calendar data cleanup...")
        
        results = {
            'years_found': 0,
            'years_deleted': 0,
            'records_deleted': 0,
            'details': {},
            'dry_run': dry_run
        }
        
        try:
            # Get all years from database
            with self.operations.db_manager.get_cursor() as cur:
                cur.execute("SELECT DISTINCT EXTRACT(YEAR FROM date) as year FROM school_calendar ORDER BY year")
                all_years = [int(row[0]) for row in cur.fetchall()]
            
            safe_print(f"📅 Found data for years: {', '.join(map(str, all_years))}")
            results['years_found'] = len(all_years)
            
            years_to_delete = []
            
            for year in all_years:
                should_delete = False
                reason = None
                
                # Check age-based deletion
                if older_than and year < older_than:
                    should_delete = True
                    reason = f"older than {older_than}"
                
                # Check validity-based deletion
                if invalid_only:
                    validation = self.operations.validate_calendar_data(year)
                    if not validation['is_valid']:
                        should_delete = True
                        reason = f"invalid data: {', '.join(validation['issues'])}"
                
                if should_delete:
                    years_to_delete.append((year, reason))
            
            # Perform deletions
            for year, reason in years_to_delete:
                safe_print(f"🗑️ {'Would delete' if dry_run else 'Deleting'} year {year}: {reason}")
                
                if not dry_run:
                    try:
                        with self.operations.db_manager.get_connection() as conn:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "DELETE FROM school_calendar WHERE EXTRACT(YEAR FROM date) = %s",
                                    (year,)
                                )
                                deleted_count = cur.rowcount
                                conn.commit()
                                
                        safe_print(f"✅ Deleted {deleted_count} records for year {year}")
                        results['records_deleted'] += deleted_count
                        results['details'][year] = {
                            'deleted': True,
                            'records': deleted_count,
                            'reason': reason
                        }
                        
                        # Clear cache for deleted year
                        if self.lookup_system:
                            try:
                                self.lookup_system.refresh_cache(year=year, strategy=self.RefreshStrategy.IMMEDIATE)
                            except Exception as e:
                                safe_print(f"⚠️ Cache refresh failed for {year}: {e}")
                                
                    except Exception as e:
                        safe_print(f"❌ Failed to delete year {year}: {e}")
                        results['details'][year] = {'deleted': False, 'error': str(e)}
                else:
                    # Dry run - count what would be deleted
                    with self.operations.db_manager.get_cursor() as cur:
                        cur.execute(
                            "SELECT COUNT(*) FROM school_calendar WHERE EXTRACT(YEAR FROM date) = %s",
                            (year,)
                        )
                        count = cur.fetchone()[0]
                    
                    results['records_deleted'] += count
                    results['details'][year] = {
                        'would_delete': True,
                        'records': count,
                        'reason': reason
                    }
                
                results['years_deleted'] += 1
            
            # Summary
            safe_print(f"\n📊 Cleanup Summary:")
            safe_print(f"   Years found: {results['years_found']}")
            safe_print(f"   Years {'would be ' if dry_run else ''}deleted: {results['years_deleted']}")
            safe_print(f"   Records {'would be ' if dry_run else ''}deleted: {results['records_deleted']}")
            
        except Exception as e:
            safe_print(f"❌ Cleanup failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def cleanup_calendar_advanced(self, cleanup_type: str = 'analyze', retention_years: int = 3, 
                                  min_school_days: int = 50, dry_run: bool = True) -> Dict[str, Any]:
        """
        Advanced cleanup with multiple strategies and analysis.
        
        Args:
            cleanup_type: Type of cleanup ('analyze', 'retention', 'invalid', 'orphaned', 'all')
            retention_years: Years to keep (past and future from current year)
            min_school_days: Minimum school days for valid data
            dry_run: Show what would be deleted without actually deleting
            
        Returns:
            Dictionary with cleanup results
        """
        results = {'success': False, 'cleanup_type': cleanup_type}
        
        try:
            if cleanup_type == 'analyze':
                safe_print("🔍 Analyzing cleanup candidates...")
                candidates = self.operations.get_cleanup_candidates(
                    retention_years=retention_years,
                    min_school_days=min_school_days
                )
                
                safe_print("✅ Cleanup Analysis Results:")
                safe_print(f"   Total years: {candidates['summary']['total_years']}")
                safe_print(f"   Years to delete: {candidates['summary']['years_to_delete']}")
                safe_print(f"   Years to keep: {candidates['summary']['years_to_keep']}")
                safe_print(f"   Records to delete: {candidates['summary']['total_records_to_delete']}")
                safe_print(f"   Retention policy: {candidates['retention_policy']}")
                safe_print(f"   Current year: {candidates['current_year']}")
                
                # Show details for each category
                if candidates['analysis']['old_data']:
                    safe_print("\n📅 Old Data (Beyond Retention):")
                    for item in candidates['analysis']['old_data']:
                        safe_print(f"   Year {item['year']}: {item['total_days']} records - {item['reason']}")
                
                if candidates['analysis']['invalid_data']:
                    safe_print("\n⚠️ Invalid Data:")
                    for item in candidates['analysis']['invalid_data']:
                        safe_print(f"   Year {item['year']}: {item['school_days']} school days - {item['reason']}")
                
                if candidates['analysis']['orphaned_data']:
                    safe_print("\n🔍 Orphaned Data:")
                    for item in candidates['analysis']['orphaned_data']:
                        safe_print(f"   Year {item['year']}: {item['total_days']} days - {item['reason']}")
                
                if candidates['analysis']['safe_data']:
                    safe_print("\n✅ Safe Data (Will Be Kept):")
                    for item in candidates['analysis']['safe_data']:
                        safe_print(f"   Year {item['year']}: {item['total_days']} records, {item['school_days']} school days")
                
                results = candidates
                results['success'] = True
                
            elif cleanup_type == 'retention':
                action = "analyze" if dry_run else "clean up"
                safe_print(f"🧹 {action.title()} old data beyond retention period ({retention_years} years)...")
                
                if dry_run:
                    candidates = self.operations.get_cleanup_candidates(retention_years=retention_years)
                    old_data = candidates['analysis']['old_data']
                    if old_data:
                        safe_print(f"📊 Would delete {len(old_data)} years with {sum(item['total_days'] for item in old_data)} records:")
                        for item in old_data:
                            safe_print(f"   Year {item['year']}: {item['total_days']} records - {item['reason']}")
                    else:
                        safe_print("✅ No old data found beyond retention period")
                    results = {'success': True, 'dry_run': True, 'candidates': old_data}
                else:
                    results = self.operations.cleanup_old_calendar_data(retention_years=retention_years)
                    if results['success']:
                        safe_print(f"✅ Cleaned up {results['total_records_deleted']} records from {len(results['years_deleted'])} years")
                        for year_info in results['years_deleted']:
                            safe_print(f"   Year {year_info['year']}: {year_info['records']} records deleted")
                    else:
                        safe_print("❌ Retention cleanup failed")
                
            elif cleanup_type == 'invalid':
                action = "analyze" if dry_run else "clean up"
                safe_print(f"🧹 {action.title()} invalid data (< {min_school_days} school days)...")
                
                if dry_run:
                    candidates = self.operations.get_cleanup_candidates(min_school_days=min_school_days)
                    invalid_data = candidates['analysis']['invalid_data']
                    if invalid_data:
                        safe_print(f"📊 Would delete {len(invalid_data)} years with invalid data:")
                        for item in invalid_data:
                            safe_print(f"   Year {item['year']}: {item['school_days']} school days - {item['reason']}")
                    else:
                        safe_print("✅ No invalid data found")
                    results = {'success': True, 'dry_run': True, 'candidates': invalid_data}
                else:
                    results = self.operations.cleanup_invalid_calendar_data(min_school_days=min_school_days)
                    if results['success']:
                        safe_print(f"✅ Cleaned up {results['total_records_deleted']} records from {len(results['years_deleted'])} invalid years")
                        for year_info in results['years_deleted']:
                            safe_print(f"   Year {year_info['year']}: {year_info['reason']}")
                    else:
                        safe_print("❌ Invalid data cleanup failed")
                
            elif cleanup_type == 'orphaned':
                action = "analyze" if dry_run else "clean up"
                safe_print(f"🧹 {action.title()} orphaned/incomplete data...")
                
                if dry_run:
                    candidates = self.operations.get_cleanup_candidates()
                    orphaned_data = candidates['analysis']['orphaned_data']
                    if orphaned_data:
                        safe_print(f"📊 Would delete {len(orphaned_data)} years with orphaned data:")
                        for item in orphaned_data:
                            safe_print(f"   Year {item['year']}: {item['total_days']} days - {item['reason']}")
                    else:
                        safe_print("✅ No orphaned data found")
                    results = {'success': True, 'dry_run': True, 'candidates': orphaned_data}
                else:
                    results = self.operations.cleanup_orphaned_calendar_data()
                    if results['success']:
                        safe_print(f"✅ Cleaned up {results['total_records_deleted']} orphaned records from {len(results['years_deleted'])} incomplete years")
                        for year_info in results['years_deleted']:
                            safe_print(f"   Year {year_info['year']}: {year_info['reason']}")
                    else:
                        safe_print("❌ Orphaned data cleanup failed")
                
            elif cleanup_type == 'all':
                safe_print(f"🧹 {'Analyzing' if dry_run else 'Performing'} comprehensive cleanup...")
                
                if dry_run:
                    candidates = self.operations.get_cleanup_candidates(
                        retention_years=retention_years,
                        min_school_days=min_school_days
                    )
                    total_to_delete = candidates['summary']['total_records_to_delete']
                    years_to_delete = candidates['summary']['years_to_delete']
                    safe_print(f"📊 Would delete {years_to_delete} years with {total_to_delete} total records")
                    results = {'success': True, 'dry_run': True, 'analysis': candidates}
                else:
                    # Perform all cleanup types
                    safe_print("🔄 Step 1: Cleaning retention data...")
                    retention_result = self.operations.cleanup_old_calendar_data(retention_years=retention_years)
                    
                    safe_print("🔄 Step 2: Cleaning invalid data...")
                    invalid_result = self.operations.cleanup_invalid_calendar_data(min_school_days=min_school_days)
                    
                    safe_print("🔄 Step 3: Cleaning orphaned data...")
                    orphaned_result = self.operations.cleanup_orphaned_calendar_data()
                    
                    total_deleted = (retention_result.get('total_records_deleted', 0) + 
                                   invalid_result.get('total_records_deleted', 0) + 
                                   orphaned_result.get('total_records_deleted', 0))
                    
                    safe_print(f"✅ Comprehensive cleanup completed: {total_deleted} total records deleted")
                    results = {
                        'success': True,
                        'total_records_deleted': total_deleted,
                        'retention_cleanup': retention_result,
                        'invalid_cleanup': invalid_result,
                        'orphaned_cleanup': orphaned_result
                    }
                
            else:
                safe_print(f"❌ Unknown cleanup type: {cleanup_type}")
                results = {'success': False, 'error': f'Unknown cleanup type: {cleanup_type}'}
            
            # Refresh cache after successful cleanup (non-dry-run)
            if results['success'] and not dry_run and cleanup_type != 'analyze':
                safe_print("🔄 Refreshing cache after cleanup...")
                try:
                    self.lookup_system.clear_cache()
                    safe_print("✅ Cache cleared and will be refreshed on next access")
                except Exception as e:
                    safe_print(f"⚠️ Cache refresh failed: {e}")
                
        except Exception as e:
            safe_print(f"❌ Cleanup failed: {e}")
            results = {'success': False, 'error': str(e)}
        
        return results
    
    def manage_cache(self, action: str, year: Optional[int] = None, strategy: str = 'immediate') -> Dict[str, Any]:
        """
        Manage cache operations.
        
        Args:
            action: Cache action (refresh, clear, stats, warm)
            year: Specific year for cache operations
            strategy: Cache refresh strategy
            
        Returns:
            Dictionary with cache operation results
        """
        if not self.lookup_system:
            safe_print("❌ School day lookup system not available")
            return {'error': 'lookup_system_unavailable'}
        
        safe_print(f"🔄 Managing cache: {action}")
        
        results = {'action': action, 'success': False}
        
        try:
            if action == 'refresh':
                if year:
                    safe_print(f"🔄 Refreshing cache for year {year} with strategy '{strategy}'...")
                    # Convert string strategy to enum
                    strategy_enum = getattr(self.RefreshStrategy, strategy.upper(), self.RefreshStrategy.IMMEDIATE)
                    self.lookup_system.refresh_cache(year=year, strategy=strategy_enum)
                    safe_print(f"✅ Cache refreshed for year {year}")
                else:
                    # Refresh current year
                    current_year = datetime.now().year
                    safe_print(f"🔄 Refreshing cache for current year {current_year}...")
                    # Convert string strategy to enum
                    strategy_enum = getattr(self.RefreshStrategy, strategy.upper(), self.RefreshStrategy.IMMEDIATE)
                    self.lookup_system.refresh_cache(year=current_year, strategy=strategy_enum)
                    safe_print(f"✅ Cache refreshed for current year {current_year}")
                
                results['success'] = True
                
            elif action == 'clear':
                safe_print("🧹 Clearing all cache data...")
                self.lookup_system.cache.clear()
                safe_print("✅ Cache cleared")
                results['success'] = True
                
            elif action == 'stats':
                safe_print("📊 Cache Statistics:")
                cache_stats = self.lookup_system.cache.get_stats()
                for key, value in cache_stats.items():
                    safe_print(f"   {key}: {value}")
                results['stats'] = cache_stats
                results['success'] = True
                
            elif action == 'warm':
                # Warm up cache with current and next year
                current_year = datetime.now().year
                years_to_warm = [current_year, current_year + 1]
                
                if year:
                    years_to_warm = [year]
                    
                safe_print(f"🔥 Warming cache for years: {', '.join(map(str, years_to_warm))}")
                
                for warm_year in years_to_warm:
                    try:
                        # Preload cache for the year
                        safe_print(f"🔥 Warming cache for {warm_year}...")
                        self.lookup_system.refresh_cache(year=warm_year, strategy=self.RefreshStrategy.IMMEDIATE)
                        safe_print(f"✅ Cache warmed for {warm_year}")
                    except Exception as e:
                        safe_print(f"⚠️ Failed to warm cache for {warm_year}: {e}")
                
                results['success'] = True
                
            else:
                safe_print(f"❌ Unknown cache action: {action}")
                results['error'] = f'unknown_action: {action}'
                
        except Exception as e:
            safe_print(f"❌ Cache operation failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def system_health(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Perform system health checks.
        
        Args:
            detailed: Include detailed diagnostic information
            
        Returns:
            Dictionary with health check results
        """
        safe_print("🏥 Performing system health checks...")
        
        health = {
            'overall_status': 'unknown',
            'components': {},
            'recommendations': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Database health
        safe_print("🔍 Checking database health...")
        try:
            db_status = self.lookup_system.get_database_status() if self.lookup_system else None
            if db_status and isinstance(db_status, dict):
                status = db_status.get('current_status', 'unknown').lower()
                health['components']['database'] = {
                    'status': status,
                    'details': db_status if detailed else None
                }
                if status != 'healthy':
                    health['recommendations'].append(f"Database status is {status} - check connection and performance")
                safe_print(f"✅ Database: {status.upper()}")
            else:
                health['components']['database'] = {'status': 'unavailable'}
                health['recommendations'].append("Database status unavailable - check connection")
                safe_print("❌ Database: UNAVAILABLE")
        except Exception as e:
            health['components']['database'] = {'status': 'error', 'error': str(e)}
            health['recommendations'].append(f"Database error: {e}")
            safe_print(f"❌ Database: ERROR - {e}")
        
        # Cache health
        safe_print("🔍 Checking cache health...")
        try:
            if self.lookup_system and hasattr(self.lookup_system, 'cache'):
                cache_stats = self.lookup_system.cache.get_stats()
                cache_size = cache_stats.get('size', 0)
                cache_hits = cache_stats.get('hits', 0)
                cache_misses = cache_stats.get('misses', 0)
                
                health['components']['cache'] = {
                    'status': 'healthy' if cache_size > 0 else 'empty',
                    'size': cache_size,
                    'hit_ratio': cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0,
                    'details': cache_stats if detailed else None
                }
                
                if cache_size == 0:
                    health['recommendations'].append("Cache is empty - consider warming up with current year data")
                
                safe_print(f"✅ Cache: {cache_size} entries, {health['components']['cache']['hit_ratio']:.1%} hit ratio")
            else:
                health['components']['cache'] = {'status': 'unavailable'}
                health['recommendations'].append("Cache system unavailable")
                safe_print("❌ Cache: UNAVAILABLE")
        except Exception as e:
            health['components']['cache'] = {'status': 'error', 'error': str(e)}
            health['recommendations'].append(f"Cache error: {e}")
            safe_print(f"❌ Cache: ERROR - {e}")
        
        # Automation health
        safe_print("🔍 Checking automation health...")
        try:
            if self.automation:
                # Check if background monitoring is running
                is_monitoring = hasattr(self.automation, '_monitoring_thread') and \
                               self.automation._monitoring_thread and \
                               self.automation._monitoring_thread.is_alive()
                
                health['components']['automation'] = {
                    'status': 'running' if is_monitoring else 'stopped',
                    'background_monitoring': is_monitoring,
                    'check_interval': getattr(self.automation, 'check_interval_seconds', 0) / 3600,
                    'details': {
                        'task_queue_size': len(getattr(self.automation, '_task_queue', [])),
                        'enable_background_monitoring': getattr(self.automation, 'enable_background_monitoring', False)
                    } if detailed else None
                }
                
                if not is_monitoring and getattr(self.automation, 'enable_background_monitoring', False):
                    health['recommendations'].append("Automation background monitoring is not running but is enabled")
                
                safe_print(f"✅ Automation: {'RUNNING' if is_monitoring else 'STOPPED'}")
            else:
                health['components']['automation'] = {'status': 'unavailable'}
                safe_print("⚠️ Automation: UNAVAILABLE (optional)")
        except Exception as e:
            health['components']['automation'] = {'status': 'error', 'error': str(e)}
            health['recommendations'].append(f"Automation error: {e}")
            safe_print(f"❌ Automation: ERROR - {e}")
        
        # Data integrity check
        safe_print("🔍 Checking data integrity...")
        try:
            current_year = datetime.now().year
            validation = self.operations.validate_calendar_data(current_year)
            
            health['components']['data_integrity'] = {
                'status': 'valid' if validation['is_valid'] else 'invalid',
                'current_year': current_year,
                'current_year_valid': validation['is_valid'],
                'current_year_days': validation['stats']['total_days'],
                'issues': validation['issues'] if not validation['is_valid'] else [],
                'details': validation if detailed else None
            }
            
            if not validation['is_valid']:
                health['recommendations'].append(f"Current year ({current_year}) data is invalid - consider regenerating")
            
            safe_print(f"✅ Data Integrity: {'VALID' if validation['is_valid'] else 'INVALID'} for {current_year}")
            
        except Exception as e:
            health['components']['data_integrity'] = {'status': 'error', 'error': str(e)}
            health['recommendations'].append(f"Data integrity check error: {e}")
            safe_print(f"❌ Data Integrity: ERROR - {e}")
        
        # Determine overall status
        component_statuses = [comp.get('status', 'unknown') for comp in health['components'].values()]
        if any(status == 'error' for status in component_statuses):
            health['overall_status'] = 'critical'
        elif any(status in ['invalid', 'unavailable'] for status in component_statuses):
            health['overall_status'] = 'warning'
        elif all(status in ['healthy', 'valid', 'running', 'stopped'] for status in component_statuses):
            health['overall_status'] = 'healthy'
        else:
            health['overall_status'] = 'unknown'
        
        # Summary
        safe_print(f"\n📊 Health Check Summary:")
        safe_print(f"   Overall Status: {health['overall_status'].upper()}")
        safe_print(f"   Components Checked: {len(health['components'])}")
        safe_print(f"   Recommendations: {len(health['recommendations'])}")
        
        if health['recommendations']:
            safe_print("💡 Recommendations:")
            for rec in health['recommendations']:
                safe_print(f"   • {rec}")
        
        return health
    
    def system_stats(self, performance: bool = False) -> Dict[str, Any]:
        """
        Display system statistics and performance metrics.
        
        Args:
            performance: Include performance benchmarks
            
        Returns:
            Dictionary with system statistics
        """
        safe_print("📊 Gathering system statistics...")
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'database': {},
            'cache': {},
            'performance': {} if performance else None
        }
        
        # Database statistics
        try:
            safe_print("📈 Database Statistics:")
            
            # Get total records and years
            with self.operations.db_manager.get_cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM school_calendar")
                total_records = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(DISTINCT EXTRACT(YEAR FROM date)) FROM school_calendar")
                total_years = cur.fetchone()[0]
                
                cur.execute("SELECT MIN(date), MAX(date) FROM school_calendar")
                date_range = cur.fetchone()
                
            stats['database'] = {
                'total_records': total_records,
                'total_years': total_years,
                'date_range': {
                    'earliest': str(date_range[0]) if date_range[0] else None,
                    'latest': str(date_range[1]) if date_range[1] else None
                }
            }
            
            safe_print(f"   Total records: {total_records:,}")
            safe_print(f"   Years covered: {total_years}")
            safe_print(f"   Date range: {date_range[0]} to {date_range[1]}")
            
        except Exception as e:
            safe_print(f"❌ Database stats error: {e}")
            stats['database']['error'] = str(e)
        
        # Cache statistics
        try:
            if self.lookup_system and hasattr(self.lookup_system, 'cache'):
                safe_print("📈 Cache Statistics:")
                cache_stats = self.lookup_system.cache.get_stats()
                stats['cache'] = cache_stats
                
                for key, value in cache_stats.items():
                    if isinstance(value, (int, float)):
                        safe_print(f"   {key}: {value:,}" if isinstance(value, int) else f"   {key}: {value:.3f}")
                    else:
                        safe_print(f"   {key}: {value}")
            else:
                safe_print("⚠️ Cache statistics unavailable")
                
        except Exception as e:
            safe_print(f"❌ Cache stats error: {e}")
            stats['cache']['error'] = str(e)
        
        # Performance benchmarks
        if performance and self.lookup_system:
            safe_print("⚡ Performance Benchmarks:")
            try:
                # Test lookup performance
                test_date = date.today()
                
                # Warm up
                self.lookup_system.is_school_day(test_date)
                
                # Benchmark lookup speed
                import time
                iterations = 1000
                start_time = time.perf_counter()
                
                for _ in range(iterations):
                    self.lookup_system.is_school_day(test_date)
                
                end_time = time.perf_counter()
                avg_lookup_time = (end_time - start_time) / iterations * 1000  # Convert to milliseconds
                
                stats['performance'] = {
                    'avg_lookup_time_ms': avg_lookup_time,
                    'lookups_per_second': 1000 / avg_lookup_time,
                    'test_iterations': iterations,
                    'test_date': str(test_date)
                }
                
                safe_print(f"   Average lookup time: {avg_lookup_time:.3f} ms")
                safe_print(f"   Lookups per second: {1000 / avg_lookup_time:,.0f}")
                safe_print(f"   Test iterations: {iterations:,}")
                
            except Exception as e:
                safe_print(f"❌ Performance benchmark error: {e}")
                stats['performance']['error'] = str(e)
        
        return stats
    
    def manage_health_monitoring(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Manage the calendar health monitoring system.
        
        Args:
            action: Health monitoring action (check, start, stop, dashboard, report)
            **kwargs: Additional parameters for specific actions
            
        Returns:
            Dictionary with health monitoring results
        """
        if not self.health_monitor:
            safe_print("❌ Calendar health monitor not available")
            return {'error': 'health_monitor_unavailable'}
        
        safe_print(f"🏥 Managing health monitoring: {action}")
        
        results = {'action': action, 'success': False}
        
        try:
            if action == 'check':
                safe_print("🔍 Performing comprehensive health check...")
                health_report = self.health_monitor.perform_health_check()
                
                overall_health = health_report['overall_health']
                safe_print(f"📊 Overall Health: {overall_health.upper()}")
                
                # Display component health
                safe_print("🔧 Component Health:")
                for component, status in health_report['components'].items():
                    status_icon = "✅" if status['status'] == 'healthy' else "⚠️" if status['status'] == 'warning' else "❌"
                    safe_print(f"   {status_icon} {component}: {status['status']}")
                    
                    if status.get('issues'):
                        for issue in status['issues']:
                            safe_print(f"      • {issue}")
                
                # Display validation issues
                if health_report['validation_issues']:
                    safe_print(f"\n⚠️ Validation Issues ({len(health_report['validation_issues'])}):")
                    for issue in health_report['validation_issues'][:5]:  # Show first 5
                        severity_icon = "🚨" if issue['severity'] == 'critical' else "⚠️" if issue['severity'] == 'error' else "ℹ️"
                        safe_print(f"   {severity_icon} {issue['message']}")
                        if issue.get('recommendation'):
                            safe_print(f"      💡 {issue['recommendation']}")
                    
                    if len(health_report['validation_issues']) > 5:
                        safe_print(f"   ... and {len(health_report['validation_issues']) - 5} more issues")
                
                # Display performance metrics
                perf = health_report['performance_metrics']
                safe_print(f"\n⚡ Performance Metrics:")
                safe_print(f"   Response Time: {perf['avg_response_time_ms']:.2f}ms")
                safe_print(f"   Cache Hit Rate: {perf['cache_hit_rate']:.1f}%")
                safe_print(f"   Total Queries: {perf['total_queries']:,}")
                safe_print(f"   Error Rate: {perf['errors_per_hour']:.1f}/hour")
                safe_print(f"   Uptime: {perf['uptime_hours']:.1f} hours")
                
                # Display recommendations
                if health_report['recommendations']:
                    safe_print(f"\n💡 Recommendations:")
                    for rec in health_report['recommendations']:
                        safe_print(f"   • {rec}")
                
                results['health_report'] = health_report
                results['success'] = True
                
            elif action == 'start':
                safe_print("🚀 Starting continuous health monitoring...")
                self.health_monitor.start_monitoring()
                safe_print("✅ Health monitoring started")
                results['success'] = True
                
            elif action == 'stop':
                safe_print("🛑 Stopping health monitoring...")
                self.health_monitor.stop_monitoring()
                safe_print("✅ Health monitoring stopped")
                results['success'] = True
                
            elif action == 'dashboard':
                safe_print("📊 Health Dashboard Data:")
                dashboard_data = self.health_monitor.get_health_dashboard_data()
                
                safe_print(f"   Overall Health: {dashboard_data['overall_health'].upper()}")
                safe_print(f"   Uptime: {dashboard_data['uptime_hours']:.1f} hours")
                safe_print(f"   Issues: {dashboard_data['validation_issues_count']} total, {dashboard_data['critical_issues_count']} critical")
                
                metrics = dashboard_data['metrics']
                safe_print(f"   Avg Response: {metrics['avg_response_time_ms']:.2f}ms")
                safe_print(f"   Cache Hit Rate: {metrics['cache_hit_rate']:.1f}%")
                safe_print(f"   Total Queries: {metrics['total_queries']:,}")
                
                if dashboard_data['recent_issues']:
                    safe_print("\n📋 Recent Issues:")
                    for issue in dashboard_data['recent_issues'][-3:]:  # Last 3 issues
                        safe_print(f"   • {issue['message']} ({issue['severity']})")
                
                results['dashboard_data'] = dashboard_data
                results['success'] = True
                
            else:
                safe_print(f"❌ Unknown health monitoring action: {action}")
                results['error'] = f'unknown_action: {action}'
                
        except Exception as e:
            safe_print(f"❌ Health monitoring operation failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def manage_performance_monitoring(self, action: str, operation: Optional[str] = None, format: str = 'summary') -> Dict[str, Any]:
        """
        Manage the performance monitoring system.
        
        Args:
            action: Performance action (stats, recommendations, reset, export)
            operation: Specific operation type to focus on
            format: Output format (summary, json)
            
        Returns:
            Dictionary with performance monitoring results
        """
        if not self.lookup_system:
            safe_print("❌ School day lookup system not available")
            return {'error': 'lookup_system_unavailable'}
        
        safe_print(f"⚡ Managing performance monitoring: {action}")
        
        results = {'action': action, 'success': False}
        
        try:
            if action == 'stats':
                safe_print("📊 Retrieving performance statistics...")
                perf_stats = self.lookup_system.get_advanced_performance_stats()
                
                if perf_stats.get('performance_monitoring') == 'disabled':
                    safe_print("⚠️ Performance monitoring is disabled")
                    safe_print("📈 Basic Statistics:")
                    basic_stats = perf_stats['basic_stats']
                    safe_print(f"   Total Lookups: {basic_stats['total_lookups']:,}")
                    safe_print(f"   Cache Hits: {basic_stats['cache_hits']:,}")
                    safe_print(f"   Database Hits: {basic_stats['database_hits']:,}")
                    safe_print(f"   Errors: {basic_stats['errors']:,}")
                    safe_print(f"   Average Lookup Time: {basic_stats['average_lookup_time_ms']:.3f}ms")
                    
                elif perf_stats.get('performance_monitoring') == 'enabled':
                    detailed_perf = perf_stats['detailed_performance']
                    
                    safe_print(f"📊 Overall Performance Summary:")
                    safe_print(f"   Total Operations: {detailed_perf['total_operations']:,}")
                    safe_print(f"   Overall Success Rate: {detailed_perf['overall_success_rate']:.1f}%")
                    
                    if operation:
                        # Show specific operation stats
                        if operation in detailed_perf['operations']:
                            op_stats = detailed_perf['operations'][operation]
                            safe_print(f"\n⚡ {operation.replace('_', ' ').title()} Statistics:")
                            safe_print(f"   Operations: {op_stats['total_operations']:,}")
                            safe_print(f"   Success Rate: {op_stats['success_rate']:.1f}%")
                            safe_print(f"   Average Duration: {op_stats['avg_duration_ms']:.3f}ms")
                            safe_print(f"   Median Duration: {op_stats['median_duration_ms']:.3f}ms")
                            safe_print(f"   95th Percentile: {op_stats['p95_duration_ms']:.3f}ms")
                            safe_print(f"   99th Percentile: {op_stats['p99_duration_ms']:.3f}ms")
                            if 'cache_hit_rate' in op_stats:
                                safe_print(f"   Cache Hit Rate: {op_stats['cache_hit_rate']:.1f}%")
                        else:
                            safe_print(f"⚠️ No statistics available for operation: {operation}")
                    else:
                        # Show all operation stats
                        safe_print(f"\n📈 By Operation Type:")
                        for op_name, op_stats in detailed_perf['operations'].items():
                            safe_print(f"   {op_name.replace('_', ' ').title()}:")
                            safe_print(f"     Operations: {op_stats['total_operations']:,}")
                            safe_print(f"     Avg Duration: {op_stats['avg_duration_ms']:.3f}ms")
                            safe_print(f"     Success Rate: {op_stats['success_rate']:.1f}%")
                    
                    # Performance levels distribution
                    perf_levels = detailed_perf.get('performance_levels', {})
                    if perf_levels:
                        safe_print(f"\n🎯 Performance Levels (Last 1000 operations):")
                        for level, count in perf_levels.items():
                            if count > 0:
                                level_icon = {"excellent": "🚀", "good": "✅", "acceptable": "🟡", "slow": "⚠️", "critical": "🚨"}.get(level, "❓")
                                safe_print(f"   {level_icon} {level.title()}: {count:,}")
                    
                    if format == 'json':
                        import json
                        safe_print(f"\n📄 JSON Export:")
                        safe_print(json.dumps(perf_stats, indent=2, default=str))
                
                results['stats'] = perf_stats
                results['success'] = True
                
            elif action == 'recommendations':
                safe_print("💡 Generating performance recommendations...")
                perf_stats = self.lookup_system.get_advanced_performance_stats()
                
                if perf_stats.get('performance_monitoring') == 'enabled':
                    recommendations = perf_stats.get('recommendations', [])
                    
                    if recommendations:
                        safe_print("💡 Performance Optimization Recommendations:")
                        for i, rec in enumerate(recommendations, 1):
                            priority_icon = "🚨" if rec['priority'] == 'critical' else "⚠️"
                            safe_print(f"\n{i}. {priority_icon} {rec['title']} ({rec['priority'].upper()})")
                            safe_print(f"   📋 {rec['description']}")
                            safe_print(f"   💡 {rec['recommendation']}")
                            safe_print(f"   📈 {rec['impact']}")
                    else:
                        safe_print("✅ No performance recommendations at this time")
                        safe_print("🎯 System is performing optimally!")
                else:
                    safe_print("⚠️ Performance monitoring is disabled - enable it for detailed recommendations")
                
                results['recommendations'] = perf_stats.get('recommendations', [])
                results['success'] = True
                
            elif action == 'reset':
                safe_print("🔄 Resetting performance statistics...")
                self.lookup_system.reset_performance_stats(operation)
                
                if operation:
                    safe_print(f"✅ Reset performance stats for {operation}")
                else:
                    safe_print("✅ Reset all performance statistics")
                
                results['success'] = True
                
            elif action == 'export':
                safe_print("📤 Exporting performance metrics...")
                perf_stats = self.lookup_system.get_advanced_performance_stats()
                
                if format == 'json':
                    import json
                    export_data = json.dumps(perf_stats, indent=2, default=str)
                    safe_print(export_data)
                else:
                    safe_print("📊 Performance Metrics Export:")
                    safe_print(f"   Timestamp: {datetime.now().isoformat()}")
                    if perf_stats.get('performance_monitoring') == 'enabled':
                        detailed = perf_stats['detailed_performance']
                        safe_print(f"   Total Operations: {detailed['total_operations']:,}")
                        safe_print(f"   Success Rate: {detailed['overall_success_rate']:.1f}%")
                        safe_print(f"   Operations by Type: {len(detailed['operations'])}")
                    else:
                        safe_print("   Performance monitoring: disabled")
                
                results['export_data'] = perf_stats
                results['success'] = True
                
            else:
                safe_print(f"❌ Unknown performance action: {action}")
                results['error'] = f'unknown_action: {action}'
                
        except Exception as e:
            safe_print(f"❌ Performance monitoring operation failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def manage_error_recovery(self, action: str, circuit_breaker_name: Optional[str] = None, detailed: bool = False) -> Dict[str, Any]:
        """
        Manage the error recovery and circuit breaker system.
        
        Args:
            action: Error recovery action (stats, reset, circuit-breaker)
            circuit_breaker_name: Specific circuit breaker name
            detailed: Show detailed information
            
        Returns:
            Dictionary with error recovery results
        """
        if not self.lookup_system:
            safe_print("❌ School day lookup system not available")
            return {'error': 'lookup_system_unavailable'}
        
        safe_print(f"🛡️ Managing error recovery: {action}")
        
        results = {'action': action, 'success': False}
        
        try:
            if action == 'stats':
                safe_print("📊 Retrieving error recovery statistics...")
                error_stats = self.lookup_system.get_error_recovery_stats()
                
                if error_stats.get('error_recovery') == 'disabled':
                    safe_print("⚠️ Error recovery system is disabled")
                    safe_print("💡 Error recovery provides circuit breakers and automatic retry mechanisms")
                    
                elif error_stats.get('error_recovery') == 'enabled':
                    safe_print(f"📈 Error Recovery Summary:")
                    safe_print(f"   Total Errors: {error_stats['total_errors']:,}")
                    safe_print(f"   Error Rate (last hour): {error_stats['error_rate_per_hour']}")
                    
                    # Show errors by severity
                    if error_stats['errors_by_severity']:
                        safe_print(f"\n🚨 Errors by Severity:")
                        severity_icons = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}
                        for severity, count in error_stats['errors_by_severity'].items():
                            if count > 0:
                                icon = severity_icons.get(severity.replace('severity_', ''), '❓')
                                safe_print(f"   {icon} {severity.replace('severity_', '').title()}: {count:,}")
                    
                    # Show errors by type
                    if error_stats['errors_by_type'] and detailed:
                        safe_print(f"\n🔍 Errors by Type:")
                        for error_type, count in sorted(error_stats['errors_by_type'].items(), key=lambda x: x[1], reverse=True)[:10]:
                            safe_print(f"   {error_type}: {count:,}")
                    
                    # Circuit breaker status
                    circuit_breakers = error_stats['circuit_breakers']
                    if circuit_breakers:
                        safe_print(f"\n🔧 Circuit Breakers:")
                        for cb_name, cb_stats in circuit_breakers.items():
                            if circuit_breaker_name and cb_name != circuit_breaker_name:
                                continue
                            
                            state_icon = {'closed': '🟢', 'open': '🔴', 'half_open': '🟡'}.get(cb_stats['state'], '❓')
                            safe_print(f"   {state_icon} {cb_name}: {cb_stats['state'].upper()}")
                            safe_print(f"     Success Rate: {cb_stats['success_rate']:.1f}%")
                            safe_print(f"     Total Operations: {cb_stats['total_operations']:,}")
                            safe_print(f"     Failure Count: {cb_stats['failure_count']}")
                            
                            if detailed and cb_stats['last_failure_time']:
                                safe_print(f"     Last Failure: {cb_stats['last_failure_time']}")
                    
                    # Recent errors
                    if error_stats['recent_errors'] and detailed:
                        safe_print(f"\n📋 Recent Errors:")
                        for error in error_stats['recent_errors']:
                            timestamp = error['timestamp'][:19]  # Remove microseconds
                            severity_icon = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}.get(error['severity'], '❓')
                            safe_print(f"   {severity_icon} {timestamp} [{error['type']}] {error['operation']}")
                            if detailed:
                                safe_print(f"      {error['message'][:100]}...")
                    
                else:
                    safe_print(f"❌ Error recovery system error: {error_stats.get('error', 'unknown')}")
                
                results['stats'] = error_stats
                results['success'] = True
                
            elif action == 'reset':
                safe_print("🔄 Resetting error recovery statistics...")
                
                # Check if error recovery is available
                error_stats = self.lookup_system.get_error_recovery_stats()
                if error_stats.get('error_recovery') == 'enabled':
                    # Access the error recovery system through the lookup system
                    if hasattr(self.lookup_system, 'error_recovery') and self.lookup_system.error_recovery:
                        self.lookup_system.error_recovery.reset_error_history()
                        safe_print("✅ Error history and statistics reset")
                    else:
                        safe_print("⚠️ Error recovery system not accessible for reset")
                else:
                    safe_print("⚠️ Error recovery system is disabled")
                
                results['success'] = True
                
            elif action == 'circuit-breaker':
                safe_print("🔧 Circuit Breaker Management:")
                error_stats = self.lookup_system.get_error_recovery_stats()
                
                if error_stats.get('error_recovery') == 'enabled':
                    circuit_breakers = error_stats['circuit_breakers']
                    
                    if circuit_breaker_name:
                        if circuit_breaker_name in circuit_breakers:
                            cb_stats = circuit_breakers[circuit_breaker_name]
                            safe_print(f"\n🔧 Circuit Breaker: {circuit_breaker_name}")
                            safe_print(f"   State: {cb_stats['state'].upper()}")
                            safe_print(f"   Success Rate: {cb_stats['success_rate']:.1f}%")
                            safe_print(f"   Failure Rate: {cb_stats['failure_rate']:.1f}%")
                            safe_print(f"   Total Operations: {cb_stats['total_operations']:,}")
                            safe_print(f"   Success Count: {cb_stats['success_count']:,}")
                            safe_print(f"   Recent Failures: {cb_stats['failure_count_recent']:,}")
                            safe_print(f"   Timeout Count: {cb_stats['timeout_count']:,}")
                            
                            if cb_stats['state'] == 'half_open':
                                safe_print(f"   Half-Open Calls: {cb_stats['half_open_calls']}")
                                safe_print(f"   Half-Open Successes: {cb_stats['half_open_successes']}")
                            
                            if cb_stats['last_failure_time']:
                                safe_print(f"   Last Failure: {cb_stats['last_failure_time']}")
                        else:
                            safe_print(f"❌ Circuit breaker '{circuit_breaker_name}' not found")
                            safe_print(f"Available circuit breakers: {', '.join(circuit_breakers.keys())}")
                    else:
                        safe_print("Available Circuit Breakers:")
                        for cb_name, cb_stats in circuit_breakers.items():
                            state_icon = {'closed': '🟢', 'open': '🔴', 'half_open': '🟡'}.get(cb_stats['state'], '❓')
                            safe_print(f"   {state_icon} {cb_name}: {cb_stats['state'].upper()} ({cb_stats['success_rate']:.1f}% success)")
                else:
                    safe_print("⚠️ Error recovery system is disabled")
                
                results['success'] = True
                
            else:
                safe_print(f"❌ Unknown error recovery action: {action}")
                results['error'] = f'unknown_action: {action}'
                
        except Exception as e:
            safe_print(f"❌ Error recovery operation failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def manage_automation(self, action: str) -> Dict[str, Any]:
        """
        Manage the calendar automation system.
        
        Args:
            action: Automation action (status, start, stop, check, process)
            
        Returns:
            Dictionary with automation management results
        """
        if not self.automation:
            safe_print("❌ Calendar automation system not available")
            return {'error': 'automation_unavailable'}
        
        safe_print(f"🤖 Managing automation: {action}")
        
        results = {'action': action, 'success': False}
        
        try:
            if action == 'status':
                is_monitoring = hasattr(self.automation, '_monitoring_thread') and \
                               self.automation._monitoring_thread and \
                               self.automation._monitoring_thread.is_alive()
                
                task_queue_size = len(getattr(self.automation, '_task_queue', []))
                
                results.update({
                    'background_monitoring': is_monitoring,
                    'task_queue_size': task_queue_size,
                    'check_interval_hours': getattr(self.automation, 'check_interval_seconds', 0) / 3600,
                    'enable_background_monitoring': getattr(self.automation, 'enable_background_monitoring', False)
                })
                
                safe_print(f"   Background monitoring: {'RUNNING' if is_monitoring else 'STOPPED'}")
                safe_print(f"   Task queue size: {task_queue_size}")
                safe_print(f"   Check interval: {results['check_interval_hours']:.1f} hours")
                safe_print(f"   Auto-monitoring enabled: {results['enable_background_monitoring']}")
                
                results['success'] = True
                
            elif action == 'start':
                safe_print("🚀 Starting background monitoring...")
                self.automation.start_background_monitoring()
                safe_print("✅ Background monitoring started")
                results['success'] = True
                
            elif action == 'stop':
                safe_print("🛑 Stopping background monitoring...")
                self.automation.stop_background_monitoring()
                safe_print("✅ Background monitoring stopped")
                results['success'] = True
                
            elif action == 'check':
                safe_print("🔍 Performing manual automation checks...")
                check_results = self.automation.perform_automatic_checks()
                results.update(check_results)
                
                safe_print(f"   Checks performed: {check_results.get('checks_performed', 0)}")
                safe_print(f"   Issues found: {len(check_results.get('issues_found', []))}")
                safe_print(f"   Tasks triggered: {check_results.get('tasks_triggered', 0)}")
                
                if check_results.get('issues_found'):
                    safe_print("   Issues:")
                    for issue in check_results['issues_found']:
                        safe_print(f"     • {issue}")
                
                results['success'] = True
                
            elif action == 'process':
                safe_print("⚙️ Processing pending tasks...")
                process_results = self.automation.process_pending_tasks()
                results.update(process_results)
                
                safe_print(f"   Tasks processed: {process_results.get('processed', 0)}")
                safe_print(f"   Tasks completed: {process_results.get('completed', 0)}")
                safe_print(f"   Tasks failed: {process_results.get('failed', 0)}")
                
                results['success'] = True
                
            else:
                safe_print(f"❌ Unknown automation action: {action}")
                results['error'] = f'unknown_action: {action}'
                
        except Exception as e:
            safe_print(f"❌ Automation operation failed: {e}")
            results['error'] = str(e)
        
        return results


def create_parser():
    """Create the argument parser for the admin CLI."""
    parser = argparse.ArgumentParser(
        description='School Calendar Administration CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s generate 2024 2025 2026
  %(prog)s validate --all-years
  %(prog)s cleanup --older-than 2020 --dry-run
  %(prog)s cache refresh --year 2025
  %(prog)s health --detailed
  %(prog)s stats --performance
  %(prog)s automation status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate calendar data for specific years')
    gen_parser.add_argument('years', nargs='+', type=int, help='Years to generate')
    gen_parser.add_argument('--force', action='store_true', help='Force regeneration even if data exists')
    gen_parser.add_argument('--batch-size', type=int, default=1000, help='Database batch size')
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate calendar data integrity')
    val_parser.add_argument('years', nargs='*', type=int, help='Specific years to validate')
    val_parser.add_argument('--all-years', action='store_true', help='Validate all years in database')
    
    # Cleanup command (legacy)
    clean_parser = subparsers.add_parser('cleanup', help='Remove old or invalid calendar data')
    clean_parser.add_argument('--older-than', type=int, help='Remove data older than this year')
    clean_parser.add_argument('--invalid-only', action='store_true', help='Only remove invalid data')
    clean_parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without deleting')
    
    # Advanced cleanup command
    cleanup_parser = subparsers.add_parser('cleanup-advanced', help='Advanced cleanup with multiple strategies')
    cleanup_parser.add_argument('type', choices=['analyze', 'retention', 'invalid', 'orphaned', 'all'], 
                               help='Type of cleanup to perform')
    cleanup_parser.add_argument('--retention-years', type=int, default=3, 
                               help='Years to keep (past and future from current year)')
    cleanup_parser.add_argument('--min-school-days', type=int, default=50, 
                               help='Minimum school days for valid data')
    cleanup_parser.add_argument('--dry-run', action='store_true', default=True,
                               help='Show what would be deleted without deleting (default: True)')
    cleanup_parser.add_argument('--execute', action='store_true',
                               help='Actually perform cleanup (overrides --dry-run)')
    
    # Cache command
    cache_parser = subparsers.add_parser('cache', help='Manage cache operations')
    cache_parser.add_argument('action', choices=['refresh', 'clear', 'stats', 'warm'], help='Cache action')
    cache_parser.add_argument('--year', type=int, help='Specific year for cache operations')
    cache_parser.add_argument('--strategy', choices=['immediate', 'lazy', 'scheduled'], default='immediate', help='Cache refresh strategy')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='System health checks and diagnostics')
    health_parser.add_argument('--detailed', action='store_true', help='Include detailed diagnostic information')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Display system statistics and performance metrics')
    stats_parser.add_argument('--performance', action='store_true', help='Include performance benchmarks')
    
    # Automation command
    auto_parser = subparsers.add_parser('automation', help='Manage calendar automation system')
    auto_parser.add_argument('action', choices=['status', 'start', 'stop', 'check', 'process'], help='Automation action')
    
    # Health monitoring command
    health_parser = subparsers.add_parser('health-monitor', help='Manage calendar health monitoring system')
    health_parser.add_argument('action', choices=['check', 'start', 'stop', 'dashboard'], help='Health monitoring action')
    health_parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    
    # Performance monitoring command
    perf_parser = subparsers.add_parser('performance', help='Manage performance monitoring and metrics')
    perf_parser.add_argument('action', choices=['stats', 'recommendations', 'reset', 'export'], help='Performance monitoring action')
    perf_parser.add_argument('--operation', choices=['cache_lookup', 'database_lookup', 'cache_write', 'batch_preload', 'validation'], help='Specific operation type')
    perf_parser.add_argument('--format', choices=['json', 'summary'], default='summary', help='Output format')
    
    # Error recovery command
    error_parser = subparsers.add_parser('error-recovery', help='Manage error recovery and circuit breakers')
    error_parser.add_argument('action', choices=['stats', 'reset', 'circuit-breaker'], help='Error recovery action')
    error_parser.add_argument('--circuit-breaker-name', help='Specific circuit breaker name for stats')
    error_parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    
    return parser


def main():
    """Main entry point for the admin CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize admin interface
    safe_print("🚀 Initializing School Calendar Administration Interface...")
    admin = SchoolCalendarAdmin()
    
    try:
        # Execute command
        if args.command == 'generate':
            result = admin.generate_calendar(
                years=args.years,
                force=args.force,
                batch_size=args.batch_size
            )
            return 0 if result['years_failed'] == 0 else 1
            
        elif args.command == 'validate':
            result = admin.validate_calendar(
                years=args.years if args.years else None,
                all_years=args.all_years
            )
            return 0 if result['years_invalid'] == 0 else 1
            
        elif args.command == 'cleanup':
            result = admin.cleanup_calendar(
                older_than=args.older_than,
                invalid_only=args.invalid_only,
                dry_run=args.dry_run
            )
            return 0 if 'error' not in result else 1
            
        elif args.command == 'cleanup-advanced':
            dry_run = args.dry_run and not args.execute  # Execute overrides dry_run
            result = admin.cleanup_calendar_advanced(
                cleanup_type=args.type,
                retention_years=args.retention_years,
                min_school_days=args.min_school_days,
                dry_run=dry_run
            )
            return 0 if result.get('success', False) else 1
            
        elif args.command == 'cache':
            result = admin.manage_cache(
                action=args.action,
                year=args.year,
                strategy=args.strategy
            )
            return 0 if result['success'] else 1
            
        elif args.command == 'health':
            result = admin.system_health(detailed=args.detailed)
            return 0 if result['overall_status'] in ['healthy', 'warning'] else 1
            
        elif args.command == 'stats':
            result = admin.system_stats(performance=args.performance)
            return 0
            
        elif args.command == 'automation':
            result = admin.manage_automation(action=args.action)
            return 0 if result['success'] else 1
            
        elif args.command == 'health-monitor':
            result = admin.manage_health_monitoring(
                action=args.action,
                detailed=getattr(args, 'detailed', False)
            )
            return 0 if result.get('success', False) else 1
            
        elif args.command == 'performance':
            result = admin.manage_performance_monitoring(
                action=args.action,
                operation=getattr(args, 'operation', None),
                format=getattr(args, 'format', 'summary')
            )
            return 0 if result.get('success', False) else 1
            
        elif args.command == 'error-recovery':
            result = admin.manage_error_recovery(
                action=args.action,
                circuit_breaker_name=getattr(args, 'circuit_breaker_name', None),
                detailed=getattr(args, 'detailed', False)
            )
            return 0 if result.get('success', False) else 1
            
    except KeyboardInterrupt:
        safe_print("\n⚠️ Operation cancelled by user")
        return 130
    except Exception as e:
        safe_print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
