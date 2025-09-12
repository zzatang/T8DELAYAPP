#!/usr/bin/env python3
"""
Automatic Calendar Generation and Maintenance System
Handles automatic year detection, calendar regeneration triggers, and data validation.
"""

import asyncio
import logging
import os
import subprocess
import sys
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import threading
import time
from dataclasses import dataclass
from enum import Enum

# Import our modules
from database.operations import get_calendar_operations
from school_day_lookup import SchoolDayLookup

# Configure logging
logger = logging.getLogger(__name__)


class TriggerReason(Enum):
    """Reasons for triggering calendar generation."""
    YEAR_ROLLOVER = "year_rollover"
    MISSING_DATA = "missing_data"
    DATA_VALIDATION_FAILED = "data_validation_failed"
    MANUAL_TRIGGER = "manual_trigger"
    SCHEDULED_MAINTENANCE = "scheduled_maintenance"
    SYSTEM_STARTUP = "system_startup"
    FUTURE_YEAR_PREPARATION = "future_year_preparation"


class GenerationPriority(Enum):
    """Priority levels for calendar generation tasks."""
    CRITICAL = "critical"    # Current year missing
    HIGH = "high"           # Next year missing near year end
    MEDIUM = "medium"       # Future year preparation
    LOW = "low"             # Maintenance/optimization


@dataclass
class GenerationTask:
    """Represents a calendar generation task."""
    year: int
    trigger_reason: TriggerReason
    priority: GenerationPriority
    created_at: datetime
    scheduled_for: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    
    @property
    def is_overdue(self) -> bool:
        """Check if the task is overdue for execution."""
        if not self.scheduled_for:
            return True  # Immediate execution
        return datetime.now() >= self.scheduled_for
    
    @property
    def can_retry(self) -> bool:
        """Check if the task can be retried."""
        return self.retry_count < self.max_retries


class CalendarAutomationSystem:
    """
    Automatic calendar generation and maintenance system.
    Handles year rollover detection, missing data detection, and automatic regeneration.
    """
    
    def __init__(self, 
                 school_day_lookup: Optional[SchoolDayLookup] = None,
                 enable_background_monitoring: bool = True,
                 check_interval_hours: int = 6):
        """
        Initialize the calendar automation system.
        
        Args:
            school_day_lookup: SchoolDayLookup instance for data validation
            enable_background_monitoring: Whether to run background monitoring
            check_interval_hours: Hours between automatic checks
        """
        self.school_day_lookup = school_day_lookup
        self.operations = get_calendar_operations()
        
        # Configuration
        self.enable_background_monitoring = enable_background_monitoring
        self.check_interval_hours = check_interval_hours
        self.check_interval_seconds = check_interval_hours * 3600
        
        # Task management
        self.pending_tasks: List[GenerationTask] = []
        self.completed_tasks: List[GenerationTask] = []
        self.failed_tasks: List[GenerationTask] = []
        self.task_lock = threading.RLock()
        
        # Background monitoring
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'years_generated': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'last_check': None,
            'last_generation': None
        }
        
        # Initialize health monitoring integration
        self.health_monitor = None
        self._initialize_health_monitoring()
        
        logger.info("🤖 Calendar Automation System initialized")
        logger.info(f"   Background monitoring: {'enabled' if enable_background_monitoring else 'disabled'}")
        logger.info(f"   Check interval: {check_interval_hours} hours")
    
    def _initialize_health_monitoring(self):
        """Initialize health monitoring integration."""
        try:
            # Lazy import to avoid circular dependencies
            from calendar_health_monitor import get_health_monitor
            
            self.health_monitor = get_health_monitor(
                school_day_lookup=self.school_day_lookup,
                operations=self.operations,
                check_interval_minutes=self.check_interval_hours * 60
            )
            logger.debug("✅ Health monitoring integration initialized")
            
        except ImportError:
            logger.debug("📋 Health monitoring not available (optional feature)")
            self.health_monitor = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize health monitoring: {e}")
            self.health_monitor = None
    
    def start_background_monitoring(self) -> None:
        """Start background monitoring for automatic calendar maintenance."""
        if self._monitoring_active:
            logger.warning("Background monitoring is already active")
            return
        
        if not self.enable_background_monitoring:
            logger.info("Background monitoring is disabled")
            return
        
        logger.info("🚀 Starting background calendar monitoring...")
        
        self._monitoring_active = True
        self._shutdown_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._background_monitoring_loop,
            name="CalendarAutomationMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info("✅ Background calendar monitoring started")
    
    def stop_background_monitoring(self) -> None:
        """Stop background monitoring gracefully."""
        if not self._monitoring_active:
            return
        
        logger.info("🛑 Stopping background calendar monitoring...")
        
        self._monitoring_active = False
        self._shutdown_event.set()
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=10)
            
        logger.info("✅ Background calendar monitoring stopped")
    
    def _background_monitoring_loop(self) -> None:
        """Background monitoring loop for automatic calendar maintenance."""
        logger.info(f"🔄 Background monitoring loop started (checking every {self.check_interval_hours}h)")
        
        while self._monitoring_active and not self._shutdown_event.is_set():
            try:
                # Perform automatic checks
                self.perform_automatic_checks()
                
                # Process pending tasks
                self.process_pending_tasks()
                
                # Wait for next check or shutdown signal
                self._shutdown_event.wait(timeout=self.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"❌ Error in background monitoring loop: {e}")
                # Continue monitoring despite errors
                time.sleep(60)  # Brief pause before retrying
        
        logger.info("🏁 Background monitoring loop ended")
    
    def perform_automatic_checks(self) -> Dict[str, Any]:
        """
        Perform comprehensive automatic checks for missing data, year rollovers, and cleanup needs.
        
        Returns:
            Dictionary with check results and any triggered tasks
        """
        logger.info("🔍 Performing automatic calendar checks...")
        
        check_results = {
            'timestamp': datetime.now().isoformat(),
            'checks_performed': [],
            'issues_found': [],
            'tasks_triggered': [],
            'recommendations': [],
            'cleanup_performed': False
        }
        
        self.stats['total_checks'] += 1
        self.stats['last_check'] = datetime.now()
        
        try:
            # Check 1: Current year data availability
            current_year = datetime.now().year
            current_year_check = self._check_year_data_availability(current_year)
            check_results['checks_performed'].append(f"current_year_{current_year}")
            
            if not current_year_check['is_complete']:
                logger.warning(f"⚠️ Current year {current_year} data is incomplete!")
                check_results['issues_found'].append(f"Current year {current_year} missing data")
                
                # Trigger high-priority generation for current year
                task = self._create_generation_task(
                    year=current_year,
                    trigger_reason=TriggerReason.MISSING_DATA,
                    priority=GenerationPriority.CRITICAL
                )
                check_results['tasks_triggered'].append(f"critical_generation_{current_year}")
            
            # Check 2: Next year data availability (especially important in Q4)
            next_year = current_year + 1
            current_month = datetime.now().month
            
            if current_month >= 10:  # October onwards - prepare for next year
                next_year_check = self._check_year_data_availability(next_year)
                check_results['checks_performed'].append(f"next_year_{next_year}")
                
                if not next_year_check['is_complete']:
                    logger.info(f"📅 Next year {next_year} data not available - scheduling generation")
                    check_results['issues_found'].append(f"Next year {next_year} missing data")
                    
                    # Trigger high-priority generation for next year
                    task = self._create_generation_task(
                        year=next_year,
                        trigger_reason=TriggerReason.FUTURE_YEAR_PREPARATION,
                        priority=GenerationPriority.HIGH
                    )
                    check_results['tasks_triggered'].append(f"high_priority_generation_{next_year}")
            
            # Check 3: Data validation for existing years
            validation_results = self._validate_existing_data()
            check_results['checks_performed'].append("data_validation")
            
            for year, validation in validation_results.items():
                if not validation['is_valid']:
                    logger.warning(f"⚠️ Data validation failed for year {year}")
                    check_results['issues_found'].append(f"Validation failed for year {year}: {validation['issues']}")
                    
                    # Trigger regeneration for failed validation
                    task = self._create_generation_task(
                        year=year,
                        trigger_reason=TriggerReason.DATA_VALIDATION_FAILED,
                        priority=GenerationPriority.HIGH
                    )
                    check_results['tasks_triggered'].append(f"validation_fix_generation_{year}")
            
            # Check 4: Year rollover detection
            rollover_check = self._check_year_rollover()
            check_results['checks_performed'].append("year_rollover_check")
            
            if rollover_check['rollover_detected']:
                logger.info(f"🗓️ Year rollover detected: {rollover_check['details']}")
                check_results['issues_found'].append(f"Year rollover: {rollover_check['details']}")
                
                # Handle year rollover
                self._handle_year_rollover(rollover_check)
                check_results['tasks_triggered'].append("year_rollover_handling")
            
            # Check 4: Perform automatic cleanup if needed
            cleanup_result = self._perform_automatic_cleanup()
            check_results['checks_performed'].append("automatic_cleanup")
            if cleanup_result['performed']:
                check_results['cleanup_performed'] = True
                check_results['issues_found'].append(f"Cleaned up {cleanup_result['records_deleted']} old records")
                logger.info(f"🧹 Automatic cleanup completed: {cleanup_result['records_deleted']} records removed")
            
            # Check 5: Perform health monitoring if available
            if self.health_monitor:
                try:
                    logger.debug("🏥 Performing integrated health check...")
                    health_report = self.health_monitor.perform_health_check()
                    check_results['checks_performed'].append("health_monitoring")
                    
                    # Add critical health issues to automation issues
                    critical_health_issues = [
                        issue for issue in health_report.get('validation_issues', [])
                        if issue.get('severity') == 'critical'
                    ]
                    
                    if critical_health_issues:
                        for issue in critical_health_issues:
                            check_results['issues_found'].append(f"Health: {issue['message']}")
                        
                        logger.warning(f"🚨 Health monitoring found {len(critical_health_issues)} critical issues")
                    
                    # Store health status for reporting
                    check_results['health_status'] = health_report.get('overall_health', 'unknown')
                    
                except Exception as e:
                    logger.warning(f"⚠️ Health monitoring check failed: {e}")
                    check_results['issues_found'].append(f"Health monitoring error: {str(e)}")
            
            # Generate recommendations
            check_results['recommendations'] = self._generate_recommendations(check_results)
            
            # Log summary
            issues_count = len(check_results['issues_found'])
            tasks_count = len(check_results['tasks_triggered'])
            
            if issues_count > 0:
                logger.warning(f"📊 Automatic check completed: {issues_count} issues found, {tasks_count} tasks triggered")
            else:
                logger.info(f"✅ Automatic check completed: No issues found")
            
            return check_results
            
        except Exception as e:
            logger.error(f"❌ Automatic checks failed: {e}")
            check_results['error'] = str(e)
            return check_results
    
    def _perform_automatic_cleanup(self, retention_years: int = 3, min_school_days: int = 50) -> Dict[str, Any]:
        """
        Perform automatic cleanup of old and invalid calendar data.
        
        Args:
            retention_years: Years to keep (past and future from current year)
            min_school_days: Minimum school days for valid data
            
        Returns:
            Dictionary with cleanup results
        """
        cleanup_result = {
            'performed': False,
            'records_deleted': 0,
            'years_deleted': [],
            'error': None
        }
        
        try:
            # Only perform cleanup if we have database operations available
            if not hasattr(self, '_operations') or self._operations is None:
                # Try to get operations from school_day_lookup
                if self.school_day_lookup and hasattr(self.school_day_lookup, 'operations'):
                    self._operations = self.school_day_lookup.operations
                else:
                    # Lazy import to avoid circular dependencies
                    from database.operations import get_calendar_operations
                    self._operations = get_calendar_operations()
            
            # Check if cleanup is needed
            candidates = self._operations.get_cleanup_candidates(
                retention_years=retention_years,
                min_school_days=min_school_days
            )
            
            total_to_delete = candidates['summary']['total_records_to_delete']
            
            if total_to_delete == 0:
                logger.debug("✅ No cleanup needed - all data is within retention policy")
                return cleanup_result
            
            # Perform cleanup (retention cleanup only for automatic runs)
            logger.info(f"🧹 Performing automatic cleanup: {total_to_delete} records beyond retention policy")
            
            retention_result = self._operations.cleanup_old_calendar_data(retention_years=retention_years)
            
            if retention_result['success']:
                cleanup_result['performed'] = True
                cleanup_result['records_deleted'] = retention_result['total_records_deleted']
                cleanup_result['years_deleted'] = [year_info['year'] for year_info in retention_result['years_deleted']]
                
                logger.info(f"✅ Automatic cleanup completed: {cleanup_result['records_deleted']} records deleted from {len(cleanup_result['years_deleted'])} years")
                
                # Refresh cache after cleanup
                if self.school_day_lookup:
                    try:
                        if hasattr(self.school_day_lookup, 'cache') and hasattr(self.school_day_lookup.cache, 'clear'):
                            self.school_day_lookup.cache.clear()
                            logger.info("🔄 Cache cleared after automatic cleanup")
                        elif hasattr(self.school_day_lookup, 'clear_cache'):
                            self.school_day_lookup.clear_cache()
                            logger.info("🔄 Cache cleared after automatic cleanup")
                        else:
                            logger.debug("ℹ️ Cache clearing not available - will refresh on next access")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to clear cache after cleanup: {e}")
            else:
                logger.warning("⚠️ Automatic cleanup failed")
                cleanup_result['error'] = retention_result.get('error', 'Unknown error')
                
        except Exception as e:
            logger.error(f"❌ Automatic cleanup failed: {e}")
            cleanup_result['error'] = str(e)
        
        return cleanup_result
    
    def _check_year_data_availability(self, year: int) -> Dict[str, Any]:
        """
        Check if complete calendar data is available for a specific year.
        
        Args:
            year: Year to check
            
        Returns:
            Dictionary with availability status and details
        """
        try:
            # Get calendar statistics for the year
            stats = self.operations.get_calendar_stats(year)
            
            # Expected number of days in the year
            expected_days = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
            
            actual_days = stats.get('total_days', 0)
            is_complete = actual_days == expected_days
            
            return {
                'year': year,
                'is_complete': is_complete,
                'expected_days': expected_days,
                'actual_days': actual_days,
                'completion_percentage': (actual_days / expected_days * 100) if expected_days > 0 else 0,
                'first_date': stats.get('first_date'),
                'last_date': stats.get('last_date'),
                'school_days': stats.get('school_days', 0),
                'non_school_days': stats.get('non_school_days', 0)
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to check year {year} data availability: {e}")
            return {
                'year': year,
                'is_complete': False,
                'error': str(e)
            }
    
    def _validate_existing_data(self) -> Dict[int, Dict[str, Any]]:
        """
        Validate existing calendar data for integrity and completeness.
        
        Returns:
            Dictionary mapping years to their validation results
        """
        validation_results = {}
        
        try:
            # Get list of years with data
            current_year = datetime.now().year
            years_to_validate = [current_year - 1, current_year, current_year + 1]
            
            for year in years_to_validate:
                try:
                    validation = self.operations.validate_calendar_data(year)
                    validation_results[year] = validation
                    
                    if not validation['is_valid']:
                        logger.warning(f"⚠️ Validation failed for year {year}: {validation['issues']}")
                    else:
                        logger.debug(f"✅ Validation passed for year {year}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not validate year {year}: {e}")
                    validation_results[year] = {
                        'is_valid': False,
                        'error': str(e)
                    }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Data validation failed: {e}")
            return {}
    
    def _check_year_rollover(self) -> Dict[str, Any]:
        """
        Check if a year rollover has occurred that requires attention.
        
        Returns:
            Dictionary with rollover detection results
        """
        try:
            current_year = datetime.now().year
            current_date = date.today()
            
            # Check if we're in a new year but don't have current year data
            year_availability = self._check_year_data_availability(current_year)
            
            rollover_detected = False
            details = ""
            
            # Scenario 1: New year started but no data available
            if current_date.month == 1 and not year_availability['is_complete']:
                rollover_detected = True
                details = f"New year {current_year} started but calendar data missing"
            
            # Scenario 2: Near end of year but next year not prepared
            elif current_date.month >= 11:  # November/December
                next_year = current_year + 1
                next_year_availability = self._check_year_data_availability(next_year)
                
                if not next_year_availability['is_complete']:
                    rollover_detected = True
                    details = f"Approaching {next_year}, but calendar data not prepared"
            
            return {
                'rollover_detected': rollover_detected,
                'current_year': current_year,
                'current_date': current_date.isoformat(),
                'details': details,
                'current_year_complete': year_availability['is_complete'],
                'recommendations': self._get_rollover_recommendations(current_year, rollover_detected)
            }
            
        except Exception as e:
            logger.error(f"❌ Year rollover check failed: {e}")
            return {
                'rollover_detected': False,
                'error': str(e)
            }
    
    def _handle_year_rollover(self, rollover_info: Dict[str, Any]) -> None:
        """
        Handle year rollover by triggering appropriate calendar generation tasks.
        
        Args:
            rollover_info: Information about the detected rollover
        """
        current_year = rollover_info['current_year']
        
        logger.info(f"🗓️ Handling year rollover for {current_year}")
        
        # Trigger current year generation if missing
        if not rollover_info.get('current_year_complete', False):
            self._create_generation_task(
                year=current_year,
                trigger_reason=TriggerReason.YEAR_ROLLOVER,
                priority=GenerationPriority.CRITICAL
            )
        
        # Trigger next year generation if we're in Q4
        if datetime.now().month >= 10:
            next_year = current_year + 1
            next_year_availability = self._check_year_data_availability(next_year)
            
            if not next_year_availability['is_complete']:
                self._create_generation_task(
                    year=next_year,
                    trigger_reason=TriggerReason.FUTURE_YEAR_PREPARATION,
                    priority=GenerationPriority.HIGH
                )
        
        # Notify school day lookup system about rollover
        if self.school_day_lookup:
            try:
                self.school_day_lookup.schedule_year_rollover_check()
                logger.info("✅ Notified SchoolDayLookup system about year rollover")
            except Exception as e:
                logger.warning(f"⚠️ Failed to notify SchoolDayLookup about rollover: {e}")
    
    def _create_generation_task(self, 
                              year: int, 
                              trigger_reason: TriggerReason, 
                              priority: GenerationPriority,
                              scheduled_for: Optional[datetime] = None) -> GenerationTask:
        """
        Create a new calendar generation task.
        
        Args:
            year: Year to generate calendar for
            trigger_reason: Reason for triggering generation
            priority: Priority level of the task
            scheduled_for: When to execute the task (None for immediate)
            
        Returns:
            Created GenerationTask
        """
        task = GenerationTask(
            year=year,
            trigger_reason=trigger_reason,
            priority=priority,
            created_at=datetime.now(),
            scheduled_for=scheduled_for
        )
        
        with self.task_lock:
            # Check if similar task already exists
            existing_task = self._find_existing_task(year, trigger_reason)
            if existing_task:
                logger.info(f"📋 Task for year {year} ({trigger_reason.value}) already exists")
                return existing_task
            
            self.pending_tasks.append(task)
            logger.info(f"📋 Created {priority.value} priority task: Generate calendar for {year} ({trigger_reason.value})")
        
        return task
    
    def _find_existing_task(self, year: int, trigger_reason: TriggerReason) -> Optional[GenerationTask]:
        """Find existing task for the same year and trigger reason."""
        for task in self.pending_tasks:
            if task.year == year and task.trigger_reason == trigger_reason:
                return task
        return None
    
    def process_pending_tasks(self) -> Dict[str, Any]:
        """
        Process all pending calendar generation tasks.
        
        Returns:
            Dictionary with processing results
        """
        with self.task_lock:
            if not self.pending_tasks:
                logger.debug("📋 No pending calendar generation tasks")
                return {'processed': 0, 'completed': 0, 'failed': 0}
        
        logger.info(f"📋 Processing {len(self.pending_tasks)} pending calendar generation tasks...")
        
        processed = 0
        completed = 0
        failed = 0
        
        # Sort tasks by priority and creation time
        sorted_tasks = sorted(
            self.pending_tasks,
            key=lambda t: (
                ['critical', 'high', 'medium', 'low'].index(t.priority.value),
                t.created_at
            )
        )
        
        with self.task_lock:
            tasks_to_process = [task for task in sorted_tasks if task.is_overdue]
        
        for task in tasks_to_process:
            try:
                logger.info(f"⚙️ Processing {task.priority.value} priority task: Generate calendar for {task.year}")
                
                # Execute the calendar generation
                success = self._execute_generation_task(task)
                
                with self.task_lock:
                    self.pending_tasks.remove(task)
                    
                    if success:
                        self.completed_tasks.append(task)
                        completed += 1
                        logger.info(f"✅ Completed calendar generation for {task.year}")
                    else:
                        if task.can_retry:
                            task.retry_count += 1
                            task.scheduled_for = datetime.now() + timedelta(hours=1)  # Retry in 1 hour
                            self.pending_tasks.append(task)
                            logger.warning(f"⚠️ Task failed, scheduled for retry {task.retry_count}/{task.max_retries}")
                        else:
                            self.failed_tasks.append(task)
                            failed += 1
                            logger.error(f"❌ Task failed permanently for {task.year}")
                
                processed += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing task for year {task.year}: {e}")
                
                with self.task_lock:
                    task.error_message = str(e)
                    if task.can_retry:
                        task.retry_count += 1
                        task.scheduled_for = datetime.now() + timedelta(hours=1)
                        logger.warning(f"⚠️ Task error, scheduled for retry {task.retry_count}/{task.max_retries}")
                    else:
                        self.pending_tasks.remove(task)
                        self.failed_tasks.append(task)
                        failed += 1
                        logger.error(f"❌ Task failed permanently due to error: {e}")
                
                processed += 1
        
        # Update statistics
        self.stats['tasks_completed'] += completed
        self.stats['tasks_failed'] += failed
        
        logger.info(f"📊 Task processing completed: {processed} processed, {completed} completed, {failed} failed")
        
        return {
            'processed': processed,
            'completed': completed,
            'failed': failed,
            'remaining_pending': len(self.pending_tasks)
        }
    
    def _execute_generation_task(self, task: GenerationTask) -> bool:
        """
        Execute a calendar generation task by running the generator script.
        
        Args:
            task: GenerationTask to execute
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Path to the calendar generator script
            generator_script = Path(__file__).parent / "school_calendar_generator.py"
            
            if not generator_script.exists():
                logger.error(f"❌ Calendar generator script not found: {generator_script}")
                return False
            
            # Build command
            cmd = [
                sys.executable,
                str(generator_script),
                str(task.year),
                "--batch-size", "1000"
            ]
            
            logger.info(f"🚀 Executing calendar generation: {' '.join(cmd)}")
            
            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Calendar generation successful for {task.year}")
                logger.debug(f"Generator output: {result.stdout}")
                
                # Update statistics
                self.stats['years_generated'] += 1
                self.stats['last_generation'] = datetime.now()
                
                # Notify SchoolDayLookup to refresh cache
                if self.school_day_lookup:
                    try:
                        self.school_day_lookup.preload_additional_years([task.year])
                        logger.info(f"✅ Refreshed SchoolDayLookup cache for {task.year}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to refresh cache for {task.year}: {e}")
                
                return True
            else:
                logger.error(f"❌ Calendar generation failed for {task.year}")
                logger.error(f"Error output: {result.stderr}")
                task.error_message = result.stderr
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Calendar generation timed out for {task.year}")
            task.error_message = "Generation timed out"
            return False
        except Exception as e:
            logger.error(f"❌ Failed to execute calendar generation for {task.year}: {e}")
            task.error_message = str(e)
            return False
    
    def _generate_recommendations(self, check_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on check results."""
        recommendations = []
        
        issues_count = len(check_results.get('issues_found', []))
        
        if issues_count == 0:
            recommendations.append("✅ All calendar data is up to date and valid")
        else:
            recommendations.append(f"⚠️ {issues_count} issues detected requiring attention")
        
        # Specific recommendations based on issues
        for issue in check_results.get('issues_found', []):
            if "Current year" in issue and "missing data" in issue:
                recommendations.append("🚨 CRITICAL: Generate current year calendar data immediately")
            elif "Next year" in issue and "missing data" in issue:
                recommendations.append("📅 HIGH: Prepare next year calendar data before year-end")
            elif "Validation failed" in issue:
                recommendations.append("🔧 Regenerate calendar data to fix validation issues")
        
        # General maintenance recommendations
        current_month = datetime.now().month
        if current_month >= 10:
            recommendations.append("📋 Consider preparing calendar data for next year")
        
        return recommendations
    
    def _get_rollover_recommendations(self, current_year: int, rollover_detected: bool) -> List[str]:
        """Get recommendations for year rollover scenarios."""
        recommendations = []
        
        if rollover_detected:
            recommendations.append(f"Generate calendar data for {current_year} immediately")
            
            if datetime.now().month >= 10:
                recommendations.append(f"Prepare calendar data for {current_year + 1}")
        
        recommendations.append("Monitor cache performance after rollover")
        recommendations.append("Validate data integrity after generation")
        
        return recommendations
    
    def trigger_manual_generation(self, year: int, priority: GenerationPriority = GenerationPriority.HIGH) -> GenerationTask:
        """
        Manually trigger calendar generation for a specific year.
        
        Args:
            year: Year to generate calendar for
            priority: Priority level for the task
            
        Returns:
            Created GenerationTask
        """
        logger.info(f"📋 Manual trigger: Calendar generation for {year}")
        
        task = self._create_generation_task(
            year=year,
            trigger_reason=TriggerReason.MANUAL_TRIGGER,
            priority=priority
        )
        
        return task
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the automation system.
        
        Returns:
            Dictionary with system status information
        """
        with self.task_lock:
            pending_count = len(self.pending_tasks)
            completed_count = len(self.completed_tasks)
            failed_count = len(self.failed_tasks)
        
        return {
            'monitoring_active': self._monitoring_active,
            'check_interval_hours': self.check_interval_hours,
            'tasks': {
                'pending': pending_count,
                'completed': completed_count,
                'failed': failed_count
            },
            'statistics': self.stats.copy(),
            'last_check': self.stats['last_check'].isoformat() if self.stats['last_check'] else None,
            'last_generation': self.stats['last_generation'].isoformat() if self.stats['last_generation'] else None
        }
    
    def cleanup_old_tasks(self, days_to_keep: int = 30) -> int:
        """
        Clean up old completed and failed tasks.
        
        Args:
            days_to_keep: Number of days of task history to keep
            
        Returns:
            Number of tasks cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned_up = 0
        
        with self.task_lock:
            # Clean up old completed tasks
            self.completed_tasks = [
                task for task in self.completed_tasks
                if task.created_at > cutoff_date
            ]
            
            # Clean up old failed tasks
            old_failed_count = len(self.failed_tasks)
            self.failed_tasks = [
                task for task in self.failed_tasks
                if task.created_at > cutoff_date
            ]
            
            cleaned_up = old_failed_count - len(self.failed_tasks)
        
        if cleaned_up > 0:
            logger.info(f"🧹 Cleaned up {cleaned_up} old task records")
        
        return cleaned_up


# Global automation system instance
_automation_system: Optional[CalendarAutomationSystem] = None


def get_automation_system(school_day_lookup: Optional[SchoolDayLookup] = None) -> CalendarAutomationSystem:
    """
    Get or create the global calendar automation system instance.
    
    Args:
        school_day_lookup: SchoolDayLookup instance for integration
        
    Returns:
        CalendarAutomationSystem instance
    """
    global _automation_system
    
    if _automation_system is None:
        _automation_system = CalendarAutomationSystem(school_day_lookup=school_day_lookup)
    
    return _automation_system


def main():
    """Command-line interface for calendar automation system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calendar Automation System')
    parser.add_argument('--check', action='store_true', help='Perform automatic checks')
    parser.add_argument('--generate', type=int, help='Manually trigger generation for specific year')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--start-monitoring', action='store_true', help='Start background monitoring')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        automation = get_automation_system()
        
        if args.check:
            logger.info("🔍 Performing manual automatic checks...")
            results = automation.perform_automatic_checks()
            print(f"✅ Check completed: {len(results['issues_found'])} issues found")
            return 0
        
        if args.generate:
            logger.info(f"📋 Manually triggering generation for {args.generate}...")
            task = automation.trigger_manual_generation(args.generate)
            
            # Process the task immediately
            processing_results = automation.process_pending_tasks()
            if processing_results['completed'] > 0:
                print(f"✅ Calendar generation completed for {args.generate}")
            else:
                print(f"❌ Calendar generation failed for {args.generate}")
            return 0
        
        if args.status:
            status = automation.get_system_status()
            print("📊 Calendar Automation System Status:")
            print(f"   Monitoring: {'Active' if status['monitoring_active'] else 'Inactive'}")
            print(f"   Check interval: {status['check_interval_hours']} hours")
            print(f"   Tasks - Pending: {status['tasks']['pending']}, Completed: {status['tasks']['completed']}, Failed: {status['tasks']['failed']}")
            print(f"   Statistics: {status['statistics']['total_checks']} checks, {status['statistics']['years_generated']} years generated")
            return 0
        
        if args.start_monitoring:
            logger.info("🚀 Starting background monitoring...")
            automation.start_background_monitoring()
            
            try:
                # Keep the script running
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                logger.info("🛑 Stopping background monitoring...")
                automation.stop_background_monitoring()
            
            return 0
        
        # Default: show help
        parser.print_help()
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
