#!/usr/bin/env python3
"""
Calendar Data Validation and Health Monitoring System

This module provides comprehensive health monitoring and validation for the school calendar system,
including data integrity checks, performance monitoring, alerting, and health dashboards.
"""

import logging
import time
import threading
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from collections import defaultdict, deque

# Configure logging
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels for different components."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in calendar data."""
    severity: ValidationSeverity
    category: str
    message: str
    year: Optional[int] = None
    date: Optional[str] = None
    count: Optional[int] = None
    recommendation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HealthMetric:
    """Represents a health metric with historical data."""
    name: str
    current_value: Union[float, int, str]
    unit: str
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """Performance metrics for calendar operations."""
    total_queries: int = 0
    avg_response_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    database_connections: int = 0
    errors_per_hour: float = 0.0
    last_error: Optional[str] = None
    uptime_hours: float = 0.0


class CalendarHealthMonitor:
    """
    Comprehensive health monitoring system for calendar data and operations.
    
    Features:
    - Data integrity validation
    - Performance monitoring
    - Health metrics collection
    - Alerting and notifications
    - Health dashboard data
    - Automated remediation suggestions
    """
    
    def __init__(self, school_day_lookup=None, operations=None, check_interval_minutes: float = 15.0):
        """
        Initialize the health monitoring system.
        
        Args:
            school_day_lookup: SchoolDayLookup instance for cache monitoring
            operations: SchoolCalendarOperations instance for database operations
            check_interval_minutes: How often to perform health checks
        """
        self.school_day_lookup = school_day_lookup
        self.operations = operations
        self.check_interval_seconds = check_interval_minutes * 60
        
        # Health monitoring state
        self.metrics: Dict[str, HealthMetric] = {}
        self.validation_issues: List[ValidationIssue] = []
        self.performance_metrics = PerformanceMetrics()
        self.overall_health = HealthStatus.UNKNOWN
        
        # Monitoring control
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._health_lock = threading.RLock()
        
        # Historical data
        self.validation_history: deque = deque(maxlen=1000)
        self.health_history: deque = deque(maxlen=500)
        
        # Configuration
        self.config = {
            'data_retention_days': 30,
            'max_response_time_ms': 100.0,
            'min_cache_hit_rate': 90.0,
            'max_error_rate_per_hour': 10.0,
            'critical_missing_data_threshold': 0.95,  # 95% completeness required
            'enable_auto_remediation': True,
            'alert_thresholds': {
                'response_time_warning_ms': 50.0,
                'response_time_critical_ms': 200.0,
                'cache_hit_rate_warning': 80.0,
                'cache_hit_rate_critical': 60.0,
                'error_rate_warning_per_hour': 5.0,
                'error_rate_critical_per_hour': 20.0
            }
        }
        
        self.start_time = datetime.now()
        logger.info("🏥 Calendar Health Monitor initialized")
        logger.info(f"   Check interval: {check_interval_minutes} minutes")
        logger.info(f"   Auto-remediation: {self.config['enable_auto_remediation']}")
    
    def start_monitoring(self):
        """Start continuous health monitoring in background thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("⚠️ Health monitoring already running")
            return
            
        logger.info("🚀 Starting calendar health monitoring...")
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="CalendarHealthMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("✅ Calendar health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        if not self._monitoring_thread or not self._monitoring_thread.is_alive():
            return
            
        logger.info("🛑 Stopping calendar health monitoring...")
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=10)
        
        if self._monitoring_thread.is_alive():
            logger.warning("⚠️ Health monitoring thread did not stop gracefully")
        else:
            logger.info("✅ Calendar health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in background thread."""
        logger.info("🔄 Health monitoring loop started")
        
        while not self._stop_monitoring.is_set():
            try:
                # Perform comprehensive health check
                health_report = self.perform_health_check()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Check for alerts
                alerts = self._check_alert_conditions()
                if alerts:
                    self._process_alerts(alerts)
                
                # Auto-remediation if enabled
                if self.config['enable_auto_remediation']:
                    self._attempt_auto_remediation()
                
                # Store health history
                with self._health_lock:
                    self.health_history.append({
                        'timestamp': datetime.now(),
                        'overall_health': self.overall_health.value,
                        'issues_count': len(self.validation_issues),
                        'performance': {
                            'avg_response_time': self.performance_metrics.avg_response_time_ms,
                            'cache_hit_rate': self.performance_metrics.cache_hit_rate,
                            'error_rate': self.performance_metrics.errors_per_hour
                        }
                    })
                
                logger.debug(f"🏥 Health check completed: {self.overall_health.value}")
                
            except Exception as e:
                logger.error(f"❌ Error in health monitoring loop: {e}")
                
            # Wait for next check or stop signal
            self._stop_monitoring.wait(self.check_interval_seconds)
        
        logger.info("🏁 Health monitoring loop ended")
    
    def perform_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of the calendar system.
        
        Returns:
            Dictionary with health check results
        """
        logger.debug("🔍 Performing comprehensive health check...")
        
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': HealthStatus.UNKNOWN.value,
            'components': {},
            'validation_issues': [],
            'performance_metrics': {},
            'recommendations': []
        }
        
        try:
            with self._health_lock:
                # Clear previous validation issues
                self.validation_issues.clear()
                
                # 1. Database Health Check
                db_health = self._check_database_health()
                health_report['components']['database'] = db_health
                
                # 2. Data Integrity Validation
                data_health = self._check_data_integrity()
                health_report['components']['data_integrity'] = data_health
                
                # 3. Cache Health Check
                cache_health = self._check_cache_health()
                health_report['components']['cache'] = cache_health
                
                # 4. Performance Check
                perf_health = self._check_performance_health()
                health_report['components']['performance'] = perf_health
                
                # 5. System Resources Check
                resource_health = self._check_system_resources()
                health_report['components']['system_resources'] = resource_health
                
                # Determine overall health status
                component_statuses = [comp['status'] for comp in health_report['components'].values()]
                self.overall_health = self._calculate_overall_health(component_statuses)
                health_report['overall_health'] = self.overall_health.value
                
                # Add validation issues
                health_report['validation_issues'] = [
                    {
                        'severity': issue.severity.value,
                        'category': issue.category,
                        'message': issue.message,
                        'year': issue.year,
                        'date': issue.date,
                        'count': issue.count,
                        'recommendation': issue.recommendation
                    }
                    for issue in self.validation_issues
                ]
                
                # Add performance metrics
                health_report['performance_metrics'] = {
                    'avg_response_time_ms': self.performance_metrics.avg_response_time_ms,
                    'cache_hit_rate': self.performance_metrics.cache_hit_rate,
                    'total_queries': self.performance_metrics.total_queries,
                    'errors_per_hour': self.performance_metrics.errors_per_hour,
                    'uptime_hours': self.performance_metrics.uptime_hours
                }
                
                # Generate recommendations
                health_report['recommendations'] = self._generate_health_recommendations()
                
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            health_report['error'] = str(e)
            self.overall_health = HealthStatus.CRITICAL
            health_report['overall_health'] = HealthStatus.CRITICAL.value
        
        return health_report
    
    def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and basic operations."""
        db_health = {
            'status': HealthStatus.UNKNOWN.value,
            'response_time_ms': None,
            'connection_pool_size': None,
            'active_connections': None,
            'issues': []
        }
        
        try:
            if not self.operations:
                # Try to get operations if not provided
                from database.operations import get_calendar_operations
                self.operations = get_calendar_operations()
            
            # Test basic connectivity and measure response time
            start_time = time.perf_counter()
            
            with self.operations.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            
            response_time_ms = (time.perf_counter() - start_time) * 1000
            db_health['response_time_ms'] = response_time_ms
            
            # Check connection pool status
            pool_info = self.operations.db_manager.get_connection_info()
            db_health['connection_pool_size'] = pool_info.get('max_connections', 'unknown')
            
            # Determine database health status
            if response_time_ms > self.config['alert_thresholds']['response_time_critical_ms']:
                db_health['status'] = HealthStatus.CRITICAL.value
                db_health['issues'].append(f"High database response time: {response_time_ms:.2f}ms")
            elif response_time_ms > self.config['alert_thresholds']['response_time_warning_ms']:
                db_health['status'] = HealthStatus.WARNING.value
                db_health['issues'].append(f"Elevated database response time: {response_time_ms:.2f}ms")
            else:
                db_health['status'] = HealthStatus.HEALTHY.value
            
        except Exception as e:
            logger.error(f"❌ Database health check failed: {e}")
            db_health['status'] = HealthStatus.CRITICAL.value
            db_health['issues'].append(f"Database connectivity error: {str(e)}")
            
            # Add validation issue
            self.validation_issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="database",
                message=f"Database connectivity failed: {str(e)}",
                recommendation="Check database connection, credentials, and network connectivity"
            ))
        
        return db_health
    
    def _check_data_integrity(self) -> Dict[str, Any]:
        """Perform comprehensive data integrity validation."""
        data_health = {
            'status': HealthStatus.UNKNOWN.value,
            'years_validated': 0,
            'years_valid': 0,
            'years_invalid': 0,
            'completeness_percentage': 0.0,
            'issues': []
        }
        
        try:
            if not self.operations:
                data_health['status'] = HealthStatus.CRITICAL.value
                data_health['issues'].append("Database operations not available")
                return data_health
            
            # Only validate current year for health monitoring
            current_year = datetime.now().year
            years_to_validate = [current_year]
            
            total_years = len(years_to_validate)
            valid_years = 0
            total_completeness = 0.0
            
            for year in years_to_validate:
                try:
                    validation_result = self.operations.validate_calendar_data(year)
                    data_health['years_validated'] += 1
                    
                    if validation_result['is_valid']:
                        valid_years += 1
                        total_completeness += 100.0  # 100% complete
                    else:
                        # Calculate completeness percentage
                        stats = validation_result.get('stats', {})
                        total_days = stats.get('total_days', 0)
                        expected_days = stats.get('expected_days', 365)
                        completeness = (total_days / expected_days * 100) if expected_days > 0 else 0
                        total_completeness += completeness
                        
                        # Add validation issues
                        for issue in validation_result.get('issues', []):
                            severity = ValidationSeverity.ERROR if completeness < 50 else ValidationSeverity.WARNING
                            self.validation_issues.append(ValidationIssue(
                                severity=severity,
                                category="data_integrity",
                                message=f"Year {year}: {issue}",
                                year=year,
                                recommendation=self._get_data_fix_recommendation(issue)
                            ))
                            data_health['issues'].append(f"Year {year}: {issue}")
                
                except Exception as e:
                    logger.warning(f"⚠️ Failed to validate year {year}: {e}")
                    self.validation_issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="validation_error",
                        message=f"Failed to validate year {year}: {str(e)}",
                        year=year,
                        recommendation="Check data generation and database integrity"
                    ))
            
            data_health['years_valid'] = valid_years
            data_health['years_invalid'] = total_years - valid_years
            data_health['completeness_percentage'] = total_completeness / total_years if total_years > 0 else 0
            
            # Determine data integrity health status
            if data_health['completeness_percentage'] >= 95.0:
                data_health['status'] = HealthStatus.HEALTHY.value
            elif data_health['completeness_percentage'] >= 80.0:
                data_health['status'] = HealthStatus.WARNING.value
            else:
                data_health['status'] = HealthStatus.CRITICAL.value
                
        except Exception as e:
            logger.error(f"❌ Data integrity check failed: {e}")
            data_health['status'] = HealthStatus.CRITICAL.value
            data_health['issues'].append(f"Data integrity check error: {str(e)}")
        
        return data_health
    
    def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache performance and health."""
        cache_health = {
            'status': HealthStatus.UNKNOWN.value,
            'hit_rate_percentage': 0.0,
            'size': 0,
            'max_size': 0,
            'years_cached': [],
            'issues': []
        }
        
        try:
            if not self.school_day_lookup or not hasattr(self.school_day_lookup, 'cache'):
                cache_health['status'] = HealthStatus.WARNING.value
                cache_health['issues'].append("Cache system not available")
                return cache_health
            
            # Get cache statistics
            cache_stats = self.school_day_lookup.cache.get_stats()
            
            cache_health['hit_rate_percentage'] = cache_stats.get('hit_rate_percent', 0.0)
            cache_health['size'] = cache_stats.get('cache_size', 0)
            cache_health['max_size'] = cache_stats.get('max_size', 0)
            cache_health['years_cached'] = cache_stats.get('years_cached', [])
            
            # Update performance metrics
            self.performance_metrics.cache_hit_rate = cache_health['hit_rate_percentage']
            
            # Determine cache health status
            hit_rate = cache_health['hit_rate_percentage']
            if hit_rate >= self.config['min_cache_hit_rate']:
                cache_health['status'] = HealthStatus.HEALTHY.value
            elif hit_rate >= self.config['alert_thresholds']['cache_hit_rate_warning']:
                cache_health['status'] = HealthStatus.WARNING.value
                cache_health['issues'].append(f"Low cache hit rate: {hit_rate:.1f}%")
            else:
                cache_health['status'] = HealthStatus.CRITICAL.value
                cache_health['issues'].append(f"Very low cache hit rate: {hit_rate:.1f}%")
                
                # Add validation issue
                self.validation_issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="cache_performance",
                    message=f"Cache hit rate below threshold: {hit_rate:.1f}%",
                    recommendation="Consider cache warming or increasing cache size"
                ))
                
        except Exception as e:
            logger.error(f"❌ Cache health check failed: {e}")
            cache_health['status'] = HealthStatus.CRITICAL.value
            cache_health['issues'].append(f"Cache health check error: {str(e)}")
        
        return cache_health
    
    def _check_performance_health(self) -> Dict[str, Any]:
        """Check system performance metrics."""
        perf_health = {
            'status': HealthStatus.UNKNOWN.value,
            'avg_response_time_ms': self.performance_metrics.avg_response_time_ms,
            'total_queries': self.performance_metrics.total_queries,
            'error_rate_per_hour': self.performance_metrics.errors_per_hour,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'issues': []
        }
        
        # Update uptime
        self.performance_metrics.uptime_hours = perf_health['uptime_hours']
        
        # Get advanced performance stats and error recovery stats from SchoolDayLookup if available
        if self.school_day_lookup and hasattr(self.school_day_lookup, 'get_advanced_performance_stats'):
            try:
                advanced_stats = self.school_day_lookup.get_advanced_performance_stats()
                if advanced_stats.get('performance_monitoring') == 'enabled':
                    detailed_perf = advanced_stats['detailed_performance']
                    
                    # Update performance metrics with real data
                    if detailed_perf['total_operations'] > 0:
                        perf_health['total_queries'] = detailed_perf['total_operations']
                        perf_health['success_rate'] = detailed_perf['overall_success_rate']
                        
                        # Calculate average response time from operation types
                        total_ops = 0
                        weighted_avg_time = 0
                        for op_name, op_stats in detailed_perf['operations'].items():
                            ops_count = op_stats['total_operations']
                            avg_time = op_stats['avg_duration_ms']
                            total_ops += ops_count
                            weighted_avg_time += ops_count * avg_time
                        
                        if total_ops > 0:
                            perf_health['avg_response_time_ms'] = weighted_avg_time / total_ops
                        
                        # Add performance level distribution
                        perf_levels = detailed_perf.get('performance_levels', {})
                        perf_health['performance_levels'] = perf_levels
                        
                        # Check for slow operations
                        slow_operations = perf_levels.get('slow', 0) + perf_levels.get('critical', 0)
                        total_recent_ops = sum(perf_levels.values())
                        if total_recent_ops > 0 and slow_operations > 0:
                            slow_percentage = (slow_operations / total_recent_ops) * 100
                            if slow_percentage > 10:  # More than 10% slow operations
                                perf_health['issues'].append(f"High percentage of slow operations: {slow_percentage:.1f}%")
                
            except Exception as e:
                logger.debug(f"Could not get advanced performance stats: {e}")
        
        # Get error recovery stats for additional health insights
        if self.school_day_lookup and hasattr(self.school_day_lookup, 'get_error_recovery_stats'):
            try:
                error_recovery_stats = self.school_day_lookup.get_error_recovery_stats()
                if error_recovery_stats.get('error_recovery') == 'enabled':
                    # Add error recovery metrics to performance health
                    perf_health['error_recovery'] = {
                        'total_errors': error_recovery_stats['total_errors'],
                        'error_rate_per_hour': error_recovery_stats['error_rate_per_hour'],
                        'circuit_breakers': error_recovery_stats['circuit_breakers']
                    }
                    
                    # Check for circuit breaker issues
                    circuit_breakers = error_recovery_stats['circuit_breakers']
                    for cb_name, cb_stats in circuit_breakers.items():
                        if cb_stats['state'] == 'open':
                            perf_health['issues'].append(f"Circuit breaker '{cb_name}' is OPEN")
                        elif cb_stats['state'] == 'half_open':
                            perf_health['issues'].append(f"Circuit breaker '{cb_name}' is testing recovery")
                        
                        # Check for high failure rates
                        if cb_stats['failure_rate'] > 25.0:  # More than 25% failures
                            perf_health['issues'].append(f"High failure rate in '{cb_name}': {cb_stats['failure_rate']:.1f}%")
                    
                    # Check overall error rate
                    if error_recovery_stats['error_rate_per_hour'] > 10:  # More than 10 errors per hour
                        perf_health['issues'].append(f"High error rate: {error_recovery_stats['error_rate_per_hour']} errors/hour")
                
            except Exception as e:
                logger.debug(f"Could not get error recovery stats: {e}")
        
        # Determine performance health status
        response_time = perf_health['avg_response_time_ms']
        error_rate = perf_health['error_rate_per_hour']
        
        issues = []
        status = HealthStatus.HEALTHY
        
        # Check response time
        if response_time > self.config['alert_thresholds']['response_time_critical_ms']:
            status = HealthStatus.CRITICAL
            issues.append(f"Critical response time: {response_time:.2f}ms")
        elif response_time > self.config['alert_thresholds']['response_time_warning_ms']:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
            issues.append(f"High response time: {response_time:.2f}ms")
        
        # Check error rate
        if error_rate > self.config['alert_thresholds']['error_rate_critical_per_hour']:
            status = HealthStatus.CRITICAL
            issues.append(f"Critical error rate: {error_rate:.1f}/hour")
        elif error_rate > self.config['alert_thresholds']['error_rate_warning_per_hour']:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
            issues.append(f"High error rate: {error_rate:.1f}/hour")
        
        # Add any existing issues from advanced performance monitoring
        if 'issues' in perf_health and perf_health['issues']:
            issues.extend(perf_health['issues'])
        
        perf_health['status'] = status.value
        perf_health['issues'] = issues
        
        return perf_health
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization."""
        resource_health = {
            'status': HealthStatus.HEALTHY.value,
            'memory_usage_mb': 0,
            'thread_count': 0,
            'issues': []
        }
        
        try:
            import psutil
            import os
            
            # Get current process info
            process = psutil.Process(os.getpid())
            
            # Memory usage
            memory_info = process.memory_info()
            resource_health['memory_usage_mb'] = memory_info.rss / 1024 / 1024
            
            # Thread count
            resource_health['thread_count'] = process.num_threads()
            
            # Check for resource issues (basic thresholds)
            if resource_health['memory_usage_mb'] > 1000:  # 1GB
                resource_health['status'] = HealthStatus.WARNING.value
                resource_health['issues'].append(f"High memory usage: {resource_health['memory_usage_mb']:.1f}MB")
            
            if resource_health['thread_count'] > 50:
                resource_health['status'] = HealthStatus.WARNING.value
                resource_health['issues'].append(f"High thread count: {resource_health['thread_count']}")
                
        except ImportError:
            resource_health['issues'].append("psutil not available for resource monitoring")
        except Exception as e:
            logger.warning(f"⚠️ Resource check failed: {e}")
            resource_health['issues'].append(f"Resource check error: {str(e)}")
        
        return resource_health
    
    def _calculate_overall_health(self, component_statuses: List[str]) -> HealthStatus:
        """Calculate overall health status from component statuses."""
        if HealthStatus.CRITICAL.value in component_statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING.value in component_statuses:
            return HealthStatus.WARNING
        elif HealthStatus.HEALTHY.value in component_statuses:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def _update_performance_metrics(self):
        """Update performance metrics from various sources."""
        # This would be called by the monitoring loop to update metrics
        # In a real implementation, this would collect metrics from various sources
        pass
    
    def _check_alert_conditions(self) -> List[Dict[str, Any]]:
        """Check for conditions that should trigger alerts."""
        alerts = []
        
        # Check for critical validation issues
        critical_issues = [issue for issue in self.validation_issues if issue.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            alerts.append({
                'type': 'critical_validation_issues',
                'count': len(critical_issues),
                'message': f"{len(critical_issues)} critical validation issues found"
            })
        
        # Check overall health
        if self.overall_health == HealthStatus.CRITICAL:
            alerts.append({
                'type': 'system_critical',
                'message': "Calendar system health is critical"
            })
        
        return alerts
    
    def _process_alerts(self, alerts: List[Dict[str, Any]]):
        """Process and handle alerts."""
        for alert in alerts:
            logger.warning(f"🚨 ALERT: {alert['message']}")
            # In a real implementation, this would send notifications, emails, etc.
    
    def _attempt_auto_remediation(self):
        """Attempt automatic remediation of known issues."""
        if not self.config['enable_auto_remediation']:
            return
        
        # Example auto-remediation actions
        for issue in self.validation_issues:
            if issue.category == "cache_performance" and "hit rate" in issue.message:
                # Try to warm the cache
                try:
                    if self.school_day_lookup:
                        current_year = datetime.now().year
                        self.school_day_lookup.refresh_cache(year=current_year)
                        logger.info("🔧 Auto-remediation: Refreshed cache")
                except Exception as e:
                    logger.warning(f"⚠️ Auto-remediation failed: {e}")
    
    def _get_data_fix_recommendation(self, issue: str) -> str:
        """Get recommendation for fixing a data integrity issue."""
        if "Missing days" in issue:
            return "Run calendar generation for the affected year"
        elif "Missing consecutive dates" in issue:
            return "Check data generation logic and regenerate calendar"
        elif "null required fields" in issue:
            return "Validate and fix data insertion process"
        else:
            return "Review data generation and validation processes"
    
    def _generate_health_recommendations(self) -> List[str]:
        """Generate actionable health recommendations."""
        recommendations = []
        
        # Check for common issues and provide recommendations
        if self.overall_health == HealthStatus.CRITICAL:
            recommendations.append("🚨 CRITICAL: Immediate attention required - check database connectivity and data integrity")
        
        if self.performance_metrics.avg_response_time_ms > 100:
            recommendations.append("⚡ Consider optimizing database queries or adding indexes")
        
        if self.performance_metrics.cache_hit_rate < 90:
            recommendations.append("🔄 Improve cache performance by warming cache or increasing cache size")
        
        if len(self.validation_issues) > 5:
            recommendations.append("📋 Multiple validation issues detected - run comprehensive data validation")
        
        return recommendations
    
    def get_health_dashboard_data(self) -> Dict[str, Any]:
        """Get formatted data for health dashboard display."""
        with self._health_lock:
            return {
                'overall_health': self.overall_health.value,
                'last_check': datetime.now().isoformat(),
                'uptime_hours': self.performance_metrics.uptime_hours,
                'metrics': {
                    'avg_response_time_ms': self.performance_metrics.avg_response_time_ms,
                    'cache_hit_rate': self.performance_metrics.cache_hit_rate,
                    'total_queries': self.performance_metrics.total_queries,
                    'errors_per_hour': self.performance_metrics.errors_per_hour
                },
                'validation_issues_count': len(self.validation_issues),
                'critical_issues_count': len([i for i in self.validation_issues if i.severity == ValidationSeverity.CRITICAL]),
                'health_history': list(self.health_history)[-20:],  # Last 20 entries
                'recent_issues': [
                    {
                        'severity': issue.severity.value,
                        'message': issue.message,
                        'timestamp': issue.timestamp.isoformat()
                    }
                    for issue in self.validation_issues[-10:]  # Last 10 issues
                ]
            }


# Global health monitor instance
_health_monitor_instance: Optional[CalendarHealthMonitor] = None


def get_health_monitor(school_day_lookup=None, operations=None, check_interval_minutes: float = 15.0) -> CalendarHealthMonitor:
    """Get or create the global health monitor instance."""
    global _health_monitor_instance
    
    if _health_monitor_instance is None:
        _health_monitor_instance = CalendarHealthMonitor(
            school_day_lookup=school_day_lookup,
            operations=operations,
            check_interval_minutes=check_interval_minutes
        )
    elif school_day_lookup and not _health_monitor_instance.school_day_lookup:
        _health_monitor_instance.school_day_lookup = school_day_lookup
    elif operations and not _health_monitor_instance.operations:
        _health_monitor_instance.operations = operations
    
    return _health_monitor_instance


if __name__ == "__main__":
    # Command-line interface for health monitoring
    import argparse
    
    def safe_print(message):
        """Safe print function to handle encoding issues."""
        try:
            print(message)
        except UnicodeEncodeError:
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            print(safe_message)
    
    parser = argparse.ArgumentParser(description='Calendar Health Monitor CLI')
    parser.add_argument('command', choices=['check', 'monitor', 'dashboard'], 
                       help='Command to execute')
    parser.add_argument('--interval', type=float, default=15.0,
                       help='Monitoring interval in minutes')
    
    args = parser.parse_args()
    
    try:
        health_monitor = get_health_monitor(check_interval_minutes=args.interval)
        
        if args.command == 'check':
            safe_print("🔍 Performing health check...")
            report = health_monitor.perform_health_check()
            safe_print(f"📊 Overall Health: {report['overall_health'].upper()}")
            safe_print(f"📈 Issues Found: {len(report['validation_issues'])}")
            
            for component, status in report['components'].items():
                safe_print(f"   {component}: {status['status']}")
                
        elif args.command == 'monitor':
            safe_print(f"🚀 Starting continuous health monitoring (interval: {args.interval}min)")
            health_monitor.start_monitoring()
            
            try:
                while True:
                    time.sleep(60)  # Keep main thread alive
            except KeyboardInterrupt:
                safe_print("\n🛑 Stopping health monitoring...")
                health_monitor.stop_monitoring()
                
        elif args.command == 'dashboard':
            safe_print("📊 Health Dashboard Data:")
            dashboard_data = health_monitor.get_health_dashboard_data()
            safe_print(json.dumps(dashboard_data, indent=2, default=str))
            
    except Exception as e:
        safe_print(f"❌ Error: {e}")
        exit(1)
