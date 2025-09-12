#!/usr/bin/env python3
"""
Performance Monitoring System for School Day Lookup Operations

This module provides comprehensive performance monitoring and metrics collection 
for the school day lookup system, including timing analysis, query optimization 
insights, and detailed performance analytics.
"""

import logging
import time
import threading
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import json
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations being monitored."""
    CACHE_LOOKUP = "cache_lookup"
    DATABASE_LOOKUP = "database_lookup"
    FALLBACK_LOOKUP = "fallback_lookup"
    CACHE_WRITE = "cache_write"
    DATABASE_QUERY = "database_query"
    BATCH_PRELOAD = "batch_preload"
    VALIDATION = "validation"


class PerformanceLevel(Enum):
    """Performance level classifications."""
    EXCELLENT = "excellent"    # < 1ms
    GOOD = "good"             # 1-10ms
    ACCEPTABLE = "acceptable"  # 10-100ms
    SLOW = "slow"             # 100-1000ms
    CRITICAL = "critical"     # > 1000ms


@dataclass
class OperationMetric:
    """Represents a single operation performance metric."""
    operation_type: OperationType
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    cache_hit: Optional[bool] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def performance_level(self) -> PerformanceLevel:
        """Classify the performance level based on duration."""
        if self.duration_ms < 1.0:
            return PerformanceLevel.EXCELLENT
        elif self.duration_ms < 10.0:
            return PerformanceLevel.GOOD
        elif self.duration_ms < 100.0:
            return PerformanceLevel.ACCEPTABLE
        elif self.duration_ms < 1000.0:
            return PerformanceLevel.SLOW
        else:
            return PerformanceLevel.CRITICAL


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    operation_type: OperationType
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    median_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    recent_durations: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_operations == 0:
            return 0.0
        return (self.successful_operations / self.total_operations) * 100.0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        total_cache_ops = self.cache_hits + self.cache_misses
        if total_cache_ops == 0:
            return 0.0
        return (self.cache_hits / total_cache_ops) * 100.0
    
    def update_with_metric(self, metric: OperationMetric):
        """Update statistics with a new metric."""
        self.total_operations += 1
        if metric.success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
        
        self.total_duration_ms += metric.duration_ms
        self.min_duration_ms = min(self.min_duration_ms, metric.duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, metric.duration_ms)
        
        self.recent_durations.append(metric.duration_ms)
        
        # Update cache statistics
        if metric.cache_hit is True:
            self.cache_hits += 1
        elif metric.cache_hit is False:
            self.cache_misses += 1
        
        # Recalculate averages and percentiles
        self._recalculate_stats()
    
    def _recalculate_stats(self):
        """Recalculate derived statistics."""
        if self.total_operations > 0:
            self.avg_duration_ms = self.total_duration_ms / self.total_operations
        
        if len(self.recent_durations) > 0:
            durations_list = list(self.recent_durations)
            self.median_duration_ms = statistics.median(durations_list)
            
            if len(durations_list) >= 20:  # Need sufficient data for percentiles
                self.p95_duration_ms = statistics.quantiles(durations_list, n=20)[18]  # 95th percentile
                if len(durations_list) >= 100:
                    self.p99_duration_ms = statistics.quantiles(durations_list, n=100)[98]  # 99th percentile


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for school day lookup operations.
    
    Features:
    - Real-time operation timing
    - Statistical analysis and trend detection
    - Performance level classification
    - Cache efficiency monitoring
    - Database query optimization insights
    - Automated performance recommendations
    """
    
    def __init__(self, max_history_size: int = 10000, enable_detailed_logging: bool = False):
        """
        Initialize the performance monitoring system.
        
        Args:
            max_history_size: Maximum number of metrics to keep in memory
            enable_detailed_logging: Whether to log detailed performance metrics
        """
        self.max_history_size = max_history_size
        self.enable_detailed_logging = enable_detailed_logging
        
        # Thread-safe storage for metrics
        self._lock = threading.RLock()
        self._metrics_history: deque = deque(maxlen=max_history_size)
        self._stats_by_operation: Dict[OperationType, PerformanceStats] = {}
        
        # Performance thresholds (configurable)
        self.thresholds = {
            'cache_lookup_warning_ms': 1.0,
            'cache_lookup_critical_ms': 5.0,
            'database_lookup_warning_ms': 50.0,
            'database_lookup_critical_ms': 200.0,
            'batch_preload_warning_ms': 1000.0,
            'batch_preload_critical_ms': 5000.0,
            'cache_hit_rate_warning': 85.0,
            'cache_hit_rate_critical': 70.0,
            'success_rate_warning': 95.0,
            'success_rate_critical': 90.0
        }
        
        # Trend analysis
        self._trend_window_minutes = 15
        self._performance_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Alerting
        self._alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self._alert_cooldown: Dict[str, datetime] = {}
        self._alert_cooldown_minutes = 5
        
        # Initialize operation stats
        for op_type in OperationType:
            self._stats_by_operation[op_type] = PerformanceStats(operation_type=op_type)
        
        logger.info("⚡ Performance Monitor initialized")
        logger.info(f"   History size: {max_history_size}")
        logger.info(f"   Detailed logging: {enable_detailed_logging}")
    
    @contextmanager
    def measure_operation(self, operation_type: OperationType, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for measuring operation performance.
        
        Args:
            operation_type: Type of operation being measured
            metadata: Additional metadata about the operation
            
        Usage:
            with monitor.measure_operation(OperationType.DATABASE_LOOKUP, {'date': '2025-01-01'}) as ctx:
                result = perform_database_lookup()
                ctx.set_cache_hit(False)
                ctx.set_success(True)
        """
        start_time = time.perf_counter()
        start_timestamp = datetime.now()
        
        class MeasurementContext:
            def __init__(self):
                self.success = True
                self.cache_hit = None
                self.error_message = None
                self.additional_metadata = {}
            
            def set_success(self, success: bool):
                self.success = success
            
            def set_cache_hit(self, cache_hit: bool):
                self.cache_hit = cache_hit
            
            def set_error(self, error_message: str):
                self.error_message = error_message
                self.success = False
            
            def add_metadata(self, key: str, value: Any):
                self.additional_metadata[key] = value
        
        context = MeasurementContext()
        
        try:
            yield context
        except Exception as e:
            context.set_error(str(e))
            raise
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Create metric
            combined_metadata = metadata or {}
            combined_metadata.update(context.additional_metadata)
            
            metric = OperationMetric(
                operation_type=operation_type,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=context.success,
                cache_hit=context.cache_hit,
                error_message=context.error_message,
                metadata=combined_metadata
            )
            
            # Record the metric
            self.record_metric(metric)
    
    def record_metric(self, metric: OperationMetric):
        """
        Record a performance metric.
        
        Args:
            metric: The performance metric to record
        """
        with self._lock:
            # Store in history
            self._metrics_history.append(metric)
            
            # Update operation statistics
            if metric.operation_type in self._stats_by_operation:
                self._stats_by_operation[metric.operation_type].update_with_metric(metric)
            
            # Update trends
            self._update_trends(metric)
            
            # Check for performance issues and alerts
            self._check_performance_alerts(metric)
            
            # Detailed logging if enabled
            if self.enable_detailed_logging:
                level_icon = self._get_performance_icon(metric.performance_level)
                logger.debug(f"{level_icon} {metric.operation_type.value}: {metric.duration_ms:.3f}ms "
                           f"{'✅' if metric.success else '❌'} "
                           f"{'🎯' if metric.cache_hit else '🔍' if metric.cache_hit is False else ''}")
    
    def get_operation_stats(self, operation_type: OperationType) -> PerformanceStats:
        """
        Get performance statistics for a specific operation type.
        
        Args:
            operation_type: The operation type to get stats for
            
        Returns:
            Performance statistics for the operation type
        """
        with self._lock:
            return self._stats_by_operation.get(operation_type, PerformanceStats(operation_type))
    
    def get_all_stats(self) -> Dict[OperationType, PerformanceStats]:
        """Get performance statistics for all operation types."""
        with self._lock:
            return dict(self._stats_by_operation)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive performance summary.
        
        Returns:
            Dictionary with performance summary data
        """
        with self._lock:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_operations': sum(stats.total_operations for stats in self._stats_by_operation.values()),
                'overall_success_rate': 0.0,
                'operations': {},
                'performance_levels': defaultdict(int),
                'trends': {},
                'alerts': self._get_recent_alerts()
            }
            
            total_ops = 0
            total_successful = 0
            
            # Aggregate stats by operation type
            for op_type, stats in self._stats_by_operation.items():
                if stats.total_operations > 0:
                    summary['operations'][op_type.value] = {
                        'total_operations': stats.total_operations,
                        'success_rate': stats.success_rate,
                        'avg_duration_ms': stats.avg_duration_ms,
                        'median_duration_ms': stats.median_duration_ms,
                        'p95_duration_ms': stats.p95_duration_ms,
                        'p99_duration_ms': stats.p99_duration_ms,
                        'cache_hit_rate': stats.cache_hit_rate,
                        'min_duration_ms': stats.min_duration_ms,
                        'max_duration_ms': stats.max_duration_ms
                    }
                    
                    total_ops += stats.total_operations
                    total_successful += stats.successful_operations
            
            # Calculate overall success rate
            if total_ops > 0:
                summary['overall_success_rate'] = (total_successful / total_ops) * 100.0
            
            # Performance level distribution
            for metric in list(self._metrics_history)[-1000:]:  # Last 1000 operations
                summary['performance_levels'][metric.performance_level.value] += 1
            
            # Recent trends
            summary['trends'] = self._calculate_trends()
            
            return summary
    
    def get_performance_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate performance optimization recommendations based on collected metrics.
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        with self._lock:
            # Analyze cache performance
            cache_stats = self._stats_by_operation.get(OperationType.CACHE_LOOKUP)
            if cache_stats and cache_stats.total_operations > 100:
                if cache_stats.cache_hit_rate < self.thresholds['cache_hit_rate_critical']:
                    recommendations.append({
                        'type': 'cache_optimization',
                        'priority': 'critical',
                        'title': 'Very Low Cache Hit Rate',
                        'description': f'Cache hit rate is {cache_stats.cache_hit_rate:.1f}%, well below optimal',
                        'recommendation': 'Consider increasing cache size or implementing cache warming strategies',
                        'impact': 'High - significant performance improvement possible'
                    })
                elif cache_stats.cache_hit_rate < self.thresholds['cache_hit_rate_warning']:
                    recommendations.append({
                        'type': 'cache_optimization',
                        'priority': 'warning',
                        'title': 'Low Cache Hit Rate',
                        'description': f'Cache hit rate is {cache_stats.cache_hit_rate:.1f}%, below optimal',
                        'recommendation': 'Monitor cache usage patterns and consider cache tuning',
                        'impact': 'Medium - moderate performance improvement possible'
                    })
            
            # Analyze database performance
            db_stats = self._stats_by_operation.get(OperationType.DATABASE_LOOKUP)
            if db_stats and db_stats.total_operations > 50:
                if db_stats.avg_duration_ms > self.thresholds['database_lookup_critical_ms']:
                    recommendations.append({
                        'type': 'database_optimization',
                        'priority': 'critical',
                        'title': 'Slow Database Queries',
                        'description': f'Average database lookup time is {db_stats.avg_duration_ms:.1f}ms',
                        'recommendation': 'Check database indexes, query optimization, and connection pool settings',
                        'impact': 'High - database optimization critical for performance'
                    })
                elif db_stats.avg_duration_ms > self.thresholds['database_lookup_warning_ms']:
                    recommendations.append({
                        'type': 'database_optimization',
                        'priority': 'warning',
                        'title': 'Elevated Database Response Time',
                        'description': f'Average database lookup time is {db_stats.avg_duration_ms:.1f}ms',
                        'recommendation': 'Monitor database performance and consider query optimization',
                        'impact': 'Medium - database tuning could improve performance'
                    })
            
            # Analyze overall performance trends
            trends = self._calculate_trends()
            if trends.get('response_time_trend', 0) > 10:  # 10% increase
                recommendations.append({
                    'type': 'performance_degradation',
                    'priority': 'warning',
                    'title': 'Performance Degradation Trend',
                    'description': f'Response times have increased by {trends["response_time_trend"]:.1f}% recently',
                    'recommendation': 'Investigate recent changes and monitor system resources',
                    'impact': 'Medium - early intervention can prevent performance issues'
                })
            
            # Check for error rates
            total_ops = sum(stats.total_operations for stats in self._stats_by_operation.values())
            total_errors = sum(stats.failed_operations for stats in self._stats_by_operation.values())
            if total_ops > 0:
                error_rate = (total_errors / total_ops) * 100.0
                if error_rate > 5.0:
                    recommendations.append({
                        'type': 'error_rate',
                        'priority': 'critical',
                        'title': 'High Error Rate',
                        'description': f'Error rate is {error_rate:.1f}%, indicating system issues',
                        'recommendation': 'Investigate error patterns and implement error handling improvements',
                        'impact': 'High - high error rates affect reliability and performance'
                    })
        
        return recommendations
    
    def reset_stats(self, operation_type: Optional[OperationType] = None):
        """
        Reset performance statistics.
        
        Args:
            operation_type: Specific operation type to reset, or None for all
        """
        with self._lock:
            if operation_type:
                self._stats_by_operation[operation_type] = PerformanceStats(operation_type)
                logger.info(f"🔄 Reset performance stats for {operation_type.value}")
            else:
                for op_type in OperationType:
                    self._stats_by_operation[op_type] = PerformanceStats(op_type)
                self._metrics_history.clear()
                logger.info("🔄 Reset all performance statistics")
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add a callback function for performance alerts."""
        self._alert_callbacks.append(callback)
    
    def _update_trends(self, metric: OperationMetric):
        """Update performance trends with new metric."""
        now = datetime.now()
        trend_key = f"{metric.operation_type.value}_duration"
        
        # Add to trend data with timestamp
        self._performance_trends[trend_key].append({
            'timestamp': now,
            'value': metric.duration_ms
        })
        
        # Clean old trend data (older than trend window)
        cutoff_time = now - timedelta(minutes=self._trend_window_minutes)
        while (self._performance_trends[trend_key] and 
               self._performance_trends[trend_key][0]['timestamp'] < cutoff_time):
            self._performance_trends[trend_key].popleft()
    
    def _calculate_trends(self) -> Dict[str, float]:
        """Calculate performance trends over the trend window."""
        trends = {}
        
        for trend_key, data_points in self._performance_trends.items():
            if len(data_points) < 10:  # Need minimum data for trend analysis
                continue
            
            # Calculate trend using linear regression slope
            values = [point['value'] for point in data_points]
            if len(values) >= 2:
                # Simple trend calculation: compare first half to second half
                mid_point = len(values) // 2
                first_half_avg = statistics.mean(values[:mid_point])
                second_half_avg = statistics.mean(values[mid_point:])
                
                if first_half_avg > 0:
                    trend_percent = ((second_half_avg - first_half_avg) / first_half_avg) * 100
                    trends[trend_key.replace('_duration', '_trend')] = trend_percent
        
        return trends
    
    def _check_performance_alerts(self, metric: OperationMetric):
        """Check if metric triggers any performance alerts."""
        alerts = []
        
        # Check duration thresholds
        if metric.operation_type == OperationType.CACHE_LOOKUP:
            if metric.duration_ms > self.thresholds['cache_lookup_critical_ms']:
                alerts.append(('critical', f'Cache lookup took {metric.duration_ms:.3f}ms (critical threshold: {self.thresholds["cache_lookup_critical_ms"]}ms)'))
            elif metric.duration_ms > self.thresholds['cache_lookup_warning_ms']:
                alerts.append(('warning', f'Cache lookup took {metric.duration_ms:.3f}ms (warning threshold: {self.thresholds["cache_lookup_warning_ms"]}ms)'))
        
        elif metric.operation_type == OperationType.DATABASE_LOOKUP:
            if metric.duration_ms > self.thresholds['database_lookup_critical_ms']:
                alerts.append(('critical', f'Database lookup took {metric.duration_ms:.3f}ms (critical threshold: {self.thresholds["database_lookup_critical_ms"]}ms)'))
            elif metric.duration_ms > self.thresholds['database_lookup_warning_ms']:
                alerts.append(('warning', f'Database lookup took {metric.duration_ms:.3f}ms (warning threshold: {self.thresholds["database_lookup_warning_ms"]}ms)'))
        
        # Check for errors
        if not metric.success:
            alerts.append(('error', f'{metric.operation_type.value} failed: {metric.error_message or "Unknown error"}'))
        
        # Fire alerts with cooldown
        for level, message in alerts:
            self._fire_alert(level, message, metric)
    
    def _fire_alert(self, level: str, message: str, metric: OperationMetric):
        """Fire a performance alert with cooldown."""
        alert_key = f"{level}_{metric.operation_type.value}"
        now = datetime.now()
        
        # Check cooldown
        if alert_key in self._alert_cooldown:
            if now - self._alert_cooldown[alert_key] < timedelta(minutes=self._alert_cooldown_minutes):
                return  # Still in cooldown
        
        # Update cooldown
        self._alert_cooldown[alert_key] = now
        
        # Create alert data
        alert_data = {
            'level': level,
            'message': message,
            'operation_type': metric.operation_type.value,
            'duration_ms': metric.duration_ms,
            'timestamp': now.isoformat(),
            'metadata': metric.metadata
        }
        
        # Log alert
        if level == 'critical':
            logger.error(f"🚨 CRITICAL PERFORMANCE ALERT: {message}")
        elif level == 'warning':
            logger.warning(f"⚠️ PERFORMANCE WARNING: {message}")
        else:
            logger.info(f"ℹ️ PERFORMANCE INFO: {message}")
        
        # Call alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(level, alert_data)
            except Exception as e:
                logger.error(f"❌ Alert callback failed: {e}")
    
    def _get_recent_alerts(self) -> List[Dict[str, Any]]:
        """Get recent alerts (placeholder for alert history)."""
        # In a full implementation, this would return recent alerts from storage
        return []
    
    def _get_performance_icon(self, level: PerformanceLevel) -> str:
        """Get emoji icon for performance level."""
        icons = {
            PerformanceLevel.EXCELLENT: "🚀",
            PerformanceLevel.GOOD: "✅",
            PerformanceLevel.ACCEPTABLE: "🟡",
            PerformanceLevel.SLOW: "⚠️",
            PerformanceLevel.CRITICAL: "🚨"
        }
        return icons.get(level, "❓")
    
    def export_metrics(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        """
        Export performance metrics in various formats.
        
        Args:
            format: Export format ('json', 'dict')
            
        Returns:
            Exported metrics data
        """
        with self._lock:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'summary': self.get_performance_summary(),
                'recommendations': self.get_performance_recommendations(),
                'thresholds': self.thresholds,
                'configuration': {
                    'max_history_size': self.max_history_size,
                    'trend_window_minutes': self._trend_window_minutes,
                    'alert_cooldown_minutes': self._alert_cooldown_minutes
                }
            }
            
            if format == 'json':
                return json.dumps(export_data, indent=2, default=str)
            else:
                return export_data


# Global performance monitor instance
_performance_monitor_instance: Optional[PerformanceMonitor] = None


def get_performance_monitor(max_history_size: int = 10000, enable_detailed_logging: bool = False) -> PerformanceMonitor:
    """Get or create the global performance monitor instance."""
    global _performance_monitor_instance
    
    if _performance_monitor_instance is None:
        _performance_monitor_instance = PerformanceMonitor(
            max_history_size=max_history_size,
            enable_detailed_logging=enable_detailed_logging
        )
    
    return _performance_monitor_instance


if __name__ == "__main__":
    # Command-line interface for performance monitoring
    import argparse
    
    def safe_print(message):
        """Safe print function to handle encoding issues."""
        try:
            print(message)
        except UnicodeEncodeError:
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            print(safe_message)
    
    parser = argparse.ArgumentParser(description='Performance Monitor CLI')
    parser.add_argument('command', choices=['stats', 'recommendations', 'export', 'reset'], 
                       help='Command to execute')
    parser.add_argument('--operation', choices=[op.value for op in OperationType],
                       help='Specific operation type for stats/reset')
    parser.add_argument('--format', choices=['json', 'dict'], default='dict',
                       help='Export format')
    
    args = parser.parse_args()
    
    try:
        monitor = get_performance_monitor()
        
        if args.command == 'stats':
            if args.operation:
                op_type = OperationType(args.operation)
                stats = monitor.get_operation_stats(op_type)
                safe_print(f"📊 Performance Stats for {args.operation}:")
                safe_print(f"   Total Operations: {stats.total_operations:,}")
                safe_print(f"   Success Rate: {stats.success_rate:.1f}%")
                safe_print(f"   Average Duration: {stats.avg_duration_ms:.3f}ms")
                safe_print(f"   Median Duration: {stats.median_duration_ms:.3f}ms")
                safe_print(f"   95th Percentile: {stats.p95_duration_ms:.3f}ms")
                safe_print(f"   Cache Hit Rate: {stats.cache_hit_rate:.1f}%")
            else:
                summary = monitor.get_performance_summary()
                safe_print("📊 Overall Performance Summary:")
                safe_print(f"   Total Operations: {summary['total_operations']:,}")
                safe_print(f"   Overall Success Rate: {summary['overall_success_rate']:.1f}%")
                safe_print("\n📈 By Operation Type:")
                for op_name, op_stats in summary['operations'].items():
                    safe_print(f"   {op_name}: {op_stats['total_operations']:,} ops, "
                             f"{op_stats['avg_duration_ms']:.3f}ms avg")
                
        elif args.command == 'recommendations':
            recommendations = monitor.get_performance_recommendations()
            if recommendations:
                safe_print("💡 Performance Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    priority_icon = "🚨" if rec['priority'] == 'critical' else "⚠️"
                    safe_print(f"\n{i}. {priority_icon} {rec['title']} ({rec['priority'].upper()})")
                    safe_print(f"   {rec['description']}")
                    safe_print(f"   💡 {rec['recommendation']}")
                    safe_print(f"   📈 {rec['impact']}")
            else:
                safe_print("✅ No performance recommendations at this time")
                
        elif args.command == 'export':
            exported = monitor.export_metrics(format=args.format)
            if args.format == 'json':
                safe_print(exported)
            else:
                safe_print("📤 Performance Metrics Export:")
                safe_print(json.dumps(exported, indent=2, default=str))
                
        elif args.command == 'reset':
            if args.operation:
                op_type = OperationType(args.operation)
                monitor.reset_stats(op_type)
                safe_print(f"🔄 Reset performance stats for {args.operation}")
            else:
                monitor.reset_stats()
                safe_print("🔄 Reset all performance statistics")
                
    except Exception as e:
        safe_print(f"❌ Error: {e}")
        exit(1)
