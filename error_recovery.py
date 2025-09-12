#!/usr/bin/env python3
"""
Comprehensive Error Recovery and Fallback System

This module provides advanced error recovery mechanisms including circuit breakers,
automatic retry with exponential backoff, graceful degradation, error escalation,
and recovery validation for the school calendar system.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import statistics
import asyncio
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error scenarios."""
    IMMEDIATE_RETRY = "immediate_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAILOVER = "failover"
    MANUAL_INTERVENTION = "manual_intervention"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


class OperationResult(Enum):
    """Operation execution results."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    DEGRADED = "degraded"


@dataclass
class ErrorRecord:
    """Represents an error occurrence with context."""
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    operation: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_time_ms: Optional[float] = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    half_open_max_calls: int = 3
    half_open_success_threshold: int = 2
    timeout_seconds: float = 30.0
    
    # Sliding window for failure tracking
    window_size: int = 100
    failure_rate_threshold: float = 0.5  # 50% failure rate


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = 3
    initial_delay_ms: float = 1000.0
    max_delay_ms: float = 30000.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    
    # Retry conditions
    retryable_exceptions: List[type] = field(default_factory=lambda: [
        ConnectionError, TimeoutError, Exception
    ])
    non_retryable_exceptions: List[type] = field(default_factory=lambda: [
        ValueError, TypeError, KeyError
    ])


class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascading failures.
    
    Provides automatic failure detection, circuit opening/closing,
    and gradual recovery testing.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self.half_open_successes = 0
        
        # Sliding window for failure tracking
        self.operation_results: deque = deque(maxlen=config.window_size)
        self._lock = threading.RLock()
        
        logger.info(f"🔧 Circuit breaker '{name}' initialized")
        logger.info(f"   Failure threshold: {config.failure_threshold}")
        logger.info(f"   Recovery timeout: {config.recovery_timeout_seconds}s")
    
    @contextmanager
    def call(self):
        """
        Context manager for circuit breaker protected calls.
        
        Usage:
            with circuit_breaker.call() as should_proceed:
                if should_proceed:
                    result = perform_operation()
                    circuit_breaker.record_success()
                else:
                    handle_circuit_open()
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    yield False
                    return
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    yield False
                    return
                self.half_open_calls += 1
        
        yield True
    
    def record_success(self):
        """Record a successful operation."""
        with self._lock:
            self.operation_results.append((datetime.now(), OperationResult.SUCCESS))
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_successes += 1
                if self.half_open_successes >= self.config.half_open_success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
    
    def record_failure(self, error: Exception):
        """Record a failed operation."""
        with self._lock:
            self.operation_results.append((datetime.now(), OperationResult.FAILURE))
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                # Failure during half-open, go back to open
                self._transition_to_open()
            elif self.state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if self._should_open_circuit():
                    self._transition_to_open()
    
    def record_timeout(self):
        """Record a timeout operation."""
        with self._lock:
            self.operation_results.append((datetime.now(), OperationResult.TIMEOUT))
            self.record_failure(TimeoutError("Operation timeout"))
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened based on failure patterns."""
        # Check immediate failure threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Check failure rate in sliding window
        if len(self.operation_results) >= 10:  # Need minimum data
            recent_failures = sum(1 for _, result in self.operation_results 
                                if result == OperationResult.FAILURE)
            failure_rate = recent_failures / len(self.operation_results)
            
            if failure_rate >= self.config.failure_rate_threshold:
                return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout_seconds
    
    def _transition_to_open(self):
        """Transition circuit breaker to OPEN state."""
        self.state = CircuitState.OPEN
        logger.warning(f"🔴 Circuit breaker '{self.name}' OPENED due to failures")
        logger.info(f"   Failure count: {self.failure_count}")
        logger.info(f"   Recovery timeout: {self.config.recovery_timeout_seconds}s")
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.half_open_successes = 0
        logger.info(f"🟡 Circuit breaker '{self.name}' HALF-OPEN - testing recovery")
    
    def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
        self.half_open_successes = 0
        logger.info(f"🟢 Circuit breaker '{self.name}' CLOSED - recovery successful")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            recent_operations = list(self.operation_results)
            total_ops = len(recent_operations)
            
            if total_ops > 0:
                success_count = sum(1 for _, result in recent_operations 
                                  if result == OperationResult.SUCCESS)
                failure_count = sum(1 for _, result in recent_operations 
                                  if result == OperationResult.FAILURE)
                timeout_count = sum(1 for _, result in recent_operations 
                                  if result == OperationResult.TIMEOUT)
                
                success_rate = (success_count / total_ops) * 100
                failure_rate = (failure_count / total_ops) * 100
            else:
                success_count = failure_count = timeout_count = 0
                success_rate = failure_rate = 0.0
            
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'total_operations': total_ops,
                'success_count': success_count,
                'failure_count_recent': failure_count,
                'timeout_count': timeout_count,
                'success_rate': success_rate,
                'failure_rate': failure_rate,
                'half_open_calls': self.half_open_calls,
                'half_open_successes': self.half_open_successes
            }


class ErrorRecoverySystem:
    """
    Comprehensive error recovery and fallback management system.
    
    Features:
    - Circuit breaker pattern for failure isolation
    - Exponential backoff retry mechanisms
    - Graceful degradation strategies
    - Error classification and escalation
    - Recovery validation and health restoration
    - Integration with monitoring systems
    """
    
    def __init__(self, 
                 default_retry_config: Optional[RetryConfig] = None,
                 enable_circuit_breakers: bool = True,
                 error_history_size: int = 1000):
        """
        Initialize the error recovery system.
        
        Args:
            default_retry_config: Default retry configuration
            enable_circuit_breakers: Whether to enable circuit breakers
            error_history_size: Maximum number of errors to keep in history
        """
        self.default_retry_config = default_retry_config or RetryConfig()
        self.enable_circuit_breakers = enable_circuit_breakers
        
        # Error tracking
        self.error_history: deque = deque(maxlen=error_history_size)
        self.error_stats = defaultdict(int)
        self._stats_lock = threading.RLock()
        
        # Circuit breakers for different operations
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        
        # Error escalation callbacks
        self.escalation_callbacks: List[Callable[[ErrorRecord], None]] = []
        
        # Health restoration callbacks
        self.health_restoration_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        logger.info("🛡️ Error Recovery System initialized")
        logger.info(f"   Circuit breakers enabled: {enable_circuit_breakers}")
        logger.info(f"   Default max retries: {self.default_retry_config.max_attempts}")
        logger.info(f"   Error history size: {error_history_size}")
    
    def register_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """
        Register a new circuit breaker for an operation.
        
        Args:
            name: Unique name for the circuit breaker
            config: Configuration for the circuit breaker
            
        Returns:
            The created circuit breaker instance
        """
        if not self.enable_circuit_breakers:
            logger.warning(f"Circuit breakers disabled, not registering '{name}'")
            return None
        
        config = config or CircuitBreakerConfig()
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        
        logger.info(f"🔧 Registered circuit breaker: {name}")
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    @contextmanager
    def protected_call(self, operation_name: str, circuit_breaker_name: Optional[str] = None):
        """
        Context manager for protected operation calls with error recovery.
        
        Args:
            operation_name: Name of the operation for tracking
            circuit_breaker_name: Name of circuit breaker to use (optional)
            
        Usage:
            with recovery_system.protected_call("database_lookup", "db_circuit") as protection:
                if protection.should_proceed:
                    result = perform_database_operation()
                    protection.record_success()
                else:
                    result = handle_circuit_open()
        """
        class ProtectionContext:
            def __init__(self, recovery_system, op_name, circuit_breaker):
                self.recovery_system = recovery_system
                self.operation_name = op_name
                self.circuit_breaker = circuit_breaker
                self.should_proceed = True
                self.start_time = time.perf_counter()
            
            def record_success(self):
                duration_ms = (time.perf_counter() - self.start_time) * 1000
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                logger.debug(f"✅ {self.operation_name} succeeded in {duration_ms:.2f}ms")
            
            def record_failure(self, error: Exception):
                duration_ms = (time.perf_counter() - self.start_time) * 1000
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(error)
                
                # Record error in recovery system
                error_record = ErrorRecord(
                    timestamp=datetime.now(),
                    error_type=type(error).__name__,
                    error_message=str(error),
                    severity=self.recovery_system._classify_error_severity(error),
                    operation=self.operation_name,
                    context={'duration_ms': duration_ms}
                )
                self.recovery_system._record_error(error_record)
                
                logger.warning(f"❌ {self.operation_name} failed in {duration_ms:.2f}ms: {error}")
            
            def record_timeout(self):
                if self.circuit_breaker:
                    self.circuit_breaker.record_timeout()
                logger.warning(f"⏱️ {self.operation_name} timed out")
        
        circuit_breaker = None
        if circuit_breaker_name and self.enable_circuit_breakers:
            circuit_breaker = self.circuit_breakers.get(circuit_breaker_name)
            if not circuit_breaker:
                logger.warning(f"Circuit breaker '{circuit_breaker_name}' not found")
        
        protection_context = ProtectionContext(self, operation_name, circuit_breaker)
        
        # Check circuit breaker status
        if circuit_breaker:
            with circuit_breaker.call() as should_proceed:
                protection_context.should_proceed = should_proceed
                yield protection_context
        else:
            yield protection_context
    
    def retry_with_backoff(self, 
                          operation: Callable,
                          operation_name: str,
                          retry_config: Optional[RetryConfig] = None,
                          **kwargs) -> Any:
        """
        Execute an operation with exponential backoff retry.
        
        Args:
            operation: Function to execute
            operation_name: Name for logging and tracking
            retry_config: Retry configuration (uses default if None)
            **kwargs: Arguments to pass to the operation
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retry attempts fail
        """
        config = retry_config or self.default_retry_config
        last_exception = None
        
        for attempt in range(config.max_attempts):
            try:
                logger.debug(f"🔄 Attempting {operation_name} (attempt {attempt + 1}/{config.max_attempts})")
                result = operation(**kwargs)
                
                if attempt > 0:
                    logger.info(f"✅ {operation_name} succeeded after {attempt + 1} attempts")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if error is retryable
                if not self._is_retryable_error(e, config):
                    logger.error(f"❌ {operation_name} failed with non-retryable error: {e}")
                    raise e
                
                if attempt < config.max_attempts - 1:
                    # Calculate delay with exponential backoff
                    delay_ms = min(
                        config.initial_delay_ms * (config.backoff_multiplier ** attempt),
                        config.max_delay_ms
                    )
                    
                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        import random
                        delay_ms *= (0.5 + random.random() * 0.5)
                    
                    delay_seconds = delay_ms / 1000.0
                    logger.warning(f"⏳ {operation_name} failed (attempt {attempt + 1}), retrying in {delay_seconds:.1f}s: {e}")
                    
                    time.sleep(delay_seconds)
                else:
                    logger.error(f"❌ {operation_name} failed after {config.max_attempts} attempts")
        
        # All attempts failed
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            error_type=type(last_exception).__name__,
            error_message=str(last_exception),
            severity=self._classify_error_severity(last_exception),
            operation=operation_name,
            context={'max_attempts': config.max_attempts, 'final_failure': True}
        )
        self._record_error(error_record)
        
        raise last_exception
    
    def graceful_degradation(self, 
                           primary_operation: Callable,
                           fallback_operation: Callable,
                           operation_name: str,
                           **kwargs) -> Tuple[Any, bool]:
        """
        Execute operation with graceful degradation to fallback.
        
        Args:
            primary_operation: Primary operation to attempt
            fallback_operation: Fallback operation if primary fails
            operation_name: Name for logging and tracking
            **kwargs: Arguments to pass to operations
            
        Returns:
            Tuple of (result, is_degraded) where is_degraded indicates fallback was used
        """
        try:
            logger.debug(f"🎯 Attempting primary operation: {operation_name}")
            result = primary_operation(**kwargs)
            logger.debug(f"✅ Primary operation succeeded: {operation_name}")
            return result, False
            
        except Exception as e:
            logger.warning(f"⚠️ Primary operation failed, attempting fallback: {operation_name}")
            logger.debug(f"Primary failure reason: {e}")
            
            try:
                result = fallback_operation(**kwargs)
                logger.info(f"🛡️ Fallback operation succeeded: {operation_name}")
                
                # Record degraded operation
                error_record = ErrorRecord(
                    timestamp=datetime.now(),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    severity=ErrorSeverity.MEDIUM,
                    operation=operation_name,
                    context={'degraded': True, 'fallback_used': True}
                )
                self._record_error(error_record)
                
                return result, True
                
            except Exception as fallback_error:
                logger.error(f"❌ Both primary and fallback operations failed: {operation_name}")
                logger.error(f"Primary error: {e}")
                logger.error(f"Fallback error: {fallback_error}")
                
                # Record critical failure
                error_record = ErrorRecord(
                    timestamp=datetime.now(),
                    error_type=f"{type(e).__name__}/{type(fallback_error).__name__}",
                    error_message=f"Primary: {e}; Fallback: {fallback_error}",
                    severity=ErrorSeverity.CRITICAL,
                    operation=operation_name,
                    context={'primary_error': str(e), 'fallback_error': str(fallback_error)}
                )
                self._record_error(error_record)
                
                raise fallback_error
    
    def add_escalation_callback(self, callback: Callable[[ErrorRecord], None]):
        """Add a callback for error escalation."""
        self.escalation_callbacks.append(callback)
        logger.info("📢 Error escalation callback registered")
    
    def add_health_restoration_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add a callback for health restoration events."""
        self.health_restoration_callbacks.append(callback)
        logger.info("🏥 Health restoration callback registered")
    
    def validate_recovery(self, operation_name: str, validation_operation: Callable) -> bool:
        """
        Validate that a system has recovered from errors.
        
        Args:
            operation_name: Name of the operation to validate
            validation_operation: Function to test system health
            
        Returns:
            True if recovery is validated, False otherwise
        """
        try:
            logger.info(f"🔍 Validating recovery for: {operation_name}")
            validation_result = validation_operation()
            
            if validation_result:
                logger.info(f"✅ Recovery validated for: {operation_name}")
                
                # Notify health restoration callbacks
                recovery_info = {
                    'operation': operation_name,
                    'validation_time': datetime.now().isoformat(),
                    'status': 'recovered'
                }
                
                for callback in self.health_restoration_callbacks:
                    try:
                        callback(operation_name, recovery_info)
                    except Exception as e:
                        logger.error(f"Health restoration callback failed: {e}")
                
                return True
            else:
                logger.warning(f"⚠️ Recovery validation failed for: {operation_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Recovery validation error for {operation_name}: {e}")
            return False
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on type and context."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        if error_type in ['SystemError', 'MemoryError', 'KeyboardInterrupt']:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in ['ConnectionError', 'TimeoutError', 'DatabaseError']:
            return ErrorSeverity.HIGH
        
        if 'critical' in error_message or 'fatal' in error_message:
            return ErrorSeverity.CRITICAL
        
        # Medium severity errors
        if error_type in ['ValueError', 'TypeError', 'AttributeError']:
            return ErrorSeverity.MEDIUM
        
        if 'timeout' in error_message or 'connection' in error_message:
            return ErrorSeverity.HIGH
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def _is_retryable_error(self, error: Exception, config: RetryConfig) -> bool:
        """Determine if an error is retryable based on configuration."""
        error_type = type(error)
        
        # Check non-retryable exceptions first
        for non_retryable in config.non_retryable_exceptions:
            if isinstance(error, non_retryable):
                return False
        
        # Check retryable exceptions
        for retryable in config.retryable_exceptions:
            if isinstance(error, retryable):
                return True
        
        # Default to non-retryable for unknown errors
        return False
    
    def _record_error(self, error_record: ErrorRecord):
        """Record an error in the system."""
        with self._stats_lock:
            self.error_history.append(error_record)
            self.error_stats[error_record.error_type] += 1
            self.error_stats[f"severity_{error_record.severity.value}"] += 1
        
        # Check for escalation
        if error_record.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            for callback in self.escalation_callbacks:
                try:
                    callback(error_record)
                except Exception as e:
                    logger.error(f"Error escalation callback failed: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self._stats_lock:
            recent_errors = list(self.error_history)
            
            if not recent_errors:
                return {
                    'total_errors': 0,
                    'error_rate_per_hour': 0.0,
                    'by_type': {},
                    'by_severity': {},
                    'recent_errors': []
                }
            
            # Calculate error rate (last hour)
            now = datetime.now()
            hour_ago = now - timedelta(hours=1)
            recent_hour_errors = [e for e in recent_errors if e.timestamp >= hour_ago]
            
            # Group by type and severity
            by_type = defaultdict(int)
            by_severity = defaultdict(int)
            
            for error in recent_errors:
                by_type[error.error_type] += 1
                by_severity[error.severity.value] += 1
            
            return {
                'total_errors': len(recent_errors),
                'error_rate_per_hour': len(recent_hour_errors),
                'by_type': dict(by_type),
                'by_severity': dict(by_severity),
                'recent_errors': [
                    {
                        'timestamp': e.timestamp.isoformat(),
                        'type': e.error_type,
                        'message': e.error_message,
                        'severity': e.severity.value,
                        'operation': e.operation
                    }
                    for e in recent_errors[-10:]  # Last 10 errors
                ],
                'circuit_breakers': {
                    name: cb.get_stats() 
                    for name, cb in self.circuit_breakers.items()
                }
            }
    
    def reset_error_history(self):
        """Reset error history and statistics."""
        with self._stats_lock:
            self.error_history.clear()
            self.error_stats.clear()
        
        logger.info("🔄 Error history and statistics reset")


# Global error recovery system instance
_error_recovery_instance: Optional[ErrorRecoverySystem] = None


def get_error_recovery_system(
    default_retry_config: Optional[RetryConfig] = None,
    enable_circuit_breakers: bool = True,
    error_history_size: int = 1000
) -> ErrorRecoverySystem:
    """Get or create the global error recovery system instance."""
    global _error_recovery_instance
    
    if _error_recovery_instance is None:
        _error_recovery_instance = ErrorRecoverySystem(
            default_retry_config=default_retry_config,
            enable_circuit_breakers=enable_circuit_breakers,
            error_history_size=error_history_size
        )
    
    return _error_recovery_instance


if __name__ == "__main__":
    # Command-line interface for error recovery system
    import argparse
    
    def safe_print(message):
        """Safe print function to handle encoding issues."""
        try:
            print(message)
        except UnicodeEncodeError:
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            print(safe_message)
    
    parser = argparse.ArgumentParser(description='Error Recovery System CLI')
    parser.add_argument('command', choices=['stats', 'reset', 'test'], 
                       help='Command to execute')
    parser.add_argument('--circuit-breaker', help='Circuit breaker name for stats')
    
    args = parser.parse_args()
    
    try:
        recovery_system = get_error_recovery_system()
        
        if args.command == 'stats':
            stats = recovery_system.get_error_statistics()
            safe_print("📊 Error Recovery Statistics:")
            safe_print(f"   Total Errors: {stats['total_errors']:,}")
            safe_print(f"   Error Rate (last hour): {stats['error_rate_per_hour']}")
            
            if stats['by_severity']:
                safe_print("\n📈 By Severity:")
                for severity, count in stats['by_severity'].items():
                    safe_print(f"   {severity.title()}: {count:,}")
            
            if stats['by_type']:
                safe_print("\n🔍 By Type:")
                for error_type, count in sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True):
                    safe_print(f"   {error_type}: {count:,}")
            
            if args.circuit_breaker:
                cb_stats = stats['circuit_breakers'].get(args.circuit_breaker)
                if cb_stats:
                    safe_print(f"\n🔧 Circuit Breaker '{args.circuit_breaker}':")
                    safe_print(f"   State: {cb_stats['state'].upper()}")
                    safe_print(f"   Success Rate: {cb_stats['success_rate']:.1f}%")
                    safe_print(f"   Failure Count: {cb_stats['failure_count']}")
                else:
                    safe_print(f"⚠️ Circuit breaker '{args.circuit_breaker}' not found")
                    
        elif args.command == 'reset':
            recovery_system.reset_error_history()
            safe_print("🔄 Error history reset")
            
        elif args.command == 'test':
            safe_print("🧪 Testing error recovery system...")
            
            # Test circuit breaker
            test_cb = recovery_system.register_circuit_breaker("test_circuit", CircuitBreakerConfig())
            
            # Simulate some failures
            for i in range(3):
                with recovery_system.protected_call("test_operation", "test_circuit") as protection:
                    if protection.should_proceed:
                        # Simulate failure
                        protection.record_failure(Exception(f"Test error {i+1}"))
                    
            safe_print("✅ Error recovery system test completed")
            
    except Exception as e:
        safe_print(f"❌ Error: {e}")
        exit(1)
