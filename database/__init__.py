#!/usr/bin/env python3
"""
Database Package for School Day Calendar System
Provides unified interface for database operations, health checks, and initialization.
"""

import logging
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime

from .connection import get_database_manager, DatabaseConnectionManager, DatabaseConfig
from .migrations import DatabaseMigrator
from .operations import get_calendar_operations, SchoolCalendarOperations

# Configure logging
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "T8 Delay Monitoring System"


class DatabaseManager:
    """
    Unified database management interface for the school calendar system.
    Combines connection management, migrations, operations, and health monitoring.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize the unified database manager.
        
        Args:
            config: Database configuration. If None, uses environment variables.
        """
        self.config = config or DatabaseConfig()
        self.connection_manager = DatabaseConnectionManager(self.config)
        self.migrator = DatabaseMigrator(self.config.get_connection_params())
        self.operations = SchoolCalendarOperations(self.connection_manager)
        
        # System status
        self._initialization_status = {
            'initialized': False,
            'initialization_time': None,
            'last_health_check': None,
            'status': 'not_initialized'
        }
    
    def initialize_system(self, force: bool = False) -> Dict[str, Any]:
        """
        Initialize the complete database system.
        
        Args:
            force: If True, force reinitialization even if already initialized
            
        Returns:
            Dictionary with initialization results
        """
        logger.info("🚀 Initializing School Day Calendar Database System...")
        
        initialization_results = {
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'steps_completed': [],
            'errors': [],
            'config': {
                'host': self.config.host,
                'port': self.config.port,
                'database': self.config.database,
                'user': self.config.user
            }
        }
        
        try:
            # Step 1: Validate configuration
            logger.info("Step 1: Validating database configuration...")
            if not self.config.validate():
                error_msg = "Database configuration validation failed"
                initialization_results['errors'].append(error_msg)
                logger.error(f"❌ {error_msg}")
                return initialization_results
            
            initialization_results['steps_completed'].append("configuration_validated")
            logger.info("✅ Configuration validated")
            
            # Step 2: Test database connectivity
            logger.info("Step 2: Testing database connectivity...")
            if not self.connection_manager.test_connection():
                error_msg = "Database connectivity test failed"
                initialization_results['errors'].append(error_msg)
                logger.error(f"❌ {error_msg}")
                return initialization_results
            
            initialization_results['steps_completed'].append("connectivity_tested")
            logger.info("✅ Database connectivity confirmed")
            
            # Step 3: Initialize database schema
            logger.info("Step 3: Initializing database schema...")
            if not self.migrator.initialize_database(force=force):
                error_msg = "Database schema initialization failed"
                initialization_results['errors'].append(error_msg)
                logger.error(f"❌ {error_msg}")
                return initialization_results
            
            initialization_results['steps_completed'].append("schema_initialized")
            logger.info("✅ Database schema initialized")
            
            # Step 4: Verify operations functionality
            logger.info("Step 4: Verifying database operations...")
            try:
                # Test basic operations
                stats = self.operations.get_calendar_stats()
                initialization_results['initial_stats'] = stats
                initialization_results['steps_completed'].append("operations_verified")
                logger.info("✅ Database operations verified")
            except Exception as e:
                error_msg = f"Database operations verification failed: {e}"
                initialization_results['errors'].append(error_msg)
                logger.error(f"❌ {error_msg}")
                return initialization_results
            
            # Step 5: Perform comprehensive health check
            logger.info("Step 5: Performing comprehensive health check...")
            health_results = self.comprehensive_health_check()
            
            if health_results['overall_status'] != 'healthy':
                error_msg = f"Health check failed: {health_results.get('issues', [])}"
                initialization_results['errors'].append(error_msg)
                logger.error(f"❌ {error_msg}")
                return initialization_results
            
            initialization_results['steps_completed'].append("health_check_passed")
            initialization_results['health_check'] = health_results
            logger.info("✅ Comprehensive health check passed")
            
            # Mark as successfully initialized
            initialization_results['success'] = True
            self._initialization_status = {
                'initialized': True,
                'initialization_time': datetime.now(),
                'last_health_check': datetime.now(),
                'status': 'healthy'
            }
            
            logger.info("🎉 Database system initialization completed successfully!")
            
        except Exception as e:
            error_msg = f"Unexpected error during initialization: {e}"
            initialization_results['errors'].append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        return initialization_results
    
    def comprehensive_health_check(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the entire database system.
        
        Args:
            force: If True, force health check even if performed recently
            
        Returns:
            Dictionary with comprehensive health check results
        """
        logger.info("🏥 Performing comprehensive database health check...")
        
        health_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'checks': {},
            'issues': [],
            'recommendations': [],
            'system_info': {}
        }
        
        try:
            # Check 1: Connection Manager Health
            logger.info("Checking connection manager health...")
            connection_health = self.connection_manager.health_check(force=force)
            health_results['checks']['connection_manager'] = connection_health
            
            if connection_health['status'] != 'healthy':
                health_results['issues'].append("Connection manager unhealthy")
            
            # Check 2: Database Schema Integrity
            logger.info("Checking database schema integrity...")
            try:
                migration_status = self.migrator.get_migration_status()
                health_results['checks']['schema_integrity'] = migration_status
                
                if not migration_status.get('table_exists', False):
                    health_results['issues'].append("School calendar table does not exist")
                    health_results['recommendations'].append("Run database initialization")
                
            except Exception as e:
                health_results['checks']['schema_integrity'] = {'error': str(e)}
                health_results['issues'].append(f"Schema integrity check failed: {e}")
            
            # Check 3: Data Operations Health
            logger.info("Checking data operations health...")
            try:
                # Test basic read operation
                current_year = datetime.now().year
                stats = self.operations.get_calendar_stats(current_year)
                health_results['checks']['data_operations'] = {
                    'status': 'healthy',
                    'current_year_stats': stats
                }
                
                # Check if we have data for current year
                if isinstance(stats, dict) and stats.get('total_days', 0) == 0:
                    health_results['issues'].append(f"No calendar data found for current year ({current_year})")
                    health_results['recommendations'].append("Generate calendar data for current year")
                
            except Exception as e:
                health_results['checks']['data_operations'] = {'error': str(e)}
                health_results['issues'].append(f"Data operations check failed: {e}")
            
            # Check 4: Performance Metrics
            logger.info("Checking performance metrics...")
            try:
                # Test query performance
                start_time = datetime.now()
                test_date = f"{datetime.now().year}-01-01"
                self.operations.get_school_day_status(test_date)
                query_time = (datetime.now() - start_time).total_seconds() * 1000
                
                health_results['checks']['performance'] = {
                    'status': 'healthy' if query_time < 10 else 'slow',
                    'query_time_ms': round(query_time, 2),
                    'meets_sla': query_time < 1.0  # Sub-1ms requirement from PRD
                }
                
                if query_time >= 10:
                    health_results['issues'].append(f"Slow query performance: {query_time:.2f}ms")
                    health_results['recommendations'].append("Check database indexes and connection pool")
                
            except Exception as e:
                health_results['checks']['performance'] = {'error': str(e)}
                health_results['issues'].append(f"Performance check failed: {e}")
            
            # Check 5: System Resources
            logger.info("Checking system resources...")
            try:
                with self.connection_manager.get_cursor() as cur:
                    # Database size
                    cur.execute("SELECT pg_size_pretty(pg_database_size(current_database())) as db_size")
                    db_size = cur.fetchone()['db_size']
                    
                    # Connection count
                    cur.execute("""
                        SELECT count(*) as active_connections
                        FROM pg_stat_activity 
                        WHERE datname = current_database()
                    """)
                    active_connections = cur.fetchone()['active_connections']
                    
                    health_results['checks']['system_resources'] = {
                        'status': 'healthy',
                        'database_size': db_size,
                        'active_connections': active_connections,
                        'max_connections': self.config.max_connections
                    }
                    
                    # Check connection usage
                    connection_usage = (active_connections / self.config.max_connections) * 100
                    if connection_usage > 80:
                        health_results['issues'].append(f"High connection usage: {connection_usage:.1f}%")
                        health_results['recommendations'].append("Consider increasing max_connections")
                
            except Exception as e:
                health_results['checks']['system_resources'] = {'error': str(e)}
                health_results['issues'].append(f"System resources check failed: {e}")
            
            # Determine overall status
            if not health_results['issues']:
                health_results['overall_status'] = 'healthy'
                logger.info("✅ All health checks passed")
            elif len(health_results['issues']) <= 2:
                health_results['overall_status'] = 'warning'
                logger.warning(f"⚠️ Health check warnings: {len(health_results['issues'])} issues found")
            else:
                health_results['overall_status'] = 'unhealthy'
                logger.error(f"❌ Health check failed: {len(health_results['issues'])} issues found")
            
            # Update system status
            self._initialization_status['last_health_check'] = datetime.now()
            self._initialization_status['status'] = health_results['overall_status']
            
        except Exception as e:
            health_results['overall_status'] = 'error'
            health_results['error'] = str(e)
            logger.error(f"❌ Health check error: {e}")
        
        return health_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and information.
        
        Returns:
            Dictionary with system status information
        """
        return {
            'version': __version__,
            'initialization_status': self._initialization_status.copy(),
            'config_summary': {
                'host': self.config.host,
                'port': self.config.port,
                'database': self.config.database,
                'user': self.config.user,
                'ssl_mode': self.config.ssl_mode,
                'connection_pool': f"{self.config.min_connections}-{self.config.max_connections}"
            },
            'components': {
                'connection_manager': 'loaded',
                'migrator': 'loaded',
                'operations': 'loaded'
            }
        }
    
    def shutdown(self):
        """Gracefully shutdown the database system."""
        logger.info("🔄 Shutting down database system...")
        
        try:
            # Close all connections
            self.connection_manager.close_all_connections()
            
            # Update status
            self._initialization_status['status'] = 'shutdown'
            
            logger.info("✅ Database system shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")


# Global database manager instance
_global_db_manager: Optional[DatabaseManager] = None


def get_database_system() -> DatabaseManager:
    """
    Get the global database system manager (singleton pattern).
    
    Returns:
        DatabaseManager instance
    """
    global _global_db_manager
    
    if _global_db_manager is None:
        _global_db_manager = DatabaseManager()
    
    return _global_db_manager


def initialize_database_system(force: bool = False) -> Dict[str, Any]:
    """Initialize the database system using the global manager."""
    return get_database_system().initialize_system(force=force)


def health_check_database_system(force: bool = False) -> Dict[str, Any]:
    """Perform health check using the global manager."""
    return get_database_system().comprehensive_health_check(force=force)


def get_system_status() -> Dict[str, Any]:
    """Get system status using the global manager."""
    return get_database_system().get_system_status()


# Convenience imports for external use
from .connection import get_database_manager, test_database_connection
from .operations import is_school_day, get_school_day_status, insert_calendar_data
from .migrations import DatabaseMigrator

__all__ = [
    'DatabaseManager',
    'get_database_system',
    'initialize_database_system',
    'health_check_database_system',
    'get_system_status',
    'get_database_manager',
    'test_database_connection',
    'is_school_day',
    'get_school_day_status',
    'insert_calendar_data',
    'DatabaseMigrator'
]
