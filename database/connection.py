#!/usr/bin/env python3
"""
Database Connection Management for School Day Calendar System
Provides centralized PostgreSQL connection handling with environment variable configuration.
"""

import os
import logging
import psycopg2
from psycopg2 import sql, pool
from psycopg2.extras import RealDictCursor
from typing import Optional, Dict, Any, ContextManager
from contextlib import contextmanager
import threading
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration management using environment variables."""
    
    def __init__(self):
        """Initialize database configuration from environment variables."""
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = int(os.getenv('DB_PORT', '5432'))
        self.database = os.getenv('DB_NAME', os.getenv('POSTGRES_DB', 'school_calendar'))
        self.user = os.getenv('DB_USER', os.getenv('POSTGRES_USER', 'postgres'))
        self.password = os.getenv('DB_PASSWORD', os.getenv('POSTGRES_PASSWORD', ''))
        
        # Connection pool settings
        self.min_connections = int(os.getenv('DB_MIN_CONNECTIONS', '2'))
        self.max_connections = int(os.getenv('DB_MAX_CONNECTIONS', '20'))
        
        # Connection timeout settings
        self.connect_timeout = int(os.getenv('DB_CONNECT_TIMEOUT', '10'))
        self.command_timeout = int(os.getenv('DB_COMMAND_TIMEOUT', '30'))
        
        # SSL settings
        self.ssl_mode = os.getenv('DB_SSL_MODE', 'prefer')
        
    def get_connection_params(self) -> Dict[str, Any]:
        """
        Get connection parameters as a dictionary.
        
        Returns:
            Dictionary of connection parameters for psycopg2
        """
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.user,
            'password': self.password,
            'connect_timeout': self.connect_timeout,
            'sslmode': self.ssl_mode,
            'cursor_factory': RealDictCursor,  # Return rows as dictionaries
        }
    
    def get_dsn(self) -> str:
        """
        Get database connection string (DSN).
        
        Returns:
            PostgreSQL connection string
        """
        return (
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}&connect_timeout={self.connect_timeout}"
        )
    
    def validate(self) -> bool:
        """
        Validate that all required configuration is present.
        
        Returns:
            True if configuration is valid
        """
        required_fields = ['host', 'database', 'user']
        missing = [field for field in required_fields if not getattr(self, field)]
        
        if missing:
            logger.error(f"Missing required database configuration: {missing}")
            return False
        
        if self.port < 1 or self.port > 65535:
            logger.error(f"Invalid database port: {self.port}")
            return False
            
        return True


class DatabaseConnectionManager:
    """
    Manages PostgreSQL database connections with connection pooling and error handling.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize the database connection manager.
        
        Args:
            config: Database configuration. If None, creates from environment variables.
        """
        self.config = config or DatabaseConfig()
        self._connection_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self._pool_lock = threading.Lock()
        self._last_health_check = None
        self._health_check_interval = timedelta(minutes=5)
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid database configuration")
    
    def _create_connection_pool(self) -> psycopg2.pool.ThreadedConnectionPool:
        """
        Create a new connection pool.
        
        Returns:
            ThreadedConnectionPool instance
            
        Raises:
            psycopg2.Error: If pool creation fails
        """
        try:
            connection_params = self.config.get_connection_params()
            
            pool_instance = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                **connection_params
            )
            
            logger.info(f"✅ Database connection pool created: {self.config.min_connections}-{self.config.max_connections} connections")
            return pool_instance
            
        except psycopg2.Error as e:
            logger.error(f"❌ Failed to create connection pool: {e}")
            raise
    
    def get_connection_pool(self) -> psycopg2.pool.ThreadedConnectionPool:
        """
        Get the connection pool, creating it if necessary.
        
        Returns:
            ThreadedConnectionPool instance
        """
        if self._connection_pool is None:
            with self._pool_lock:
                if self._connection_pool is None:
                    self._connection_pool = self._create_connection_pool()
        
        return self._connection_pool
    
    @contextmanager
    def get_connection(self) -> ContextManager[psycopg2.extensions.connection]:
        """
        Get a database connection from the pool as a context manager.
        
        Yields:
            PostgreSQL connection object
            
        Example:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
        """
        pool = self.get_connection_pool()
        conn = None
        
        try:
            conn = pool.getconn()
            if conn is None:
                raise psycopg2.Error("Failed to get connection from pool")
            
            # Test connection is still alive
            if conn.closed:
                logger.warning("Got closed connection from pool, reconnecting...")
                pool.putconn(conn, close=True)
                conn = pool.getconn()
            
            yield conn
            
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                pool.putconn(conn, close=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error with database connection: {e}")
            if conn:
                pool.putconn(conn, close=True)
            raise
        else:
            # Return connection to pool
            if conn:
                pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, connection: Optional[psycopg2.extensions.connection] = None) -> ContextManager[psycopg2.extensions.cursor]:
        """
        Get a database cursor as a context manager.
        
        Args:
            connection: Existing connection to use. If None, gets from pool.
            
        Yields:
            PostgreSQL cursor object
            
        Example:
            with db_manager.get_cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM school_calendar")
                count = cur.fetchone()['count']
        """
        if connection:
            # Use provided connection
            try:
                with connection.cursor() as cursor:
                    yield cursor
            except psycopg2.Error as e:
                logger.error(f"Cursor error: {e}")
                raise
        else:
            # Get connection from pool
            with self.get_connection() as conn:
                try:
                    with conn.cursor() as cursor:
                        yield cursor
                except psycopg2.Error as e:
                    logger.error(f"Cursor error: {e}")
                    raise
    
    def test_connection(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_cursor() as cur:
                cur.execute("SELECT 1 as test")
                result = cur.fetchone()
                
                if result and result['test'] == 1:
                    logger.info("✅ Database connection test successful")
                    self._last_health_check = datetime.now()
                    return True
                else:
                    logger.error("❌ Database connection test failed: unexpected result")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False
    
    def health_check(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform a comprehensive database health check.
        
        Args:
            force: If True, skip interval check and always perform health check
            
        Returns:
            Dictionary with health check results
        """
        now = datetime.now()
        
        # Check if we need to run health check
        if (not force and 
            self._last_health_check and 
            now - self._last_health_check < self._health_check_interval):
            return {
                'status': 'skipped',
                'message': 'Health check performed recently',
                'last_check': self._last_health_check.isoformat()
            }
        
        health_info = {
            'timestamp': now.isoformat(),
            'status': 'unknown',
            'connection_test': False,
            'pool_info': {},
            'database_info': {}
        }
        
        try:
            # Test basic connectivity
            health_info['connection_test'] = self.test_connection()
            
            if health_info['connection_test']:
                # Get pool information
                if self._connection_pool:
                    pool = self._connection_pool
                    health_info['pool_info'] = {
                        'min_connections': self.config.min_connections,
                        'max_connections': self.config.max_connections,
                        'closed': pool.closed
                    }
                
                # Get database information
                with self.get_cursor() as cur:
                    # Database version
                    cur.execute("SELECT version()")
                    version = cur.fetchone()['version']
                    
                    # Database size
                    cur.execute("SELECT pg_size_pretty(pg_database_size(current_database())) as size")
                    db_size = cur.fetchone()['size']
                    
                    # Check if school_calendar table exists
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'school_calendar'
                        ) as table_exists
                    """)
                    table_exists = cur.fetchone()['table_exists']
                    
                    # Get record count if table exists
                    record_count = 0
                    if table_exists:
                        cur.execute("SELECT COUNT(*) as count FROM school_calendar")
                        record_count = cur.fetchone()['count']
                    
                    health_info['database_info'] = {
                        'version': version,
                        'size': db_size,
                        'table_exists': table_exists,
                        'record_count': record_count
                    }
                
                health_info['status'] = 'healthy'
                logger.info("✅ Database health check passed")
            else:
                health_info['status'] = 'unhealthy'
                logger.warning("⚠️ Database health check failed")
                
        except Exception as e:
            health_info['status'] = 'error'
            health_info['error'] = str(e)
            logger.error(f"❌ Database health check error: {e}")
        
        return health_info
    
    def close_all_connections(self):
        """Close all connections in the pool."""
        if self._connection_pool:
            try:
                self._connection_pool.closeall()
                self._connection_pool = None
                logger.info("✅ All database connections closed")
            except Exception as e:
                logger.error(f"❌ Error closing connections: {e}")
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get current connection information.
        
        Returns:
            Dictionary with connection configuration and status
        """
        return {
            'config': {
                'host': self.config.host,
                'port': self.config.port,
                'database': self.config.database,
                'user': self.config.user,
                'ssl_mode': self.config.ssl_mode,
                'min_connections': self.config.min_connections,
                'max_connections': self.config.max_connections,
            },
            'pool_created': self._connection_pool is not None,
            'last_health_check': self._last_health_check.isoformat() if self._last_health_check else None
        }


# Global database manager instance
_db_manager: Optional[DatabaseConnectionManager] = None
_manager_lock = threading.Lock()


def get_database_manager() -> DatabaseConnectionManager:
    """
    Get the global database manager instance (singleton pattern).
    
    Returns:
        DatabaseConnectionManager instance
    """
    global _db_manager
    
    if _db_manager is None:
        with _manager_lock:
            if _db_manager is None:
                _db_manager = DatabaseConnectionManager()
    
    return _db_manager


def reset_database_manager():
    """Reset the global database manager (useful for testing)."""
    global _db_manager
    with _manager_lock:
        if _db_manager:
            _db_manager.close_all_connections()
        _db_manager = None


# Convenience functions for common operations
def get_connection():
    """Get a database connection from the global manager."""
    return get_database_manager().get_connection()


def get_cursor(connection=None):
    """Get a database cursor from the global manager."""
    return get_database_manager().get_cursor(connection)


def test_database_connection() -> bool:
    """Test database connectivity using the global manager."""
    return get_database_manager().test_connection()


def main():
    """Command-line interface for database connection testing."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Database Connection Testing Tool')
    parser.add_argument('--test', action='store_true', help='Test database connection')
    parser.add_argument('--health', action='store_true', help='Perform health check')
    parser.add_argument('--info', action='store_true', help='Show connection info')
    parser.add_argument('--force', action='store_true', help='Force health check')
    
    args = parser.parse_args()
    
    try:
        db_manager = get_database_manager()
        
        if args.test:
            success = db_manager.test_connection()
            print(f"Connection test: {'✅ SUCCESS' if success else '❌ FAILED'}")
            return 0 if success else 1
        
        if args.health:
            health = db_manager.health_check(force=args.force)
            print(json.dumps(health, indent=2))
            return 0 if health['status'] == 'healthy' else 1
        
        if args.info:
            info = db_manager.get_connection_info()
            print(json.dumps(info, indent=2))
            return 0
        
        # Default: run connection test
        success = db_manager.test_connection()
        print(f"Connection test: {'✅ SUCCESS' if success else '❌ FAILED'}")
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
