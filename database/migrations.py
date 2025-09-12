#!/usr/bin/env python3
"""
Database Migration Utilities for School Day Calendar System
Handles schema setup, migrations, and database initialization for PostgreSQL.
"""

import os
import sys
import logging
import psycopg2
from psycopg2 import sql
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """
    Handles database migrations and schema management for the school calendar system.
    """
    
    def __init__(self, connection_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the database migrator.
        
        Args:
            connection_params: Database connection parameters. If None, uses environment variables.
        """
        self.connection_params = connection_params or self._get_connection_params()
        self.schema_file = Path(__file__).parent / "schema.sql"
        self.migration_history_table = "schema_migrations"
        
    def _get_connection_params(self) -> Dict[str, Any]:
        """Get database connection parameters from environment variables."""
        return {
            'host': os.getenv('DB_HOST', '192.168.68.55'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', os.getenv('POSTGRES_DB', 'aidb')),
            'user': os.getenv('DB_USER', os.getenv('POSTGRES_USER', 'aiuser')),
            'password': os.getenv('DB_PASSWORD', os.getenv('POSTGRES_PASSWORD', 'aipass'))
        }
    
    def get_connection(self) -> psycopg2.extensions.connection:
        """
        Create a database connection.
        
        Returns:
            PostgreSQL connection object
            
        Raises:
            psycopg2.Error: If connection fails
        """
        try:
            conn = psycopg2.connect(**self.connection_params)
            conn.autocommit = True
            return conn
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    result = cur.fetchone()
                    if result and result[0] == 1:
                        logger.info("✅ Database connection successful")
                        return True
                    else:
                        logger.error(f"❌ Database connection failed: unexpected result {result}")
                        return False
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def create_migration_history_table(self):
        """Create the schema migrations tracking table if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id SERIAL PRIMARY KEY,
            migration_name VARCHAR(255) NOT NULL UNIQUE,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            checksum VARCHAR(64),
            status VARCHAR(20) DEFAULT 'success'
        );
        
        CREATE INDEX IF NOT EXISTS idx_schema_migrations_name ON schema_migrations(migration_name);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_table_sql)
                    logger.info("✅ Migration history table created/verified")
        except psycopg2.Error as e:
            logger.error(f"❌ Failed to create migration history table: {e}")
            raise
    
    def is_migration_applied(self, migration_name: str) -> bool:
        """
        Check if a migration has already been applied.
        
        Args:
            migration_name: Name of the migration to check
            
        Returns:
            True if migration is already applied
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM schema_migrations WHERE migration_name = %s AND status = 'success'",
                        (migration_name,)
                    )
                    return cur.fetchone()[0] > 0
        except psycopg2.Error as e:
            logger.warning(f"Could not check migration status: {e}")
            return False
    
    def record_migration(self, migration_name: str, checksum: str, status: str = 'success'):
        """
        Record a migration in the history table.
        
        Args:
            migration_name: Name of the migration
            checksum: Checksum of the migration content
            status: Status of the migration (success/failed)
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO schema_migrations (migration_name, checksum, status)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (migration_name) 
                        DO UPDATE SET 
                            applied_at = CURRENT_TIMESTAMP,
                            checksum = EXCLUDED.checksum,
                            status = EXCLUDED.status
                        """,
                        (migration_name, checksum, status)
                    )
                    logger.info(f"✅ Migration '{migration_name}' recorded with status: {status}")
        except psycopg2.Error as e:
            logger.error(f"❌ Failed to record migration: {e}")
            raise
    
    def calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate a simple checksum for file content."""
        import hashlib
        content = file_path.read_text(encoding='utf-8')
        return hashlib.md5(content.encode()).hexdigest()
    
    def run_schema_migration(self, force: bool = False) -> bool:
        """
        Run the main schema migration from schema.sql.
        
        Args:
            force: If True, run migration even if already applied
            
        Returns:
            True if successful, False otherwise
        """
        migration_name = "initial_schema"
        
        if not self.schema_file.exists():
            logger.error(f"❌ Schema file not found: {self.schema_file}")
            return False
        
        # Calculate checksum
        checksum = self.calculate_file_checksum(self.schema_file)
        
        # Check if already applied
        if not force and self.is_migration_applied(migration_name):
            logger.info(f"✅ Migration '{migration_name}' already applied, skipping")
            return True
        
        try:
            # Read and execute schema
            schema_content = self.schema_file.read_text(encoding='utf-8')
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Execute the schema in a transaction
                    conn.autocommit = False
                    try:
                        cur.execute(schema_content)
                        conn.commit()
                        logger.info("✅ Schema migration executed successfully")
                        
                        # Record the migration
                        conn.autocommit = True
                        self.record_migration(migration_name, checksum, 'success')
                        return True
                        
                    except Exception as e:
                        conn.rollback()
                        conn.autocommit = True
                        logger.error(f"❌ Schema migration failed: {e}")
                        self.record_migration(migration_name, checksum, 'failed')
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Failed to run schema migration: {e}")
            return False
    
    def initialize_database(self, force: bool = False) -> bool:
        """
        Initialize the database with schema and migration tracking.
        
        Args:
            force: If True, recreate schema even if it exists
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("🚀 Initializing school calendar database...")
        
        try:
            # Test connection first
            if not self.test_connection():
                return False
            
            # Create migration history table
            self.create_migration_history_table()
            
            # Run schema migration
            if not self.run_schema_migration(force=force):
                return False
            
            logger.info("✅ Database initialization completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False
    
    def get_migration_status(self) -> Dict[str, Any]:
        """
        Get the current migration status and database info.
        
        Returns:
            Dictionary with migration status information
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Get migration history
                    cur.execute("""
                        SELECT migration_name, applied_at, status, checksum
                        FROM schema_migrations 
                        ORDER BY applied_at DESC
                    """)
                    migrations = cur.fetchall()
                    
                    # Get table info
                    cur.execute("""
                        SELECT COUNT(*) FROM information_schema.tables 
                        WHERE table_name = 'school_calendar'
                    """)
                    table_exists = cur.fetchone()[0] > 0
                    
                    # Get record count if table exists
                    record_count = 0
                    if table_exists:
                        cur.execute("SELECT COUNT(*) FROM school_calendar")
                        record_count = cur.fetchone()[0]
                    
                    return {
                        'database_connected': True,
                        'table_exists': table_exists,
                        'record_count': record_count,
                        'migrations': [
                            {
                                'name': m[0],
                                'applied_at': m[1].isoformat() if m[1] else None,
                                'status': m[2],
                                'checksum': m[3]
                            } for m in migrations
                        ]
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            return {
                'database_connected': False,
                'error': str(e)
            }
    
    def drop_all_tables(self, confirm: bool = False) -> bool:
        """
        Drop all tables (for development/testing only).
        
        Args:
            confirm: Must be True to actually drop tables
            
        Returns:
            True if successful
        """
        if not confirm:
            logger.warning("⚠️  drop_all_tables requires confirm=True parameter")
            return False
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Drop tables in correct order (handle dependencies)
                    tables = ['school_calendar', 'schema_migrations']
                    
                    for table in tables:
                        cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                        logger.info(f"✅ Dropped table: {table}")
                    
                    # Drop views
                    cur.execute("DROP VIEW IF EXISTS school_calendar_stats CASCADE")
                    logger.info("✅ Dropped view: school_calendar_stats")
                    
                    # Drop functions
                    cur.execute("DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE")
                    logger.info("✅ Dropped function: update_updated_at_column")
                    
            logger.info("✅ All tables dropped successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to drop tables: {e}")
            return False


def main():
    """Command-line interface for database migrations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='School Calendar Database Migration Tool')
    parser.add_argument('--init', action='store_true', help='Initialize database with schema')
    parser.add_argument('--status', action='store_true', help='Show migration status')
    parser.add_argument('--test', action='store_true', help='Test database connection')
    parser.add_argument('--force', action='store_true', help='Force migration even if already applied')
    parser.add_argument('--drop-all', action='store_true', help='Drop all tables (DESTRUCTIVE)')
    parser.add_argument('--confirm-drop', action='store_true', help='Confirm table dropping')
    
    args = parser.parse_args()
    
    # Create migrator instance
    migrator = DatabaseMigrator()
    
    if args.test:
        success = migrator.test_connection()
        sys.exit(0 if success else 1)
    
    if args.status:
        status = migrator.get_migration_status()
        print(json.dumps(status, indent=2))
        return
    
    if args.drop_all:
        if not args.confirm_drop:
            print("❌ --drop-all requires --confirm-drop flag for safety")
            sys.exit(1)
        success = migrator.drop_all_tables(confirm=True)
        sys.exit(0 if success else 1)
    
    if args.init:
        success = migrator.initialize_database(force=args.force)
        sys.exit(0 if success else 1)
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
