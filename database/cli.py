#!/usr/bin/env python3
"""
Command-line interface for School Day Calendar Database System
Provides unified CLI for initialization, health checks, and system management.
"""

import argparse
import json
import sys
import logging
from datetime import datetime
from typing import Dict, Any

from . import (
    get_database_system, 
    initialize_database_system, 
    health_check_database_system,
    get_system_status
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_output(data: Dict[str, Any], format_type: str = 'json') -> str:
    """
    Format output data for display.
    
    Args:
        data: Data to format
        format_type: Output format ('json', 'summary', 'table')
        
    Returns:
        Formatted string
    """
    if format_type == 'json':
        return json.dumps(data, indent=2, default=str)
    
    elif format_type == 'summary':
        output = []
        
        if 'overall_status' in data:
            # Health check summary
            status_emoji = {
                'healthy': '✅',
                'warning': '⚠️',
                'unhealthy': '❌',
                'error': '💥'
            }
            
            status = data['overall_status']
            output.append(f"{status_emoji.get(status, '❓')} Overall Status: {status.upper()}")
            
            if data.get('issues'):
                output.append(f"\nIssues Found ({len(data['issues'])}):")
                for issue in data['issues']:
                    output.append(f"  • {issue}")
            
            if data.get('recommendations'):
                output.append(f"\nRecommendations ({len(data['recommendations'])}):")
                for rec in data['recommendations']:
                    output.append(f"  → {rec}")
                    
        elif 'success' in data:
            # Initialization summary
            success_emoji = '✅' if data['success'] else '❌'
            output.append(f"{success_emoji} Initialization: {'SUCCESS' if data['success'] else 'FAILED'}")
            
            if data.get('steps_completed'):
                output.append(f"\nSteps Completed ({len(data['steps_completed'])}):")
                for step in data['steps_completed']:
                    output.append(f"  ✓ {step.replace('_', ' ').title()}")
            
            if data.get('errors'):
                output.append(f"\nErrors ({len(data['errors'])}):")
                for error in data['errors']:
                    output.append(f"  ✗ {error}")
                    
        elif 'initialization_status' in data:
            # System status summary
            init_status = data['initialization_status']
            status_emoji = '✅' if init_status['initialized'] else '❌'
            output.append(f"{status_emoji} System Status: {'INITIALIZED' if init_status['initialized'] else 'NOT INITIALIZED'}")
            
            if init_status.get('initialization_time'):
                output.append(f"Initialized: {init_status['initialization_time']}")
            
            config = data.get('config_summary', {})
            output.append(f"\nDatabase: {config.get('user', 'unknown')}@{config.get('host', 'unknown')}:{config.get('port', 'unknown')}/{config.get('database', 'unknown')}")
            output.append(f"Connection Pool: {config.get('connection_pool', 'unknown')}")
        
        return '\n'.join(output)
    
    else:
        return str(data)


def cmd_init(args) -> int:
    """Handle database initialization command."""
    print("🚀 Initializing School Day Calendar Database System...")
    
    try:
        result = initialize_database_system(force=args.force)
        
        if args.output == 'json':
            print(format_output(result, 'json'))
        else:
            print(format_output(result, 'summary'))
        
        return 0 if result['success'] else 1
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_health(args) -> int:
    """Handle health check command."""
    print("🏥 Performing Database Health Check...")
    
    try:
        result = health_check_database_system(force=args.force)
        
        if args.output == 'json':
            print(format_output(result, 'json'))
        else:
            print(format_output(result, 'summary'))
        
        # Return appropriate exit code
        status_codes = {
            'healthy': 0,
            'warning': 1,
            'unhealthy': 2,
            'error': 3
        }
        
        return status_codes.get(result['overall_status'], 3)
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 3


def cmd_status(args) -> int:
    """Handle system status command."""
    try:
        result = get_system_status()
        
        if args.output == 'json':
            print(format_output(result, 'json'))
        else:
            print(format_output(result, 'summary'))
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to get system status: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_test_connection(args) -> int:
    """Handle connection test command."""
    print("🔗 Testing Database Connection...")
    
    try:
        db_system = get_database_system()
        success = db_system.connection_manager.test_connection()
        
        if success:
            print("✅ Database connection successful!")
            
            if args.verbose:
                conn_info = db_system.connection_manager.get_connection_info()
                print(f"\nConnection Details:")
                print(f"  Host: {conn_info['config']['host']}:{conn_info['config']['port']}")
                print(f"  Database: {conn_info['config']['database']}")
                print(f"  User: {conn_info['config']['user']}")
                print(f"  Pool: {conn_info['config']['min_connections']}-{conn_info['config']['max_connections']} connections")
            
            return 0
        else:
            print("❌ Database connection failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_validate_data(args) -> int:
    """Handle data validation command."""
    print(f"🔍 Validating Calendar Data for {args.year}...")
    
    try:
        db_system = get_database_system()
        result = db_system.operations.validate_calendar_data(args.year)
        
        if args.output == 'json':
            print(format_output(result, 'json'))
        else:
            status_emoji = '✅' if result['is_valid'] else '❌'
            print(f"{status_emoji} Validation: {'PASSED' if result['is_valid'] else 'FAILED'}")
            
            stats = result.get('stats', {})
            if stats:
                print(f"\nStatistics:")
                print(f"  Total Days: {stats.get('total_days', 'N/A')}")
                print(f"  Expected Days: {stats.get('expected_days', 'N/A')}")
                print(f"  Date Range: {stats.get('first_date', 'N/A')} to {stats.get('last_date', 'N/A')}")
            
            if result.get('issues'):
                print(f"\nIssues Found ({len(result['issues'])}):")
                for issue in result['issues']:
                    print(f"  • {issue}")
        
        return 0 if result['is_valid'] else 1
        
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_check_date(args) -> int:
    """Handle date checking command."""
    print(f"📅 Checking School Day Status for {args.date}...")
    
    try:
        db_system = get_database_system()
        result = db_system.operations.get_school_day_status(args.date)
        
        if result:
            school_day = result['school_day']
            status_emoji = '✅' if school_day else '❌'
            print(f"{status_emoji} {args.date} is {'a SCHOOL DAY' if school_day else 'NOT a school day'}")
            
            if args.verbose:
                print(f"\nDetails:")
                print(f"  Day of Week: {result.get('day_of_week', 'N/A')}")
                print(f"  Reason: {result.get('reason', 'N/A')}")
                print(f"  Term: {result.get('term', 'N/A')}")
                print(f"  Week of Term: {result.get('week_of_term', 'N/A')}")
        else:
            print(f"❌ No data found for date: {args.date}")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Date check failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='School Day Calendar Database System CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s init                    # Initialize database system
  %(prog)s health                  # Perform health check
  %(prog)s status                  # Show system status
  %(prog)s test                    # Test database connection
  %(prog)s validate 2025           # Validate 2025 calendar data
  %(prog)s check 2025-09-08        # Check if date is school day
  %(prog)s health --output json    # Get health check as JSON
        """
    )
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output', choices=['json', 'summary'], default='summary', help='Output format')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize database system')
    init_parser.add_argument('--force', action='store_true', help='Force reinitialization')
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Perform comprehensive health check')
    health_parser.add_argument('--force', action='store_true', help='Force health check')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    # Test connection command
    test_parser = subparsers.add_parser('test', help='Test database connection')
    
    # Validate data command
    validate_parser = subparsers.add_parser('validate', help='Validate calendar data')
    validate_parser.add_argument('year', type=int, help='Year to validate')
    
    # Check date command
    check_parser = subparsers.add_parser('check', help='Check if date is school day')
    check_parser.add_argument('date', help='Date to check (YYYY-MM-DD)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Execute command
    try:
        if args.command == 'init':
            return cmd_init(args)
        elif args.command == 'health':
            return cmd_health(args)
        elif args.command == 'status':
            return cmd_status(args)
        elif args.command == 'test':
            return cmd_test_connection(args)
        elif args.command == 'validate':
            return cmd_validate_data(args)
        elif args.command == 'check':
            return cmd_check_date(args)
        else:
            parser.print_help()
            return 0
            
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
