#!/usr/bin/env python3
"""
Twitter API Quota Calculator for T8 Monitor
Helps determine optimal polling intervals based on your API quota.
"""

def calculate_quota_usage(polling_interval_minutes):
    """Calculate monthly API usage based on polling interval."""
    # Monitoring windows: 7:00-8:45 AM (1h 45m) + 1:00-4:00 PM (3h) = 4h 45m = 285 minutes per day
    # School days per month: approximately 20 days
    
    monitoring_minutes_per_day = 285  # 4h 45m
    school_days_per_month = 20
    
    calls_per_day = monitoring_minutes_per_day / polling_interval_minutes
    calls_per_month = calls_per_day * school_days_per_month
    
    return calls_per_day, calls_per_month

def print_quota_table():
    """Print a table of different polling intervals and their quota usage."""
    print("📊 T8 Monitor API Quota Calculator")
    print("=" * 60)
    print()
    print("Monitoring Schedule:")
    print("• Morning: 7:00-8:45 AM (1h 45m)")
    print("• Afternoon: 1:00-4:00 PM (3h)")
    print("• Total: 4h 45m per day × 20 school days = 95 hours/month")
    print()
    print("Quota Usage by Polling Interval:")
    print("-" * 60)
    print(f"{'Interval':<12} {'Calls/Day':<12} {'Calls/Month':<15} {'Status'}")
    print("-" * 60)
    
    intervals = [2, 5, 10, 15, 20, 30, 45, 60, 90, 120]
    
    for interval in intervals:
        calls_per_day, calls_per_month = calculate_quota_usage(interval)
        
        if calls_per_month <= 100:
            status = "✅ Within quota"
        elif calls_per_month <= 200:
            status = "⚠️  Moderate usage"
        elif calls_per_month <= 500:
            status = "🔶 High usage"
        else:
            status = "❌ Exceeds quota"
        
        print(f"{interval:>2} minutes   {calls_per_day:>8.1f}    {calls_per_month:>10.0f}      {status}")

def recommend_interval(quota_limit):
    """Recommend optimal interval for given quota."""
    print(f"\n🎯 Recommendations for {quota_limit} calls/month:")
    print("-" * 50)
    
    # Find intervals that fit within quota
    suitable_intervals = []
    
    for interval in range(1, 181):  # Test 1-180 minutes
        _, calls_per_month = calculate_quota_usage(interval)
        if calls_per_month <= quota_limit:
            suitable_intervals.append((interval, calls_per_month))
    
    if not suitable_intervals:
        print("❌ No suitable interval found for this quota limit.")
        return
    
    # Find the most frequent (smallest interval) that fits
    best_interval, best_usage = suitable_intervals[0]
    
    print(f"💚 Recommended interval: {best_interval} minutes")
    print(f"📊 Expected usage: {best_usage:.0f} calls/month ({best_usage/quota_limit*100:.1f}% of quota)")
    
    # Show some alternatives
    print(f"\n📋 Alternative intervals:")
    for interval, usage in suitable_intervals[:5]:
        if interval != best_interval:
            print(f"   • {interval} minutes: {usage:.0f} calls/month ({usage/quota_limit*100:.1f}% of quota)")

def interactive_calculator():
    """Interactive quota calculator."""
    print("\n🧮 Interactive Quota Calculator")
    print("=" * 40)
    
    try:
        quota = int(input("Enter your monthly API quota limit: "))
        
        if quota <= 0:
            print("❌ Please enter a positive number.")
            return
        
        recommend_interval(quota)
        
        print(f"\n💡 To use this interval, set in your .env file:")
        best_interval = None
        for interval in range(1, 181):
            _, calls_per_month = calculate_quota_usage(interval)
            if calls_per_month <= quota:
                best_interval = interval
                break
        
        if best_interval:
            print(f"POLLING_INTERVAL_MINUTES={best_interval}")
        
    except ValueError:
        print("❌ Please enter a valid number.")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

def main():
    """Main function."""
    print_quota_table()
    
    print("\n" + "=" * 60)
    print("📋 Common Quota Limits:")
    print("• Free Twitter API: 100 calls/month")
    print("• Basic Twitter API: 10,000 calls/month")
    print("• Pro Twitter API: 1,000,000 calls/month")
    
    # Show recommendations for common quotas
    common_quotas = [100, 500, 1000, 10000]
    
    for quota in common_quotas:
        recommend_interval(quota)
    
    # Interactive mode
    while True:
        print("\n" + "=" * 60)
        choice = input("Would you like to calculate for a custom quota? (y/n): ").lower().strip()
        
        if choice in ['y', 'yes']:
            interactive_calculator()
        elif choice in ['n', 'no']:
            break
        else:
            print("Please enter 'y' or 'n'")
    
    print("\n🚆 Happy monitoring!")

if __name__ == "__main__":
    main() 