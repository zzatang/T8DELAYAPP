from sydney_school_day_checker import SchoolDayChecker
import datetime

checker = SchoolDayChecker()
test_date = datetime.date(2025, 9, 8)

print('Holiday periods around September 2025:')
holidays = checker.term_dates.get('holidays', [])
for start, end in sorted(holidays):
    if start.month >= 7 and start.month <= 11:
        print(f'  {start} to {end}')

print(f'\nAnalyzing {test_date}:')
print(f'  Day of week: {test_date.strftime("%A")}')
print(f'  Is weekend: {test_date.weekday() >= 5}')

in_holiday = any(start <= test_date <= end for start, end in holidays)
print(f'  In holiday period: {in_holiday}')

# Since terms are empty, the checker says not in term
term_data = checker.term_dates.get('terms', {})
has_term_data = any(term_info for term_info in term_data.values())
print(f'  Term data available: {has_term_data}')

# The issue is that the ICS parser found holidays but not term dates
print('\nThe issue: ICS parser found holidays but not term start/end dates')
print('This means the checker falls back to "not in term" for any date')
print('Because the term dictionaries are empty: {}, {}, {}, {}')

