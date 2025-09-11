-- School Day Calendar Database Schema
-- Optimized for fast lookups with sub-1ms performance
-- Created for T8 Delay Monitoring System

-- Drop existing table if it exists (for clean reinstalls)
DROP TABLE IF EXISTS school_calendar CASCADE;

-- Create the main school calendar table
CREATE TABLE school_calendar (
    date DATE PRIMARY KEY,
    day_of_week VARCHAR(10) NOT NULL,
    school_day BOOLEAN NOT NULL,
    reason VARCHAR(100),
    term VARCHAR(20),
    week_of_term INTEGER,
    month INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    week_of_year INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optimized indexes for fast lookups
-- Primary index on date (already created by PRIMARY KEY)
CREATE INDEX idx_school_calendar_school_day ON school_calendar(school_day);
CREATE INDEX idx_school_calendar_month ON school_calendar(month);
CREATE INDEX idx_school_calendar_quarter ON school_calendar(quarter);
CREATE INDEX idx_school_calendar_term ON school_calendar(term) WHERE term IS NOT NULL;
CREATE INDEX idx_school_calendar_year ON school_calendar(EXTRACT(YEAR FROM date));

-- Composite indexes for common query patterns
CREATE INDEX idx_school_calendar_month_school_day ON school_calendar(month, school_day);
CREATE INDEX idx_school_calendar_year_month ON school_calendar(EXTRACT(YEAR FROM date), month);

-- Partial index for school days only (most common lookup)
CREATE INDEX idx_school_calendar_school_days_only ON school_calendar(date) WHERE school_day = true;

-- Create a function to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at on row updates
CREATE TRIGGER update_school_calendar_updated_at 
    BEFORE UPDATE ON school_calendar 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create a view for quick statistics
CREATE OR REPLACE VIEW school_calendar_stats AS
SELECT 
    EXTRACT(YEAR FROM date) as year,
    COUNT(*) as total_days,
    COUNT(*) FILTER (WHERE school_day = true) as school_days,
    COUNT(*) FILTER (WHERE school_day = false) as non_school_days,
    COUNT(*) FILTER (WHERE reason = 'weekend') as weekend_days,
    COUNT(*) FILTER (WHERE reason LIKE '%holiday%') as holiday_days,
    COUNT(*) FILTER (WHERE term IS NOT NULL) as term_days,
    MIN(date) as first_date,
    MAX(date) as last_date,
    MAX(updated_at) as last_updated
FROM school_calendar
GROUP BY EXTRACT(YEAR FROM date)
ORDER BY year;

-- Grant permissions for the application user (will be configured via environment variables)
-- Note: These will be executed by the migration script with actual username
-- GRANT SELECT, INSERT, UPDATE, DELETE ON school_calendar TO app_user;
-- GRANT SELECT ON school_calendar_stats TO app_user;

-- Add comments for documentation
COMMENT ON TABLE school_calendar IS 'Pre-computed school day calendar for NSW, Australia with optimized indexes for sub-1ms lookups';
COMMENT ON COLUMN school_calendar.date IS 'Calendar date (primary key)';
COMMENT ON COLUMN school_calendar.day_of_week IS 'Day name (Monday, Tuesday, etc.)';
COMMENT ON COLUMN school_calendar.school_day IS 'True if this date is a school day in NSW';
COMMENT ON COLUMN school_calendar.reason IS 'Reason why it is/is not a school day (weekend, public holiday, school holiday, etc.)';
COMMENT ON COLUMN school_calendar.term IS 'School term identifier (Term 1, Term 2, etc.) if during term time';
COMMENT ON COLUMN school_calendar.week_of_term IS 'Week number within the school term';
COMMENT ON COLUMN school_calendar.month IS 'Month number (1-12) for quick filtering';
COMMENT ON COLUMN school_calendar.quarter IS 'Quarter number (1-4) for quick filtering';
COMMENT ON COLUMN school_calendar.week_of_year IS 'Week number within the year (1-53)';
COMMENT ON COLUMN school_calendar.created_at IS 'Timestamp when record was created';
COMMENT ON COLUMN school_calendar.updated_at IS 'Timestamp when record was last updated';

-- Create an index on the comments for better introspection
-- This helps with debugging and maintenance
SELECT 'Schema created successfully. Ready for calendar data generation.' as status;
