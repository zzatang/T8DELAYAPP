#!/usr/bin/env python3
"""
Test script to verify access to Data.NSW school and public holidays dataset
This demonstrates the primary data source for dynamic calendar integration.
"""

import aiohttp
import asyncio
import csv
import io
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataNSWFetcher:
    """
    Fetches school term and public holiday data from Data.NSW portal
    """
    
    # Data.NSW dataset URL for school and public holidays
    BASE_URL = "https://www.data.nsw.gov.au/data/dataset/2-school-and-public-holidays"
    
    # Direct CSV download URLs (these may change - need to verify)
    CSV_URLS = [
        "https://www.data.nsw.gov.au/data/dataset/2-school-and-public-holidays/resource/xyz/download/school-public-holidays.csv",
        # Alternative: Check the actual resource URLs on the website
    ]
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'T8Monitor-CalendarIntegration/1.0'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_dataset_page(self) -> str:
        """
        Fetch the main dataset page to find current CSV download links
        """
        try:
            async with self.session.get(self.BASE_URL) as response:
                if response.status == 200:
                    content = await response.text()
                    logger.info(f"✅ Successfully fetched dataset page ({len(content)} chars)")
                    return content
                else:
                    logger.error(f"❌ Failed to fetch dataset page: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"❌ Error fetching dataset page: {e}")
            return None
    
    def extract_csv_urls(self, html_content: str) -> List[str]:
        """
        Extract CSV download URLs from the dataset page HTML
        This is a simplified version - would need proper HTML parsing
        """
        csv_urls = []
        
        # Look for CSV download links in the HTML
        # This is a basic pattern - would need BeautifulSoup for robust parsing
        import re
        
        # Pattern to find CSV download URLs
        csv_patterns = [
            r'href="([^"]*\.csv[^"]*)"',
            r'href="([^"]*resource/[^"]*download[^"]*)"',
        ]
        
        for pattern in csv_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            for match in matches:
                if 'csv' in match.lower() or 'download' in match.lower():
                    # Convert relative URLs to absolute
                    if match.startswith('/'):
                        match = f"https://www.data.nsw.gov.au{match}"
                    elif not match.startswith('http'):
                        match = f"https://www.data.nsw.gov.au/data/dataset/2-school-and-public-holidays/{match}"
                    
                    csv_urls.append(match)
        
        # Remove duplicates
        csv_urls = list(set(csv_urls))
        
        logger.info(f"🔍 Found {len(csv_urls)} potential CSV URLs")
        for url in csv_urls:
            logger.info(f"   - {url}")
        
        return csv_urls
    
    async def download_csv_data(self, csv_url: str) -> Optional[str]:
        """
        Download CSV data from a specific URL
        """
        try:
            logger.info(f"📥 Attempting to download: {csv_url}")
            
            async with self.session.get(csv_url) as response:
                if response.status == 200:
                    content = await response.text()
                    logger.info(f"✅ Successfully downloaded CSV ({len(content)} chars)")
                    return content
                else:
                    logger.warning(f"⚠️  Failed to download CSV: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"❌ Error downloading CSV: {e}")
            return None
    
    def parse_csv_data(self, csv_content: str) -> Dict:
        """
        Parse CSV content to extract school terms and public holidays
        """
        try:
            # Parse CSV content
            csv_reader = csv.DictReader(io.StringIO(csv_content))
            
            school_terms = []
            public_holidays = []
            
            # This is a placeholder - actual CSV structure needs to be analyzed
            for row in csv_reader:
                logger.debug(f"CSV Row: {row}")
                
                # Example parsing logic (needs to be adapted to actual CSV structure)
                if 'type' in row and row['type'].lower() == 'school_term':
                    school_terms.append({
                        'start_date': row.get('start_date'),
                        'end_date': row.get('end_date'),
                        'term': row.get('term'),
                        'year': row.get('year')
                    })
                elif 'type' in row and row['type'].lower() == 'public_holiday':
                    public_holidays.append({
                        'date': row.get('date'),
                        'name': row.get('name'),
                        'year': row.get('year')
                    })
            
            result = {
                'school_terms': school_terms,
                'public_holidays': public_holidays,
                'raw_rows': len(list(csv.DictReader(io.StringIO(csv_content))))
            }
            
            logger.info(f"📊 Parsed CSV: {len(school_terms)} terms, {len(public_holidays)} holidays")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error parsing CSV: {e}")
            return None
    
    async def fetch_calendar_data(self, year: int = None) -> Optional[Dict]:
        """
        Main method to fetch and parse calendar data
        """
        if year is None:
            year = datetime.now().year
        
        logger.info(f"🗓️  Fetching calendar data for {year}...")
        
        # Step 1: Get the dataset page to find current CSV URLs
        html_content = await self.fetch_dataset_page()
        if not html_content:
            return None
        
        # Step 2: Extract CSV download URLs
        csv_urls = self.extract_csv_urls(html_content)
        if not csv_urls:
            logger.error("❌ No CSV URLs found on dataset page")
            return None
        
        # Step 3: Try to download and parse CSV data
        for csv_url in csv_urls:
            csv_content = await self.download_csv_data(csv_url)
            if csv_content:
                parsed_data = self.parse_csv_data(csv_content)
                if parsed_data:
                    return parsed_data
        
        logger.error("❌ Failed to download and parse any CSV files")
        return None

async def test_data_nsw_access():
    """
    Test function to verify Data.NSW access and data structure
    """
    logger.info("🧪 Testing Data.NSW calendar data access...")
    
    async with DataNSWFetcher() as fetcher:
        # Test current year
        current_year = datetime.now().year
        data = await fetcher.fetch_calendar_data(current_year)
        
        if data:
            logger.info("✅ Successfully accessed Data.NSW calendar data!")
            logger.info(f"📊 Data summary:")
            logger.info(f"   - School terms: {len(data.get('school_terms', []))}")
            logger.info(f"   - Public holidays: {len(data.get('public_holidays', []))}")
            logger.info(f"   - Raw CSV rows: {data.get('raw_rows', 0)}")
            
            # Display sample data
            if data.get('school_terms'):
                logger.info("📚 Sample school terms:")
                for term in data['school_terms'][:3]:  # First 3 terms
                    logger.info(f"   - {term}")
            
            if data.get('public_holidays'):
                logger.info("🎉 Sample public holidays:")
                for holiday in data['public_holidays'][:5]:  # First 5 holidays
                    logger.info(f"   - {holiday}")
            
            return True
        else:
            logger.error("❌ Failed to access Data.NSW calendar data")
            return False

async def main():
    """
    Main function to run the test
    """
    logger.info("🚀 Starting Data.NSW calendar integration test...")
    
    try:
        success = await test_data_nsw_access()
        
        if success:
            logger.info("✅ Phase 1 Research validation: Data.NSW access confirmed")
            logger.info("🔄 Ready to proceed with Phase 2: Design Architecture")
        else:
            logger.error("❌ Phase 1 Research validation failed")
            logger.info("🔄 May need to investigate alternative data sources")
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")

if __name__ == "__main__":
    # Run the test
    asyncio.run(main())

