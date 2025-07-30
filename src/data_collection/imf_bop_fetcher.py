"""
IMF Balance of Payments Data Fetcher
=====================================

This module fetches missing BOP data series from the IMF API to complete
the Financial Account components for comprehensive "Overall Capital Flows" calculation.

Missing series:
- Other Investment (Net, Assets, Liabilities)
- Financial Derivatives (Net, Assets, Liabilities)  
- Reserve Assets

Author: Claude Code Assistant
Date: 2025-01-30
"""

import requests
import pandas as pd
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IMFBOPFetcher:
    """
    Fetches Balance of Payments data from IMF API for missing Financial Account components
    """
    
    def __init__(self):
        self.base_url = "http://dataservices.imf.org/REST/SDMX_JSON.svc"
        self.database = "BOP"  # Balance of Payments database
        
        # Countries for Case Study 1 (same as existing data)
        self.countries = {
            'Iceland': 'IS',
            'Austria': 'AT',
            'Belgium': 'BE', 
            'Finland': 'FI',
            'France': 'FR',
            'Germany': 'DE',
            'Ireland': 'IE',
            'Italy': 'IT',
            'Netherlands': 'NL',
            'Portugal': 'PT',
            'Spain': 'ES'
            # Note: Excluding Luxembourg as in existing analysis
        }
        
        # Missing BOP series we need to fetch (corrected codes from IMF SDMX)
        self.missing_series = {
            # Other Investment (these are the correct IMF codes)
            'BFOA_BP6_USD': 'Other Investment, Net Acquisition of Financial Assets',
            'BFOL_BP6_USD': 'Other Investment, Net Incurrence of Liabilities', 
            
            # Financial Derivatives (these are the correct IMF codes)
            'BFFA_BP6_USD': 'Financial Derivatives (Other Than Reserves) and Employee Stock Options, Net Acquisition of Financial Assets',
            'BFFL_BP6_USD': 'Financial Derivatives (Other Than Reserves) and Employee Stock Options, Net Incurrence of Liabilities',
            
            # Reserve Assets (this is the correct IMF code)
            'BFRA_BP6_USD': 'Reserve Assets'
        }
    
    def build_api_url(self, country_code: str, indicators: List[str], 
                      start_year: str = "1999", end_year: str = "2024") -> str:
        """
        Build IMF API URL for data request
        
        Args:
            country_code: ISO country code (e.g., 'IS' for Iceland)
            indicators: List of BOP indicator codes
            start_year: Start year for data
            end_year: End year for data
            
        Returns:
            Complete API URL string
        """
        # Format: /REST/SDMX_JSON.svc/CompactData/{database}/{frequency}.{country}.{indicator}/{period}
        frequency = "Q"  # Quarterly data
        
        # Join multiple indicators with plus signs
        indicator_string = "+".join(indicators)
        
        # Time period format: startPeriod-endPeriod
        period = f"?startPeriod={start_year}&endPeriod={end_year}"
        
        url = f"{self.base_url}/CompactData/{self.database}/{frequency}.{country_code}.{indicator_string}{period}"
        
        return url
    
    def fetch_country_data(self, country_code: str, country_name: str, 
                          max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch BOP data for a single country
        
        Args:
            country_code: ISO country code
            country_name: Full country name
            max_retries: Maximum number of retry attempts
            
        Returns:
            DataFrame with country's BOP data or None if failed
        """
        indicators = list(self.missing_series.keys())
        url = self.build_api_url(country_code, indicators)
        
        logger.info(f"Fetching data for {country_name} ({country_code})")
        logger.info(f"API URL: {url}")
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Parse JSON response
                data = response.json()
                
                # Extract data from IMF SDMX JSON structure
                if 'CompactData' in data and 'DataSet' in data['CompactData']:
                    dataset = data['CompactData']['DataSet']
                    
                    if 'Series' in dataset:
                        series_list = dataset['Series']
                        if not isinstance(series_list, list):
                            series_list = [series_list]
                        
                        # Parse series data
                        country_data = []
                        for series in series_list:
                            indicator = series.get('@INDICATOR', '')
                            
                            # Get observations
                            if 'Obs' in series:
                                observations = series['Obs']
                                if not isinstance(observations, list):
                                    observations = [observations]
                                
                                for obs in observations:
                                    time_period = obs.get('@TIME_PERIOD', '')
                                    value = obs.get('@OBS_VALUE', '')
                                    
                                    if time_period and value:
                                        try:
                                            numeric_value = float(value)
                                            country_data.append({
                                                'COUNTRY': country_name,
                                                'INDICATOR': indicator,
                                                'TIME_PERIOD': time_period,
                                                'OBS_VALUE': numeric_value,
                                                'UNIT': 'US dollar',
                                                'FREQUENCY': 'Quarterly',
                                                'SCALE': 'Millions'
                                            })
                                        except ValueError:
                                            # Skip non-numeric values
                                            continue
                        
                        if country_data:
                            df = pd.DataFrame(country_data)
                            logger.info(f"Successfully fetched {len(df)} observations for {country_name}")
                            return df
                        else:
                            logger.warning(f"No data found for {country_name}")
                            return None
                
                logger.warning(f"No data series found in response for {country_name}")
                return None
                
            except requests.RequestException as e:
                logger.error(f"Attempt {attempt + 1} failed for {country_name}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                continue
            except Exception as e:
                logger.error(f"Unexpected error for {country_name}: {str(e)}")
                return None
        
        logger.error(f"Failed to fetch data for {country_name} after {max_retries} attempts")
        return None
    
    def fetch_all_countries(self) -> pd.DataFrame:
        """
        Fetch BOP data for all countries in Case Study 1
        
        Returns:
            Combined DataFrame with all countries' data
        """
        all_data = []
        
        logger.info(f"Starting data collection for {len(self.countries)} countries")
        logger.info(f"Fetching indicators: {list(self.missing_series.keys())}")
        
        for country_name, country_code in self.countries.items():
            country_df = self.fetch_country_data(country_code, country_name)
            
            if country_df is not None:
                all_data.append(country_df)
            
            # Be respectful to IMF API
            time.sleep(1)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Successfully collected data for {len(all_data)} countries")
            logger.info(f"Total observations: {len(combined_df)}")
            return combined_df
        else:
            logger.error("No data collected from any country")
            return pd.DataFrame()
    
    def process_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and map indicator codes to readable names
        
        Args:
            df: Raw DataFrame from API
            
        Returns:
            DataFrame with processed indicator names
        """
        if df.empty:
            return df
        
        # Map indicator codes to BOP accounting entries and descriptions
        indicator_mapping = {
            'BFOA_BP6_USD': ('Assets, Net acquisition of financial assets', 'Other investment, Total financial assets/liabilities'),
            'BFOL_BP6_USD': ('Liabilities, Net incurrence of liabilities', 'Other investment, Total financial assets/liabilities'),
            'BFON_BP6_USD': ('Net (net acquisition of financial assets less net incurrence of liabilities), Transactions', 'Other investment, Total financial assets/liabilities'),
            'BFFA_BP6_USD': ('Assets, Net acquisition of financial assets', 'Financial derivatives, Total financial assets/liabilities'),
            'BFFL_BP6_USD': ('Liabilities, Net incurrence of liabilities', 'Financial derivatives, Total financial assets/liabilities'),
            'BFFN_BP6_USD': ('Net (net acquisition of financial assets less net incurrence of liabilities), Transactions', 'Financial derivatives, Total financial assets/liabilities'),
            'BFRA_BP6_USD': ('Assets, Net acquisition of financial assets', 'Reserve assets, Total financial assets/liabilities')
        }
        
        # Add BOP_ACCOUNTING_ENTRY and readable INDICATOR columns
        df['BOP_ACCOUNTING_ENTRY'] = df['INDICATOR'].map(lambda x: indicator_mapping.get(x, ('Unknown', ''))[0])
        df['INDICATOR_DESCRIPTION'] = df['INDICATOR'].map(lambda x: indicator_mapping.get(x, ('', 'Unknown'))[1])
        
        # Replace INDICATOR with description for consistency with existing data
        df['INDICATOR'] = df['INDICATOR_DESCRIPTION']
        df = df.drop('INDICATOR_DESCRIPTION', axis=1)
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save fetched data to CSV file
        
        Args:
            df: DataFrame to save
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"imf_bop_missing_series_{timestamp}.csv"
        
        # Save in data_collection directory
        output_dir = Path(__file__).parent
        output_path = output_dir / filename
        
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to: {output_path}")
        
        return str(output_path)
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for fetched data
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {"status": "No data available"}
        
        summary = {
            "total_observations": len(df),
            "countries": df['COUNTRY'].nunique(),
            "country_list": sorted(df['COUNTRY'].unique().tolist()),
            "indicators": df['INDICATOR'].nunique(),
            "indicator_list": sorted(df['INDICATOR'].unique().tolist()),
            "time_range": {
                "start": df['TIME_PERIOD'].min(),
                "end": df['TIME_PERIOD'].max()
            },
            "data_coverage": df.groupby('COUNTRY')['TIME_PERIOD'].count().to_dict()
        }
        
        return summary


def main():
    """
    Main function to execute data collection
    """
    fetcher = IMFBOPFetcher()
    
    # Fetch all data
    print("Starting IMF BOP data collection...")
    df = fetcher.fetch_all_countries()
    
    if not df.empty:
        # Process indicators
        df_processed = fetcher.process_indicators(df)
        
        # Save data
        output_file = fetcher.save_data(df_processed)
        
        # Print summary
        summary = fetcher.get_data_summary(df_processed)
        print("\n" + "="*50)
        print("DATA COLLECTION SUMMARY")
        print("="*50)
        print(f"Total observations: {summary['total_observations']}")
        print(f"Countries: {summary['countries']}")
        print(f"Indicators: {summary['indicators']}")
        print(f"Time range: {summary['time_range']['start']} to {summary['time_range']['end']}")
        print(f"Output file: {output_file}")
        
        print("\nCountry coverage:")
        for country, obs_count in summary['data_coverage'].items():
            print(f"  {country}: {obs_count} observations")
        
        print("\nIndicators collected:")
        for indicator in summary['indicator_list']:
            print(f"  {indicator}")
    else:
        print("❌ No data collected. Check API connectivity and country codes.")


if __name__ == "__main__":
    main()