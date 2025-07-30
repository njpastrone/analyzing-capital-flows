"""
BOP Data Integration Module
===========================

This module integrates newly fetched IMF BOP data with existing Case Study 1 data
to create a comprehensive dataset for "Overall Capital Flows" calculation.

Author: Claude Code Assistant
Date: 2025-01-30
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)

class BOPDataIntegrator:
    """
    Integrates new IMF BOP data with existing Case Study 1 dataset
    """
    
    def __init__(self, project_root: Path = None):
        if project_root is None:
            # Assume we're in src/data_collection, so project root is 2 levels up
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = project_root
            
        self.data_dir = self.project_root / "data"
        self.existing_data_file = self.data_dir / "case_study_1_data_july_24_2025.csv"
        
    def load_existing_data(self) -> pd.DataFrame:
        """
        Load existing Case Study 1 BOP data
        
        Returns:
            DataFrame with existing BOP data
        """
        logger.info(f"Loading existing data from: {self.existing_data_file}")
        
        if not self.existing_data_file.exists():
            raise FileNotFoundError(f"Existing data file not found: {self.existing_data_file}")
        
        df = pd.read_csv(self.existing_data_file)
        logger.info(f"Loaded {len(df)} observations from existing dataset")
        
        return df
    
    def validate_new_data(self, new_df: pd.DataFrame) -> bool:
        """
        Validate new BOP data structure and content
        
        Args:
            new_df: New BOP data from IMF API
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['COUNTRY', 'BOP_ACCOUNTING_ENTRY', 'INDICATOR', 
                           'UNIT', 'FREQUENCY', 'TIME_PERIOD', 'OBS_VALUE', 'SCALE']
        
        # Check required columns
        missing_columns = set(required_columns) - set(new_df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for empty data
        if new_df.empty:
            logger.error("New data is empty")
            return False
        
        # Check time period format (should be YYYY-QN)
        time_periods = new_df['TIME_PERIOD'].dropna()
        if not time_periods.str.match(r'^\d{4}-Q[1-4]$').all():
            logger.warning("Some time periods don't match expected format YYYY-QN")
        
        # Check for numeric values
        if not pd.api.types.is_numeric_dtype(new_df['OBS_VALUE']):
            logger.error("OBS_VALUE column is not numeric")
            return False
        
        logger.info("New data validation passed")
        return True
    
    def harmonize_data_structure(self, new_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonize new data structure to match existing data format
        
        Args:
            new_df: New BOP data from IMF API
            existing_df: Existing Case Study 1 data
            
        Returns:
            Harmonized new data
        """
        harmonized_df = new_df.copy()
        
        # Ensure all required columns exist and are in the same order as existing data
        existing_columns = existing_df.columns.tolist()
        
        # Reorder columns to match existing data
        harmonized_df = harmonized_df.reindex(columns=existing_columns, fill_value=None)
        
        # Ensure data types match
        for col in existing_columns:
            if col in harmonized_df.columns and col in existing_df.columns:
                existing_dtype = existing_df[col].dtype
                try:
                    harmonized_df[col] = harmonized_df[col].astype(existing_dtype)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {col} to {existing_dtype}")
        
        logger.info(f"Harmonized {len(harmonized_df)} observations")
        return harmonized_df
    
    def integrate_datasets(self, new_data_path: str) -> pd.DataFrame:
        """
        Integrate new BOP data with existing dataset
        
        Args:
            new_data_path: Path to new BOP data CSV file
            
        Returns:
            Combined dataset with existing + new data
        """
        # Load datasets
        existing_df = self.load_existing_data()
        new_df = pd.read_csv(new_data_path)
        
        logger.info(f"Integrating datasets:")
        logger.info(f"  Existing: {len(existing_df)} observations")
        logger.info(f"  New: {len(new_df)} observations")
        
        # Validate new data
        if not self.validate_new_data(new_df):
            raise ValueError("New data validation failed")
        
        # Harmonize structure
        new_df_harmonized = self.harmonize_data_structure(new_df, existing_df)
        
        # Combine datasets
        combined_df = pd.concat([existing_df, new_df_harmonized], ignore_index=True)
        
        # Remove any duplicate rows (in case of overlap)
        initial_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(
            subset=['COUNTRY', 'BOP_ACCOUNTING_ENTRY', 'INDICATOR', 'TIME_PERIOD'],
            keep='first'
        )
        final_len = len(combined_df)
        
        if initial_len != final_len:
            logger.info(f"Removed {initial_len - final_len} duplicate observations")
        
        logger.info(f"Integration complete: {len(combined_df)} total observations")
        
        return combined_df
    
    def calculate_overall_capital_flows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive Overall Capital Flows with all Financial Account components
        
        Args:
            df: Integrated BOP dataset
            
        Returns:
            Dataset with Overall Capital Flows indicator added
        """
        logger.info("Calculating comprehensive Overall Capital Flows...")
        
        # Process data to wide format for calculation (similar to existing dashboard logic)
        df_work = df.copy()
        
        # Create full indicator names
        df_work['ENTRY_FIRST_WORD'] = df_work['BOP_ACCOUNTING_ENTRY'].str.extract(r'^([^,]+)')
        df_work['FULL_INDICATOR'] = df_work['ENTRY_FIRST_WORD'] + ' - ' + df_work['INDICATOR']
        
        # Identify available Net indicators for Overall Capital Flows
        net_indicators = df_work[df_work['FULL_INDICATOR'].str.startswith('Net ')]['FULL_INDICATOR'].unique()
        
        # Define comprehensive Overall Capital Flows components
        comprehensive_components = [
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Direct investment, Total financial assets/liabilities',
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Portfolio investment, Total financial assets/liabilities',
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Other investment, Total financial assets/liabilities',
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Financial derivatives, Total financial assets/liabilities'
            # Note: Reserve Assets excluded as they represent policy intervention, not market flows
        ]
        
        # Check which components are available
        available_components = [comp for comp in comprehensive_components if comp in net_indicators]
        missing_components = [comp for comp in comprehensive_components if comp not in net_indicators]
        
        logger.info(f"Available components for Overall Capital Flows: {len(available_components)}")
        for comp in available_components:
            logger.info(f"  ✅ {comp}")
        
        if missing_components:
            logger.warning(f"Missing components: {len(missing_components)}")
            for comp in missing_components:
                logger.warning(f"  ❌ {comp}")
        
        if not available_components:
            logger.error("No components available for Overall Capital Flows calculation")
            return df
        
        # Create Overall Capital Flows by summing available Net components
        # We need to work with the data in wide format for calculation
        
        # Pivot to wide format
        metadata_cols = ['COUNTRY', 'UNIT', 'FREQUENCY', 'SCALE', 'TIME_PERIOD']
        pivot_df = df_work.pivot_table(
            index=metadata_cols,
            columns='FULL_INDICATOR',
            values='OBS_VALUE',
            aggfunc='first'
        ).reset_index()
        
        # Calculate Overall Capital Flows
        if available_components:
            # Use skipna=True and only sum where we have at least one component
            # Create mask for rows that have at least one non-null component
            component_mask = ~pivot_df[available_components].isna().all(axis=1)
            pivot_df['Overall Capital Flows (Net, Comprehensive)'] = 0.0
            pivot_df.loc[component_mask, 'Overall Capital Flows (Net, Comprehensive)'] = pivot_df.loc[component_mask, available_components].sum(axis=1, skipna=True)
            
            # Convert back to long format for the new indicator
            overall_cf_data = []
            for _, row in pivot_df.iterrows():
                # Only include if we have a meaningful calculation (not zero from missing data)
                if (not pd.isna(row['Overall Capital Flows (Net, Comprehensive)']) and 
                    (row['Overall Capital Flows (Net, Comprehensive)'] != 0.0 or 
                     any(not pd.isna(row[comp]) for comp in available_components))):
                    overall_cf_data.append({
                        'COUNTRY': row['COUNTRY'],
                        'BOP_ACCOUNTING_ENTRY': 'Net (net acquisition of financial assets less net incurrence of liabilities), Transactions',
                        'INDICATOR': 'Overall Capital Flows (Net, Comprehensive)',
                        'UNIT': row['UNIT'],
                        'FREQUENCY': row['FREQUENCY'],
                        'TIME_PERIOD': row['TIME_PERIOD'],
                        'OBS_VALUE': row['Overall Capital Flows (Net, Comprehensive)'],
                        'SCALE': row['SCALE']
                    })
            
            if overall_cf_data:
                overall_cf_df = pd.DataFrame(overall_cf_data)
                
                # Add Overall Capital Flows to main dataset
                enhanced_df = pd.concat([df, overall_cf_df], ignore_index=True)
                
                logger.info(f"Added {len(overall_cf_df)} Overall Capital Flows observations")
                logger.info(f"Components used: {', '.join([comp.split(' - ')[1] for comp in available_components])}")
                
                return enhanced_df
        
        logger.warning("Could not calculate Overall Capital Flows")
        return df
    
    def save_integrated_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save integrated dataset to data folder
        
        Args:
            df: Integrated dataset
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"case_study_1_comprehensive_bop_data_{timestamp}.csv"
        
        output_path = self.data_dir / filename
        df.to_csv(output_path, index=False)
        
        logger.info(f"Integrated data saved to: {output_path}")
        logger.info(f"Total observations: {len(df)}")
        logger.info(f"Countries: {df['COUNTRY'].nunique()}")
        logger.info(f"Indicators: {df['INDICATOR'].nunique()}")
        
        return str(output_path)
    
    def generate_integration_report(self, original_df: pd.DataFrame, enhanced_df: pd.DataFrame) -> Dict:
        """
        Generate a report comparing original vs enhanced dataset
        
        Args:
            original_df: Original dataset
            enhanced_df: Enhanced dataset with new components
            
        Returns:
            Dictionary with comparison statistics
        """
        report = {
            "original_data": {
                "observations": len(original_df),
                "countries": original_df['COUNTRY'].nunique(),
                "indicators": original_df['INDICATOR'].nunique(),
                "unique_indicators": sorted(original_df['INDICATOR'].unique().tolist())
            },
            "enhanced_data": {
                "observations": len(enhanced_df),
                "countries": enhanced_df['COUNTRY'].nunique(), 
                "indicators": enhanced_df['INDICATOR'].nunique(),
                "unique_indicators": sorted(enhanced_df['INDICATOR'].unique().tolist())
            },
            "additions": {
                "new_observations": len(enhanced_df) - len(original_df),
                "new_indicators": enhanced_df['INDICATOR'].nunique() - original_df['INDICATOR'].nunique(),
                "new_indicator_list": list(set(enhanced_df['INDICATOR'].unique()) - set(original_df['INDICATOR'].unique()))
            }
        }
        
        return report


def main():
    """
    Main integration workflow
    """
    integrator = BOPDataIntegrator()
    
    # Check if we have new data to integrate
    data_collection_dir = Path(__file__).parent
    new_data_files = list(data_collection_dir.glob("imf_bop_missing_series_*.csv"))
    
    if not new_data_files:
        print("❌ No new BOP data files found. Run imf_bop_fetcher.py first.")
        return
    
    # Use the most recent file
    latest_file = max(new_data_files, key=lambda x: x.stat().st_mtime)
    print(f"Using new data file: {latest_file}")
    
    try:
        # Load existing data for comparison
        original_df = integrator.load_existing_data()
        
        # Integrate datasets
        integrated_df = integrator.integrate_datasets(str(latest_file))
        
        # Calculate comprehensive Overall Capital Flows
        enhanced_df = integrator.calculate_overall_capital_flows(integrated_df)
        
        # Save integrated data
        output_file = integrator.save_integrated_data(enhanced_df)
        
        # Generate report
        report = integrator.generate_integration_report(original_df, enhanced_df)
        
        # Print summary
        print("\n" + "="*60)
        print("BOP DATA INTEGRATION SUMMARY")
        print("="*60)
        print(f"Original data: {report['original_data']['observations']} observations, {report['original_data']['indicators']} indicators")
        print(f"Enhanced data: {report['enhanced_data']['observations']} observations, {report['enhanced_data']['indicators']} indicators")
        print(f"Added: {report['additions']['new_observations']} observations, {report['additions']['new_indicators']} indicators")
        print(f"Output file: {output_file}")
        
        if report['additions']['new_indicator_list']:
            print("\nNew indicators added:")
            for indicator in report['additions']['new_indicator_list']:
                print(f"  ✅ {indicator}")
        
        print("\n🎉 Integration complete! Enhanced dataset ready for use.")
        
    except Exception as e:
        print(f"❌ Integration failed: {str(e)}")
        logger.error(f"Integration error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()