"""
Core data processing classes for Capital Flows Research
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

class DataProcessor:
    """Base class for data processing operations"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir
        self.raw_data = {}
        self.processed_data = {}
        self.metadata = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path: Union[str, Path], data_key: str) -> pd.DataFrame:
        """Load data from file and store in raw_data dict"""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            self.raw_data[data_key] = df
            self.logger.info(f"Loaded {data_key}: {df.shape[0]} rows, {df.shape[1]} columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate that DataFrame has required columns"""
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return True
    
    def get_summary(self, data_key: str) -> Dict:
        """Get summary information about a dataset"""
        if data_key not in self.raw_data:
            raise KeyError(f"Data key '{data_key}' not found")
        
        df = self.raw_data[data_key]
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': dict(df.dtypes),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }


class BOPDataProcessor(DataProcessor):
    """Specialized processor for Balance of Payments data"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        super().__init__(data_dir)
        self.bop_indicators = []
        self.gdp_data = None
        
    def process_bop_data(self, bop_data: pd.DataFrame) -> pd.DataFrame:
        """Process raw BOP data into analysis-ready format"""
        self.logger.info("Processing BOP data...")
        
        # Validate required columns
        required_cols = ['COUNTRY', 'TIME_PERIOD', 'BOP_ACCOUNTING_ENTRY', 
                        'INDICATOR', 'OBS_VALUE']
        self.validate_data(bop_data, required_cols)
        
        # Create indicator names
        df = bop_data.copy()
        df['ENTRY_FIRST_WORD'] = df['BOP_ACCOUNTING_ENTRY'].str.extract(r'^([^,]+)')
        df['FULL_INDICATOR'] = df['ENTRY_FIRST_WORD'] + ' - ' + df['INDICATOR']
        
        # Clean unnecessary columns
        columns_to_drop = ['BOP_ACCOUNTING_ENTRY', 'INDICATOR', 'ENTRY_FIRST_WORD']
        # Only drop columns that exist
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=columns_to_drop)
        
        # Process time period
        df[['YEAR', 'QUARTER']] = df['TIME_PERIOD'].str.split('-', expand=True)
        df['YEAR'] = df['YEAR'].astype(int)
        df['QUARTER'] = df['QUARTER'].str.extract(r'(\d+)').astype(int)
        df = df.drop('TIME_PERIOD', axis=1)
        
        # Pivot to wide format
        df_pivot = df.pivot_table(
            index=['COUNTRY', 'YEAR', 'QUARTER', 'UNIT'],
            columns='FULL_INDICATOR',
            values='OBS_VALUE',
            aggfunc='first'
        ).reset_index()
        
        self.processed_data['bop_pivot'] = df_pivot
        self.bop_indicators = [col for col in df_pivot.columns 
                              if col not in ['COUNTRY', 'YEAR', 'QUARTER', 'UNIT']]
        
        self.logger.info(f"BOP processing complete: {len(self.bop_indicators)} indicators")
        return df_pivot
    
    def process_gdp_data(self, gdp_data: pd.DataFrame) -> pd.DataFrame:
        """Process GDP data"""
        self.logger.info("Processing GDP data...")
        
        required_cols = ['COUNTRY', 'TIME_PERIOD', 'INDICATOR', 'OBS_VALUE']
        self.validate_data(gdp_data, required_cols)
        
        # Select and pivot GDP data
        df = gdp_data[required_cols].copy()
        df_pivot = df.pivot_table(
            index=['COUNTRY', 'TIME_PERIOD'],
            columns='INDICATOR',
            values='OBS_VALUE',
            aggfunc='first'
        ).reset_index()
        
        self.processed_data['gdp_pivot'] = df_pivot
        self.gdp_data = df_pivot
        
        self.logger.info("GDP processing complete")
        return df_pivot
    
    def join_bop_gdp(self, remove_luxembourg: bool = True) -> pd.DataFrame:
        """Join BOP and GDP data and normalize to % of GDP"""
        if 'bop_pivot' not in self.processed_data or 'gdp_pivot' not in self.processed_data:
            raise ValueError("Both BOP and GDP data must be processed first")
        
        self.logger.info("Joining BOP and GDP data...")
        
        bop_df = self.processed_data['bop_pivot']
        gdp_df = self.processed_data['gdp_pivot']
        
        # Merge datasets
        merged_data = bop_df.merge(
            gdp_df,
            left_on=['COUNTRY', 'YEAR'],
            right_on=['COUNTRY', 'TIME_PERIOD'],
            how='left'
        ).drop('TIME_PERIOD', axis=1, errors='ignore')
        
        # Identify GDP column (flexible naming)
        gdp_col = None
        for col in merged_data.columns:
            if 'GDP' in col.upper() and 'CURRENT' in col.upper():
                gdp_col = col
                break
        
        if gdp_col is None:
            raise ValueError("GDP column not found in merged data")
        
        # Normalize BOP indicators to % of GDP
        metadata_cols = ['COUNTRY', 'YEAR', 'QUARTER', 'UNIT']
        indicator_cols = [col for col in merged_data.columns 
                         if col not in metadata_cols + [gdp_col]]
        
        normalized_data = merged_data[metadata_cols + [gdp_col]].copy()
        
        for col in indicator_cols:
            # Annualize (×4) and convert to % of GDP (×100)
            normalized_data[f"{col}_PGDP"] = (
                merged_data[col] * 4 / merged_data[gdp_col]
            ) * 100
        
        normalized_data['UNIT'] = "% of GDP (annualized)"
        
        # Remove Luxembourg if requested
        if remove_luxembourg:
            normalized_data = normalized_data[
                normalized_data['COUNTRY'] != 'Luxembourg'
            ].copy()
        
        self.processed_data['normalized'] = normalized_data
        
        self.logger.info(f"Data joining complete: {normalized_data.shape}")
        return normalized_data
    
    def create_groups(self, group_definitions: Dict[str, List[str]]) -> pd.DataFrame:
        """Create country groups for analysis"""
        if 'normalized' not in self.processed_data:
            raise ValueError("Normalized data must be created first")
        
        df = self.processed_data['normalized'].copy()
        
        # Create group column
        def assign_group(country):
            for group_name, countries in group_definitions.items():
                if country in countries:
                    return group_name
            return 'Other'
        
        df['GROUP'] = df['COUNTRY'].apply(assign_group)
        
        # Get analysis indicators
        analysis_indicators = [col for col in df.columns if col.endswith('_PGDP')]
        
        self.processed_data['final'] = df
        self.metadata['analysis_indicators'] = analysis_indicators
        self.metadata['groups'] = group_definitions
        
        self.logger.info(f"Groups created: {df['GROUP'].value_counts().to_dict()}")
        return df
    
    def get_analysis_ready_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Get final analysis-ready data and indicator list"""
        if 'final' not in self.processed_data:
            raise ValueError("Data processing pipeline must be completed first")
        
        df = self.processed_data['final']
        indicators = self.metadata.get('analysis_indicators', [])
        
        return df, indicators