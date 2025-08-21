"""
Winsorized Data Loader for Capital Flows Research

Utility functions to load winsorized datasets and support comparative analysis
between original and outlier-adjusted data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_winsorized_comprehensive_data(include_crisis_years: bool = True) -> Tuple[pd.DataFrame, List[str], Dict]:
    """
    Load winsorized comprehensive dataset
    
    Args:
        include_crisis_years: Whether to include crisis periods
        
    Returns:
        Tuple of (data, indicators, metadata)
    """
    try:
        # Load winsorized comprehensive dataset
        data_dir = Path(__file__).parents[2] / "updated_data" / "Clean"
        winsorized_file = data_dir / "comprehensive_df_PGDP_labeled_winsorized.csv"
        
        if not winsorized_file.exists():
            raise FileNotFoundError(f"Winsorized dataset not found: {winsorized_file}")
        
        df = pd.read_csv(winsorized_file)
        logger.info(f"Loaded winsorized comprehensive dataset: {len(df)} rows, {len(df.columns)} columns")
        
        # Apply crisis period filtering if requested
        if not include_crisis_years:
            crisis_years = [2008, 2009, 2010, 2020, 2021, 2022]
            original_size = len(df)
            df = df[~df['YEAR'].isin(crisis_years)]
            excluded_count = original_size - len(df)
            logger.info(f"Excluded {excluded_count} crisis period observations")
        else:
            excluded_count = 0
        
        # Identify indicator columns
        metadata_cols = ['COUNTRY', 'INDICATOR', 'UNIT', 'YEAR', 'QUARTER', 'TIME_PERIOD',
                        'CS1_GROUP', 'CS2_GROUP', 'CS3_GROUP', 'CS4_GROUP', 'CS5_GROUP']
        
        indicator_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Generate metadata
        metadata = {
            'total_observations': len(df),
            'countries': df['COUNTRY'].nunique(),
            'indicators': len(indicator_cols),
            'time_period': f"{df['YEAR'].min()}-{df['YEAR'].max()}",
            'crisis_years_included': include_crisis_years,
            'excluded_observations': excluded_count,
            'winsorized': True,
            'winsorization_level': 0.05  # 5% from each tail
        }
        
        return df, indicator_cols, metadata
        
    except Exception as e:
        logger.error(f"Error loading winsorized comprehensive data: {str(e)}")
        raise

def load_original_vs_winsorized_comparison(case_study: str = "CS1", 
                                         include_crisis_years: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both original and winsorized datasets for comparison
    
    Args:
        case_study: Case study identifier (CS1, CS2, etc.)
        include_crisis_years: Whether to include crisis periods
        
    Returns:
        Tuple of (original_data, winsorized_data)
    """
    try:
        data_dir = Path(__file__).parents[2] / "updated_data" / "Clean"
        
        # Load original data
        original_file = data_dir / "comprehensive_df_PGDP_labeled.csv"
        winsorized_file = data_dir / "comprehensive_df_PGDP_labeled_winsorized.csv"
        
        if not original_file.exists() or not winsorized_file.exists():
            raise FileNotFoundError("Original or winsorized dataset not found")
        
        df_original = pd.read_csv(original_file)
        df_winsorized = pd.read_csv(winsorized_file)
        
        # Apply crisis period filtering
        if not include_crisis_years:
            crisis_years = [2008, 2009, 2010, 2020, 2021, 2022]
            df_original = df_original[~df_original['YEAR'].isin(crisis_years)]
            df_winsorized = df_winsorized[~df_winsorized['YEAR'].isin(crisis_years)]
        
        logger.info(f"Loaded comparison datasets - Original: {len(df_original)} rows, Winsorized: {len(df_winsorized)} rows")
        
        return df_original, df_winsorized
        
    except Exception as e:
        logger.error(f"Error loading comparison data: {str(e)}")
        raise

def load_winsorized_cs4_data(dataset_type: str = "net_capital_flows", 
                           include_crisis_years: bool = True) -> pd.DataFrame:
    """
    Load winsorized CS4 statistical modeling data
    
    Args:
        dataset_type: Type of dataset (net_capital_flows, net_direct_investment, etc.)
        include_crisis_years: Whether to include crisis periods
        
    Returns:
        Winsorized CS4 dataset
    """
    try:
        data_dir = Path(__file__).parents[2] / "updated_data" / "Clean" / "CS4_Statistical_Modeling_winsorized"
        
        if include_crisis_years:
            filename = f"{dataset_type}_full.csv"
        else:
            filename = f"{dataset_type}_no_crises.csv"
        
        file_path = data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Winsorized CS4 dataset not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded winsorized CS4 {dataset_type}: {len(df)} rows")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading winsorized CS4 data: {str(e)}")
        raise

def load_winsorized_cs5_data(data_type: str = "capital_controls", 
                           dataset: str = "sd_yearly_flows") -> pd.DataFrame:
    """
    Load winsorized CS5 data (capital controls or regime analysis)
    
    Args:
        data_type: Either "capital_controls" or "regime_analysis" 
        dataset: Specific dataset name
        
    Returns:
        Winsorized CS5 dataset
    """
    try:
        if data_type == "capital_controls":
            data_dir = Path(__file__).parents[2] / "updated_data" / "Clean" / "CS5_Capital_Controls_winsorized"
        else:
            data_dir = Path(__file__).parents[2] / "updated_data" / "Clean" / "CS5_Regime_Analysis_winsorized"
        
        file_path = data_dir / f"{dataset}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Winsorized CS5 dataset not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded winsorized CS5 {data_type} {dataset}: {len(df)} rows")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading winsorized CS5 data: {str(e)}")
        raise

def calculate_winsorization_impact(df_original: pd.DataFrame, 
                                 df_winsorized: pd.DataFrame, 
                                 indicator_cols: List[str]) -> pd.DataFrame:
    """
    Calculate the impact of winsorization on key statistical measures
    
    Args:
        df_original: Original dataset
        df_winsorized: Winsorized dataset  
        indicator_cols: List of indicator columns to analyze
        
    Returns:
        DataFrame with winsorization impact statistics
    """
    try:
        impact_stats = []
        
        for indicator in indicator_cols:
            if indicator not in df_original.columns or indicator not in df_winsorized.columns:
                continue
                
            orig_values = df_original[indicator].dropna()
            wins_values = df_winsorized[indicator].dropna()
            
            if len(orig_values) == 0 or len(wins_values) == 0:
                continue
            
            # Calculate impact metrics
            impact = {
                'Indicator': indicator,
                'Original_Mean': orig_values.mean(),
                'Winsorized_Mean': wins_values.mean(),
                'Mean_Change_Pct': ((wins_values.mean() - orig_values.mean()) / orig_values.mean()) * 100,
                'Original_Std': orig_values.std(),
                'Winsorized_Std': wins_values.std(),
                'Std_Change_Pct': ((wins_values.std() - orig_values.std()) / orig_values.std()) * 100,
                'Original_Min': orig_values.min(),
                'Winsorized_Min': wins_values.min(),
                'Original_Max': orig_values.max(),
                'Winsorized_Max': wins_values.max(),
                'Values_Changed': (orig_values != wins_values).sum(),
                'Pct_Values_Changed': ((orig_values != wins_values).sum() / len(orig_values)) * 100
            }
            
            impact_stats.append(impact)
        
        impact_df = pd.DataFrame(impact_stats)
        
        # Round numeric columns
        numeric_cols = impact_df.select_dtypes(include=[np.number]).columns
        impact_df[numeric_cols] = impact_df[numeric_cols].round(3)
        
        logger.info(f"Calculated winsorization impact for {len(impact_stats)} indicators")
        
        return impact_df
        
    except Exception as e:
        logger.error(f"Error calculating winsorization impact: {str(e)}")
        raise

def load_winsorization_summary() -> pd.DataFrame:
    """
    Load the comprehensive winsorization summary statistics
    
    Returns:
        DataFrame with winsorization summary statistics
    """
    try:
        data_dir = Path(__file__).parents[2] / "updated_data" / "Clean"
        summary_file = data_dir / "comprehensive_df_PGDP_labeled_winsorization_summary.csv"
        
        if not summary_file.exists():
            logger.warning("Winsorization summary file not found")
            return pd.DataFrame()
        
        summary_df = pd.read_csv(summary_file)
        logger.info(f"Loaded winsorization summary: {len(summary_df)} indicators")
        
        return summary_df
        
    except Exception as e:
        logger.error(f"Error loading winsorization summary: {str(e)}")
        return pd.DataFrame()

def get_available_winsorized_datasets() -> Dict[str, List[str]]:
    """
    Get list of available winsorized datasets
    
    Returns:
        Dictionary mapping data categories to available datasets
    """
    try:
        data_dir = Path(__file__).parents[2] / "updated_data" / "Clean"
        
        available_datasets = {
            'Comprehensive': [],
            'CS4_Statistical_Modeling': [],
            'CS5_Capital_Controls': [],
            'CS5_Regime_Analysis': []
        }
        
        # Check comprehensive dataset
        if (data_dir / "comprehensive_df_PGDP_labeled_winsorized.csv").exists():
            available_datasets['Comprehensive'].append("comprehensive_df_PGDP_labeled_winsorized.csv")
        
        # Check CS4 datasets
        cs4_dir = data_dir / "CS4_Statistical_Modeling_winsorized"
        if cs4_dir.exists():
            available_datasets['CS4_Statistical_Modeling'] = [f.name for f in cs4_dir.glob("*.csv")]
        
        # Check CS5 Capital Controls datasets
        cs5_controls_dir = data_dir / "CS5_Capital_Controls_winsorized"
        if cs5_controls_dir.exists():
            available_datasets['CS5_Capital_Controls'] = [f.name for f in cs5_controls_dir.glob("*.csv")]
        
        # Check CS5 Regime Analysis datasets
        cs5_regime_dir = data_dir / "CS5_Regime_Analysis_winsorized"
        if cs5_regime_dir.exists():
            available_datasets['CS5_Regime_Analysis'] = [f.name for f in cs5_regime_dir.glob("*.csv")]
        
        total_datasets = sum(len(datasets) for datasets in available_datasets.values())
        logger.info(f"Found {total_datasets} available winsorized datasets")
        
        return available_datasets
        
    except Exception as e:
        logger.error(f"Error checking available datasets: {str(e)}")
        return {}

def validate_winsorized_data_integrity() -> Dict[str, bool]:
    """
    Validate integrity of winsorized datasets
    
    Returns:
        Dictionary with validation results for each dataset category
    """
    try:
        validation_results = {}
        
        # Check comprehensive dataset
        try:
            df_winsorized, _, _ = load_winsorized_comprehensive_data()
            validation_results['Comprehensive'] = len(df_winsorized) > 0
        except:
            validation_results['Comprehensive'] = False
        
        # Check CS4 datasets
        try:
            df_cs4 = load_winsorized_cs4_data("net_capital_flows")
            validation_results['CS4'] = len(df_cs4) > 0
        except:
            validation_results['CS4'] = False
        
        # Check CS5 datasets
        try:
            df_cs5 = load_winsorized_cs5_data("capital_controls", "sd_yearly_flows")
            validation_results['CS5'] = len(df_cs5) > 0
        except:
            validation_results['CS5'] = False
        
        logger.info(f"Data integrity validation completed: {sum(validation_results.values())}/{len(validation_results)} passed")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error validating data integrity: {str(e)}")
        return {}

if __name__ == "__main__":
    # Test the winsorized data loader
    print("Testing Winsorized Data Loader...")
    
    # Test comprehensive data loading
    try:
        df, indicators, metadata = load_winsorized_comprehensive_data()
        print(f"✓ Loaded comprehensive winsorized data: {len(df)} rows, {len(indicators)} indicators")
        print(f"  Metadata: {metadata}")
    except Exception as e:
        print(f"✗ Error loading comprehensive data: {e}")
    
    # Test data integrity
    validation = validate_winsorized_data_integrity()
    print(f"\n✓ Data integrity validation: {validation}")
    
    # Test available datasets
    available = get_available_winsorized_datasets()
    print(f"\n✓ Available datasets: {sum(len(v) for v in available.values())} total files")
    
    print("\nWinsorized Data Loader test completed!")