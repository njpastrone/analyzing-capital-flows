"""
Centralized Data Loading Utilities for Capital Flows Analysis

This module provides unified data loading functions for Case Studies 1-3,
eliminating duplication across report files while maintaining flexibility.

Version: 0.2.0-phase2
"""

import pandas as pd
import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dashboard_config import get_data_paths
from config.constants import (
    CRISIS_YEARS_LIST,
    sort_indicators_by_type
)


# ==================== CS2 Helper Functions ====================

def _create_expanded_euro_adoption_timeline():
    """
    Create expanded timeline with maximized data periods (including adoption years in post-Euro).

    Used internally for CS2 Euro adoption period labeling.
    """
    return {
        'Estonia, Republic of': {
            'adoption_date': '2011-01-01',
            'adoption_year': 2011,
            'pre_period_full': (1999, 2010),
            'post_period_full': (2011, 2025),              # Include 2011 adoption year
            'pre_period_crisis_excluded': (1999, 2007),    # Excludes 2008-2010
            'post_period_crisis_excluded': (2011, 2025),   # Include 2011, excludes 2020-2022, includes 2023-2025
            'crisis_years': [2008, 2009, 2010, 2020, 2021, 2022]
        },
        'Latvia, Republic of': {
            'adoption_date': '2014-01-01',
            'adoption_year': 2014,
            'pre_period_full': (1999, 2013),
            'post_period_full': (2014, 2025),              # Include 2014 adoption year
            'pre_period_crisis_excluded': (1999, 2013),    # Include 2013 - crisis years (2008-2012) filtered by non_crisis_mask
            'post_period_crisis_excluded': (2014, 2025),   # Include 2014, excludes 2020-2022, includes 2023-2025
            'crisis_years': [2008, 2009, 2010, 2011, 2012, 2020, 2021, 2022]  # Add Latvian Banking Crisis (2011-2012)
        },
        'Lithuania, Republic of': {
            'adoption_date': '2015-01-01',
            'adoption_year': 2015,
            'pre_period_full': (1999, 2014),
            'post_period_full': (2015, 2025),              # Include 2015 adoption year
            'pre_period_crisis_excluded': (1999, 2014),    # Excludes 2008-2010 within range
            'post_period_crisis_excluded': (2015, 2025),   # Include 2015, excludes 2020-2022, includes 2023-2025
            'crisis_years': [2008, 2009, 2010, 2020, 2021, 2022]
        }
    }


def _create_euro_periods(data, include_crisis_years=True):
    """
    Create Euro adoption period labels for CS2 dataset.

    Adds a period column (EURO_PERIOD_FULL or EURO_PERIOD_CRISIS_EXCLUDED)
    labeling observations as 'Pre-Euro' or 'Post-Euro' based on country-specific
    adoption dates.

    Parameters
    ----------
    data : DataFrame
        CS2 dataset with COUNTRY and YEAR columns
    include_crisis_years : bool
        If True, use full periods; if False, use crisis-excluded periods

    Returns
    -------
    DataFrame
        Data with added Euro period column
    """
    timeline = _create_expanded_euro_adoption_timeline()

    # Create period column based on crisis inclusion
    period_col = 'EURO_PERIOD_FULL' if include_crisis_years else 'EURO_PERIOD_CRISIS_EXCLUDED'
    data[period_col] = 'Unknown'

    for country, info in timeline.items():
        country_mask = data['COUNTRY'] == country

        if include_crisis_years:
            # Full series: use pre_period_full and post_period_full
            pre_start, pre_end = info['pre_period_full']
            post_start, post_end = info['post_period_full']

            pre_mask = country_mask & (data['YEAR'] >= pre_start) & (data['YEAR'] <= pre_end)
            post_mask = country_mask & (data['YEAR'] >= post_start) & (data['YEAR'] <= post_end)
        else:
            # Crisis-excluded: use pre_period_crisis_excluded and post_period_crisis_excluded
            pre_start, pre_end = info['pre_period_crisis_excluded']
            post_start, post_end = info['post_period_crisis_excluded']

            # Exclude crisis years
            crisis_years = info['crisis_years']
            non_crisis_mask = ~data['YEAR'].isin(crisis_years)

            pre_mask = country_mask & (data['YEAR'] >= pre_start) & (data['YEAR'] <= pre_end) & non_crisis_mask
            post_mask = country_mask & (data['YEAR'] >= post_start) & (data['YEAR'] <= post_end) & non_crisis_mask

        data.loc[pre_mask, period_col] = 'Pre-Euro'
        data.loc[post_mask, period_col] = 'Post-Euro'

    return data


# ==================== Public API Functions ====================


def load_case_study_data(
    case_study: int,
    analysis_type: str = 'full',
    include_crisis_years: bool = True
) -> tuple:
    """
    Load data for Case Studies 1, 2, or 3 with unified interface.

    This function provides a centralized data loading pattern that handles:
    - Full vs winsorized (outlier-adjusted) datasets
    - Crisis year filtering
    - Case study-specific processing
    - Indicator standardization

    Parameters
    ----------
    case_study : int
        Case study number (1, 2, or 3)
        - 1: Iceland vs Eurozone comparison
        - 2: Baltic Euro adoption analysis
        - 3: Iceland vs Small Open Economies
    analysis_type : str, default='full'
        Type of analysis dataset to load
        - 'full': Original unmodified data
        - 'winsorized': Outlier-adjusted data (5th-95th percentile capping)
    include_crisis_years : bool, default=True
        Whether to include crisis periods in analysis
        - True: Include all years
        - False: Exclude GFC (2008-2010) and COVID (2020-2022)

    Returns
    -------
    tuple
        (data, indicators, metadata) where:
        - data (DataFrame): Processed dataset ready for analysis
        - indicators (list): List of indicator column names
        - metadata (dict): Information about data processing

    Raises
    ------
    ValueError
        If case_study not in [1, 2, 3] or analysis_type not in ['full', 'winsorized']
    FileNotFoundError
        If required data files cannot be found

    Examples
    --------
    >>> # Load CS1 full dataset with crisis years
    >>> data, indicators, meta = load_case_study_data(1, 'full', True)

    >>> # Load CS2 winsorized dataset without crisis years
    >>> data, indicators, meta = load_case_study_data(2, 'winsorized', False)

    Notes
    -----
    - CS1: Removes Luxembourg from analysis (per original methodology)
    - CS2: Creates Euro adoption period labels
    - CS3: Simpler processing with no country exclusions
    - All case studies apply consistent indicator naming and sorting
    """
    # Validate inputs
    if case_study not in [1, 2, 3]:
        raise ValueError(f"case_study must be 1, 2, or 3. Got: {case_study}")

    if analysis_type not in ['full', 'winsorized']:
        raise ValueError(f"analysis_type must be 'full' or 'winsorized'. Got: {analysis_type}")

    # Route to case study-specific loader
    if case_study == 1:
        return _load_cs1_data(analysis_type, include_crisis_years)
    elif case_study == 2:
        return _load_cs2_data(analysis_type, include_crisis_years)
    elif case_study == 3:
        return _load_cs3_data(analysis_type, include_crisis_years)


def _load_cs1_data(analysis_type: str, include_crisis_years: bool) -> tuple:
    """Load Case Study 1: Iceland vs Eurozone data."""
    # Get data paths for specified analysis type
    data_paths = get_data_paths(analysis_type)
    comprehensive_file = data_paths['master_dataset']

    if not comprehensive_file.exists():
        raise FileNotFoundError(f"Data file not found: {comprehensive_file}")

    # Load comprehensive labeled data
    comprehensive_df = pd.read_csv(comprehensive_file)
    original_shape = comprehensive_df.shape

    # Filter for Case Study 1 data (CS1_GROUP not null)
    case_one_data = comprehensive_df[comprehensive_df['CS1_GROUP'].notna()].copy()

    # Remove Luxembourg as per original analysis
    final_data = case_one_data[case_one_data['COUNTRY'] != 'Luxembourg'].copy()
    filtered_shape = final_data.shape

    # Apply crisis filtering if requested
    excluded_observations = 0
    if not include_crisis_years:
        # Use centralized crisis years definition: GFC (2008-2010) + COVID (2020-2022)
        crisis_years = CRISIS_YEARS_LIST

        # Filter out crisis years
        original_count = len(final_data)
        final_data = final_data[~final_data['YEAR'].isin(crisis_years)].copy()
        excluded_observations = original_count - len(final_data)

    # Create GROUP column for analysis
    final_data['GROUP'] = final_data['CS1_GROUP'].copy()

    # Get PGDP indicator columns (% of GDP normalized)
    pgdp_columns = [col for col in final_data.columns if col.endswith('_PGDP')]

    # Exclude discontinued indicators
    exclude_patterns = ['Financial derivatives', 'Financial account balance']
    analysis_indicators = [
        col for col in pgdp_columns
        if not any(pattern in col for pattern in exclude_patterns)
    ]

    # Apply standardized indicator naming
    final_data = _standardize_indicator_names(final_data)

    # Update analysis indicators list to match renamed columns
    # Use the same rename map as _standardize_indicator_names()
    rename_map = {
        'Net (net acquisition of financial assets less net incurrence of liabilities) - Direct investment, Total financial assets/liabilities_PGDP':
            'Net - Direct investment, Total financial assets/liabilities_PGDP',
        'Net (net acquisition of financial assets less net incurrence of liabilities) - Portfolio investment, Total financial assets/liabilities_PGDP':
            'Net - Portfolio investment, Total financial assets/liabilities_PGDP',
        'Net (net acquisition of financial assets less net incurrence of liabilities) - Other investment, Total financial assets/liabilities_PGDP':
            'Net - Other investment, Total financial assets/liabilities_PGDP'
    }
    analysis_indicators = [rename_map.get(col, col) for col in analysis_indicators]

    # Sort indicators using centralized sorting function
    analysis_indicators = sort_indicators_by_type(analysis_indicators)

    # Build metadata
    metadata = {
        'case_study': 1,
        'analysis_type': analysis_type,
        'original_shape': original_shape,
        'filtered_shape': filtered_shape,
        'final_shape': final_data.shape,
        'n_indicators': len(analysis_indicators),
        'study_version': 'Full Time Period' if include_crisis_years else 'Crisis-Excluded',
        'include_crisis_years': include_crisis_years,
        'excluded_observations': excluded_observations,
        'countries_excluded': ['Luxembourg'],
        'is_outlier_adjusted': (analysis_type == 'winsorized')
    }

    return final_data, analysis_indicators, metadata


def _load_cs2_data(analysis_type: str, include_crisis_years: bool) -> tuple:
    """Load Case Study 2: Baltic Euro Adoption data."""
    # Get data paths for specified analysis type
    data_paths = get_data_paths(analysis_type)
    comprehensive_file = data_paths['master_dataset']

    if not comprehensive_file.exists():
        raise FileNotFoundError(f"Data file not found: {comprehensive_file}")

    # Load comprehensive labeled data
    comprehensive_df = pd.read_csv(comprehensive_file)
    original_shape = comprehensive_df.shape

    # Filter for Case Study 2 data (Baltic countries)
    final_data = comprehensive_df[comprehensive_df['CS2_GROUP'].notna()].copy()

    # Create Euro adoption periods using internal helper
    final_data = _create_euro_periods(final_data, include_crisis_years)

    # Get PGDP indicator columns
    pgdp_columns = [col for col in final_data.columns if col.endswith('_PGDP')]

    # Exclude discontinued indicators
    exclude_patterns = ['Financial derivatives', 'Financial account balance']
    analysis_indicators = [
        col for col in pgdp_columns
        if not any(pattern in col for pattern in exclude_patterns)
    ]

    # Apply standardized indicator naming
    final_data = _standardize_indicator_names(final_data)

    # Update analysis indicators list to match renamed columns
    rename_map = {
        'Net (net acquisition of financial assets less net incurrence of liabilities) - Direct investment, Total financial assets/liabilities_PGDP':
            'Net - Direct investment, Total financial assets/liabilities_PGDP',
        'Net (net acquisition of financial assets less net incurrence of liabilities) - Portfolio investment, Total financial assets/liabilities_PGDP':
            'Net - Portfolio investment, Total financial assets/liabilities_PGDP',
        'Net (net acquisition of financial assets less net incurrence of liabilities) - Other investment, Total financial assets/liabilities_PGDP':
            'Net - Other investment, Total financial assets/liabilities_PGDP'
    }
    analysis_indicators = [rename_map.get(col, col) for col in analysis_indicators]

    # Sort indicators
    analysis_indicators = sort_indicators_by_type(analysis_indicators)

    # Build metadata
    metadata = {
        'case_study': 2,
        'analysis_type': analysis_type,
        'original_shape': original_shape,
        'final_shape': final_data.shape,
        'n_indicators': len(analysis_indicators),
        'study_version': 'Full Series' if include_crisis_years else 'Crisis-Excluded',
        'include_crisis_years': include_crisis_years,
        'countries': ['Estonia, Republic of', 'Latvia, Republic of', 'Lithuania, Republic of'],
        'euro_adoption_years': {'Estonia': 2011, 'Latvia': 2014, 'Lithuania': 2015},
        'is_outlier_adjusted': (analysis_type == 'winsorized')
    }

    return final_data, analysis_indicators, metadata


def _load_cs3_data(analysis_type: str, include_crisis_years: bool) -> tuple:
    """Load Case Study 3: Iceland vs Small Open Economies data."""
    # Get data paths for specified analysis type
    data_paths = get_data_paths(analysis_type)
    comprehensive_file = data_paths['master_dataset']

    if not comprehensive_file.exists():
        raise FileNotFoundError(f"Data file not found: {comprehensive_file}")

    # Load comprehensive labeled data
    comprehensive_df = pd.read_csv(comprehensive_file)
    original_shape = comprehensive_df.shape

    # Filter for Case Study 3 data (CS3_GROUP not null)
    cs3_data = comprehensive_df[comprehensive_df['CS3_GROUP'].notna()].copy()

    if len(cs3_data) == 0:
        raise ValueError("No Case Study 3 data found in dataset")

    # Apply crisis filtering if requested
    excluded_observations = 0
    if not include_crisis_years:
        # Define crisis years: GFC (2008-2010) + COVID (2020-2022)
        crisis_years = CRISIS_YEARS_LIST

        # Filter out crisis years
        original_count = len(cs3_data)
        cs3_data = cs3_data[~cs3_data['YEAR'].isin(crisis_years)].copy()
        excluded_observations = original_count - len(cs3_data)

    # Create GROUP column for analysis
    cs3_data['GROUP'] = cs3_data['CS3_GROUP'].apply(
        lambda x: 'Iceland' if x == 'Iceland' else 'Small Open Economies'
    )

    # Get PGDP indicator columns
    pgdp_columns = [col for col in cs3_data.columns if col.endswith('_PGDP')]

    # Exclude discontinued indicators
    exclude_patterns = ['Financial derivatives', 'Financial account balance']
    analysis_indicators = [
        col for col in pgdp_columns
        if not any(pattern in col for pattern in exclude_patterns)
    ]

    # Apply standardized indicator naming
    cs3_data = _standardize_indicator_names(cs3_data)

    # Update analysis indicators list to match renamed columns
    rename_map = {
        'Net (net acquisition of financial assets less net incurrence of liabilities) - Direct investment, Total financial assets/liabilities_PGDP':
            'Net - Direct investment, Total financial assets/liabilities_PGDP',
        'Net (net acquisition of financial assets less net incurrence of liabilities) - Portfolio investment, Total financial assets/liabilities_PGDP':
            'Net - Portfolio investment, Total financial assets/liabilities_PGDP',
        'Net (net acquisition of financial assets less net incurrence of liabilities) - Other investment, Total financial assets/liabilities_PGDP':
            'Net - Other investment, Total financial assets/liabilities_PGDP'
    }
    analysis_indicators = [rename_map.get(col, col) for col in analysis_indicators]

    # Sort indicators
    analysis_indicators = sort_indicators_by_type(analysis_indicators)

    # Build metadata
    metadata = {
        'case_study': 3,
        'analysis_type': analysis_type,
        'original_shape': original_shape,
        'final_shape': cs3_data.shape,
        'n_indicators': len(analysis_indicators),
        'study_version': 'Full Time Period' if include_crisis_years else 'Crisis-Excluded',
        'include_crisis_years': include_crisis_years,
        'excluded_observations': excluded_observations,
        'comparator_group': 'Small Open Economies',
        'soe_countries': ['Aruba', 'Bahamas', 'Brunei Darussalam', 'Malta', 'Mauritius', 'Seychelles'],
        'countries_excluded': ['Bermuda (missing GDP data)'],
        'is_outlier_adjusted': (analysis_type == 'winsorized')
    }

    return cs3_data, analysis_indicators, metadata


def _standardize_indicator_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply consistent indicator naming across all case studies.

    Renames long "Net (net acquisition of financial assets less net incurrence
    of liabilities)" indicators to shorter "Net -" format for consistency.

    Parameters
    ----------
    data : DataFrame
        Data with original indicator column names

    Returns
    -------
    DataFrame
        Data with standardized indicator column names
    """
    # Define indicator name mappings
    indicator_renames = {
        'Net (net acquisition of financial assets less net incurrence of liabilities) - Direct investment, Total financial assets/liabilities_PGDP':
            'Net - Direct investment, Total financial assets/liabilities_PGDP',
        'Net (net acquisition of financial assets less net incurrence of liabilities) - Portfolio investment, Total financial assets/liabilities_PGDP':
            'Net - Portfolio investment, Total financial assets/liabilities_PGDP',
        'Net (net acquisition of financial assets less net incurrence of liabilities) - Other investment, Total financial assets/liabilities_PGDP':
            'Net - Other investment, Total financial assets/liabilities_PGDP'
    }

    # Apply renames (only rename columns that exist)
    existing_renames = {old: new for old, new in indicator_renames.items() if old in data.columns}
    if existing_renames:
        data = data.rename(columns=existing_renames)

    return data


def load_overall_capital_flows_data(
    analysis_type: str = 'full',
    include_crisis_years: bool = True
) -> tuple:
    """
    Load data for overall capital flows analysis (4 aggregate indicators).

    This function loads the main capital flow categories for CS1 analysis:
    - Net Direct Investment
    - Net Portfolio Investment
    - Net Other Investment
    - Net Capital Flows (computed as sum of above three)

    Parameters
    ----------
    analysis_type : str, default='full'
        'full' or 'winsorized'
    include_crisis_years : bool, default=True
        Whether to include crisis periods

    Returns
    -------
    tuple
        (data, indicators_mapping) where:
        - data (DataFrame): Processed dataset with computed Net Capital Flows
        - indicators_mapping (dict): Maps display names to column names

    Examples
    --------
    >>> data, mapping = load_overall_capital_flows_data('full', True)
    >>> print(mapping.keys())
    dict_keys(['Net Direct Investment', 'Net Portfolio Investment',
               'Net Other Investment', 'Net Capital Flows'])
    """
    # Load CS1 data using centralized function
    final_data, all_indicators, metadata = load_case_study_data(
        case_study=1,
        analysis_type=analysis_type,
        include_crisis_years=include_crisis_years
    )

    # Define the 4 main capital flow indicators
    indicators_mapping = {
        'Net Direct Investment': 'Net - Direct investment, Total financial assets/liabilities_PGDP',
        'Net Portfolio Investment': 'Net - Portfolio investment, Total financial assets/liabilities_PGDP',
        'Net Other Investment': 'Net - Other investment, Total financial assets/liabilities_PGDP'
    }

    # Compute Net Capital Flows as sum of the three components
    final_data['Net Capital Flows_PGDP'] = (
        final_data[indicators_mapping['Net Direct Investment']] +
        final_data[indicators_mapping['Net Portfolio Investment']] +
        final_data[indicators_mapping['Net Other Investment']]
    )

    # Add to mapping
    indicators_mapping['Net Capital Flows'] = 'Net Capital Flows_PGDP'

    return final_data, indicators_mapping


# Module version information
__version__ = "0.2.0-phase2"
__all__ = [
    'load_case_study_data',
    'load_overall_capital_flows_data'
]
