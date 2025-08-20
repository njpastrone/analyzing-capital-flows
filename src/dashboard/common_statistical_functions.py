"""
Common Statistical Functions - Unified statistical analysis functions for all case studies
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dashboard_config import SIGNIFICANCE_LEVELS

# ============================================================================
# F-TEST UTILITIES
# ============================================================================

def perform_f_test(series1: pd.Series, series2: pd.Series, 
                   group1_name: str = "Iceland", group2_name: str = "Comparator") -> Dict[str, any]:
    """
    Perform F-test for variance equality (homoscedasticity)
    
    Null Hypothesis: σ²(group1) = σ²(group2)
    Alternative: σ²(group1) ≠ σ²(group2)
    
    Args:
        series1: First time series (typically Iceland)
        series2: Second time series (comparator group)
        group1_name: Name of first group
        group2_name: Name of second group
        
    Returns:
        Dictionary with test results
    """
    try:
        # Remove NaN values
        s1 = series1.dropna()
        s2 = series2.dropna()
        
        if len(s1) < 2 or len(s2) < 2:
            return {
                'f_statistic': np.nan,
                'p_value': np.nan,
                'significance': '',
                'variance_1': np.nan,
                'variance_2': np.nan,
                'std_1': np.nan,
                'std_2': np.nan,
                'group_1': group1_name,
                'group_2': group2_name,
                'error': 'Insufficient data points'
            }
        
        # Calculate variances and standard deviations
        var1 = np.var(s1, ddof=1)  # Sample variance
        var2 = np.var(s2, ddof=1)
        std1 = np.std(s1, ddof=1)
        std2 = np.std(s2, ddof=1)
        
        # F-statistic (larger variance in numerator)
        if var1 >= var2:
            f_stat = var1 / var2
            df1, df2 = len(s1) - 1, len(s2) - 1
        else:
            f_stat = var2 / var1
            df1, df2 = len(s2) - 1, len(s1) - 1
        
        # Two-tailed p-value
        p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
        
        # Determine significance
        significance = format_significance(p_value)
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significance': significance,
            'variance_1': var1,
            'variance_2': var2,
            'std_1': std1,
            'std_2': std2,
            'group_1': group1_name,
            'group_2': group2_name,
            'error': None
        }
        
    except Exception as e:
        return {
            'f_statistic': np.nan,
            'p_value': np.nan,
            'significance': '',
            'variance_1': np.nan,
            'variance_2': np.nan,
            'std_1': np.nan,
            'std_2': np.nan,
            'group_1': group1_name,
            'group_2': group2_name,
            'error': str(e)
        }

def format_significance(p_value: float) -> str:
    """Format p-value with significance stars"""
    if pd.isna(p_value):
        return ''
    
    for alpha, symbol in SIGNIFICANCE_LEVELS.items():
        if p_value < alpha:
            return symbol
    return ''

def format_value_with_significance(value: float, p_value: float, decimals: int = 4) -> str:
    """Format numerical value with significance stars"""
    if pd.isna(value):
        return 'N/A'
    
    base_value = f"{value:.{decimals}f}"
    significance = format_significance(p_value)
    return base_value + significance

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

def calculate_comprehensive_stats(data: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive descriptive statistics"""
    clean_data = data.dropna()
    
    if len(clean_data) == 0:
        return {stat: np.nan for stat in ['count', 'mean', 'std', 'min', 'q25', 'median', 'q75', 'max', 'skew', 'kurt']}
    
    return {
        'count': len(clean_data),
        'mean': clean_data.mean(),
        'std': clean_data.std(),
        'min': clean_data.min(),
        'q25': clean_data.quantile(0.25),
        'median': clean_data.median(),
        'q75': clean_data.quantile(0.75),
        'max': clean_data.max(),
        'skew': clean_data.skew(),
        'kurt': clean_data.kurtosis()
    }

def calculate_group_statistics(df: pd.DataFrame, group_col: str, value_cols: List[str]) -> pd.DataFrame:
    """Calculate statistics by group for multiple value columns"""
    results = []
    
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group]
        
        for col in value_cols:
            stats_dict = calculate_comprehensive_stats(group_data[col])
            stats_dict['group'] = group
            stats_dict['indicator'] = col
            results.append(stats_dict)
    
    return pd.DataFrame(results)

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def calculate_correlation(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Calculate Pearson correlation coefficient with p-value"""
    try:
        # Remove NaN values
        df_clean = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(df_clean) < 2:
            return np.nan, np.nan
        
        corr, p_value = stats.pearsonr(df_clean['x'], df_clean['y'])
        return corr, p_value
        
    except Exception:
        return np.nan, np.nan

# ============================================================================
# OUTLIER DETECTION
# ============================================================================

def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (data < lower_bound) | (data > upper_bound)

def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(data.dropna()))
    return pd.Series(z_scores > threshold, index=data.dropna().index).reindex(data.index, fill_value=False)

def remove_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr', **kwargs) -> pd.DataFrame:
    """Remove outliers from specified columns"""
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            outliers = detect_outliers_iqr(df_clean[col], **kwargs)
        elif method == 'zscore':
            outliers = detect_outliers_zscore(df_clean[col], **kwargs)
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        df_clean = df_clean[~outliers]
    
    return df_clean

# ============================================================================
# TIME SERIES UTILITIES
# ============================================================================

def filter_crisis_periods(df: pd.DataFrame, year_col: str = 'YEAR', 
                         crisis_periods: List[Tuple[int, int]] = None) -> pd.DataFrame:
    """Filter out crisis periods from data"""
    if crisis_periods is None:
        from dashboard_config import CRISIS_PERIODS
        crisis_periods = [CRISIS_PERIODS['gfc'], CRISIS_PERIODS['covid']]
    
    df_filtered = df.copy()
    
    for start_year, end_year in crisis_periods:
        df_filtered = df_filtered[~((df_filtered[year_col] >= start_year) & (df_filtered[year_col] <= end_year))]
    
    return df_filtered

def create_before_after_split(df: pd.DataFrame, split_year: int, year_col: str = 'YEAR') -> Dict[str, pd.DataFrame]:
    """Split data into before and after periods around a specific year"""
    before = df[df[year_col] < split_year].copy()
    after = df[df[year_col] >= split_year].copy()
    
    return {'before': before, 'after': after}

# ============================================================================
# HYPOTHESIS TESTING
# ============================================================================

def perform_t_test(group1: pd.Series, group2: pd.Series, 
                   equal_var: bool = True) -> Dict[str, float]:
    """Perform two-sample t-test"""
    try:
        clean_group1 = group1.dropna()
        clean_group2 = group2.dropna()
        
        if len(clean_group1) < 2 or len(clean_group2) < 2:
            return {'t_statistic': np.nan, 'p_value': np.nan}
        
        t_stat, p_val = stats.ttest_ind(clean_group1, clean_group2, equal_var=equal_var)
        return {'t_statistic': t_stat, 'p_value': p_val}
        
    except Exception:
        return {'t_statistic': np.nan, 'p_value': np.nan}

def perform_levene_test(*groups) -> Dict[str, float]:
    """Perform Levene test for equality of variances"""
    try:
        clean_groups = [group.dropna() for group in groups if len(group.dropna()) > 1]
        
        if len(clean_groups) < 2:
            return {'levene_statistic': np.nan, 'p_value': np.nan}
        
        stat, p_val = stats.levene(*clean_groups)
        return {'levene_statistic': stat, 'p_value': p_val}
        
    except Exception:
        return {'levene_statistic': np.nan, 'p_value': np.nan}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_division(numerator: float, denominator: float, default: float = np.nan) -> float:
    """Safely perform division with default for zero denominator"""
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator
    except:
        return default

def validate_data_requirements(data: pd.Series, min_observations: int = 2) -> bool:
    """Validate that data meets minimum requirements for analysis"""
    clean_data = data.dropna()
    return len(clean_data) >= min_observations and clean_data.std() > 0

def prepare_data_for_analysis(df: pd.DataFrame, value_cols: List[str], 
                            remove_outliers_flag: bool = False, 
                            filter_crisis: bool = False) -> pd.DataFrame:
    """Prepare data for analysis with optional outlier removal and crisis filtering"""
    df_prepared = df.copy()
    
    if filter_crisis:
        df_prepared = filter_crisis_periods(df_prepared)
    
    if remove_outliers_flag:
        df_prepared = remove_outliers(df_prepared, value_cols)
    
    return df_prepared