"""
Minimal Statistical Functions Extracted from Dashboard
~150 lines of pure statistics from 19,506 lines of code

Sources:
- cs1_report.py: F-test calculations (lines 240-248)
- cs4_statistical_analysis.py: AR(4) models (lines 254-294)
- cs2_shared_functions.py: Euro adoption dates (lines 33-61)
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg
import warnings

def calculate_f_statistic(group1_data, group2_data, group1_name="Group1", group2_name="Group2"):
    """
    F-test for equality of variances between two groups.

    Source: cs1_report.py lines 240-248

    Parameters
    ----------
    group1_data : array-like
        First group's data
    group2_data : array-like
        Second group's data
    group1_name : str
        Name of first group for reporting
    group2_name : str
        Name of second group for reporting

    Returns
    -------
    dict
        Contains f_statistic, p_value, var1, var2, n1, n2
    """
    # Remove NaN values
    g1_clean = pd.Series(group1_data).dropna()
    g2_clean = pd.Series(group2_data).dropna()

    # Check sufficient data
    if len(g1_clean) < 2 or len(g2_clean) < 2:
        return {
            'f_statistic': np.nan,
            'p_value': np.nan,
            'var1': np.nan,
            'var2': np.nan,
            'n1': len(g1_clean),
            'n2': len(g2_clean),
            'error': 'Insufficient data'
        }

    # Calculate variances
    var1 = g1_clean.var()
    var2 = g2_clean.var()

    # F-statistic (always put larger variance in numerator for two-tailed test)
    if var2 == 0:
        f_stat = np.inf if var1 > 0 else 1.0
        p_value = 0.0 if var1 > 0 else 1.0
    else:
        f_stat = var1 / var2
        df1 = len(g1_clean) - 1
        df2 = len(g2_clean) - 1

        # Two-tailed p-value
        p_value = 2 * min(stats.f.cdf(f_stat, df1, df2),
                          1 - stats.f.cdf(f_stat, df1, df2))

    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'var1': var1,
        'var2': var2,
        'n1': len(g1_clean),
        'n2': len(g2_clean),
        'group1': group1_name,
        'group2': group2_name
    }

def fit_ar4_model(series):
    """
    Fit AR(4) model to time series.

    Source: cs4_statistical_analysis.py lines 254-294

    Parameters
    ----------
    series : array-like
        Time series data

    Returns
    -------
    dict or None
        Model results including coefficients, AIC, BIC
    """
    # Clean series
    clean_series = pd.Series(series).dropna()

    # Need at least 8 observations for AR(4)
    if len(clean_series) < 8:
        return None

    try:
        # Fit AR(4) model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AutoReg(clean_series, lags=4, trend='c')
            fitted_model = model.fit()

        # Extract coefficients (excluding constant)
        ar_coeffs = fitted_model.params[1:5].values

        return {
            'coefficients': ar_coeffs,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'n_obs': len(clean_series)
        }
    except Exception as e:
        return {'error': str(e)}

def calculate_temporal_change(pre_data, post_data, indicator_name=""):
    """
    Calculate before/after statistics for temporal analysis (CS2).

    Parameters
    ----------
    pre_data : array-like
        Pre-period data
    post_data : array-like
        Post-period data
    indicator_name : str
        Name of indicator for reporting

    Returns
    -------
    dict
        Pre/post statistics including variance change
    """
    pre_clean = pd.Series(pre_data).dropna()
    post_clean = pd.Series(post_data).dropna()

    pre_var = pre_clean.var() if len(pre_clean) > 1 else np.nan
    post_var = post_clean.var() if len(post_clean) > 1 else np.nan

    # Calculate percentage change
    if pre_var > 0:
        var_change_pct = ((post_var - pre_var) / pre_var) * 100
        var_ratio = post_var / pre_var
    else:
        var_change_pct = np.nan
        var_ratio = np.nan

    return {
        'indicator': indicator_name,
        'pre_mean': pre_clean.mean() if len(pre_clean) > 0 else np.nan,
        'post_mean': post_clean.mean() if len(post_clean) > 0 else np.nan,
        'pre_var': pre_var,
        'post_var': post_var,
        'pre_n': len(pre_clean),
        'post_n': len(post_clean),
        'var_ratio': var_ratio,
        'var_change_pct': var_change_pct,
        'volatility_decreased': post_var < pre_var if not np.isnan(post_var) else None
    }

# Constants from cs2_shared_functions.py lines 33-61
EURO_ADOPTION_DATES = {
    'Estonia, Republic of': 2011,
    'Latvia, Republic of': 2014,
    'Lithuania, Republic of': 2015
}

# Crisis years for optional exclusion
CRISIS_YEARS = {
    'GFC': [2008, 2009, 2010],
    'COVID': [2020, 2021, 2022],
    'ALL': [2008, 2009, 2010, 2020, 2021, 2022]
}

def get_significance_stars(p_value):
    """
    Convert p-value to significance stars.

    Returns
    -------
    str
        '***' if p < 0.01, '**' if p < 0.05, '*' if p < 0.10, '' otherwise
    """
    if p_value < 0.01:
        return '***'
    elif p_value < 0.05:
        return '**'
    elif p_value < 0.10:
        return '*'
    else:
        return ''