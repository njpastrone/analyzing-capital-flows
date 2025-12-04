"""
Shared functions for CS2 Baltic Countries Euro Adoption Analysis
Used by cs2_estonia_report.py, cs2_latvia_report.py, cs2_lithuania_report.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from pathlib import Path
import sys
from datetime import datetime

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

# Import centralized configuration and constants
from dashboard_config import get_data_paths, COLORBLIND_SAFE
from config.constants import (
    CRISIS_YEARS_LIST,
    get_indicator_nickname,
    get_investment_type_sort_key,
    sort_indicators_by_type
)

def create_euro_adoption_timeline():
    """Define Euro adoption dates and analysis periods for all Baltic countries"""
    return {
        'Estonia, Republic of': {
            'adoption_date': '2011-01-01',
            'adoption_year': 2011,
            'pre_period_full': (1999, 2010),
            'post_period_full': (2011, 2025),
            'pre_period_crisis_excluded': (1999, 2007),
            'post_period_crisis_excluded': (2011, 2025),
            'crisis_years': [2008, 2009, 2010, 2020, 2021, 2022]
        },
        'Latvia, Republic of': {
            'adoption_date': '2014-01-01',
            'adoption_year': 2014,
            'pre_period_full': (1999, 2013),
            'post_period_full': (2014, 2025),
            'pre_period_crisis_excluded': (1999, 2013),
            'post_period_crisis_excluded': (2014, 2025),
            'crisis_years': [2008, 2009, 2010, 2011, 2012, 2020, 2021, 2022]
        },
        'Lithuania, Republic of': {
            'adoption_date': '2015-01-01',
            'adoption_year': 2015,
            'pre_period_full': (1999, 2014),
            'post_period_full': (2015, 2025),
            'pre_period_crisis_excluded': (1999, 2014),
            'post_period_crisis_excluded': (2015, 2025),
            'crisis_years': [2008, 2009, 2010, 2020, 2021, 2022]
        }
    }

def create_euro_periods(data, include_crisis_years=True):
    """Create Euro adoption period labels for comprehensive dataset"""
    timeline = create_euro_adoption_timeline()

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

def get_country_specific_crisis_text(country):
    """Get crisis exclusion text for a specific country"""
    timeline = create_euro_adoption_timeline()

    if country in timeline:
        crisis_years = timeline[country]['crisis_years']

        crisis_labels = []
        if any(year in crisis_years for year in [2008, 2009, 2010]):
            crisis_labels.append("GFC (2008-2010)")
        if any(year in crisis_years for year in [2011, 2012]) and country == 'Latvia, Republic of':
            crisis_labels.append("Latvian Banking Crisis (2011-2012)")
        if any(year in crisis_years for year in [2020, 2021, 2022]):
            crisis_labels.append("COVID (2020-2022)")

        if crisis_labels:
            return " + ".join(crisis_labels)
        else:
            return "No crisis periods"
    else:
        return "GFC (2008-2010) + COVID (2020-2022)"

def add_country_specific_crisis_shading(ax, country, include_labels=False):
    """Add country-specific crisis period shading to time series charts"""
    timeline = create_euro_adoption_timeline()

    if country in timeline and timeline[country]['crisis_years']:
        crisis_years = timeline[country]['crisis_years']

        # Define crisis periods and their colors
        crisis_periods = {
            'GFC': {'years': [2008, 2009, 2010], 'color': 'red', 'label': 'GFC (excluded)'},
            'COVID': {'years': [2020, 2021, 2022], 'color': 'orange', 'label': 'COVID (excluded)'}
        }

        # Add Latvia-specific banking crisis
        if country == 'Latvia, Republic of':
            crisis_periods['Latvian Banking'] = {
                'years': [2011, 2012],
                'color': 'purple',
                'label': 'Latvian Banking Crisis (excluded)'
            }

        # Add shading for each crisis period that affects this country
        for crisis_name, crisis_info in crisis_periods.items():
            crisis_years_set = set(crisis_info['years'])
            country_crisis_years_set = set(crisis_years)

            # Only add shading if this crisis affects this country
            if crisis_years_set.intersection(country_crisis_years_set):
                start_year = min(crisis_info['years'])
                end_year = max(crisis_info['years'])

                ax.axvspan(
                    pd.to_datetime(f'{start_year}-01-01'),
                    pd.to_datetime(f'{end_year}-12-31'),
                    alpha=0.15,
                    color=crisis_info['color'],
                    label=crisis_info['label'] if include_labels else ""
                )
    else:
        # Default crisis shading for countries not in timeline
        ax.axvspan(pd.to_datetime('2008-01-01'), pd.to_datetime('2010-12-31'),
                  alpha=0.15, color='red', label='GFC (excluded)' if include_labels else "")
        ax.axvspan(pd.to_datetime('2020-01-01'), pd.to_datetime('2022-12-31'),
                  alpha=0.15, color='orange', label='COVID (excluded)' if include_labels else "")

def calculate_temporal_statistics(data, country, indicators, period_column='EURO_PERIOD'):
    """Calculate pre/post Euro statistics for specified indicators"""
    country_data = data[data['COUNTRY'] == country].copy()

    stats = {}
    for indicator in indicators:
        indicator_data = country_data[country_data['INDICATOR'] == indicator].copy()

        pre_euro_data = indicator_data[indicator_data[period_column] == 'Pre-Euro']['VALUE']
        post_euro_data = indicator_data[indicator_data[period_column] == 'Post-Euro']['VALUE']

        stats[indicator] = {
            'Pre-Euro': {
                'mean': pre_euro_data.mean() if len(pre_euro_data) > 0 else np.nan,
                'std': pre_euro_data.std() if len(pre_euro_data) > 0 else np.nan,
                'count': len(pre_euro_data)
            },
            'Post-Euro': {
                'mean': post_euro_data.mean() if len(post_euro_data) > 0 else np.nan,
                'std': post_euro_data.std() if len(post_euro_data) > 0 else np.nan,
                'count': len(post_euro_data)
            }
        }

    return stats

def create_temporal_boxplot_data(data, country, indicators, period_column='EURO_PERIOD'):
    """Create data for temporal boxplots showing pre/post Euro volatility"""
    boxplot_data = []

    for indicator in indicators:
        indicator_data = data[(data['COUNTRY'] == country) &
                             (data['INDICATOR'] == indicator)].copy()

        for period in ['Pre-Euro', 'Post-Euro']:
            period_data = indicator_data[indicator_data[period_column] == period]['VALUE']

            if len(period_data) > 0:
                for value in period_data:
                    boxplot_data.append({
                        'Indicator': get_indicator_nickname(indicator),
                        'Period': period,
                        'Value': value
                    })

    return pd.DataFrame(boxplot_data)

def perform_temporal_volatility_tests(data, country, indicators, period_column='EURO_PERIOD'):
    """Perform F-tests comparing pre/post Euro volatility"""
    country_data = data[data['COUNTRY'] == country].copy()

    test_results = {}
    for indicator in indicators:
        indicator_data = country_data[country_data['INDICATOR'] == indicator].copy()

        pre_euro = indicator_data[indicator_data[period_column] == 'Pre-Euro']['VALUE'].dropna()
        post_euro = indicator_data[indicator_data[period_column] == 'Post-Euro']['VALUE'].dropna()

        if len(pre_euro) > 1 and len(post_euro) > 1:
            # Calculate variances
            var_pre = np.var(pre_euro, ddof=1)
            var_post = np.var(post_euro, ddof=1)

            # F-test
            if var_pre > 0 and var_post > 0:
                f_stat = var_pre / var_post
                df1 = len(pre_euro) - 1
                df2 = len(post_euro) - 1
                p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
            else:
                f_stat = np.nan
                p_value = np.nan
        else:
            f_stat = np.nan
            p_value = np.nan

        test_results[indicator] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'pre_variance': var_pre if 'var_pre' in locals() else np.nan,
            'post_variance': var_post if 'var_post' in locals() else np.nan,
            'pre_n': len(pre_euro),
            'post_n': len(post_euro)
        }

    return test_results

def load_case_study_2_data(include_crisis_years=True, data_type='full'):
    """Load Euro adoption analysis data from comprehensive dataset"""
    try:
        # Load comprehensive dataset based on data type
        data_dir = Path(__file__).parent.parent.parent.parent / "updated_data" / "Clean"

        if data_type == 'winsorized':
            comprehensive_file = data_dir / "comprehensive_df_PGDP_labeled_winsorized.csv"
        else:
            comprehensive_file = data_dir / "comprehensive_df_PGDP_labeled.csv"

        if not comprehensive_file.exists():
            raise FileNotFoundError(f"Data file not found: {comprehensive_file}")

        data = pd.read_csv(comprehensive_file)

        # Filter for Baltic countries
        baltic_countries = ['Estonia, Republic of', 'Latvia, Republic of', 'Lithuania, Republic of']
        data = data[data['COUNTRY'].isin(baltic_countries)].copy()

        # Add quarter and date columns
        if 'QUARTER' in data.columns:
            data['DATE'] = pd.to_datetime(data['YEAR'].astype(str) + '-Q' + data['QUARTER'].astype(str))

        # Create Euro periods
        data = create_euro_periods(data, include_crisis_years)

        return data

    except Exception as e:
        print(f"Error loading CS2 data: {str(e)}")
        return pd.DataFrame()

def load_overall_capital_flows_data_cs2(include_crisis_years=True, data_type='full'):
    """Load data for overall capital flows analysis"""
    return load_case_study_2_data(include_crisis_years, data_type)