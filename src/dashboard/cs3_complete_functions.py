"""
CS3 Complete Analysis Functions - Exact Copy of CS1 with CS3 Groupings

This module contains the complete CS3 analysis implementation copied from CS1
with adaptations for Iceland vs Small Open Economies comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from pathlib import Path
import sys
import io
from datetime import datetime
import base64
import zipfile

# Import all CS1 utility functions
def create_indicator_nicknames():
    """Create readable nicknames for indicators"""
    return {
        'Assets - Direct investment, Total financial assets/liabilities': 'Assets - Direct Investment',
        'Assets - Other investment, Debt instruments': 'Assets - Other Investment (Debt)',
        'Assets - Other investment, Debt instruments, Deposit taking corporations, except the Central Bank': 'Assets - Other Investment (Banks)',
        'Assets - Portfolio investment, Debt securities': 'Assets - Portfolio (Debt)',
        'Assets - Portfolio investment, Equity and investment fund shares': 'Assets - Portfolio (Equity)',
        'Assets - Portfolio investment, Total financial assets/liabilities': 'Assets - Portfolio (Total)',
        'Liabilities - Direct investment, Total financial assets/liabilities': 'Liabilities - Direct Investment',
        'Liabilities - Other investment, Debt instruments, Deposit taking corporations, except the Central Bank': 'Liabilities - Other Investment (Banks)',
        'Liabilities - Portfolio investment, Debt securities': 'Liabilities - Portfolio (Debt)',
        'Liabilities - Portfolio investment, Equity and investment fund shares': 'Liabilities - Portfolio (Equity)',
        'Liabilities - Portfolio investment, Total financial assets/liabilities': 'Liabilities - Portfolio (Total)',
        'Net - Direct investment, Total financial assets/liabilities': 'Net - Direct Investment',
        'Net - Portfolio investment, Total financial assets/liabilities': 'Net - Portfolio Investment',
        'Net - Other investment, Total financial assets/liabilities': 'Net - Other Investment',
        'Net (net acquisition of financial assets less net incurrence of liabilities) - Direct investment, Total financial assets/liabilities': 'Net - Direct Investment',
        'Net (net acquisition of financial assets less net incurrence of liabilities) - Portfolio investment, Total financial assets/liabilities': 'Net - Portfolio Investment',
        'Net (net acquisition of financial assets less net incurrence of liabilities) - Other investment, Total financial assets/liabilities': 'Net - Other Investment'
    }

def get_nickname(indicator_name):
    """Get nickname for indicator, fallback to shortened version"""
    nicknames = create_indicator_nicknames()
    nickname = nicknames.get(indicator_name, indicator_name[:25] + '...' if len(indicator_name) > 25 else indicator_name)
    return nickname

def load_cs3_data(include_crisis_years=True):
    """Load CS3 data: Iceland vs Small Open Economies"""
    try:
        # Load the comprehensive labeled dataset
        data_dir = Path(__file__).parent.parent.parent / "updated_data" / "Clean"
        file_path = data_dir / "comprehensive_df_PGDP_labeled.csv "  # Note: space in filename
        
        if not file_path.exists():
            st.error(f"❌ Data file not found: {file_path}")
            return None, None, None
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Filter for Case Study 3 countries only
        cs3_data = df[df['CS3_GROUP'].notna()].copy()
        
        if len(cs3_data) == 0:
            st.error("❌ No Case Study 3 data found in dataset")
            return None, None, None
        
        # Apply crisis filtering if requested (matching CS1 methodology)
        if not include_crisis_years:
            # Define crisis years: GFC (2008-2010) + COVID (2020-2022)
            crisis_years = [2008, 2009, 2010, 2020, 2021, 2022]
            
            # Filter out crisis years
            original_count = len(cs3_data)
            cs3_data = cs3_data[~cs3_data['YEAR'].isin(crisis_years)].copy()
            excluded_count = original_count - len(cs3_data)
            
            if len(cs3_data) == 0:
                st.error("No data remaining after crisis period exclusion.")
                return None, None, None
        
        # Create GROUP column for compatibility with CS1 functions
        cs3_data['GROUP'] = cs3_data['CS3_GROUP'].map({
            'Iceland': 'Iceland',
            'Comparator': 'Small Open Economies'
        })
        
        # Get indicator columns (ending with _PGDP)
        all_indicators = [col for col in cs3_data.columns if col.endswith('_PGDP')]
        
        # Remove excluded indicators (matching CS1)
        indicators_to_exclude = [
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Financial derivatives (other than reserves) and employee stock options_PGDP',
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Financial account balance, excluding reserves and related items_PGDP'
        ]
        analysis_indicators = [ind for ind in all_indicators if ind not in indicators_to_exclude]
        
        # Rename indicators to consistent format (matching CS1)
        indicator_renames = {
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Direct investment, Total financial assets/liabilities_PGDP': 'Net - Direct investment, Total financial assets/liabilities_PGDP',
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Portfolio investment, Total financial assets/liabilities_PGDP': 'Net - Portfolio investment, Total financial assets/liabilities_PGDP',
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Other investment, Total financial assets/liabilities_PGDP': 'Net - Other investment, Total financial assets/liabilities_PGDP'
        }
        
        # Apply renames to dataframe
        cs3_data = cs3_data.rename(columns=indicator_renames)
        
        # Update indicator list with new names
        analysis_indicators = [indicator_renames.get(ind, ind) for ind in analysis_indicators]
        
        # Create metadata
        study_version = "Full Time Period" if include_crisis_years else "Crisis-Excluded"
        metadata = {
            'original_shape': df.shape,
            'filtered_shape': cs3_data.shape,
            'final_shape': cs3_data.shape,
            'n_indicators': len(analysis_indicators),
            'study_version': study_version,
            'include_crisis_years': include_crisis_years
        }
        
        if not include_crisis_years:
            metadata['excluded_observations'] = excluded_count
            metadata['crisis_years'] = [2008, 2009, 2010, 2020, 2021, 2022]
        
        st.success(f"✅ Loaded CS3 data: {len(cs3_data)} observations, {len(analysis_indicators)} indicators")
        
        return cs3_data, analysis_indicators, metadata
        
    except Exception as e:
        st.error(f"❌ Error loading CS3 data: {str(e)}")
        return None, None, None

def calculate_group_statistics(data, group_col, indicators):
    """Calculate comprehensive statistics by group (copied from CS1)"""
    results = []
    
    for group in data[group_col].unique():
        group_data = data[data[group_col] == group]
        
        for indicator in indicators:
            if indicator in data.columns:
                values = group_data[indicator].dropna()
                
                if len(values) > 1:
                    mean_val = values.mean()
                    std_val = values.std()
                    cv = (std_val / abs(mean_val)) * 100 if mean_val != 0 else np.inf
                    
                    results.append({
                        'Group': group,
                        'Indicator': indicator.replace('_PGDP', ''),
                        'N': len(values),
                        'Mean': mean_val,
                        'Std_Dev': std_val,
                        'Skewness': stats.skew(values),
                        'CV_Percent': cv
                    })
    
    return pd.DataFrame(results)

def create_boxplot_data(data, indicators):
    """Create dataset for boxplot visualization (adapted for CS3)"""
    stats_data = []
    
    for group in ['Iceland', 'Small Open Economies']:
        group_data = data[data['GROUP'] == group]
        
        for indicator in indicators:
            if indicator in data.columns:
                values = group_data[indicator].dropna()
                
                if len(values) > 1:
                    mean_val = values.mean()
                    std_val = values.std()
                    
                    stats_data.extend([
                        {'GROUP': group, 'Statistic': 'Mean', 'Value': mean_val, 'Indicator': indicator.replace('_PGDP', '')},
                        {'GROUP': group, 'Statistic': 'Standard Deviation', 'Value': std_val, 'Indicator': indicator.replace('_PGDP', '')}
                    ])
    
    return pd.DataFrame(stats_data)

def perform_volatility_tests(data, indicators):
    """Perform F-tests for variance equality (adapted for CS3)"""
    results = []
    
    for indicator in indicators:
        if indicator in data.columns:
            iceland_data = data[data['GROUP'] == 'Iceland'][indicator].dropna()
            soe_data = data[data['GROUP'] == 'Small Open Economies'][indicator].dropna()
            
            if len(iceland_data) > 1 and len(soe_data) > 1:
                # Perform F-test for equality of variances
                iceland_var = np.var(iceland_data, ddof=1)
                soe_var = np.var(soe_data, ddof=1)
                
                # F-statistic (larger variance in numerator)
                if iceland_var >= soe_var:
                    f_stat = iceland_var / soe_var
                    df1, df2 = len(iceland_data) - 1, len(soe_data) - 1
                else:
                    f_stat = soe_var / iceland_var
                    df1, df2 = len(soe_data) - 1, len(iceland_data) - 1
                
                # Two-tailed p-value
                p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
                
                results.append({
                    'Indicator': indicator.replace('_PGDP', ''),
                    'Iceland_Std': np.std(iceland_data, ddof=1),
                    'SOE_Std': np.std(soe_data, ddof=1),
                    'F_Statistic': f_stat,
                    'P_Value': p_value,
                    'Significant_5pct': p_value < 0.05,
                    'Significant_1pct': p_value < 0.01,
                    'Iceland_Higher_Volatility': np.std(iceland_data, ddof=1) > np.std(soe_data, ddof=1)
                })
    
    return pd.DataFrame(results)

def create_plot_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_base64

def sort_indicators_by_type(indicators):
    """Sort indicators by investment type (Assets, Liabilities, Net)"""
    # Define sorting order
    order = ['Assets', 'Liabilities', 'Net']
    
    def get_sort_key(indicator):
        clean_name = indicator.replace('_PGDP', '')
        for i, prefix in enumerate(order):
            if clean_name.startswith(prefix):
                return (i, clean_name)  # Primary sort by type, secondary by name
        return (999, clean_name)  # Unknown types go last
    
    return sorted(indicators, key=get_sort_key)

def get_investment_type_order(indicator_name):
    """Get sorting order for investment types"""
    if indicator_name.startswith('Assets'):
        return 0
    elif indicator_name.startswith('Liabilities'):
        return 1
    elif indicator_name.startswith('Net'):
        return 2
    else:
        return 999

def create_individual_country_boxplot_data(data, indicators):
    """Create dataset for individual country boxplot visualization (CS3 version)"""
    stats_data = []
    
    for country in data['COUNTRY'].unique():
        country_data = data[data['COUNTRY'] == country]
        
        for indicator in indicators:
            if indicator in data.columns:
                values = country_data[indicator].dropna()
                
                if len(values) > 1:
                    mean_val = values.mean()
                    std_val = values.std()
                    
                    stats_data.extend([
                        {'COUNTRY': country, 'Statistic': 'Mean', 'Value': mean_val, 'Indicator': indicator.replace('_PGDP', '')},
                        {'COUNTRY': country, 'Statistic': 'Standard Deviation', 'Value': std_val, 'Indicator': indicator.replace('_PGDP', '')}
                    ])
    
    return pd.DataFrame(stats_data)

def load_overall_capital_flows_data_cs3(include_crisis_years=True):
    """Load overall capital flows data for CS3 - aggregate net flows only"""
    try:
        # Load CS3 data
        cs3_data, _, metadata = load_cs3_data(include_crisis_years=include_crisis_years)
        
        if cs3_data is None:
            return None, None
        
        # Define overall capital flow indicators (net flows only)
        overall_indicators = {
            'Net Direct Investment': 'Net - Direct investment, Total financial assets/liabilities_PGDP',
            'Net Portfolio Investment': 'Net - Portfolio investment, Total financial assets/liabilities_PGDP',
            'Net Other Investment': 'Net - Other investment, Total financial assets/liabilities_PGDP',
            'Total Net Flows': None  # Will be calculated
        }
        
        # Calculate Total Net Flows as sum of the three components
        net_di = 'Net - Direct investment, Total financial assets/liabilities_PGDP'
        net_pi = 'Net - Portfolio investment, Total financial assets/liabilities_PGDP'
        net_oi = 'Net - Other investment, Total financial assets/liabilities_PGDP'
        
        # Check which columns exist
        available_cols = []
        for col in [net_di, net_pi, net_oi]:
            if col in cs3_data.columns:
                available_cols.append(col)
        
        if available_cols:
            # Calculate total net flows as sum of available components
            cs3_data['Total_Net_Flows_PGDP'] = cs3_data[available_cols].sum(axis=1, skipna=False)
            overall_indicators['Total Net Flows'] = 'Total_Net_Flows_PGDP'
        
        # Filter to only include available indicators
        available_indicators = {k: v for k, v in overall_indicators.items() if v and v in cs3_data.columns}
        
        return cs3_data, available_indicators
        
    except Exception as e:
        st.error(f"Error loading overall capital flows data for CS3: {str(e)}")
        return None, None