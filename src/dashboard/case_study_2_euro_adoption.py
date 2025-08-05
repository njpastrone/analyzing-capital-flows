"""
Case Study 2: Euro Adoption Impact Analysis - Baltic Countries
Temporal comparison of capital flow volatility before and after Euro adoption
Estonia (2011), Latvia (2014), Lithuania (2015)
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

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

warnings.filterwarnings('ignore')

# Import shared functions from Case Study 1  
import simple_report_app
from simple_report_app import (
    create_indicator_nicknames, 
    get_nickname,
    get_investment_type_order,
    sort_indicators_by_type,
    COLORBLIND_SAFE
)

def create_euro_adoption_timeline():
    """Define Euro adoption dates and analysis periods"""
    return {
        'Estonia, Republic of': {
            'adoption_date': '2011-01-01',
            'pre_period': (2005, 2010),
            'post_period': (2012, 2017),
            'adoption_year': 2011
        },
        'Latvia, Republic of': {
            'adoption_date': '2014-01-01', 
            'pre_period': (2007, 2012),
            'post_period': (2015, 2020),
            'adoption_year': 2014
        },
        'Lithuania, Republic of': {
            'adoption_date': '2015-01-01',
            'pre_period': (2008, 2013), 
            'post_period': (2016, 2021),
            'adoption_year': 2015
        }
    }

def create_euro_periods(data, include_crisis_years=True):
    """Create Euro adoption period labels for comprehensive dataset"""
    timeline = create_expanded_euro_adoption_timeline()
    
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

def load_case_study_2_data(include_crisis_years=True):
    """Load Euro adoption analysis data from comprehensive dataset"""
    try:
        # Load comprehensive dataset
        data_dir = Path(__file__).parent.parent.parent / "updated_data" / "Clean"
        comprehensive_file = data_dir / "comprehensive_df_PGDP_labeled.csv "  # Note trailing space
        
        if not comprehensive_file.exists():
            st.error(f"Comprehensive dataset not found at {comprehensive_file}")
            return None, None, None
        
        # Load comprehensive data and filter for CS2 countries
        df = pd.read_csv(comprehensive_file)
        cs2_data = df[df['CS2_GROUP'].notna()].copy()
        
        if cs2_data.empty:
            st.error("No CS2 countries found in comprehensive dataset")
            return None, None, None
        
        # Create Euro adoption periods for each country
        cs2_data = create_euro_periods(cs2_data, include_crisis_years)
        study_version = "Full Series" if include_crisis_years else "Crisis-Excluded"
        
        # Use the processed data format
        final_data = cs2_data
        
        # Get analysis indicators (columns ending with _PGDP)
        all_indicators = [col for col in final_data.columns if col.endswith('_PGDP')]
        # Filter to only indicators with data for CS2 countries
        all_indicators = [col for col in all_indicators if final_data[col].notna().any()]
        
        # Match CS1 indicator set exactly - map comprehensive dataset names to CS1 names
        cs1_indicator_mapping = {
            'Assets - Direct investment, Total financial assets/liabilities': 'Assets - Direct investment, Total financial assets/liabilities',
            'Liabilities - Direct investment, Total financial assets/liabilities': 'Liabilities - Direct investment, Total financial assets/liabilities', 
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Direct investment, Total financial assets/liabilities': 'Net - Direct investment, Total financial assets/liabilities',
            'Assets - Portfolio investment, Total financial assets/liabilities': 'Assets - Portfolio investment, Total financial assets/liabilities',
            'Liabilities - Portfolio investment, Total financial assets/liabilities': 'Liabilities - Portfolio investment, Total financial assets/liabilities',
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Portfolio investment, Total financial assets/liabilities': 'Net - Portfolio investment, Total financial assets/liabilities',
            'Assets - Portfolio investment, Debt securities': 'Assets - Portfolio investment, Debt securities',
            'Liabilities - Portfolio investment, Debt securities': 'Liabilities - Portfolio investment, Debt securities',
            'Assets - Portfolio investment, Equity and investment fund shares': 'Assets - Portfolio investment, Equity and investment fund shares',
            'Liabilities - Portfolio investment, Equity and investment fund shares': 'Liabilities - Portfolio investment, Equity and investment fund shares',
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Other investment, Total financial assets/liabilities': 'Net - Other investment, Total financial assets/liabilities',
            'Assets - Other investment, Debt instruments': 'Assets - Other investment, Debt instruments',
            'Assets - Other investment, Debt instruments, Deposit taking corporations, except the Central Bank': 'Assets - Other investment, Debt instruments, Deposit taking corporations, except the Central Bank',
            'Liabilities - Other investment, Debt instruments, Deposit taking corporations, except the Central Bank': 'Liabilities - Other investment, Debt instruments, Deposit taking corporations, except the Central Bank'
        }
        
        # Create renamed columns for the 14 CS1 indicators
        analysis_indicators = []
        for original_col in all_indicators:
            clean_name = original_col.replace('_PGDP', '')
            if clean_name in cs1_indicator_mapping:
                cs1_name = cs1_indicator_mapping[clean_name]
                new_col = cs1_name + '_PGDP'
                # Rename the column in the dataset
                final_data = final_data.rename(columns={original_col: new_col})
                analysis_indicators.append(new_col)
        
        analysis_indicators = sort_indicators_by_type(analysis_indicators)
        
        # Create timeline information with full period structure
        timeline = create_expanded_euro_adoption_timeline()
        
        # Analysis countries (simplified country names for display)
        analysis_countries = ['Estonia', 'Latvia', 'Lithuania']
        
        # Determine which period column to use
        period_column = 'EURO_PERIOD_CRISIS_EXCLUDED' if not include_crisis_years else 'EURO_PERIOD_FULL'
        
        # Create metadata
        metadata = {
            'final_shape': final_data.shape,
            'n_indicators': len(analysis_indicators),
            'countries': analysis_countries,
            'timeline': timeline,
            'study_version': study_version,
            'include_crisis_years': include_crisis_years,
            'period_column': period_column
        }
        
        return final_data, analysis_indicators, metadata
        
    except Exception as e:
        st.error(f"Error loading Case Study 2 data: {str(e)}")
        return None, None, None

def create_expanded_euro_adoption_timeline():
    """Create expanded timeline with maximized data periods (including adoption years in post-Euro)"""
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

def get_country_specific_crisis_text(country):
    """Get crisis exclusion text for a specific country"""
    timeline = create_expanded_euro_adoption_timeline()
    
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
    timeline = create_expanded_euro_adoption_timeline()
    
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
    """Calculate statistics comparing pre-Euro vs post-Euro periods for a country"""
    results = []
    
    country_data = data[data['COUNTRY'] == country]
    
    for period in ['Pre-Euro', 'Post-Euro']:
        period_data = country_data[country_data[period_column] == period]
        
        for indicator in indicators:
            values = period_data[indicator].dropna()
            
            if len(values) > 1:
                mean_val = values.mean()
                std_val = values.std()
                cv = (std_val / abs(mean_val)) * 100 if mean_val != 0 else np.inf
                
                results.append({
                    'Country': country,
                    'Period': period,
                    'Indicator': indicator.replace('_PGDP', ''),
                    'N': len(values),
                    'Mean': mean_val,
                    'Std_Dev': std_val,
                    'Skewness': stats.skew(values),
                    'CV_Percent': cv
                })
    
    return pd.DataFrame(results)

def create_temporal_boxplot_data(data, country, indicators, period_column='EURO_PERIOD'):
    """Create dataset for temporal boxplot visualization"""
    stats_data = []
    
    country_data = data[data['COUNTRY'] == country]
    
    for period in ['Pre-Euro', 'Post-Euro']:
        period_data = country_data[country_data[period_column] == period]
        
        for indicator in indicators:
            values = period_data[indicator].dropna()
            if len(values) > 1:
                mean_val = values.mean()
                std_val = values.std()
                
                stats_data.append({
                    'PERIOD': period,
                    'Indicator': indicator.replace('_PGDP', ''),
                    'Statistic': 'Mean',
                    'Value': mean_val
                })
                
                stats_data.append({
                    'PERIOD': period,
                    'Indicator': indicator.replace('_PGDP', ''),
                    'Statistic': 'Standard Deviation', 
                    'Value': std_val
                })
    
    return pd.DataFrame(stats_data)

def perform_temporal_volatility_tests(data, country, indicators, period_column='EURO_PERIOD'):
    """Perform F-tests comparing pre-Euro vs post-Euro volatility for a country"""
    test_results = []
    
    country_data = data[data['COUNTRY'] == country]
    
    for indicator in indicators:
        pre_data = country_data[
            (country_data[period_column] == 'Pre-Euro')
        ][indicator].dropna()
        post_data = country_data[
            (country_data[period_column] == 'Post-Euro') 
        ][indicator].dropna()
        
        if len(pre_data) > 1 and len(post_data) > 1:
            pre_var = pre_data.var()
            post_var = post_data.var()
            
            f_stat = pre_var / post_var if post_var != 0 else np.inf
            df1, df2 = len(pre_data) - 1, len(post_data) - 1
            
            # Two-tailed p-value
            p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
            
            test_results.append({
                'Country': country,
                'Indicator': indicator.replace('_PGDP', ''),
                'F_Statistic': f_stat,
                'P_Value': p_value,
                'Pre_Euro_Higher_Volatility': pre_var > post_var,
                'Significant_5pct': p_value < 0.05,
                'Significant_1pct': p_value < 0.01,
                'Pre_Euro_Variance': pre_var,
                'Post_Euro_Variance': post_var
            })
    
    return pd.DataFrame(test_results)

def load_overall_capital_flows_data_cs2(include_crisis_years=True):
    """Load data specifically for Case Study 2 Overall Capital Flows Analysis"""
    try:
        # Use comprehensive dataset with robust path finding
        current_dir = Path(__file__).parent
        # Navigate up to find the project root (contains updated_data)
        project_root = current_dir.parent.parent
        data_dir = project_root / "updated_data" / "Clean"
        comprehensive_file = data_dir / "comprehensive_df_PGDP_labeled.csv "
        
        if not comprehensive_file.exists():
            return None, None, None
        
        # Load comprehensive labeled data
        comprehensive_df = pd.read_csv(comprehensive_file)
        
        # Filter for Case Study 2 data (CS2_GROUP not null)
        case_two_data = comprehensive_df[comprehensive_df['CS2_GROUP'].notna()].copy()
        
        # Create Euro adoption timeline with full 1999-2025 periods
        timeline = create_expanded_euro_adoption_timeline()
        
        # Add period classification using CS1 methodology
        def classify_period(row, timeline):
            country = row['COUNTRY']
            year = row['YEAR']
            
            if country in timeline:
                adoption_year = timeline[country]['adoption_year']
                
                # Always use full 1999-2025 timeline, just classify periods
                if year < adoption_year:
                    return 'Pre-Euro'
                elif year >= adoption_year:
                    return 'Post-Euro'
            return 'Unknown'
        
        case_two_data['EURO_PERIOD'] = case_two_data.apply(
            lambda row: classify_period(row, timeline), axis=1
        )
        
        # Apply crisis filtering using country-specific crisis periods
        if not include_crisis_years:
            # Filter data based on country-specific crisis years
            rows_to_keep = []
            
            for _, row in case_two_data.iterrows():
                country = row['COUNTRY']
                year = row['YEAR']
                
                if country in timeline:
                    country_crisis_years = timeline[country]['crisis_years']
                    # Keep row if year is not in country's crisis years
                    if year not in country_crisis_years:
                        rows_to_keep.append(True)
                    else:
                        rows_to_keep.append(False)
                else:
                    # For unknown countries, apply default crisis exclusion
                    default_crisis_years = [2008, 2009, 2010, 2020, 2021, 2022]
                    rows_to_keep.append(year not in default_crisis_years)
            
            case_two_data = case_two_data[rows_to_keep].copy()
        
        # Define the 4 overall capital flows indicators (3 base + 1 computed)
        base_indicators_mapping = {
            'Net Portfolio Investment': 'Net (net acquisition of financial assets less net incurrence of liabilities) - Portfolio investment, Total financial assets/liabilities_PGDP',
            'Net Direct Investment': 'Net (net acquisition of financial assets less net incurrence of liabilities) - Direct investment, Total financial assets/liabilities_PGDP',
            'Net Other Investment': 'Net (net acquisition of financial assets less net incurrence of liabilities) - Other investment, Total financial assets/liabilities_PGDP'
        }
        
        # Compute the new Net Capital Flows indicator
        net_direct_col = base_indicators_mapping['Net Direct Investment']
        net_portfolio_col = base_indicators_mapping['Net Portfolio Investment'] 
        net_other_col = base_indicators_mapping['Net Other Investment']
        
        # Create the computed column
        case_two_data['Net Capital Flows (Direct + Portfolio + Other Investment)_PGDP'] = (
            case_two_data[net_direct_col].fillna(0) + 
            case_two_data[net_portfolio_col].fillna(0) + 
            case_two_data[net_other_col].fillna(0)
        )
        
        # Complete indicators mapping including computed indicator
        overall_indicators_mapping = {
            **base_indicators_mapping,
            'Net Capital Flows (Direct + Portfolio + Other Investment)': 'Net Capital Flows (Direct + Portfolio + Other Investment)_PGDP'
        }
        
        # Create metadata
        study_version = "Full Series" if include_crisis_years else "Crisis-Excluded"
        metadata = {
            'final_shape': case_two_data.shape,
            'countries': sorted(case_two_data['COUNTRY'].unique()),
            'timeline': timeline,
            'study_version': study_version,
            'include_crisis_years': include_crisis_years
        }
        
        return case_two_data, overall_indicators_mapping, metadata
        
    except Exception as e:
        st.error(f"Error loading Case Study 2 overall capital flows data: {str(e)}")
        return None, None, None

def show_overall_capital_flows_analysis_cs2(selected_country, selected_display_country, include_crisis_years=True):
    """Display Overall Capital Flows Analysis section for Case Study 2 - country-specific version"""
    study_version = "Full Series" if include_crisis_years else "Crisis-Excluded"
    st.markdown(f"*Aggregate net capital flows summary - {study_version}*")
    
    # Load data and filter for selected country
    overall_data, indicators_mapping, metadata = load_overall_capital_flows_data_cs2(include_crisis_years)
    
    if overall_data is not None:
        # Filter for selected country only
        overall_data = overall_data[overall_data['COUNTRY'] == selected_country].copy()
    
    if overall_data is None or indicators_mapping is None:
        st.error("Failed to load overall capital flows data.")
        return
    
    # Use consistent COLORBLIND_SAFE palette
    colors = {'Pre-Euro': COLORBLIND_SAFE[0], 'Post-Euro': COLORBLIND_SAFE[1]}
    
    # Summary statistics (matching CS1 format exactly)
    st.subheader("📊 Summary Statistics by Period")
    
    summary_stats = []
    for clean_name, col_name in indicators_mapping.items():
        if col_name in overall_data.columns:
            for period in ['Pre-Euro', 'Post-Euro']:
                period_data = overall_data[overall_data['EURO_PERIOD'] == period][col_name].dropna()
                if len(period_data) > 0:
                    summary_stats.append({
                        'Indicator': clean_name,
                        'Period': period,
                        'Mean': period_data.mean(),
                        'Std Dev': period_data.std(),
                        'Median': period_data.median(),
                        'Min': period_data.min(),
                        'Max': period_data.max(),
                        'Count': len(period_data)
                    })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Display summary table (matching CS1 pivot structure)
    pivot_summary = summary_df.pivot_table(
        index='Indicator', 
        columns='Period', 
        values=['Mean', 'Std Dev', 'Median'],
        aggfunc='first'
    ).round(2)
    
    # Create custom HTML table with strict column width control for CS2
    st.markdown("""
    <style>
    .cs2-summary-table {
        width: 100% !important;
        border-collapse: collapse !important;
        table-layout: fixed !important;
        font-size: 8px !important;
        font-family: Arial, sans-serif !important;
    }
    .cs2-summary-table th, .cs2-summary-table td {
        border: 1px solid #ddd !important;
        padding: 2px !important;
        text-align: center !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    .cs2-summary-table th {
        background-color: #f0f0f0 !important;
        font-weight: bold !important;
        font-size: 9px !important;
    }
    .cs2-summary-table th:first-child, .cs2-summary-table td:first-child {
        width: 220px !important;
        max-width: 220px !important;
        text-align: left !important;
        font-weight: bold !important;
    }
    .cs2-summary-table th:not(:first-child), .cs2-summary-table td:not(:first-child) {
        width: 70px !important;
        max-width: 70px !important;
    }
    .cs2-summary-table tr:nth-child(even) {
        background-color: #f9f9f9 !important;
    }
    @media print {
        .cs2-summary-table {
            font-size: 7px !important;
        }
        .cs2-summary-table th:first-child, .cs2-summary-table td:first-child {
            width: 180px !important;
            max-width: 180px !important;
        }
        .cs2-summary-table th:not(:first-child), .cs2-summary-table td:not(:first-child) {
            width: 60px !important;
            max-width: 60px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Convert pivot table to regular DataFrame for HTML generation
    pivot_df = pivot_summary.reset_index()
    
    # Generate HTML table content
    html_table = '<table class="cs2-summary-table">'
    html_table += '<thead><tr>'
    for col in pivot_df.columns:
        col_name = str(col).replace('(', '').replace(')', '').replace("'", '') if isinstance(col, tuple) else str(col)
        html_table += f'<th>{col_name}</th>'
    html_table += '</tr></thead><tbody>'
    
    for _, row in pivot_df.iterrows():
        html_table += '<tr>'
        for col in pivot_df.columns:
            value = row[col]
            if pd.isna(value):
                value = '-'
            elif isinstance(value, (int, float)):
                value = f'{value:.2f}'
            html_table += f'<td>{value}</td>'
        html_table += '</tr>'
    
    html_table += '</tbody></table>'
    
    # Display the custom HTML table
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Distribution Comparison - 2x2 matrix of boxplots (matching CS1 exactly)
    st.subheader("📦 Distribution Comparison by Period")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (clean_name, col_name) in enumerate(indicators_mapping.items()):
        if col_name in overall_data.columns and i < 4:
            ax = axes[i]
            
            # Prepare data for boxplot
            pre_data = overall_data[overall_data['EURO_PERIOD'] == 'Pre-Euro'][col_name].dropna()
            post_data = overall_data[overall_data['EURO_PERIOD'] == 'Post-Euro'][col_name].dropna()
            
            if len(pre_data) > 0 and len(post_data) > 0:
                # Create boxplot
                bp = ax.boxplot([pre_data, post_data], 
                               labels=['Pre-Euro', 'Post-Euro'], 
                               patch_artist=True)
                
                # Color the boxes
                bp['boxes'][0].set_facecolor(colors['Pre-Euro'])
                bp['boxes'][1].set_facecolor(colors['Post-Euro'])
                for box in bp['boxes']:
                    box.set_alpha(0.7)
                
                ax.set_title(clean_name, fontweight='bold', fontsize=10)
                ax.set_ylabel('% of GDP (annualized)', fontsize=9)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
                ax.tick_params(labelsize=8)
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Time Series - 2x2 matrix (matching CS1 exactly)
    st.subheader("📈 Time Series by Period")
    
    # Create date column
    overall_data_ts = overall_data.copy()
    overall_data_ts['DATE'] = pd.to_datetime(overall_data_ts['YEAR'].astype(str) + '-Q' + overall_data_ts['QUARTER'].astype(str))
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    axes2 = axes2.flatten()
    
    for i, (clean_name, col_name) in enumerate(indicators_mapping.items()):
        if col_name in overall_data.columns and i < 4:
            ax = axes2[i]
            
            # Plot time series with proper crisis period handling
            sorted_data = overall_data_ts.sort_values('DATE')
            
            # Since we're now working with single-country data, period overlap is resolved
            if not include_crisis_years:
                # Crisis-excluded: plot data in segments to avoid connecting across excluded periods
                for period in ['Pre-Euro', 'Post-Euro']:
                    period_data = sorted_data[sorted_data['EURO_PERIOD'] == period]
                    if len(period_data) == 0:
                        continue
                    
                    # Aggregate by date (single country, so just group by DATE)
                    period_agg = period_data.groupby('DATE')[col_name].mean().reset_index()
                    
                    if len(period_agg) == 0:
                        continue
                    
                    # Find segments separated by crisis periods
                    segments = []
                    current_segment = []
                    
                    for _, row in period_agg.iterrows():
                        year = row['DATE'].year
                        # Check if this year is adjacent to the previous year (allowing for gaps)
                        if len(current_segment) == 0:
                            current_segment.append(row)
                        else:
                            prev_year = current_segment[-1]['DATE'].year
                            # If there's a gap of more than 1 year, start a new segment
                            if year - prev_year > 1:
                                segments.append(pd.DataFrame(current_segment))
                                current_segment = [row]
                            else:
                                current_segment.append(row)
                    
                    # Add the last segment
                    if current_segment:
                        segments.append(pd.DataFrame(current_segment))
                    
                    # Plot each segment separately
                    for j, segment in enumerate(segments):
                        if len(segment) > 0:
                            ax.plot(segment['DATE'], segment[col_name], 
                                   color=colors[period], linewidth=2, alpha=0.8,
                                   label=period if j == 0 else "")
            else:
                # Full series: simple aggregation by period (single country, no overlap possible)
                for period in ['Pre-Euro', 'Post-Euro']:
                    period_data = sorted_data[sorted_data['EURO_PERIOD'] == period]
                    if len(period_data) > 0:
                        # Aggregate by date (single country)
                        period_agg = period_data.groupby('DATE')[col_name].mean().reset_index()
                        if len(period_agg) > 0:
                            ax.plot(period_agg['DATE'], period_agg[col_name], 
                                   color=colors[period], label=period, linewidth=2, alpha=0.8)
            
            # Add crisis period shading for crisis-excluded version
            if not include_crisis_years:
                add_country_specific_crisis_shading(ax, selected_country, include_labels=(i == 0))
            
            # Add Euro adoption line for selected country only
            country_adoption_years = {
                'Estonia, Republic of': 2011, 
                'Latvia, Republic of': 2014, 
                'Lithuania, Republic of': 2015
            }
            adoption_year = country_adoption_years.get(selected_country)
            if adoption_year:
                adoption_date = pd.to_datetime(f'{adoption_year}-01-01')
                ax.axvline(x=adoption_date, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                          label='Euro Adoption' if i == 0 else "")
            
            ax.set_title(clean_name, fontweight='bold', fontsize=10)
            ax.set_ylabel('% of GDP (annualized)', fontsize=9)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            ax.legend(loc='upper right', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
    
    fig2.tight_layout()
    st.pyplot(fig2)

# Helper functions for CS1-CS2 consistency
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
        'Net - Other investment, Total financial assets/liabilities': 'Net - Other Investment'
    }

def get_nickname(indicator_name):
    """Get nickname for indicator, fallback to shortened version"""
    nicknames = create_indicator_nicknames()
    nickname = nicknames.get(indicator_name, indicator_name[:25] + '...' if len(indicator_name) > 25 else indicator_name)
    # Truncate for table display while maintaining readability
    return nickname[:35] + '...' if len(nickname) > 35 else nickname

def get_investment_type_order(indicator_name):
    """
    Extract sorting key for indicators: Type of Investment -> Disaggregation -> Accounting Entry
    Returns tuple for sorting: (investment_type_order, disaggregation_order, accounting_entry_order)
    """
    # Investment type mapping
    if 'Direct investment' in indicator_name:
        inv_type = 0  # Direct
    elif 'Portfolio investment' in indicator_name:
        inv_type = 1  # Portfolio  
    elif 'Other investment' in indicator_name:
        inv_type = 2  # Other
    else:
        inv_type = 9  # Unknown
    
    # Disaggregation mapping (for Portfolio and Other)
    if 'Total financial assets/liabilities' in indicator_name:
        disagg = 0  # Total (comes first)
    elif 'Debt' in indicator_name:
        if 'Deposit taking corporations' in indicator_name:
            disagg = 2  # Debt - Banks (more specific)
        else:
            disagg = 1  # Debt - General
    elif 'Equity' in indicator_name:
        disagg = 3  # Equity
    else:
        disagg = 9  # No disaggregation or other
    
    # Accounting entry mapping
    if indicator_name.startswith('Assets'):
        acc_entry = 0
    elif indicator_name.startswith('Liabilities'):
        acc_entry = 1
    elif indicator_name.startswith('Net'):
        acc_entry = 2
    else:
        acc_entry = 9
    
    return (inv_type, disagg, acc_entry)

def sort_indicators_by_type(indicators):
    """Sort indicators by investment type, disaggregation, then accounting entry"""
    # Convert to clean names if they have _PGDP suffix
    clean_indicators = [ind.replace('_PGDP', '') if ind.endswith('_PGDP') else ind for ind in indicators]
    
    # Sort using the custom key
    sorted_clean = sorted(clean_indicators, key=get_investment_type_order)
    
    # Convert back to original format if needed
    if any(ind.endswith('_PGDP') for ind in indicators):
        return [ind + '_PGDP' for ind in sorted_clean]
    else:
        return sorted_clean

def show_indicator_level_analysis_cs2(selected_country, include_crisis_years=True):
    """Show indicator-level analysis for a specific country - sections 1-6"""
    
    # Load data using standardized CS2 data loading
    final_data, analysis_indicators, metadata = load_case_study_2_data(include_crisis_years)
    
    if final_data is None:
        st.error("❌ Failed to load crisis-excluded data.")
        return
    
    # Country mapping for display
    country_mapping = {
        'Estonia, Republic of': 'Estonia',
        'Latvia, Republic of': 'Latvia', 
        'Lithuania, Republic of': 'Lithuania'
    }
    
    selected_display_country = country_mapping.get(selected_country, selected_country.replace(', Republic of', ''))
    
    # Get timeline information
    timeline = metadata['timeline']
    country_info = timeline[selected_country]
    
    # Get period information based on mode
    if include_crisis_years:
        pre_period = country_info['pre_period_full']
        post_period = country_info['post_period_full']
        period_label = "Full Series"
    else:
        pre_period = country_info['pre_period_crisis_excluded']
        post_period = country_info['post_period_crisis_excluded']
        period_label = "Crisis-Excluded"
    
    st.info(f"""
    **{selected_display_country} Analysis ({period_label}):** Euro adoption on {country_info['adoption_date']}
    - **Pre-Euro Period:** {pre_period[0]} to {pre_period[1]}
    - **Post-Euro Period:** {post_period[0]} to {post_period[1]} (includes adoption year {country_info['adoption_year']})
    """)
    
    # Generate unique session ID for widget keys
    import time
    import random
    session_id = f"{selected_display_country}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    
    # Calculate statistics for selected country
    period_column = metadata['period_column']
    country_stats = calculate_temporal_statistics(final_data, selected_country, analysis_indicators, period_column)
    boxplot_data = create_temporal_boxplot_data(final_data, selected_country, analysis_indicators, period_column) 
    test_results = perform_temporal_volatility_tests(final_data, selected_country, analysis_indicators, period_column)
    
    # Sort indicators by investment type for consistent ordering
    sorted_indicators = sort_indicators_by_type(analysis_indicators)
    
    # 1. Summary Statistics and Boxplots
    st.header("1. Summary Statistics and Boxplots")
    
    # Create side-by-side boxplots for compact layout (matching CS1 optimization)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Mean data boxplot
    mean_data = boxplot_data[boxplot_data['Statistic'] == 'Mean']
    mean_pre = mean_data[mean_data['PERIOD'] == 'Pre-Euro']['Value']
    mean_post = mean_data[mean_data['PERIOD'] == 'Post-Euro']['Value']
    
    bp1 = ax1.boxplot([mean_pre, mean_post], labels=['Pre-Euro', 'Post-Euro'], patch_artist=True)
    bp1['boxes'][0].set_facecolor(COLORBLIND_SAFE[0])
    bp1['boxes'][1].set_facecolor(COLORBLIND_SAFE[1])
    
    study_title_suffix = " (Crisis-Excluded)" if not include_crisis_years else ""
    ax1.set_title(f'Panel A: Distribution of Means\nAcross All Capital Flow Indicators{study_title_suffix}', 
                 fontweight='bold', fontsize=10, pad=10)
    ax1.set_ylabel('Mean (% of GDP, annualized)', fontsize=9)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax1.text(0.02, 0.98, f'Pre-Euro Avg: {mean_pre.mean():.2f}%\nPost-Euro Avg: {mean_post.mean():.2f}%', 
            transform=ax1.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Standard deviations boxplot
    std_data = boxplot_data[boxplot_data['Statistic'] == 'Standard Deviation']
    std_pre = std_data[std_data['PERIOD'] == 'Pre-Euro']['Value']
    std_post = std_data[std_data['PERIOD'] == 'Post-Euro']['Value']
    
    bp2 = ax2.boxplot([std_pre, std_post], labels=['Pre-Euro', 'Post-Euro'], patch_artist=True)
    bp2['boxes'][0].set_facecolor(COLORBLIND_SAFE[0])
    bp2['boxes'][1].set_facecolor(COLORBLIND_SAFE[1])
    
    ax2.set_title(f'Panel B: Distribution of Standard Deviations\nAcross All Capital Flow Indicators{study_title_suffix}', 
                 fontweight='bold', fontsize=10, pad=10)
    ax2.set_ylabel('Std Dev. (% of GDP, annualized)', fontsize=9)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    volatility_ratio = std_pre.mean() / std_post.mean() if std_post.mean() != 0 else float('inf')
    ax2.text(0.02, 0.98, f'Pre-Euro Avg: {std_pre.mean():.2f}%\nPost-Euro Avg: {std_post.mean():.2f}%\nRatio: {volatility_ratio:.2f}x', 
            transform=ax2.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Download buttons in columns for compact layout (matching CS1)
    col1, col2 = st.columns(2)
    
    version_suffix = "_crisis_excluded" if not include_crisis_years else ""
    
    with col1:
        # Download combined figure
        buf_full = io.BytesIO()
        fig.savefig(buf_full, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf_full.seek(0)
        
        st.download_button(
            label="📥 Download Combined Boxplots (PNG)",
            data=buf_full.getvalue(),
            file_name=f"{selected_display_country}_boxplots_combined{version_suffix}.png",
            mime="image/png",
            key=f"download_combined_cs2_{selected_display_country.lower()}{'_crisis_excluded' if not include_crisis_years else '_full'}_{session_id}"
        )
    
    with col2:
        # Option to download individual std dev plot
        fig2_ind, ax2_ind = plt.subplots(1, 1, figsize=(6, 4))
        bp2_ind = ax2_ind.boxplot([std_pre, std_post], labels=['Pre-Euro', 'Post-Euro'], patch_artist=True)
        bp2_ind['boxes'][0].set_facecolor(COLORBLIND_SAFE[0])
        bp2_ind['boxes'][1].set_facecolor(COLORBLIND_SAFE[1])
        ax2_ind.set_title(f'Panel B: Distribution of Standard Deviations Across All Capital Flow Indicators{study_title_suffix}', 
                     fontweight='bold', fontsize=10, pad=10)
        ax2_ind.set_ylabel('Std Dev. (% of GDP, annualized)', fontsize=9)
        ax2_ind.tick_params(axis='both', which='major', labelsize=8)
        ax2_ind.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax2_ind.text(0.02, 0.98, f'Pre-Euro Avg: {std_pre.mean():.2f}%\nPost-Euro Avg: {std_post.mean():.2f}%\nRatio: {volatility_ratio:.2f}x', 
                transform=ax2_ind.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        fig2_ind.tight_layout()
        
        buf2 = io.BytesIO()
        fig2_ind.savefig(buf2, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf2.seek(0)
        plt.close(fig2_ind)
        
        st.download_button(
            label="📥 Download Std Dev Boxplot (PNG)",
            data=buf2.getvalue(),
            file_name=f"{selected_display_country}_stddev_boxplot{version_suffix}.png",
            mime="image/png",
            key=f"download_stddev_cs2_{selected_display_country.lower()}{'_crisis_excluded' if not include_crisis_years else '_full'}_{session_id}"
        )
    
    # Comprehensive Statistical Summary from Boxplots (matching CS1)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Means Across All Indicators:**
        - Pre-Euro: {mean_pre.mean():.2f}% (median: {mean_pre.median():.2f}%)
        - Post-Euro: {mean_post.mean():.2f}% (median: {mean_post.median():.2f}%)
        """)
    
    with col2:
        st.markdown(f"""
        **Standard Deviations Across All Indicators:**
        - Pre-Euro: {std_pre.mean():.2f}% (median: {std_pre.median():.2f}%)
        - Post-Euro: {std_post.mean():.2f}% (median: {std_post.median():.2f}%)
        """)
    
    # Volatility comparison info box
    change_direction = "reduced" if volatility_ratio > 1 else "increased"
    st.info(f"**Volatility Impact:** Euro adoption {change_direction} average volatility by {abs(1-1/volatility_ratio)*100:.1f}%")
    
    st.markdown("---")
    
    # 2. Comprehensive Statistical Summary Table
    st.header("2. Comprehensive Statistical Summary Table")
    
    st.markdown(f"**{selected_display_country} - Pre-Euro vs Post-Euro Statistics{' (Crisis-Excluded)' if not include_crisis_years else ''}**")
    
    # Create side-by-side comparison table
    table_data = []
    for indicator in sorted_indicators:
        clean_name = indicator.replace('_PGDP', '')
        nickname = get_nickname(clean_name)
        
        # Get statistics from country_stats
        indicator_stats = country_stats[country_stats['Indicator'] == clean_name]
        
        pre_stats = indicator_stats[indicator_stats['Period'] == 'Pre-Euro'].iloc[0] if len(indicator_stats[indicator_stats['Period'] == 'Pre-Euro']) > 0 else None
        post_stats = indicator_stats[indicator_stats['Period'] == 'Post-Euro'].iloc[0] if len(indicator_stats[indicator_stats['Period'] == 'Post-Euro']) > 0 else None
        
        if pre_stats is not None and post_stats is not None:
            cv_ratio = pre_stats['CV_Percent']/post_stats['CV_Percent'] if post_stats['CV_Percent'] != 0 else float('inf')
            
            table_data.append({
                'Indicator': nickname,
                'Pre-Euro Mean': f"{pre_stats['Mean']:.2f}",
                'Pre-Euro Std Dev': f"{pre_stats['Std_Dev']:.2f}",
                'Pre-Euro CV%': f"{pre_stats['CV_Percent']:.1f}",
                'Post-Euro Mean': f"{post_stats['Mean']:.2f}",
                'Post-Euro Std Dev': f"{post_stats['Std_Dev']:.2f}",
                'Post-Euro CV%': f"{post_stats['CV_Percent']:.1f}",
                'CV Ratio (Pre/Post)': f"{cv_ratio:.2f}"
            })
    
    summary_df = pd.DataFrame(table_data)
    
    # Apply CS1 styling
    styled_table = summary_df.style.set_properties(**{
        'text-align': 'center',
        'font-size': '10px',
        'border': '1px solid #ddd'
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('font-weight', 'bold'), ('font-size', '11px')]},
        {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},
        {'selector': 'td:first-child', 'props': [('text-align', 'left'), ('font-weight', 'bold')]}
    ])
    
    # Create custom HTML table with strict column width control for CS2 indicator summary
    st.markdown("""
    <style>
    .cs2-indicator-table {
        width: 100% !important;
        border-collapse: collapse !important;
        table-layout: fixed !important;
        font-size: 8px !important;
        font-family: Arial, sans-serif !important;
    }
    .cs2-indicator-table th, .cs2-indicator-table td {
        border: 1px solid #ddd !important;
        padding: 2px !important;
        text-align: center !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    .cs2-indicator-table th {
        background-color: #f0f0f0 !important;
        font-weight: bold !important;
        font-size: 9px !important;
    }
    .cs2-indicator-table th:first-child, .cs2-indicator-table td:first-child {
        width: 220px !important;
        max-width: 220px !important;
        text-align: left !important;
        font-weight: bold !important;
    }
    .cs2-indicator-table th:not(:first-child), .cs2-indicator-table td:not(:first-child) {
        width: 70px !important;
        max-width: 70px !important;
    }
    .cs2-indicator-table tr:nth-child(even) {
        background-color: #f9f9f9 !important;
    }
    @media print {
        .cs2-indicator-table {
            font-size: 7px !important;
        }
        .cs2-indicator-table th:first-child, .cs2-indicator-table td:first-child {
            width: 180px !important;
            max-width: 180px !important;
        }
        .cs2-indicator-table th:not(:first-child), .cs2-indicator-table td:not(:first-child) {
            width: 60px !important;
            max-width: 60px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Convert styled DataFrame to regular DataFrame for HTML generation
    table_df = styled_table.data
    
    # Generate HTML table content
    html_table = '<table class="cs2-indicator-table">'
    html_table += '<thead><tr>'
    for col in table_df.columns:
        html_table += f'<th>{col}</th>'
    html_table += '</tr></thead><tbody>'
    
    for _, row in table_df.iterrows():
        html_table += '<tr>'
        for col in table_df.columns:
            html_table += f'<td>{row[col]}</td>'
        html_table += '</tr>'
    
    html_table += '</tbody></table>'
    
    # Display the custom HTML table
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Summary statistics
    cv_ratios = [float(row['CV Ratio (Pre/Post)'].replace('inf', '999')) for row in table_data]
    cv_ratios = [r for r in cv_ratios if r < 999]  # Exclude infinite ratios
    avg_cv_ratio = np.mean(cv_ratios) if cv_ratios else 0
    
    indicators_pre_higher = sum(1 for r in cv_ratios if r > 1)
    total_indicators = len(cv_ratios)
    
    st.info(f"""
    **Summary:** Statistics for all {len(sorted_indicators)} capital flow indicators comparing pre and post Euro adoption periods.
    - **CV% = Coefficient of Variation** (Std Dev/Mean × 100) - measures relative volatility
    - **Average CV Ratio:** {avg_cv_ratio:.2f} - values >1 indicate higher pre-Euro volatility
    - **Indicators with higher pre-Euro volatility:** {indicators_pre_higher}/{total_indicators} ({indicators_pre_higher/total_indicators*100:.1f}%)
    """)
    
    st.markdown("---")
    
    # 3. Hypothesis Testing Results
    st.header("3. Hypothesis Testing Results")
    
    country_mapping = {'Estonia': 'Estonia, Republic of', 'Latvia': 'Latvia, Republic of', 'Lithuania': 'Lithuania, Republic of'}
    selected_country_full = country_mapping[selected_display_country]
    crisis_text = get_country_specific_crisis_text(selected_country_full) if not include_crisis_years else ""
    
    st.markdown(f"**F-Tests for Equal Variances: {selected_display_country} Pre-Euro vs Post-Euro{' (Crisis-Excluded)' if not include_crisis_years else ''}** | H₀: Equal variances | H₁: Different variances | α = 0.05{f' | Excludes: {crisis_text}' if not include_crisis_years else ''}")
    
    # Create hypothesis test results table (matching CS1 format)
    results_display = test_results.copy()
    results_display['Sort_Key'] = results_display['Indicator'].apply(get_investment_type_order)
    results_display = results_display.sort_values('Sort_Key')
    
    test_table_data = []
    for _, row in results_display.iterrows():
        nickname = get_nickname(row['Indicator'])
        
        # Significance symbols
        if row['P_Value'] < 0.001:
            significance = '***'
        elif row['P_Value'] < 0.01:
            significance = '**'
        elif row['P_Value'] < 0.05:
            significance = '*'
        else:
            significance = ''
        
        higher_vol = 'Pre-Euro' if row['Pre_Euro_Higher_Volatility'] else 'Post-Euro'
        
        test_table_data.append({
            'Indicator': nickname,
            'F-Statistic': f"{row['F_Statistic']:.2f}",
            'P-Value': f"{row['P_Value']:.4f}",
            'Significance': significance,
            'Higher Volatility': higher_vol
        })
    
    test_df = pd.DataFrame(test_table_data)
    
    # Two-column layout (matching CS1)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Apply CS1 styling to test results table
        styled_test_table = test_df.style.set_properties(**{
            'text-align': 'center',
            'font-size': '11px',
            'border': '1px solid #ddd'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#e6f3ff'), ('font-weight', 'bold'), ('text-align', 'center')]},
            {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},
            {'selector': 'td:first-child', 'props': [('text-align', 'left')]}
        ])
        
        # Create custom HTML table with strict column width control for CS2 test results
        st.markdown("""
        <style>
        .cs2-test-table {
            width: 100% !important;
            border-collapse: collapse !important;
            table-layout: fixed !important;
            font-size: 8px !important;
            font-family: Arial, sans-serif !important;
        }
        .cs2-test-table th, .cs2-test-table td {
            border: 1px solid #ddd !important;
            padding: 2px !important;
            text-align: center !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
        .cs2-test-table th {
            background-color: #f0f0f0 !important;
            font-weight: bold !important;
            font-size: 9px !important;
        }
        .cs2-test-table th:first-child, .cs2-test-table td:first-child {
            width: 220px !important;
            max-width: 220px !important;
            text-align: left !important;
            font-weight: bold !important;
        }
        .cs2-test-table th:not(:first-child), .cs2-test-table td:not(:first-child) {
            width: 70px !important;
            max-width: 70px !important;
        }
        .cs2-test-table tr:nth-child(even) {
            background-color: #f9f9f9 !important;
        }
        @media print {
            .cs2-test-table {
                font-size: 7px !important;
            }
            .cs2-test-table th:first-child, .cs2-test-table td:first-child {
                width: 180px !important;
                max-width: 180px !important;
            }
            .cs2-test-table th:not(:first-child), .cs2-test-table td:not(:first-child) {
                width: 60px !important;
                max-width: 60px !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Convert styled DataFrame to regular DataFrame for HTML generation
        test_df = styled_test_table.data
        
        # Generate HTML table content
        html_table = '<table class="cs2-test-table">'
        html_table += '<thead><tr>'
        for col in test_df.columns:
            html_table += f'<th>{col}</th>'
        html_table += '</tr></thead><tbody>'
        
        for _, row in test_df.iterrows():
            html_table += '<tr>'
            for col in test_df.columns:
                html_table += f'<td>{row[col]}</td>'
            html_table += '</tr>'
        
        html_table += '</tbody></table>'
        
        # Display the custom HTML table
        st.markdown(html_table, unsafe_allow_html=True)
        st.caption("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    
    with col2:
        st.markdown("**Legend:**")
        st.markdown("- **F-Statistic**: Ratio of variances")
        st.markdown("- **P-Value**: Probability of observing this difference by chance")
        st.markdown("- **Higher Volatility**: Period with greater variance")
    
    # Summary metrics (three columns matching CS1)
    total_indicators = len(test_results)
    pre_higher_count = test_results['Pre_Euro_Higher_Volatility'].sum()
    sig_5pct_count = test_results['Significant_5pct'].sum()
    sig_1pct_count = test_results['Significant_1pct'].sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pre-Euro Higher Volatility", f"{pre_higher_count}/{total_indicators}", f"{pre_higher_count/total_indicators*100:.1f}%")
    with col2:
        st.metric("Significant (5%)", f"{sig_5pct_count}/{total_indicators}", f"{sig_5pct_count/total_indicators*100:.1f}%")
    with col3:
        st.metric("Significant (1%)", f"{sig_1pct_count}/{total_indicators}", f"{sig_1pct_count/total_indicators*100:.1f}%")
    
    # Color-coded conclusion box (matching CS1)
    if pre_higher_count/total_indicators > 0.7:
        conclusion = "Strong evidence that Euro adoption reduced capital flow volatility"
        conclusion_type = "success"
    elif pre_higher_count/total_indicators > 0.5:
        conclusion = "Moderate evidence that Euro adoption reduced capital flow volatility"
        conclusion_type = "info"
    else:
        conclusion = "Mixed evidence for Euro adoption's impact on capital flow volatility"
        conclusion_type = "warning"
    
    if conclusion_type == "success":
        st.success(f"**Conclusion:** {conclusion} in {selected_display_country}.")
    elif conclusion_type == "info":
        st.info(f"**Conclusion:** {conclusion} in {selected_display_country}.")
    else:
        st.warning(f"**Conclusion:** {conclusion} in {selected_display_country}.")
    
    st.markdown("---")
    
    # 4. Time Series Analysis
    st.header("4. Time Series Analysis")
    
    # Create date column for plotting
    final_data_copy = final_data.copy()
    final_data_copy['Date'] = pd.to_datetime(
        final_data_copy['YEAR'].astype(str) + '-' + 
        ((final_data_copy['QUARTER'] - 1) * 3 + 1).astype(str) + '-01'
    )
    
    # Filter for selected country
    country_data = final_data_copy[final_data_copy['COUNTRY'] == selected_country]
    
    # Euro adoption date
    adoption_year = country_info['adoption_year']
    adoption_date = pd.to_datetime(f'{adoption_year}-01-01')
    
    version_suffix = "_crisis_excluded" if not include_crisis_years else ""
    
    
    # Create grid layout for time series - process in groups of 4 for 2x2 grids (matching CS1)
    n_indicators = len(sorted_indicators)
    
    # Process indicators in groups of 4 for 2x2 grids
    for group_idx in range(0, n_indicators, 4):
        group_indicators = sorted_indicators[group_idx:min(group_idx+4, n_indicators)]
        n_in_group = len(group_indicators)
        
        # Create 2x2 grid (or smaller if less than 4 indicators remain)
        n_cols = min(2, n_in_group)
        n_rows = (n_in_group + 1) // 2
        
        fig_group, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3.5))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, indicator in enumerate(group_indicators):
            ax = axes[idx]
            i = group_idx + idx  # Overall index
            
            clean_name = indicator.replace('_PGDP', '').replace('_', ' ')
            
            # Get test statistic for title
            indicator_clean = indicator.replace('_PGDP', '')
            f_stat = test_results[test_results['Indicator'] == indicator_clean]['F_Statistic'].iloc[0] if len(test_results[test_results['Indicator'] == indicator_clean]) > 0 else 0
            
            # Panel letter (A, B, C, etc.)
            panel_letter = chr(65 + i)
            
            # Add crisis period shading for crisis-excluded version (only on first plot of group)
            if not include_crisis_years:
                add_country_specific_crisis_shading(ax, selected_country, include_labels=(idx == 0))
            
            # Plot pre-Euro data
            pre_data = country_data[country_data[period_column] == 'Pre-Euro']
            if not include_crisis_years and len(pre_data) > 0:
                # Crisis-excluded: segment plotting to avoid connecting across excluded periods
                pre_data_sorted = pre_data.sort_values('Date')
                segments = []
                current_segment = []
                
                for _, row in pre_data_sorted.iterrows():
                    if len(current_segment) == 0:
                        current_segment.append(row)
                    else:
                        last_date = current_segment[-1]['Date']
                        current_date = row['Date']
                        gap_years = (current_date - last_date).days / 365.25
                        
                        if gap_years > 2:  # Gap indicates crisis exclusion
                            if current_segment:
                                segments.append(pd.DataFrame(current_segment))
                            current_segment = [row]
                        else:
                            current_segment.append(row)
                
                if current_segment:
                    segments.append(pd.DataFrame(current_segment))
                
                # Plot each segment separately
                for j, segment in enumerate(segments):
                    if len(segment) > 0:
                        label = 'Pre-Euro' if j == 0 and idx == 0 else None  # Only label first segment of first plot
                        ax.plot(segment['Date'], segment[indicator], 
                               color=COLORBLIND_SAFE[0], linewidth=1.5, label=label)
            else:
                # Normal plotting for full series
                if len(pre_data) > 0:
                    label = 'Pre-Euro' if idx == 0 else None  # Only label first plot of group
                    ax.plot(pre_data['Date'], pre_data[indicator], 
                           color=COLORBLIND_SAFE[0], linewidth=1.5, label=label)
            
            # Plot post-Euro data
            post_data = country_data[country_data[period_column] == 'Post-Euro']
            if not include_crisis_years and len(post_data) > 0:
                # Crisis-excluded: segment plotting
                post_data_sorted = post_data.sort_values('Date')
                segments = []
                current_segment = []
                
                for _, row in post_data_sorted.iterrows():
                    if len(current_segment) == 0:
                        current_segment.append(row)
                    else:
                        last_date = current_segment[-1]['Date']
                        current_date = row['Date']
                        gap_years = (current_date - last_date).days / 365.25
                        
                        if gap_years > 2:  # Gap indicates crisis exclusion
                            if current_segment:
                                segments.append(pd.DataFrame(current_segment))
                            current_segment = [row]
                        else:
                            current_segment.append(row)
                
                if current_segment:
                    segments.append(pd.DataFrame(current_segment))
                
                # Plot each segment separately
                for j, segment in enumerate(segments):
                    if len(segment) > 0:
                        label = 'Post-Euro' if j == 0 and idx == 0 else None  # Only label first segment of first plot
                        ax.plot(segment['Date'], segment[indicator], 
                               color=COLORBLIND_SAFE[1], linewidth=1.5, label=label)
            else:
                # Normal plotting for full series
                if len(post_data) > 0:
                    label = 'Post-Euro' if idx == 0 else None  # Only label first plot of group
                    ax.plot(post_data['Date'], post_data[indicator], 
                           color=COLORBLIND_SAFE[1], linewidth=1.5, label=label)
            
            # Add Euro adoption line
            ax.axvline(x=adoption_date, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                      label='Euro Adoption' if idx == 0 else "")
            
            # Formatting (following CS1 standards exactly)
            ax.set_title(f'{panel_letter}: {clean_name}\n(F-stat: {f_stat:.2f}){study_title_suffix}', 
                        fontweight='bold', fontsize=9, pad=5)
            ax.set_ylabel('% of GDP', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            
            # Show legend only on first plot of each group
            if idx == 0:
                ax.legend(loc='best', fontsize=8, frameon=True)
        
        # Hide unused subplots if any
        for idx in range(len(group_indicators), len(axes)):
            axes[idx].set_visible(False)
        
        fig_group.tight_layout()
        st.pyplot(fig_group)
        
        # Download button for this group
        buf_group = io.BytesIO()
        fig_group.savefig(buf_group, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf_group.seek(0)
        
        group_letter = chr(65 + group_idx // 4)  # A, B, C for each group
        st.download_button(
            label=f"📥 Download Time Series Group {group_letter} ({selected_display_country}) (PNG)",
            data=buf_group.getvalue(),
            file_name=f"{selected_display_country}_timeseries_group_{group_letter}{version_suffix}.png",
            mime="image/png",
            key=f"download_ts_group_cs2_{selected_display_country.lower()}_{group_idx}{'_crisis_excluded' if not include_crisis_years else '_full'}_{session_id}"
        )
    
    # Create individual figures for detailed downloads (matching CS1)
    with st.expander(f"📥 Download Individual Time Series Charts ({selected_display_country})"):
        n_cols_download = 3
        
        for batch_idx in range(0, n_indicators, n_cols_download):
            cols = st.columns(n_cols_download)
            batch_indicators = sorted_indicators[batch_idx:min(batch_idx+n_cols_download, n_indicators)]
            
            for col_idx, indicator in enumerate(batch_indicators):
                i = batch_idx + col_idx
                clean_name = indicator.replace('_PGDP', '').replace('_', ' ')
                
                # Create individual figure
                fig_ind, ax_ind = plt.subplots(1, 1, figsize=(6, 3))
                
                # Get test statistic for title
                indicator_clean = indicator.replace('_PGDP', '')
                f_stat = test_results[test_results['Indicator'] == indicator_clean]['F_Statistic'].iloc[0] if len(test_results[test_results['Indicator'] == indicator_clean]) > 0 else 0
                
                # Add crisis period shading if applicable
                if not include_crisis_years:
                    add_country_specific_crisis_shading(ax_ind, selected_country, include_labels=True)
                
                # Plot data with proper segmentation for crisis-excluded
                pre_data = country_data[country_data[period_column] == 'Pre-Euro']
                post_data = country_data[country_data[period_column] == 'Post-Euro']
                
                # Use the same plotting logic as above for individual charts
                if not include_crisis_years and len(pre_data) > 0:
                    # Segmented plotting for crisis-excluded
                    pre_data_sorted = pre_data.sort_values('Date')
                    segments = []
                    current_segment = []
                    
                    for _, row in pre_data_sorted.iterrows():
                        if len(current_segment) == 0:
                            current_segment.append(row)
                        else:
                            last_date = current_segment[-1]['Date']
                            current_date = row['Date']
                            gap_years = (current_date - last_date).days / 365.25
                            
                            if gap_years > 2:
                                if current_segment:
                                    segments.append(pd.DataFrame(current_segment))
                                current_segment = [row]
                            else:
                                current_segment.append(row)
                    
                    if current_segment:
                        segments.append(pd.DataFrame(current_segment))
                    
                    for j, segment in enumerate(segments):
                        if len(segment) > 0:
                            label = 'Pre-Euro' if j == 0 else None
                            ax_ind.plot(segment['Date'], segment[indicator], 
                                       color=COLORBLIND_SAFE[0], linewidth=1.5, label=label)
                else:
                    if len(pre_data) > 0:
                        ax_ind.plot(pre_data['Date'], pre_data[indicator], 
                                   color=COLORBLIND_SAFE[0], linewidth=1.5, label='Pre-Euro')
                
                # Same for post-Euro data
                if not include_crisis_years and len(post_data) > 0:
                    post_data_sorted = post_data.sort_values('Date')
                    segments = []
                    current_segment = []
                    
                    for _, row in post_data_sorted.iterrows():
                        if len(current_segment) == 0:
                            current_segment.append(row)
                        else:
                            last_date = current_segment[-1]['Date']
                            current_date = row['Date']
                            gap_years = (current_date - last_date).days / 365.25
                            
                            if gap_years > 2:
                                if current_segment:
                                    segments.append(pd.DataFrame(current_segment))
                                current_segment = [row]
                            else:
                                current_segment.append(row)
                    
                    if current_segment:
                        segments.append(pd.DataFrame(current_segment))
                    
                    for j, segment in enumerate(segments):
                        if len(segment) > 0:
                            label = 'Post-Euro' if j == 0 else None
                            ax_ind.plot(segment['Date'], segment[indicator], 
                                       color=COLORBLIND_SAFE[1], linewidth=1.5, label=label)
                else:
                    if len(post_data) > 0:
                        ax_ind.plot(post_data['Date'], post_data[indicator], 
                                   color=COLORBLIND_SAFE[1], linewidth=1.5, label='Post-Euro')
                
                # Add Euro adoption line
                ax_ind.axvline(x=adoption_date, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Euro Adoption')
                
                # Formatting
                ax_ind.set_title(f'{clean_name} (F-statistic: {f_stat:.2f}){study_title_suffix}', fontweight='bold', fontsize=9)
                ax_ind.set_ylabel('% of GDP (annualized)', fontsize=8)
                ax_ind.set_xlabel('Year', fontsize=8)
                ax_ind.tick_params(axis='both', which='major', labelsize=7)
                ax_ind.legend(loc='best', fontsize=8)
                ax_ind.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
                fig_ind.tight_layout()
                
                # Save to buffer
                buf_ind = io.BytesIO()
                fig_ind.savefig(buf_ind, format='png', dpi=300, bbox_inches='tight', facecolor='white')
                buf_ind.seek(0)
                
                # Add download button
                clean_filename = clean_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
                with cols[col_idx]:
                    st.download_button(
                        label=f"📥 {clean_name}",
                        data=buf_ind.getvalue(),
                        file_name=f"{selected_display_country}_{clean_filename}_timeseries{version_suffix}.png",
                        mime="image/png",
                        key=f"download_ts_cs2_{selected_display_country.lower()}_{i}_{clean_filename}{'_crisis_excluded' if not include_crisis_years else '_full'}_{session_id}"
                    )
                
                plt.close(fig_ind)
    
    st.markdown("---")
    
    # 5. Key Findings Summary
    st.header("5. Key Findings Summary")
    
    # Generate dynamic findings based on actual results
    volatility_summary = "reduced" if pre_higher_count/total_indicators > 0.5 else "increased"
    significant_indicators = sig_5pct_count
    most_significant_indicators = sig_1pct_count
    
    # Crisis period label for findings
    crisis_label = " (excluding crisis periods)" if not include_crisis_years else ""
    
    # Two-column layout (matching CS1)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### Statistical Evidence for {selected_display_country}{crisis_label}:
        
        - **{pre_higher_count}/{total_indicators} capital flow indicators** ({pre_higher_count/total_indicators*100:.1f}%) showed higher volatility before Euro adoption
        - **{significant_indicators}/{total_indicators} indicators** ({significant_indicators/total_indicators*100:.1f}%) show statistically significant differences (p<0.05)
        - **{most_significant_indicators} indicators** show highly significant differences (p<0.01)
        - **Average volatility change** of {abs(1-1/volatility_ratio)*100:.1f}% after Euro adoption in {adoption_year}
        
        **Most significant flow types:** {', '.join([get_nickname(row['Indicator']) for _, row in test_results.nsmallest(3, 'P_Value').iterrows()])}
        """)
    
    with col2:
        st.markdown(f"""
        ### Additional Statistical Context:
        
        - **Temporal analysis:** Before/after comparison using {adoption_year} as adoption threshold
        - **Statistical methodology:** F-test for variance equality at 5% significance level
        - **Data completeness:** {len(final_data)} observations across {total_indicators} capital flow indicators
        - **Cross-validation:** Results consistent across multiple volatility measures (CV%, standard deviation)
        
        **Analytical approach:** Temporal comparison focusing on structural changes in volatility patterns.
        """)
    
    st.markdown("---")
    
    # 6. Download Results
    st.header("6. Download Results")
    
    # Four-column download layout (matching CS1)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Summary Statistics CSV
        summary_csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📊 Summary Statistics CSV",
            data=summary_csv,
            file_name=f"{selected_display_country}_summary_statistics{version_suffix}.csv",
            mime="text/csv",
            key=f"download_summary_csv_cs2_{selected_display_country.lower()}{'_crisis_excluded' if not include_crisis_years else '_full'}_{session_id}",
            help="Comprehensive statistical summary table with CV ratios"
        )
    
    with col2:
        # Hypothesis Test Results CSV
        test_csv = test_results.to_csv(index=False)
        st.download_button(
            label="🧪 Hypothesis Test Results CSV",
            data=test_csv,
            file_name=f"{selected_display_country}_hypothesis_tests{version_suffix}.csv",
            mime="text/csv",
            key=f"download_tests_csv_cs2_{selected_display_country.lower()}{'_crisis_excluded' if not include_crisis_years else '_full'}_{session_id}",
            help="F-test results with significance levels and conclusions"
        )
    
    with col3:
        # Country Statistics CSV
        country_csv = country_stats.to_csv(index=False)
        st.download_button(
            label="🇱🇹 Country Statistics CSV",
            data=country_csv,
            file_name=f"{selected_display_country}_country_statistics{version_suffix}.csv",
            mime="text/csv",
            key=f"download_country_csv_cs2_{selected_display_country.lower()}{'_crisis_excluded' if not include_crisis_years else '_full'}_{session_id}",
            help="Detailed temporal statistics by indicator and period"
        )
    
    with col4:
        # HTML Report Generator
        if st.button(
            "📄 Generate HTML Report",
            key=f"generate_html_cs2_{selected_display_country.lower()}{'_crisis_excluded' if not include_crisis_years else '_full'}_{session_id}",
            help="Generate comprehensive HTML report with embedded charts"
        ):
            with st.spinner('Generating comprehensive HTML report...'):
                try:
                    # Generate HTML report content
                    html_content = generate_cs2_html_report(
                        selected_display_country, 
                        include_crisis_years,
                        summary_df,
                        test_results,
                        country_stats,
                        pre_higher_count,
                        total_indicators,
                        significant_indicators,
                        adoption_year,
                        volatility_ratio
                    )
                    
                    # Create download button for HTML
                    import datetime
                    current_date = datetime.datetime.now().strftime("%Y%m%d")
                    html_filename = f"{selected_display_country}_euro_adoption_report_{current_date}{version_suffix}.html"
                    
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=html_content,
                        file_name=html_filename,
                        mime="text/html",
                        key=f"download_html_cs2_{selected_display_country.lower()}{'_crisis_excluded' if not include_crisis_years else '_full'}_{session_id}"
                    )
                    
                    st.success("✅ HTML report generated successfully!")
                    st.info("💡 **Tip:** Open the HTML file in your browser and print to PDF for a professional report.")
                    
                except Exception as e:
                    st.error(f"❌ Error generating HTML report: {str(e)}")
                    st.info("📊 Individual CSV downloads are still available above.")

def generate_cs2_html_report(selected_display_country, include_crisis_years, summary_df, test_results, country_stats, pre_higher_count, total_indicators, significant_indicators, adoption_year, volatility_ratio):
    """Generate comprehensive HTML report for CS2 analysis"""
    import base64
    import io
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Load and filter data  
    final_data, analysis_indicators, metadata = load_case_study_2_data(include_crisis_years)
    
    # Map display name to data name
    country_mapping = {
        'Estonia': 'Estonia, Republic of',
        'Latvia': 'Latvia, Republic of', 
        'Lithuania': 'Lithuania, Republic of'
    }
    selected_country = country_mapping[selected_display_country]
    
    df_filtered = final_data[final_data['COUNTRY'] == selected_country].copy()
    
    # Create date column
    df_filtered['Date'] = pd.to_datetime(
        df_filtered['YEAR'].astype(str) + '-' + 
        ((df_filtered['QUARTER'] - 1) * 3 + 1).astype(str) + '-01'
    )
    
    # Get period column based on study version
    period_col = metadata['period_column']
    
    # Generate charts and encode to base64
    def fig_to_base64(fig):
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)
        return img_base64
    
    # Chart 1: Summary Statistics Boxplots
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Panel A: Distribution of Means
    boxplot_data = create_temporal_boxplot_data(final_data, selected_country, analysis_indicators, period_col)
    
    mean_data = boxplot_data[boxplot_data['Statistic'] == 'Mean']
    mean_pre = mean_data[mean_data['PERIOD'] == 'Pre-Euro']['Value']
    mean_post = mean_data[mean_data['PERIOD'] == 'Post-Euro']['Value']
    
    bp1 = ax1.boxplot([mean_pre, mean_post], labels=['Pre-Euro', 'Post-Euro'], patch_artist=True)
    bp1['boxes'][0].set_facecolor(COLORBLIND_SAFE[0])
    bp1['boxes'][1].set_facecolor(COLORBLIND_SAFE[1])
    
    study_title_suffix = " (Crisis-Excluded)" if not include_crisis_years else ""
    ax1.set_title(f'Panel A: Distribution of Means Across All Indicators{study_title_suffix}', fontweight='bold')
    ax1.set_ylabel('Mean (% of GDP, annualized)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Panel B: Distribution of Standard Deviations
    std_data = boxplot_data[boxplot_data['Statistic'] == 'Standard Deviation']
    std_pre = std_data[std_data['PERIOD'] == 'Pre-Euro']['Value']
    std_post = std_data[std_data['PERIOD'] == 'Post-Euro']['Value']
    
    bp2 = ax2.boxplot([std_pre, std_post], labels=['Pre-Euro', 'Post-Euro'], patch_artist=True)
    bp2['boxes'][0].set_facecolor(COLORBLIND_SAFE[0])
    bp2['boxes'][1].set_facecolor(COLORBLIND_SAFE[1])
    ax2.set_title(f'Panel B: Distribution of Standard Deviations Across All Indicators{study_title_suffix}', fontweight='bold')
    ax2.set_ylabel('Standard Deviation (% of GDP, annualized)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{selected_display_country} Capital Flows Analysis - Summary Statistics{study_title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart1_base64 = fig_to_base64(fig1)
    
    # Chart 2: Time Series Analysis (top 6 indicators)
    sorted_indicators = sort_indicators_by_type(analysis_indicators)[:6]
    
    n_rows = 3
    fig2, axes = plt.subplots(n_rows, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    adoption_date = pd.to_datetime(f'{adoption_year}-01-01')
    
    for i, indicator in enumerate(sorted_indicators):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        clean_name = indicator.replace('_PGDP', '')
        nickname = get_nickname(clean_name)
        
        # Plot pre-Euro and post-Euro data
        pre_data = df_filtered[df_filtered[period_col] == 'Pre-Euro'].sort_values('Date')
        post_data = df_filtered[df_filtered[period_col] == 'Post-Euro'].sort_values('Date')
        
        if not include_crisis_years:
            # For crisis-excluded: plot data in segments to avoid connecting across excluded periods
            for period_name, data in [('Pre-Euro', pre_data), ('Post-Euro', post_data)]:
                if len(data) == 0:
                    continue
                    
                # Find segments separated by crisis periods
                segments = []
                current_segment = []
                
                for _, row in data.iterrows():
                    if len(current_segment) == 0:
                        current_segment.append(row)
                    else:
                        # Check for gaps indicating crisis exclusion
                        last_date = current_segment[-1]['Date']
                        current_date = row['Date']
                        gap_years = (current_date - last_date).days / 365.25
                        
                        if gap_years > 2:  # Gap indicates crisis exclusion
                            segments.append(pd.DataFrame(current_segment))
                            current_segment = [row]
                        else:
                            current_segment.append(row)
                
                if current_segment:
                    segments.append(pd.DataFrame(current_segment))
                
                # Plot each segment
                color = COLORBLIND_SAFE[0] if period_name == 'Pre-Euro' else COLORBLIND_SAFE[1]
                for j, segment in enumerate(segments):
                    if len(segment) > 0:
                        ax.plot(segment['Date'], segment[indicator], 
                               color=color, linewidth=2.5, 
                               label=period_name if j == 0 else "", marker='o', markersize=2)
        else:
            # Normal plotting for full series
            if len(pre_data) > 0:
                ax.plot(pre_data['Date'], pre_data[indicator], 
                       color=COLORBLIND_SAFE[0], linewidth=2.5, 
                       label='Pre-Euro', marker='o', markersize=2)
            if len(post_data) > 0:
                ax.plot(post_data['Date'], post_data[indicator], 
                       color=COLORBLIND_SAFE[1], linewidth=2.5, 
                       label='Post-Euro', marker='o', markersize=2)
        
        # Add Euro adoption line
        ax.axvline(x=adoption_date, color='green', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(adoption_date, ax.get_ylim()[1] * 0.9, f'Euro Adoption\n{adoption_year}', 
               ha='center', va='top', fontsize=8, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        # Add crisis period shading if showing crisis-excluded version
        if not include_crisis_years:
            selected_country = {'Estonia': 'Estonia, Republic of', 'Latvia': 'Latvia, Republic of', 'Lithuania': 'Lithuania, Republic of'}[selected_display_country]
            add_country_specific_crisis_shading(ax, selected_country, include_labels=(i == 0))
        
        # Formatting
        ax.set_title(nickname, fontweight='bold', fontsize=10)
        ax.set_ylabel('% of GDP (annualized)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    
    # Hide unused subplots
    for i in range(len(sorted_indicators), len(axes)):
        axes[i].set_visible(False)
    
    crisis_text = " (Crisis Years Excluded)" if not include_crisis_years else ""
    plt.suptitle(f'{selected_display_country} - Time Series Analysis{crisis_text}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    chart2_base64 = fig_to_base64(fig2)
    
    # Generate HTML content
    crisis_title = " (Crisis Years Excluded)" if not include_crisis_years else ""
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Case Study 2: {selected_display_country} Euro Adoption Analysis{crisis_title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .section {{ margin: 30px 0; }}
            .chart {{ text-align: center; margin: 20px 0; }}
            .chart img {{ max-width: 100%; height: auto; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .significant {{ background-color: #ffeb3b; }}
            .summary-box {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .methodology {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Case Study 2: {selected_display_country} Euro Adoption Impact Analysis</h1>
            <h2>Capital Flow Volatility Comparison{crisis_title}</h2>
            <p><strong>Generated:</strong> {report_date}</p>
        </div>
        
        <div class="methodology">
            <h3>Methodology</h3>
            <p><strong>Research Question:</strong> How did Euro adoption affect capital flow volatility in {selected_display_country}?</p>
            <p><strong>Approach:</strong> Temporal comparison of capital flow volatility before and after Euro adoption ({adoption_year})</p>
            <p><strong>Statistical Test:</strong> F-tests for equality of variances between Pre-Euro and Post-Euro periods</p>
            <p><strong>Data Period:</strong> 1999-2025 quarterly data, normalized to % of GDP (annualized)</p>
            {"<p><strong>Crisis Exclusion:</strong> Global Financial Crisis (2008-2010) and COVID-19 (2020-2022) periods excluded from analysis</p>" if not include_crisis_years else ""}
        </div>
        
        <div class="section">
            <h2>1. Summary Statistics and Distributions</h2>
            <div class="chart">
                <img src="data:image/png;base64,{chart1_base64}" alt="Summary Statistics Boxplots">
            </div>
        </div>
        
        <div class="section">
            <h2>2. Statistical Summary Table</h2>
            <table>
                <thead>
                    <tr>
                        <th rowspan="2">Capital Flow Indicator</th>
                        <th colspan="3">Pre-Euro Period</th>
                        <th colspan="3">Post-Euro Period</th>
                        <th rowspan="2">CV Ratio</th>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <th>Std Dev</th>
                        <th>CV%</th>
                        <th>Mean</th>
                        <th>Std Dev</th>
                        <th>CV%</th>
                    </tr>
                </thead>
                <tbody>"""
    
    for _, row in summary_df.iterrows():
        html_content += f"""
                    <tr>
                        <td>{row['Indicator']}</td>
                        <td>{row['Pre-Euro Mean']}%</td>
                        <td>{row['Pre-Euro Std Dev']}%</td>
                        <td>{row['Pre-Euro CV%']}%</td>
                        <td>{row['Post-Euro Mean']}%</td>
                        <td>{row['Post-Euro Std Dev']}%</td>
                        <td>{row['Post-Euro CV%']}%</td>
                        <td>{row['CV Ratio (Pre/Post)']}</td>
                    </tr>"""
    
    html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>3. Hypothesis Testing Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Capital Flow Indicator</th>
                        <th>F-Statistic</th>
                        <th>P-Value</th>
                        <th>Significant (α=0.05)</th>
                        <th>Higher Volatility Period</th>
                    </tr>
                </thead>
                <tbody>"""
    
    significant_count = 0
    for _, row in test_results.iterrows():
        clean_name = row['Indicator']
        nickname = get_nickname(clean_name)
        
        sig_class = "significant" if row['Significant_5pct'] else ""
        if row['Significant_5pct']:
            significant_count += 1
        
        higher_period = 'Pre-Euro' if row['Pre_Euro_Higher_Volatility'] else 'Post-Euro'
        
        html_content += f"""
                    <tr class="{sig_class}">
                        <td>{nickname}</td>
                        <td>{row['F_Statistic']:.3f}</td>
                        <td>{row['P_Value']:.4f}</td>
                        <td>{'Yes' if row['Significant_5pct'] else 'No'}</td>
                        <td>{higher_period}</td>
                    </tr>"""
    
    html_content += f"""
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>4. Time Series Analysis</h2>
            <div class="chart">
                <img src="data:image/png;base64,{chart2_base64}" alt="Time Series Analysis">
            </div>
        </div>
        
        <div class="section">
            <h2>5. Key Findings Summary</h2>
            <div class="summary-box">
                <h3>Statistical Evidence</h3>
                <ul>
                    <li><strong>Significant Results:</strong> {significant_count} out of {total_indicators} indicators show statistically significant volatility differences (α = 0.05)</li>
                    <li><strong>Pre-Euro Higher Volatility:</strong> {pre_higher_count} out of {total_indicators} indicators ({pre_higher_count/total_indicators*100:.1f}%)</li>
                    <li><strong>Volatility Change:</strong> Average {abs(1-1/volatility_ratio)*100:.1f}% {"reduction" if volatility_ratio > 1 else "increase"} in volatility after Euro adoption</li>
                    <li><strong>Policy Implication:</strong> {"Euro adoption appears to have reduced capital flow volatility" if pre_higher_count/total_indicators > 0.5 else "Euro adoption shows mixed effects on capital flow volatility"}</li>
                </ul>
                
                <h3>Methodological Notes</h3>
                <ul>
                    <li>Analysis uses quarterly Balance of Payments data normalized to % of GDP (annualized)</li>
                    <li>F-tests compare variance equality between Pre-Euro and Post-Euro periods</li>
                    <li>Yellow highlighting indicates statistically significant results at 5% level</li>
                    {"<li>Crisis periods (2008-2010, 2020-2022) excluded to focus on structural changes from Euro adoption</li>" if not include_crisis_years else ""}
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>6. Technical Appendix</h2>
            <p><strong>Data Source:</strong> IMF Balance of Payments Statistics, World Economic Outlook Database</p>
            <p><strong>Coverage:</strong> {len(analysis_indicators)} capital flow indicators across quarterly observations (1999-2025)</p>
            <p><strong>Statistical Software:</strong> Python with SciPy statistical package</p>
            <p><strong>Euro Adoption Date:</strong> {selected_display_country} adopted the Euro on January 1, {adoption_year}</p>
            <p><strong>Report Generation:</strong> Automated analysis pipeline with Claude Code assistance</p>
        </div>
    </body>
    </html>"""
    
    return html_content


def main():
    """Main Case Study 2 application"""
    
    # Initialize session state for unique widget keys
    import time
    import hashlib
    
    # Create a truly unique base session ID only once per session
    if 'cs2_base_session_id' not in st.session_state:
        import random
        current_time = str(int(time.time() * 1000))
        random_component = str(random.randint(10000, 99999))
        st.session_state.cs2_base_session_id = f"{current_time[-6:]}_{random_component}"
    
    base_session_id = st.session_state.cs2_base_session_id
    
    # Get tab-specific identifier if available (for unique widget keys across tabs)
    tab_id = st.session_state.get('current_cs2_tab_id', 'default')
    unique_key_prefix = f"{base_session_id}_{tab_id}"
    
    # Title and header
    st.title("🇪🇺 Euro Adoption Impact Analysis")
    st.subheader("Case Study 2: Baltic Countries Capital Flow Volatility")
    
    st.markdown("""
    **Research Question:** How does Euro adoption affect capital flow volatility?
    
    **Hypothesis:** Euro adoption reduces capital flow volatility through increased monetary stability
    
    **Countries:** Estonia (2011), Latvia (2014), Lithuania (2015)
    """)
    
    # Study version selector
    st.markdown("---")
    st.subheader("📊 Study Configuration")
    
    study_version = st.radio(
        "Select study version:",
        ["Full Series", "Crisis-Excluded"],
        index=0,
        help="Full Series uses all available data. Crisis-Excluded removes major crisis periods (GFC 2008-2010 + COVID 2020-2022).",
        key=f"cs2_study_version_cs2_{unique_key_prefix}"
    )
    
    include_crisis_years = (study_version == "Full Series")
    
    # Create run-specific counter to handle app reruns
    run_key = f"cs2_run_counter_{study_version}"
    if run_key not in st.session_state:
        st.session_state[run_key] = 0
    st.session_state[run_key] += 1
    
    # Build final session ID with all components for uniqueness
    version_key = "full" if include_crisis_years else "crisis"
    session_id = f"{base_session_id}_{version_key}_{st.session_state[run_key]}"
    
    if study_version == "Full Series":
        st.info("📈 **Full Series:** Maximizes data usage with all available pre/post Euro periods (asymmetric windows)")
    else:
        st.warning(f"🚫 **Crisis-Excluded:** Removes major crisis periods ({get_country_specific_crisis_text(selected_country)}) to isolate Euro adoption effects")
    
    st.markdown("---")
    
    # Data and Methodology section  
    with st.expander("📋 Data and Methodology", expanded=False):
        expanded_timeline = create_expanded_euro_adoption_timeline()
        
        st.markdown(f"""
        ### Temporal Analysis Design ({study_version})
        - **Methodology:** Before-after comparison for each country
        - **Analysis Periods:** Asymmetric windows maximizing available data
        - **Crisis Handling:** {'Includes all available data' if include_crisis_years else f'Excludes major crisis periods ({get_country_specific_crisis_text(selected_country)})'}
        - **Data Normalization:** All BOP flows converted to annualized % of GDP
        
        ### Country-Specific Analysis Periods ({study_version})
        """)
        
        for country, info in expanded_timeline.items():
            country_short = country.split(',')[0]
            
            if include_crisis_years:
                pre_start, pre_end = info['pre_period_full']
                post_start, post_end = info['post_period_full']
            else:
                pre_start, pre_end = info['pre_period_crisis_excluded']
                post_start, post_end = info['post_period_crisis_excluded']
            
            st.markdown(f"""
            **{country_short} (Euro adoption: {info['adoption_year']})**
            - Pre-Euro: {pre_start}-{pre_end} ({pre_end - pre_start + 1} years)
            - Post-Euro: {post_start}-{post_end} ({post_end - post_start + 1} years, includes adoption year)
            """)
    
    # Load data with selected version
    with st.spinner(f"Loading and processing Euro adoption data ({study_version})..."):
        final_data, analysis_indicators, metadata = load_case_study_2_data(include_crisis_years)
    
    # Note: Overall Capital Flows Analysis moved to after country selection for country-specific display
    
    if final_data is None:
        st.stop()
    
    # Data overview
    st.success(f"✅ Data loaded successfully! ({metadata['study_version']})")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Observations", f"{metadata['final_shape'][0]:,}")
    with col2:
        st.metric("Indicators", metadata['n_indicators'])
    with col3:
        st.metric("Countries", len(metadata['countries']))
    with col4:
        st.metric("Study Version", metadata['study_version'])
    
    # Show data period summary
    period_col = metadata['period_column']
    if period_col in final_data.columns:
        period_counts = final_data[period_col].value_counts()
        st.info(f"📊 **Data Distribution:** {period_counts.to_dict()}")
    
    st.markdown("---")
    
    # Country selection
    st.header("Select Country for Analysis")
    
    # Country mapping for display vs data
    country_mapping = {
        'Estonia': 'Estonia, Republic of',
        'Latvia': 'Latvia, Republic of', 
        'Lithuania': 'Lithuania, Republic of'
    }
    
    selected_display_country = st.selectbox(
        "Choose a Baltic country to analyze:",
        ['Estonia', 'Latvia', 'Lithuania'],
        index=0,
        key=f"cs2_country_select_cs2_{unique_key_prefix}"
    )
    
    selected_country = country_mapping[selected_display_country]
    
    # Final session_id includes all variable components for complete uniqueness
    session_id = f"{session_id}_{selected_display_country.lower()}"
    
    # Overall Capital Flows Analysis (Country-Specific)
    st.header("📈 Overall Capital Flows Analysis")
    show_overall_capital_flows_analysis_cs2(selected_country, selected_display_country, include_crisis_years)
    
    timeline = metadata['timeline']
    country_info = timeline[selected_country]
    
    # Get the appropriate period keys based on study version
    if include_crisis_years:
        pre_period = country_info['pre_period_full']
        post_period = country_info['post_period_full']
        period_label = "Full Series"
    else:
        pre_period = country_info['pre_period_crisis_excluded']
        post_period = country_info['post_period_crisis_excluded']
        period_label = "Crisis-Excluded"
    
    st.info(f"""
    **{selected_display_country} Analysis ({period_label}):** Euro adoption on {country_info['adoption_date']}
    - **Pre-Euro Period:** {pre_period[0]} to {pre_period[1]}
    - **Post-Euro Period:** {post_period[0]} to {post_period[1]} (includes adoption year {country_info['adoption_year']})
    """)
    
    # Detailed analysis sections are handled by the optimized functions below
    
if __name__ == "__main__":
    main()
