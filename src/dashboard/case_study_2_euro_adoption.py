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

warnings.filterwarnings('ignore')

# Import shared functions from Case Study 1
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

def load_case_study_2_data(include_crisis_years=True):
    """Load processed Euro adoption analysis data with version selection"""
    try:
        # Load processed data files
        data_dir = Path(__file__).parent.parent.parent / "data"
        
        if include_crisis_years:
            processed_file = data_dir / "case_study_2_euro_adoption_data.csv"
            study_version = "Full Series"
        else:
            processed_file = data_dir / "case_study_2_euro_adoption_data_crisis_excluded.csv"
            study_version = "Crisis-Excluded"
        
        if not processed_file.exists():
            st.error(f"Case Study 2 {study_version.lower()} data not found. Please run data_processor_case_study_2.py first.")
            return None, None, None
        
        # Load processed data
        final_data = pd.read_csv(processed_file)
        
        # Get analysis indicators (columns ending with _PGDP)
        analysis_indicators = [col for col in final_data.columns if col.endswith('_PGDP')]
        analysis_indicators = sort_indicators_by_type(analysis_indicators)
        
        # Create timeline information (updated with new periods)
        timeline = create_expanded_euro_adoption_timeline()
        
        # Analysis countries (simplified country names for display)
        analysis_countries = ['Estonia', 'Latvia', 'Lithuania']
        
        # Determine which period column to use
        period_column = 'EURO_PERIOD_CRISIS_EXCLUDED' if not include_crisis_years else 'EURO_PERIOD_FULL'
        if period_column not in final_data.columns:
            period_column = 'EURO_PERIOD'  # Fallback for backward compatibility
        
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
            'post_period_full': (2011, 2024),              # Include 2011 adoption year
            'pre_period_crisis_excluded': (1999, 2007),    # Excludes 2008-2010
            'post_period_crisis_excluded': (2011, 2019),   # Include 2011, excludes 2020-2022
            'crisis_years': [2008, 2009, 2010, 2020, 2021, 2022]
        },
        'Latvia, Republic of': {
            'adoption_date': '2014-01-01',
            'adoption_year': 2014,
            'pre_period_full': (1999, 2013),
            'post_period_full': (2014, 2024),              # Include 2014 adoption year
            'pre_period_crisis_excluded': (1999, 2013),    # Excludes 2008-2010 within range
            'post_period_crisis_excluded': (2014, 2019),   # Include 2014, excludes 2020-2022
            'crisis_years': [2008, 2009, 2010, 2020, 2021, 2022]
        },
        'Lithuania, Republic of': {
            'adoption_date': '2015-01-01',
            'adoption_year': 2015,
            'pre_period_full': (1999, 2014),
            'post_period_full': (2015, 2024),              # Include 2015 adoption year
            'pre_period_crisis_excluded': (1999, 2014),    # Excludes 2008-2010 within range
            'post_period_crisis_excluded': (2015, 2019),   # Include 2015, excludes 2020-2022
            'crisis_years': [2008, 2009, 2010, 2020, 2021, 2022]
        }
    }

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
        
        # Create Euro adoption timeline
        timeline = create_euro_adoption_timeline()
        
        # Add period classification
        def classify_period(row, timeline, include_crisis_years):
            country = row['COUNTRY']
            year = row['YEAR']
            
            if country in timeline:
                adoption_year = timeline[country]['adoption_year']
                
                if include_crisis_years:
                    # Full series analysis
                    if year < adoption_year:
                        return 'Pre-Euro'
                    elif year >= adoption_year:
                        return 'Post-Euro'
                else:
                    # Crisis-excluded analysis
                    pre_start, pre_end = timeline[country]['pre_period']
                    post_start, post_end = timeline[country]['post_period']
                    
                    if pre_start <= year <= pre_end:
                        return 'Pre-Euro'
                    elif post_start <= year <= post_end:
                        return 'Post-Euro'
                    else:
                        return 'Excluded'
            return 'Unknown'
        
        case_two_data['EURO_PERIOD'] = case_two_data.apply(
            lambda row: classify_period(row, timeline, include_crisis_years), axis=1
        )
        
        # Filter out excluded periods if crisis-excluded version
        if not include_crisis_years:
            case_two_data = case_two_data[case_two_data['EURO_PERIOD'] != 'Excluded'].copy()
        
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

def show_overall_capital_flows_analysis_cs2(include_crisis_years=True):
    """Display Overall Capital Flows Analysis section for Case Study 2 - matches CS1 template exactly"""
    st.header("📈 Overall Capital Flows Analysis")
    study_version = "Full Series" if include_crisis_years else "Crisis-Excluded"
    st.markdown(f"*High-level summary of aggregate net capital flows before detailed disaggregated analysis - {study_version}*")
    
    # Load data
    overall_data, indicators_mapping, metadata = load_overall_capital_flows_data_cs2(include_crisis_years)
    
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
    
    st.dataframe(pivot_summary, use_container_width=True)
    
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
            
            # Plot time series by period
            sorted_data = overall_data_ts.sort_values('DATE')
            
            for period in ['Pre-Euro', 'Post-Euro']:
                period_data = sorted_data[sorted_data['EURO_PERIOD'] == period]
                if len(period_data) > 0:
                    # Aggregate by date (average across countries for each period)
                    period_agg = period_data.groupby('DATE')[col_name].mean().reset_index()
                    if len(period_agg) > 0:
                        ax.plot(period_agg['DATE'], period_agg[col_name], 
                               color=colors[period], label=period, linewidth=2, alpha=0.8)
            
            ax.set_title(clean_name, fontweight='bold', fontsize=10)
            ax.set_ylabel('% of GDP (annualized)', fontsize=9)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            ax.legend(loc='upper right', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
    
    fig2.tight_layout()
    st.pyplot(fig2)

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
    
    # 1. Summary Statistics and Boxplots (matching CS1 exactly)
    st.subheader("📊 Summary Statistics by Time Period")
    
    # Create temporal boxplots (matching CS1 sizing)
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    mean_data = boxplot_data[boxplot_data['Statistic'] == 'Mean']
    mean_pre = mean_data[mean_data['PERIOD'] == 'Pre-Euro']['Value']
    mean_post = mean_data[mean_data['PERIOD'] == 'Post-Euro']['Value']
    
    bp1 = ax1.boxplot([mean_pre, mean_post], labels=['Pre-Euro', 'Post-Euro'], patch_artist=True)
    bp1['boxes'][0].set_facecolor(COLORBLIND_SAFE[0])
    bp1['boxes'][1].set_facecolor(COLORBLIND_SAFE[1])
    
    study_title_suffix = " (Crisis-Excluded)" if not include_crisis_years else ""
    ax1.set_title(f'Panel A: Distribution of Means Across All Capital Flow Indicators{study_title_suffix}', 
                 fontweight='bold', fontsize=10, pad=10)
    ax1.set_ylabel('Mean (% of GDP, annualized)', fontsize=9)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax1.text(0.02, 0.98, f'Pre-Euro Avg: {mean_pre.mean():.2f}%\\nPost-Euro Avg: {mean_post.mean():.2f}%', 
            transform=ax1.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    st.pyplot(fig1)
    
    # Standard deviations boxplot (matching CS1 sizing)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    std_data = boxplot_data[boxplot_data['Statistic'] == 'Standard Deviation']
    std_pre = std_data[std_data['PERIOD'] == 'Pre-Euro']['Value']
    std_post = std_data[std_data['PERIOD'] == 'Post-Euro']['Value']
    
    bp2 = ax2.boxplot([std_pre, std_post], labels=['Pre-Euro', 'Post-Euro'], patch_artist=True)
    bp2['boxes'][0].set_facecolor(COLORBLIND_SAFE[0])
    bp2['boxes'][1].set_facecolor(COLORBLIND_SAFE[1])
    
    ax2.set_title(f'Panel B: Distribution of Standard Deviations Across All Capital Flow Indicators{study_title_suffix}', 
                 fontweight='bold', fontsize=10, pad=10)
    ax2.set_ylabel('Std Dev. (% of GDP, annualized)', fontsize=9)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    volatility_ratio = std_pre.mean() / std_post.mean() if std_post.mean() != 0 else float('inf')
    ax2.text(0.02, 0.98, f'Pre-Euro Avg: {std_pre.mean():.2f}%\\nPost-Euro Avg: {std_post.mean():.2f}%\\nRatio: {volatility_ratio:.2f}x', 
            transform=ax2.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    # 2. Time Series - Individual Charts (following CS1 standards exactly)
    st.subheader("📈 Time Series by Indicator")
    
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
    
    # Create individual time series charts (following CS1 format exactly)
    for i, indicator in enumerate(analysis_indicators):
        fig_ts, ax = plt.subplots(1, 1, figsize=(6, 2.5))  # CS1 standard dimensions
        
        clean_name = indicator.replace('_PGDP', '').replace('_', ' ')
        
        # Get test statistic for title
        indicator_clean = indicator.replace('_PGDP', '')
        f_stat = test_results[test_results['Indicator'] == indicator_clean]['F_Statistic'].iloc[0] if len(test_results[test_results['Indicator'] == indicator_clean]) > 0 else 0
        
        # Panel letter (A, B, C, etc.)
        panel_letter = chr(65 + i)
        
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
                    ax.plot(segment['Date'], segment[indicator], 
                           color=COLORBLIND_SAFE[0], linewidth=1.5, label='Pre-Euro' if j == 0 else "")
        else:
            # Normal plotting for full series
            if len(pre_data) > 0:
                ax.plot(pre_data['Date'], pre_data[indicator], 
                       color=COLORBLIND_SAFE[0], linewidth=1.5, label='Pre-Euro')
        
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
                    ax.plot(segment['Date'], segment[indicator], 
                           color=COLORBLIND_SAFE[1], linewidth=1.5, label='Post-Euro' if j == 0 else "")
        else:
            # Normal plotting for full series
            if len(post_data) > 0:
                ax.plot(post_data['Date'], post_data[indicator], 
                       color=COLORBLIND_SAFE[1], linewidth=1.5, label='Post-Euro')
        
        # Add Euro adoption line
        ax.axvline(x=adoption_date, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Euro Adoption')
        
        # Add crisis period shading for crisis-excluded version
        if not include_crisis_years:
            ax.axvspan(pd.Timestamp('2008-01-01'), pd.Timestamp('2010-12-31'), 
                      alpha=0.15, color='red', label='GFC (excluded)' if i == 0 else "")
            ax.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2022-12-31'), 
                      alpha=0.15, color='orange', label='COVID (excluded)' if i == 0 else "")
        
        # Formatting (following CS1 standards exactly)
        ax.set_title(f'Panel {panel_letter}: {clean_name}{study_title_suffix} (F-statistic: {f_stat:.2f})', 
                    fontweight='bold', fontsize=9, pad=8)
        ax.set_ylabel('% of GDP (annualized)', fontsize=8)
        ax.set_xlabel('Year', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.legend(loc='best', fontsize=7, frameon=True, fancybox=False, shadow=False)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        st.pyplot(fig_ts)
        
        # Individual download button (following CS1 pattern)
        buf_ts = io.BytesIO()
        fig_ts.savefig(buf_ts, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf_ts.seek(0)
        
        clean_filename = clean_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        
        st.download_button(
            label=f"📥 Download {clean_name} Time Series (PNG)",
            data=buf_ts.getvalue(),
            file_name=f"{selected_display_country}_{clean_filename}_timeseries{version_suffix}.png",
            mime="image/png",
            key=f"download_ts_{selected_display_country}_{i}_{clean_filename}{'_crisis_excluded' if not include_crisis_years else '_full'}_{session_id}"
        )

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
        key=f"cs2_study_version_{unique_key_prefix}"
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
        st.warning("🚫 **Crisis-Excluded:** Removes major crisis periods (GFC 2008-2010 + COVID 2020-2022) to isolate Euro adoption effects")
    
    st.markdown("---")
    
    # Data and Methodology section  
    with st.expander("📋 Data and Methodology", expanded=False):
        expanded_timeline = create_expanded_euro_adoption_timeline()
        
        st.markdown(f"""
        ### Temporal Analysis Design ({study_version})
        - **Methodology:** Before-after comparison for each country
        - **Analysis Periods:** Asymmetric windows maximizing available data
        - **Crisis Handling:** {'Includes all available data' if include_crisis_years else 'Excludes major crisis periods (GFC 2008-2010 + COVID 2020-2022)'}
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
    
    # Overall Capital Flows Analysis (NEW SECTION)
    show_overall_capital_flows_analysis_cs2(include_crisis_years)
    
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
        key=f"cs2_country_select_{unique_key_prefix}"
    )
    
    selected_country = country_mapping[selected_display_country]
    
    # Final session_id includes all variable components for complete uniqueness
    session_id = f"{session_id}_{selected_display_country.lower()}"
    
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
    
    # Calculate statistics for selected country using the appropriate period column
    period_column = metadata['period_column']
    country_stats = calculate_temporal_statistics(final_data, selected_country, analysis_indicators, period_column)
    boxplot_data = create_temporal_boxplot_data(final_data, selected_country, analysis_indicators, period_column) 
    test_results = perform_temporal_volatility_tests(final_data, selected_country, analysis_indicators, period_column)
    
    # 1. Summary Statistics and Boxplots
    st.header("1. Summary Statistics and Boxplots")
    
    # Create temporal boxplots (matching CS1 sizing)
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    mean_data = boxplot_data[boxplot_data['Statistic'] == 'Mean']
    mean_pre = mean_data[mean_data['PERIOD'] == 'Pre-Euro']['Value']
    mean_post = mean_data[mean_data['PERIOD'] == 'Post-Euro']['Value']
    
    bp1 = ax1.boxplot([mean_pre, mean_post], labels=['Pre-Euro', 'Post-Euro'], patch_artist=True)
    bp1['boxes'][0].set_facecolor(COLORBLIND_SAFE[0])
    bp1['boxes'][1].set_facecolor(COLORBLIND_SAFE[1])
    study_title_suffix = " (Crisis-Excluded)" if not include_crisis_years else ""
    ax1.set_title(f'Panel A: Distribution of Means Across All Capital Flow Indicators{study_title_suffix}', 
                 fontweight='bold', fontsize=10, pad=10)
    ax1.set_ylabel('Mean (% of GDP, annualized)', fontsize=9)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)  # Add horizontal line at y=0
    ax1.text(0.02, 0.98, f'Pre-Euro Avg: {mean_pre.mean():.2f}%\\nPost-Euro Avg: {mean_post.mean():.2f}%', 
            transform=ax1.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.tight_layout()
    st.pyplot(fig1)
    
    # Download button for means boxplot
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf1.seek(0)
    
    version_suffix = "_crisis_excluded" if not include_crisis_years else ""
    st.download_button(
        label="📥 Download Means Boxplot (PNG)",
        data=buf1.getvalue(),
        file_name=f"{selected_display_country}_means_boxplot{version_suffix}.png",
        mime="image/png",
        key=f"download_means_{selected_display_country}{'_crisis_excluded' if not include_crisis_years else '_full'}_{session_id}"
    )
    
    # Standard deviations boxplot (matching CS1 sizing)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    std_data = boxplot_data[boxplot_data['Statistic'] == 'Standard Deviation']
    std_pre = std_data[std_data['PERIOD'] == 'Pre-Euro']['Value']
    std_post = std_data[std_data['PERIOD'] == 'Post-Euro']['Value']
    
    bp2 = ax2.boxplot([std_pre, std_post], labels=['Pre-Euro', 'Post-Euro'], patch_artist=True)
    bp2['boxes'][0].set_facecolor(COLORBLIND_SAFE[0])
    bp2['boxes'][1].set_facecolor(COLORBLIND_SAFE[1])
    ax2.set_title(f'Panel B: Distribution of Standard Deviations Across All Capital Flow Indicators{study_title_suffix}', 
                 fontweight='bold', fontsize=10, pad=10)
    ax2.set_ylabel('Std Dev. (% of GDP, annualized)', fontsize=9)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)  # Add horizontal line at y=0
    
    volatility_ratio = std_pre.mean() / std_post.mean() if std_post.mean() != 0 else float('inf')
    ax2.text(0.02, 0.98, f'Pre-Euro Avg: {std_pre.mean():.2f}%\\nPost-Euro Avg: {std_post.mean():.2f}%\\nRatio: {volatility_ratio:.2f}x', 
            transform=ax2.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax2.set_title('Distribution of Standard Deviations: Pre vs Post Euro Adoption Volatility', 
                 fontweight='bold', fontsize=10, pad=10)
    plt.tight_layout()
    
    st.pyplot(fig2)
    
    # Download button for std dev boxplot
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf2.seek(0)
    
    st.download_button(
        label="📥 Download Std Dev Boxplot (PNG)",
        data=buf2.getvalue(),
        file_name=f"{selected_display_country}_stddev_boxplot{version_suffix}.png",
        mime="image/png",
        key=f"download_stddev_{selected_display_country}{'_crisis_excluded' if not include_crisis_years else '_full'}_{session_id}"
    )
    
    # Summary statistics
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
    
    change_direction = "reduced" if volatility_ratio > 1 else "increased"
    st.info(f"**Volatility Impact:** Euro adoption {change_direction} average volatility by {abs(1-1/volatility_ratio)*100:.1f}%")
    
    st.markdown("---")
    
    # 2. Comprehensive Statistical Summary Table
    st.header("2. Comprehensive Statistical Summary Table")
    
    st.markdown(f"**{selected_display_country} - Pre-Euro vs Post-Euro Statistics{' (Crisis-Excluded)' if not include_crisis_years else ''}**")
    
    # Create side-by-side comparison table
    table_data = []
    sorted_indicators = sort_indicators_by_type(analysis_indicators)
    
    for indicator in sorted_indicators:
        clean_name = indicator.replace('_PGDP', '')
        nickname = get_nickname(clean_name)
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
    
    # Style the table
    styled_table = summary_df.style.set_properties(**{
        'text-align': 'center',
        'font-size': '10px',
        'border': '1px solid #ddd'
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('font-weight', 'bold'), ('font-size', '11px')]},
        {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},
        {'selector': 'td:first-child', 'props': [('text-align', 'left'), ('font-weight', 'bold')]}
    ])
    
    st.dataframe(styled_table, use_container_width=True, hide_index=True)
    
    st.info(f"**Summary:** Statistics for all {len(analysis_indicators)} capital flow indicators comparing pre and post Euro adoption periods. CV% = Coefficient of Variation. Values >1 indicate higher pre-Euro volatility.")
    
    st.markdown("---")
    
    # 3. Hypothesis Testing Results
    st.header("3. Hypothesis Testing Results")
    
    st.markdown(f"""
    **F-Tests for Equal Variances: {selected_display_country} Pre-Euro vs Post-Euro{' (Crisis-Excluded)' if not include_crisis_years else ''}**
    
    - **H₀:** Equal volatility pre and post Euro adoption
    - **H₁:** Different volatility pre and post Euro adoption  
    - **α = 0.05**
    {f'- **Crisis Exclusions:** GFC (2008-2010) + COVID (2020-2022)' if not include_crisis_years else ''}
    """)
    
    # Create hypothesis test results table
    results_display = test_results.copy()
    results_display['Sort_Key'] = results_display['Indicator'].apply(get_investment_type_order)
    results_display = results_display.sort_values('Sort_Key')
    
    test_table_data = []
    for _, row in results_display.iterrows():
        nickname = get_nickname(row['Indicator'])
        significance = '***' if row['P_Value'] < 0.001 else '**' if row['P_Value'] < 0.01 else '*' if row['P_Value'] < 0.05 else ''
        higher_vol = 'Pre-Euro' if row['Pre_Euro_Higher_Volatility'] else 'Post-Euro'
        
        test_table_data.append({
            'Indicator': nickname,
            'F-Statistic': f"{row['F_Statistic']:.2f}",
            'P-Value': f"{row['P_Value']:.4f}",
            'Significance': significance,
            'Higher Volatility': higher_vol
        })
    
    test_df = pd.DataFrame(test_table_data)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        styled_test_table = test_df.style.set_properties(**{
            'text-align': 'center',
            'font-size': '11px',
            'border': '1px solid #ddd'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#e6f3ff'), ('font-weight', 'bold'), ('text-align', 'center')]},
            {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},
            {'selector': 'td:first-child', 'props': [('text-align', 'left')]}
        ])
        
        st.dataframe(styled_test_table, use_container_width=True, hide_index=True)
        st.caption("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    
    with col2:
        st.markdown("**Legend:**")
        st.markdown("- **F-Statistic**: Ratio of variances")
        st.markdown("- **P-Value**: Statistical significance") 
        st.markdown("- **Higher Volatility**: Which period shows more volatility")
    
    # Test summary
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
    
    conclusion = "Strong evidence that Euro adoption reduced" if pre_higher_count/total_indicators > 0.6 else "Mixed evidence for Euro adoption reducing"
    st.success(f"**Conclusion:** {conclusion} capital flow volatility in {selected_display_country}.")
    
    st.markdown("---")
    
    # 4. Time Series Analysis
    st.header("4. Time Series Analysis")
    
    # Create date column
    final_data_copy = final_data.copy()
    final_data_copy['Date'] = pd.to_datetime(
        final_data_copy['YEAR'].astype(str) + '-' + 
        ((final_data_copy['QUARTER'] - 1) * 3 + 1).astype(str) + '-01'
    )
    
    country_data = final_data_copy[final_data_copy['COUNTRY'] == selected_country]
    adoption_date = pd.to_datetime(country_info['adoption_date'])
    
    crisis_label = " - Crisis-Excluded Version" if not include_crisis_years else ""
    st.markdown(f"**Showing all {len(analysis_indicators)} indicators for {selected_display_country} (Euro adoption: {country_info['adoption_year']}){crisis_label}**")
    
    # Create individual time series plots
    for i, indicator in enumerate(analysis_indicators):
        fig_ts, ax = plt.subplots(1, 1, figsize=(6, 2.5))
        
        clean_name = indicator.replace('_PGDP', '')
        nickname = get_nickname(clean_name)
        
        # Use the appropriate period column for data filtering
        period_col = metadata['period_column']
        
        # Plot pre-Euro data (break at crisis periods if crisis-excluded)
        pre_data = country_data[country_data[period_col] == 'Pre-Euro']
        if not include_crisis_years and len(pre_data) > 0:
            # For crisis-excluded: plot pre-Euro data in segments to avoid connecting across excluded periods
            pre_data_sorted = pre_data.sort_values('Date')
            segments = []
            current_segment = []
            
            for _, row in pre_data_sorted.iterrows():
                if len(current_segment) == 0:
                    current_segment.append(row)
                else:
                    # Check if there's a gap indicating excluded crisis period
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
            
            # Plot each segment separately
            for i, segment in enumerate(segments):
                if len(segment) > 0:
                    ax.plot(segment['Date'], segment[indicator], 
                           color=COLORBLIND_SAFE[0], linewidth=2.5, label='Pre-Euro' if i == 0 else "")
        else:
            # Normal plotting for full series
            ax.plot(pre_data['Date'], pre_data[indicator], 
                   color=COLORBLIND_SAFE[0], linewidth=2.5, label='Pre-Euro')
        
        # Plot post-Euro data (break at crisis periods if crisis-excluded)
        post_data = country_data[country_data[period_col] == 'Post-Euro']
        if not include_crisis_years and len(post_data) > 0:
            # For crisis-excluded: plot post-Euro data in segments
            post_data_sorted = post_data.sort_values('Date')
            segments = []
            current_segment = []
            
            for _, row in post_data_sorted.iterrows():
                if len(current_segment) == 0:
                    current_segment.append(row)
                else:
                    # Check if there's a gap indicating excluded crisis period
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
            
            # Plot each segment separately
            for i, segment in enumerate(segments):
                if len(segment) > 0:
                    ax.plot(segment['Date'], segment[indicator], 
                           color=COLORBLIND_SAFE[1], linewidth=2.5, label='Post-Euro' if i == 0 else "")
        else:
            # Normal plotting for full series
            ax.plot(post_data['Date'], post_data[indicator], 
                   color=COLORBLIND_SAFE[1], linewidth=2.5, label='Post-Euro')
        
        # Add Euro adoption line
        ax.axvline(x=adoption_date, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Euro Adoption')
        
        # Add crisis period shading if showing crisis-excluded version
        if not include_crisis_years:
            # Add shaded regions for excluded crisis periods
            ax.axvspan(pd.to_datetime('2008-01-01'), pd.to_datetime('2010-12-31'), 
                      alpha=0.2, color='gray', label='GFC (excluded)')
            ax.axvspan(pd.to_datetime('2020-01-01'), pd.to_datetime('2022-12-31'), 
                      alpha=0.2, color='orange', label='COVID (excluded)')
        
        # Formatting
        f_stat = test_results[test_results['Indicator'] == clean_name]['F_Statistic'].iloc[0]
        panel_letter = chr(65 + i)  # A, B, C, etc.
        study_label = " (Crisis-Excluded)" if not include_crisis_years else ""
        ax.set_title(f'Panel {panel_letter}: {nickname}{study_label} (F-statistic: {f_stat:.2f})', 
                    fontweight='bold', fontsize=9, pad=8)
        ax.set_ylabel('% of GDP (annualized)', fontsize=8)
        ax.set_xlabel('Year', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.legend(loc='best', fontsize=7, frameon=True, fancybox=False, shadow=False)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        st.pyplot(fig_ts)
        
        # Individual download button
        buf_ts = io.BytesIO()
        fig_ts.savefig(buf_ts, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf_ts.seek(0)
        
        clean_filename = nickname.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        
        # Create a widget-specific key that includes the indicator name for extra uniqueness
        widget_key = f"download_ts_{selected_display_country}_{i}_{clean_filename}{'_crisis_excluded' if not include_crisis_years else '_full'}_{session_id}"
        
        st.download_button(
            label=f"📥 Download {nickname} Time Series (PNG)",
            data=buf_ts.getvalue(),
            file_name=f"{selected_display_country}_{clean_filename}_timeseries{version_suffix}.png",
            mime="image/png",
            key=widget_key
        )
    
    st.markdown("---")
    
    # 5. Key Findings Summary
    st.header("5. Key Findings Summary")
    
    col1, col2 = st.columns(2)
    
    crisis_notice = f" *(Excludes GFC 2008-2010 + COVID 2020-2022)*" if not include_crisis_years else ""
    
    with col1:
        st.markdown(f"""
        ### Statistical Evidence for {selected_display_country}{crisis_notice}:
        - **{pre_higher_count/total_indicators*100:.1f}% of capital flow indicators** showed higher volatility before Euro adoption
        - **{sig_5pct_count/total_indicators*100:.1f}% of indicators** show statistically significant differences (p<0.05)
        - **Average volatility change** of {abs(1-1/volatility_ratio)*100:.1f}% after Euro adoption
        - **Euro adoption in {country_info['adoption_year']}** marked a clear structural break
        """)
    
    with col2:
        st.markdown("""
        ### Policy Implications:
        - Evidence supports Euro adoption reducing capital flow volatility
        - Monetary union provides stabilizing effect on external financing
        - Integration with larger currency area reduces country-specific shocks
        - Supports theoretical predictions of currency union benefits
        """)
    
    # Download section
    st.markdown("---")
    st.header("6. Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download summary table
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary Table (CSV)",
            data=csv,
            file_name=f"{selected_display_country}_euro_adoption_summary{version_suffix}.csv",
            mime="text/csv",
            key=f"download_summary_{selected_display_country}{'_crisis_excluded' if not include_crisis_years else '_full'}_{session_id}"
        )
    
    with col2:
        # Download test results
        csv = test_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Test Results (CSV)",
            data=csv,
            file_name=f"{selected_display_country}_hypothesis_tests{version_suffix}.csv",
            mime="text/csv",
            key=f"download_tests_{selected_display_country}{'_crisis_excluded' if not include_crisis_years else '_full'}_{session_id}"
        )
    
    with col3:
        # Download country statistics
        csv = country_stats.to_csv(index=False)
        st.download_button(
            label="📥 Download Country Statistics (CSV)",
            data=csv,
            file_name=f"{selected_display_country}_temporal_statistics{version_suffix}.csv",
            mime="text/csv",
            key=f"download_stats_{selected_display_country}{'_crisis_excluded' if not include_crisis_years else '_full'}_{session_id}"
        )

if __name__ == "__main__":
    main()