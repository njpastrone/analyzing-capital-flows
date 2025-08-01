"""
Simple Capital Flows Report Dashboard - Case Study 1
Streamlit version of the Case_Study_1_Report_Template
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
    return nicknames.get(indicator_name, indicator_name[:25] + '...' if len(indicator_name) > 25 else indicator_name)

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

# Set styling for econometrics (clean, academic style)
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

# Colorblind-friendly econometrics palette (blues, oranges, teals)
ECON_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
# More accessible: blue, orange, green, red, purple, brown
COLORBLIND_SAFE = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#FBAFE4']
sns.set_palette(COLORBLIND_SAFE)

# Page configuration - removed to avoid conflicts when imported into main_app.py
# st.set_page_config() is now handled by main_app.py

def load_default_data():
    """Load default Case Study 1 data from cleaned datasets"""
    try:
        # Use new cleaned data path
        data_dir = Path(__file__).parent.parent.parent / "updated_data" / "Clean"
        comprehensive_file = data_dir / "comprehensive_df_PGDP_labeled.csv "
        
        if not comprehensive_file.exists():
            st.error("Cleaned data file not found. Please check file paths.")
            return None, None, None
        
        # Load comprehensive labeled data
        comprehensive_df = pd.read_csv(comprehensive_file)
        
        # Filter for Case Study 1 data (CS1_GROUP not null)
        case_one_data = comprehensive_df[comprehensive_df['CS1_GROUP'].notna()].copy()
        
        # Remove Luxembourg as per original analysis
        final_data = case_one_data[case_one_data['COUNTRY'] != 'Luxembourg'].copy()
        
        # Create GROUP column using CS1_GROUP mapping
        final_data['GROUP'] = final_data['COUNTRY'].apply(
            lambda x: 'Iceland' if x == 'Iceland' else 'Eurozone'
        )
        
        # Get analysis indicators (columns ending with _PGDP)
        all_indicators = [col for col in final_data.columns if col.endswith('_PGDP')]
        
        # Remove the last two indicators (Financial derivatives and Financial account balance - now discontinued)
        indicators_to_exclude = [
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Financial derivatives (other than reserves) and employee stock options_PGDP',
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Financial account balance, excluding reserves and related items_PGDP'
        ]
        analysis_indicators = [ind for ind in all_indicators if ind not in indicators_to_exclude]
        
        # Rename indicators to consistent format
        indicator_renames = {
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Direct investment, Total financial assets/liabilities_PGDP': 'Net - Direct investment, Total financial assets/liabilities_PGDP',
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Portfolio investment, Total financial assets/liabilities_PGDP': 'Net - Portfolio investment, Total financial assets/liabilities_PGDP',
            'Net (net acquisition of financial assets less net incurrence of liabilities) - Other investment, Total financial assets/liabilities_PGDP': 'Net - Other investment, Total financial assets/liabilities_PGDP'
        }
        
        # Apply renames to dataframe
        final_data = final_data.rename(columns=indicator_renames)
        
        # Update indicator list with new names
        analysis_indicators = [indicator_renames.get(ind, ind) for ind in analysis_indicators]
        analysis_indicators = sort_indicators_by_type(analysis_indicators)
        
        return final_data, analysis_indicators, {
            'original_shape': comprehensive_df.shape,
            'filtered_shape': case_one_data.shape,
            'final_shape': final_data.shape,
            'n_indicators': len(analysis_indicators)
        }
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def calculate_group_statistics(data, group_col, indicators):
    """Calculate comprehensive statistics by group"""
    results = []
    
    for group in data[group_col].unique():
        group_data = data[data[group_col] == group]
        
        for indicator in indicators:
            # Make sure we only process indicators that actually exist in the data
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
            else:
                print(f"Warning: Indicator {indicator} not found in data columns")
    
    return pd.DataFrame(results)

def create_boxplot_data(data, indicators):
    """Create dataset for boxplot visualization"""
    stats_data = []
    
    for group in ['Iceland', 'Eurozone']:
        group_data = data[data['GROUP'] == group]
        
        for indicator in indicators:
            values = group_data[indicator].dropna()
            if len(values) > 1:
                mean_val = values.mean()
                std_val = values.std()
                
                stats_data.append({
                    'GROUP': group,
                    'Indicator': indicator.replace('_PGDP', ''),
                    'Statistic': 'Mean',
                    'Value': mean_val
                })
                
                stats_data.append({
                    'GROUP': group,
                    'Indicator': indicator.replace('_PGDP', ''),
                    'Statistic': 'Standard Deviation', 
                    'Value': std_val
                })
    
    return pd.DataFrame(stats_data)

def create_individual_country_boxplot_data(data, indicators):
    """Create dataset for individual country boxplot visualization"""
    stats_data_individual = []
    
    # Get list of Eurozone countries (exclude Iceland)
    eurozone_countries = data[data['GROUP'] == 'Eurozone']['COUNTRY'].unique().tolist()
    all_countries = ['Iceland'] + sorted(eurozone_countries)
    
    for country in all_countries:
        country_data = data[data['COUNTRY'] == country]
        
        for indicator in indicators:
            values = country_data[indicator].dropna()
            if len(values) > 1:
                mean_val = values.mean()
                std_val = values.std()
                
                stats_data_individual.append({
                    'COUNTRY': country,
                    'Indicator': indicator.replace('_PGDP', ''),
                    'Statistic': 'Mean',
                    'Value': mean_val
                })
                
                stats_data_individual.append({
                    'COUNTRY': country,
                    'Indicator': indicator.replace('_PGDP', ''),
                    'Statistic': 'Standard Deviation', 
                    'Value': std_val
                })
    
    return pd.DataFrame(stats_data_individual)

def perform_volatility_tests(data, indicators):
    """Perform F-tests comparing Iceland vs Eurozone volatility"""
    test_results = []
    
    for indicator in indicators:
        iceland_data = data[data['GROUP'] == 'Iceland'][indicator].dropna()
        eurozone_data = data[data['GROUP'] == 'Eurozone'][indicator].dropna()
        
        if len(iceland_data) > 1 and len(eurozone_data) > 1:
            iceland_var = iceland_data.var()
            eurozone_var = eurozone_data.var()
            
            f_stat = iceland_var / eurozone_var if eurozone_var != 0 else np.inf
            df1, df2 = len(iceland_data) - 1, len(eurozone_data) - 1
            
            # Two-tailed p-value
            p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
            
            test_results.append({
                'Indicator': indicator.replace('_PGDP', ''),
                'F_Statistic': f_stat,
                'P_Value': p_value,
                'Iceland_Higher_Volatility': iceland_var > eurozone_var,
                'Significant_5pct': p_value < 0.05,
                'Significant_1pct': p_value < 0.01
            })
    
    return pd.DataFrame(test_results)

def load_overall_capital_flows_data():
    """Load data specifically for Overall Capital Flows Analysis"""
    try:
        # Use comprehensive dataset
        data_dir = Path(__file__).parent.parent.parent / "updated_data" / "Clean"
        comprehensive_file = data_dir / "comprehensive_df_PGDP_labeled.csv "
        
        if not comprehensive_file.exists():
            return None, None
        
        # Load comprehensive labeled data
        comprehensive_df = pd.read_csv(comprehensive_file)
        
        # Filter for Case Study 1 data (CS1_GROUP not null)
        case_one_data = comprehensive_df[comprehensive_df['CS1_GROUP'].notna()].copy()
        
        # Remove Luxembourg as per original analysis
        final_data = case_one_data[case_one_data['COUNTRY'] != 'Luxembourg'].copy()
        
        # Create GROUP column
        final_data['GROUP'] = final_data['COUNTRY'].apply(
            lambda x: 'Iceland' if x == 'Iceland' else 'Eurozone'
        )
        
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
        final_data['Net Capital Flows (Direct + Portfolio + Other Investment)_PGDP'] = (
            final_data[net_direct_col].fillna(0) + 
            final_data[net_portfolio_col].fillna(0) + 
            final_data[net_other_col].fillna(0)
        )
        
        # Complete indicators mapping including computed indicator
        overall_indicators_mapping = {
            **base_indicators_mapping,
            'Net Capital Flows (Direct + Portfolio + Other Investment)': 'Net Capital Flows (Direct + Portfolio + Other Investment)_PGDP'
        }
        
        return final_data, overall_indicators_mapping
        
    except Exception as e:
        st.error(f"Error loading overall capital flows data: {str(e)}")
        return None, None

def show_overall_capital_flows_analysis():
    """Display Overall Capital Flows Analysis section"""
    st.header("📈 Overall Capital Flows Analysis")
    st.markdown("*High-level summary of aggregate net capital flows before detailed disaggregated analysis*")
    
    # Load data
    overall_data, indicators_mapping = load_overall_capital_flows_data()
    
    if overall_data is None or indicators_mapping is None:
        st.error("Failed to load overall capital flows data.")
        return
    
    # Color scheme
    colors = {'Iceland': '#FF6B6B', 'Eurozone': '#4ECDC4'}
    
    # Summary statistics
    st.subheader("📊 Summary Statistics by Group")
    
    summary_stats = []
    for clean_name, col_name in indicators_mapping.items():
        if col_name in overall_data.columns:
            for group in ['Iceland', 'Eurozone']:
                group_data = overall_data[overall_data['GROUP'] == group][col_name].dropna()
                summary_stats.append({
                    'Indicator': clean_name,
                    'Group': group,
                    'Mean': group_data.mean(),
                    'Std Dev': group_data.std(),
                    'Median': group_data.median(),
                    'Min': group_data.min(),
                    'Max': group_data.max(),
                    'Count': len(group_data)
                })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Display summary table
    pivot_summary = summary_df.pivot_table(
        index='Indicator', 
        columns='Group', 
        values=['Mean', 'Std Dev', 'Median'],
        aggfunc='first'
    ).round(2)
    
    st.dataframe(pivot_summary, use_container_width=True)
    
    # Side-by-side boxplots
    st.subheader("📦 Distribution Comparison by Group")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (clean_name, col_name) in enumerate(indicators_mapping.items()):
        if col_name in overall_data.columns and i < 4:
            ax = axes[i]
            
            # Prepare data for boxplot
            iceland_data = overall_data[overall_data['GROUP'] == 'Iceland'][col_name].dropna()
            eurozone_data = overall_data[overall_data['GROUP'] == 'Eurozone'][col_name].dropna()
            
            # Create boxplot
            bp = ax.boxplot([iceland_data, eurozone_data], 
                           labels=['Iceland', 'Eurozone'], 
                           patch_artist=True)
            
            # Color the boxes
            bp['boxes'][0].set_facecolor(colors['Iceland'])
            bp['boxes'][1].set_facecolor(colors['Eurozone'])
            for box in bp['boxes']:
                box.set_alpha(0.7)
            
            ax.set_title(clean_name, fontweight='bold', fontsize=10)
            ax.set_ylabel('% of GDP (annualized)', fontsize=9)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Time series plots
    st.subheader("📈 Time Series by Group")
    
    # Create date column
    overall_data_ts = overall_data.copy()
    overall_data_ts['DATE'] = pd.to_datetime(overall_data_ts['YEAR'].astype(str) + '-Q' + overall_data_ts['QUARTER'].astype(str))
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    axes2 = axes2.flatten()
    
    for i, (clean_name, col_name) in enumerate(indicators_mapping.items()):
        if col_name in overall_data.columns and i < 4:
            ax = axes2[i]
            
            for group in ['Iceland', 'Eurozone']:
                group_data = overall_data_ts[overall_data_ts['GROUP'] == group].sort_values('DATE')
                ax.plot(group_data['DATE'], group_data[col_name], 
                       color=colors[group], label=group, linewidth=2, alpha=0.8)
            
            ax.set_title(clean_name, fontweight='bold', fontsize=10)
            ax.set_ylabel('% of GDP (annualized)', fontsize=9)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            ax.legend(loc='upper right', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Key insights
    st.subheader("🔍 Key Insights")
    
    # Calculate volatility comparison
    volatility_comparison = []
    for clean_name, col_name in indicators_mapping.items():
        if col_name in overall_data.columns:
            iceland_std = overall_data[overall_data['GROUP'] == 'Iceland'][col_name].std()
            eurozone_std = overall_data[overall_data['GROUP'] == 'Eurozone'][col_name].std()
            volatility_comparison.append({
                'Indicator': clean_name,
                'Iceland Volatility': iceland_std,
                'Eurozone Volatility': eurozone_std,
                'Volatility Ratio': iceland_std / eurozone_std if eurozone_std != 0 else float('inf')
            })
    
    vol_df = pd.DataFrame(volatility_comparison)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Volatility Comparison (Standard Deviation)**")
        for _, row in vol_df.iterrows():
            ratio = row['Volatility Ratio']
            if ratio > 1:
                st.write(f"• **{row['Indicator']}**: Iceland {ratio:.1f}x more volatile")
            else:
                st.write(f"• **{row['Indicator']}**: Similar volatility levels")
    
    with col2:
        st.markdown("**Overall Pattern**")
        high_vol_count = sum(1 for _, row in vol_df.iterrows() if row['Volatility Ratio'] > 1.5)
        total_indicators = len(vol_df)
        
        if high_vol_count >= total_indicators * 0.75:
            st.write("🔴 **Iceland shows consistently higher volatility** across most capital flow categories")
        elif high_vol_count >= total_indicators * 0.5:
            st.write("🟡 **Mixed volatility patterns** between Iceland and Eurozone")
        else:
            st.write("🟢 **Similar volatility levels** between Iceland and Eurozone")
    
    st.markdown("---")

def main():
    """Main report application"""
    
    # Title and header
    st.title("📊 Capital Flow Volatility Analysis")
    st.subheader("Case Study 1: Iceland vs. Eurozone Comparison")
    
    st.markdown("""
    **Research Question:** Should Iceland adopt the Euro as its currency?
    
    **Hypothesis:** Iceland's capital flows show more volatility than the Eurozone bloc average
    
    ---
    """)
    
    # Data and Methodology section
    with st.expander("📋 Data and Methodology", expanded=False):
        st.markdown("""
        ### Data Sources
        - **Balance of Payments Data:** IMF, quarterly frequency (1999-2024)
        - **GDP Data:** IMF World Economic Outlook, annual frequency
        - **Countries:** Iceland vs. 10 initial Euro adopters (excluding Luxembourg)
        
        ### Methodology
        1. **Data Normalization:** All BOP flows converted to annualized % of GDP
        2. **Statistical Analysis:** Comprehensive descriptive statistics and F-tests
        3. **Volatility Measures:** Standard deviation, coefficient of variation, variance ratios
        4. **Hypothesis Testing:** F-tests for equality of variances between groups
        
        ### Countries Analyzed
        - **Iceland:** Independent monetary policy with floating exchange rate
        - **Eurozone Bloc:** Austria, Belgium, Finland, France, Germany, Ireland, Italy, Netherlands, Portugal, Spain
        """)
    
    # Load data
    with st.spinner("Loading and processing data..."):
        final_data, analysis_indicators, metadata = load_default_data()
    
    if final_data is None:
        st.stop()
    
    # Data overview
    st.success("✅ Data loaded successfully!")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Observations", f"{metadata['final_shape'][0]:,}")
    with col2:
        st.metric("Indicators", metadata['n_indicators'])
    with col3:
        st.metric("Countries", final_data['COUNTRY'].nunique())
    with col4:
        st.metric("Time Period", f"{final_data['YEAR'].min()}-{final_data['YEAR'].max()}")
    
    st.markdown("---")
    
    # Overall Capital Flows Analysis (NEW SECTION)
    show_overall_capital_flows_analysis()
    
    # Calculate all statistics
    group_stats = calculate_group_statistics(final_data, 'GROUP', analysis_indicators)
    boxplot_data = create_boxplot_data(final_data, analysis_indicators)
    individual_country_data = create_individual_country_boxplot_data(final_data, analysis_indicators)
    test_results = perform_volatility_tests(final_data, analysis_indicators)
    
    # 1. Summary Statistics and Boxplots
    st.header("1. Summary Statistics and Boxplots")
    
    # Create individual boxplots with standard sizing
    # First boxplot - Means (very compact size)
    fig1, ax1 = plt.subplots(1, 1, figsize=(4, 3))
    
    # Boxplot for Means
    mean_data = boxplot_data[boxplot_data['Statistic'] == 'Mean']
    mean_iceland = mean_data[mean_data['GROUP'] == 'Iceland']['Value']
    mean_eurozone = mean_data[mean_data['GROUP'] == 'Eurozone']['Value']
    
    bp1 = ax1.boxplot([mean_eurozone, mean_iceland], labels=['Eurozone', 'Iceland'], patch_artist=True)
    bp1['boxes'][0].set_facecolor(COLORBLIND_SAFE[0])
    bp1['boxes'][1].set_facecolor(COLORBLIND_SAFE[1])
    
    ax1.set_title('Panel A: Distribution of Means Across All Capital Flow Indicators', 
                  fontweight='bold', fontsize=10, pad=10)
    ax1.set_ylabel('Mean (% of GDP, annualized)', fontsize=9)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add summary stats to plot
    ax1.text(0.02, 0.98, f'Eurozone Avg: {mean_eurozone.mean():.2f}%\nIceland Avg: {mean_iceland.mean():.2f}%', 
             transform=ax1.transAxes, verticalalignment='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    st.pyplot(fig1)
    
    # Download button for means boxplot
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf1.seek(0)
    
    st.download_button(
        label="📥 Download Means Boxplot (PNG)",
        data=buf1.getvalue(),
        file_name="case_study_1_means_boxplot.png",
        mime="image/png",
        key="download_means"
    )
    
    # Second boxplot - Standard Deviations (very compact size)
    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 3))
    
    std_data = boxplot_data[boxplot_data['Statistic'] == 'Standard Deviation']
    std_iceland = std_data[std_data['GROUP'] == 'Iceland']['Value']
    std_eurozone = std_data[std_data['GROUP'] == 'Eurozone']['Value']
    
    bp2 = ax2.boxplot([std_eurozone, std_iceland], labels=['Eurozone', 'Iceland'], patch_artist=True)
    bp2['boxes'][0].set_facecolor(COLORBLIND_SAFE[0])
    bp2['boxes'][1].set_facecolor(COLORBLIND_SAFE[1])
    
    ax2.set_title('Panel B: Distribution of Standard Deviations Across All Capital Flow Indicators', 
                  fontweight='bold', fontsize=10, pad=10)
    ax2.set_ylabel('Std Dev. (% of GDP, annualized)', fontsize=9)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    
    # Add summary stats to plot (fix newline character display)
    volatility_ratio = std_iceland.mean() / std_eurozone.mean()
    ax2.text(0.02, 0.98, f'Eurozone Avg: {std_eurozone.mean():.2f}%\nIceland Avg: {std_iceland.mean():.2f}%\nRatio: {volatility_ratio:.2f}x', 
             transform=ax2.transAxes, verticalalignment='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax2.set_title('Distribution of Standard Deviations: Iceland vs Eurozone Capital Flow Volatility', 
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
        file_name="case_study_1_stddev_boxplot.png",
        mime="image/png",
        key="download_stddev"
    )
    
    # Summary statistics from boxplots
    iceland_mean_avg = mean_iceland.mean()
    eurozone_mean_avg = mean_eurozone.mean()
    iceland_std_avg = std_iceland.mean()
    eurozone_std_avg = std_eurozone.mean()
    volatility_ratio = iceland_std_avg / eurozone_std_avg
    
    st.markdown("### Comprehensive Statistical Summary from Boxplots:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Means Across All Indicators:**
        - Eurozone: {eurozone_mean_avg:.2f}% (median: {mean_eurozone.median():.2f}%)
        - Iceland: {iceland_mean_avg:.2f}% (median: {mean_iceland.median():.2f}%)
        """)
    
    with col2:
        st.markdown(f"""
        **Standard Deviations Across All Indicators:**
        - Eurozone: {eurozone_std_avg:.2f}% (median: {std_eurozone.median():.2f}%)
        - Iceland: {iceland_std_avg:.2f}% (median: {std_iceland.median():.2f}%)
        """)
    
    st.info(f"**Volatility Comparison:** Iceland volatility is {volatility_ratio:.2f}x higher than Eurozone on average")
    
    # 1b. Individual Country Comparisons
    st.subheader("1b. Individual Country Comparisons: Iceland vs Each Eurozone Country")
    
    st.markdown("""
    **Enhanced Analysis:** Rather than comparing Iceland to the Eurozone as an aggregate group, 
    this section compares Iceland's values to each individual Eurozone country separately.
    """)
    
    # Prepare data for individual country boxplots
    mean_data_individual = individual_country_data[individual_country_data['Statistic'] == 'Mean']
    std_data_individual = individual_country_data[individual_country_data['Statistic'] == 'Standard Deviation']
    
    # Calculate median values for ordering
    mean_medians = mean_data_individual.groupby('COUNTRY')['Value'].median().sort_values(ascending=False)
    std_medians = std_data_individual.groupby('COUNTRY')['Value'].median().sort_values(ascending=False)
    
    # Create boxplots ordered by descending median
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 4))
    
    # Prepare data for means boxplot, ordered by median
    mean_boxplot_data = []
    mean_boxplot_labels = []
    iceland_mean_position = None
    
    for i, (country, _) in enumerate(mean_medians.items()):
        country_means = mean_data_individual[mean_data_individual['COUNTRY'] == country]['Value']
        mean_boxplot_data.append(country_means)
        mean_boxplot_labels.append(country)
        if country == 'Iceland':
            iceland_mean_position = i
    
    # Create means boxplot
    bp3 = ax3.boxplot(mean_boxplot_data, labels=mean_boxplot_labels, patch_artist=True)
    
    # Color Iceland distinctly (red) and others (blue)
    for i, box in enumerate(bp3['boxes']):
        if i == iceland_mean_position:
            box.set_facecolor(COLORBLIND_SAFE[3])  # Red for Iceland
            box.set_alpha(0.8)
        else:
            box.set_facecolor(COLORBLIND_SAFE[0])  # Blue for Eurozone countries
            box.set_alpha(0.6)
    
    ax3.set_title('Panel C: Distribution of Means - Iceland vs Individual Eurozone Countries\n(Ordered by Descending Median Value)', 
                  fontweight='bold', fontsize=11, pad=10)
    ax3.set_ylabel('Mean (% of GDP, annualized)', fontsize=9)
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add reference line for Iceland's median
    iceland_median_mean = mean_medians['Iceland']
    ax3.axhline(y=iceland_median_mean, color=COLORBLIND_SAFE[3], linestyle='--', alpha=0.7, linewidth=1.5, 
                label=f'Iceland Median: {iceland_median_mean:.2f}%')
    ax3.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig3)
    
    # Download button for individual means boxplot
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf3.seek(0)
    
    st.download_button(
        label="📥 Download Individual Country Means Boxplot (PNG)",
        data=buf3.getvalue(),
        file_name="case_study_1_individual_means_boxplot.png",
        mime="image/png",
        key="download_individual_means"
    )
    
    # Create standard deviations boxplot, ordered by median
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 4))
    
    # Prepare data for std dev boxplot, ordered by median
    std_boxplot_data = []
    std_boxplot_labels = []
    iceland_std_position = None
    
    for i, (country, _) in enumerate(std_medians.items()):
        country_stds = std_data_individual[std_data_individual['COUNTRY'] == country]['Value']
        std_boxplot_data.append(country_stds)
        std_boxplot_labels.append(country)
        if country == 'Iceland':
            iceland_std_position = i
    
    # Create std dev boxplot
    bp4 = ax4.boxplot(std_boxplot_data, labels=std_boxplot_labels, patch_artist=True)
    
    # Color Iceland distinctly (red) and others (blue)
    for i, box in enumerate(bp4['boxes']):
        if i == iceland_std_position:
            box.set_facecolor(COLORBLIND_SAFE[3])  # Red for Iceland
            box.set_alpha(0.8)
        else:
            box.set_facecolor(COLORBLIND_SAFE[0])  # Blue for Eurozone countries
            box.set_alpha(0.6)
    
    ax4.set_title('Panel D: Distribution of Standard Deviations - Iceland vs Individual Eurozone Countries\n(Ordered by Descending Median Value)', 
                  fontweight='bold', fontsize=11, pad=10)
    ax4.set_ylabel('Std Dev. (% of GDP, annualized)', fontsize=9)
    ax4.tick_params(axis='x', rotation=45, labelsize=8)
    ax4.tick_params(axis='y', labelsize=8)
    
    # Add reference line for Iceland's median
    iceland_median_std = std_medians['Iceland']
    ax4.axhline(y=iceland_median_std, color=COLORBLIND_SAFE[3], linestyle='--', alpha=0.7, linewidth=1.5,
                label=f'Iceland Median: {iceland_median_std:.2f}%')
    ax4.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig4)
    
    # Download button for individual std dev boxplot
    buf4 = io.BytesIO()
    fig4.savefig(buf4, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf4.seek(0)
    
    st.download_button(
        label="📥 Download Individual Country Std Dev Boxplot (PNG)",
        data=buf4.getvalue(),
        file_name="case_study_1_individual_stddev_boxplot.png",
        mime="image/png",
        key="download_individual_stddev"
    )
    
    # Summary of individual country comparison
    iceland_mean_rank = list(mean_medians.index).index('Iceland') + 1
    iceland_std_rank = list(std_medians.index).index('Iceland') + 1
    total_countries = len(mean_medians)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Iceland's Mean Rank", f"{iceland_mean_rank} of {total_countries}", 
                 f"{'Higher' if iceland_mean_rank <= total_countries/2 else 'Lower'} than average")
    with col2:
        st.metric("Iceland's Volatility Rank", f"{iceland_std_rank} of {total_countries}", 
                 f"{'More volatile' if iceland_std_rank <= total_countries/2 else 'Less volatile'} than average")
    
    st.markdown(f"""
    **Individual Country Analysis Summary:**
    - **Means:** Iceland ranks #{iceland_mean_rank} out of {total_countries} countries by median mean across all indicators
    - **Volatility:** Iceland ranks #{iceland_std_rank} out of {total_countries} countries by median standard deviation
    - **Key Insight:** This shows Iceland's position relative to each individual Eurozone member rather than the aggregate
    """)
    
    st.markdown("---")
    
    # 2. Comprehensive Statistical Summary Table
    st.header("2. Comprehensive Statistical Summary Table")
    
    st.markdown("**All Indicators - Iceland vs Eurozone Statistics**")
    
    # Create a clean table with one row per indicator (both groups side-by-side)
    # Sort indicators properly first
    sorted_indicators = sort_indicators_by_type(analysis_indicators)
    table_data = []
    for indicator in sorted_indicators:
        clean_name = indicator.replace('_PGDP', '')
        nickname = get_nickname(clean_name)
        indicator_stats = group_stats[group_stats['Indicator'] == clean_name]
        
        # Get stats for both groups
        iceland_stats = indicator_stats[indicator_stats['Group'] == 'Iceland'].iloc[0] if len(indicator_stats[indicator_stats['Group'] == 'Iceland']) > 0 else None
        eurozone_stats = indicator_stats[indicator_stats['Group'] == 'Eurozone'].iloc[0] if len(indicator_stats[indicator_stats['Group'] == 'Eurozone']) > 0 else None
        
        if iceland_stats is not None and eurozone_stats is not None:
            table_data.append({
                'Indicator': nickname,
                'Iceland Mean': f"{iceland_stats['Mean']:.2f}",
                'Iceland Std Dev': f"{iceland_stats['Std_Dev']:.2f}",
                'Iceland CV%': f"{iceland_stats['CV_Percent']:.1f}",
                'Eurozone Mean': f"{eurozone_stats['Mean']:.2f}",
                'Eurozone Std Dev': f"{eurozone_stats['Std_Dev']:.2f}",
                'Eurozone CV%': f"{eurozone_stats['CV_Percent']:.1f}",
                'CV Ratio (Ice/Euro)': f"{iceland_stats['CV_Percent']/eurozone_stats['CV_Percent']:.2f}" if eurozone_stats['CV_Percent'] != 0 else "∞"
            })
    
    # Create DataFrame and display as table
    summary_df = pd.DataFrame(table_data)
    
    # Style the table for better readability
    styled_table = summary_df.style.set_properties(**{
        'text-align': 'center',
        'font-size': '10px',
        'border': '1px solid #ddd'
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('font-weight', 'bold'), ('font-size', '11px')]},
        {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},
        {'selector': 'td:first-child', 'props': [('text-align', 'left'), ('font-weight', 'bold')]}  # Left-align indicator names
    ])
    
    st.dataframe(styled_table, use_container_width=True, hide_index=True)
    
    st.info(f"**Summary:** Statistics for all {len(analysis_indicators)} capital flow indicators. CV% = Coefficient of Variation (Std Dev / |Mean| × 100). Higher CV% indicates greater volatility relative to mean.")
    
    # Create summary with CV ratios for download
    summary_pivot = group_stats.pivot_table(
        index='Indicator',
        columns='Group',
        values=['Mean', 'Std_Dev', 'Skewness', 'CV_Percent'],
        aggfunc='first'
    )
    
    comprehensive_table = pd.DataFrame({
        'Mean_Eurozone': summary_pivot[('Mean', 'Eurozone')],
        'Mean_Iceland': summary_pivot[('Mean', 'Iceland')],
        'StdDev_Eurozone': summary_pivot[('Std_Dev', 'Eurozone')],
        'StdDev_Iceland': summary_pivot[('Std_Dev', 'Iceland')],
        'Skew_Eurozone': summary_pivot[('Skewness', 'Eurozone')],
        'Skew_Iceland': summary_pivot[('Skewness', 'Iceland')],
        'CV_Eurozone': summary_pivot[('CV_Percent', 'Eurozone')],
        'CV_Iceland': summary_pivot[('CV_Percent', 'Iceland')]
    })
    
    comprehensive_table['CV_Ratio_Iceland_Eurozone'] = (
        comprehensive_table['CV_Iceland'] / comprehensive_table['CV_Eurozone']
    ).round(2)
    
    avg_cv_ratio = comprehensive_table['CV_Ratio_Iceland_Eurozone'].mean()
    higher_cv_count = (comprehensive_table['CV_Ratio_Iceland_Eurozone'] > 1).sum()
    
    st.markdown(f"""
    **CV Ratio Summary (Iceland/Eurozone):**
    - Average CV Ratio: {avg_cv_ratio:.2f}
    - Indicators where Iceland > Eurozone: {higher_cv_count}/{len(comprehensive_table)} ({higher_cv_count/len(comprehensive_table)*100:.1f}%)
    """)
    
    st.markdown("---")
    
    # 3. Hypothesis Testing Results
    st.header("3. Hypothesis Testing Results")
    
    st.markdown("""
    **F-Tests for Equal Variances (Iceland vs. Eurozone)**
    
    - **H₀:** Equal volatility  
    - **H₁:** Different volatility  
    - **α = 0.05**
    """)
    
    # Create a clean static table for hypothesis tests
    results_display = test_results.copy()
    
    # Sort by investment type rather than F-statistic
    results_display['Sort_Key'] = results_display['Indicator'].apply(get_investment_type_order)
    results_display = results_display.sort_values('Sort_Key')
    
    # Add nicknames and format for display
    results_display['Indicator_Nick'] = results_display['Indicator'].apply(get_nickname)
    results_display['Significant'] = results_display.apply(
        lambda row: '***' if row['P_Value'] < 0.001 else '**' if row['P_Value'] < 0.01 else '*' if row['P_Value'] < 0.05 else '', 
        axis=1
    )
    results_display['Higher Volatility'] = results_display['Iceland_Higher_Volatility'].map({True: 'Iceland', False: 'Eurozone'})
    
    # Create formatted table
    test_table_data = []
    for _, row in results_display.iterrows():
        test_table_data.append({
            'Indicator': row['Indicator_Nick'],
            'F-Statistic': f"{row['F_Statistic']:.2f}",
            'P-Value': f"{row['P_Value']:.4f}",
            'Significance': row['Significant'],
            'Higher Volatility': row['Higher Volatility']
        })
    
    test_df = pd.DataFrame(test_table_data)
    
    # Display as static table with better formatting
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Style the test results table
        styled_test_table = test_df.style.set_properties(**{
            'text-align': 'center',
            'font-size': '11px',
            'border': '1px solid #ddd'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#e6f3ff'), ('font-weight', 'bold'), ('text-align', 'center')]},
            {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},
            {'selector': 'td:first-child', 'props': [('text-align', 'left')]}  # Left-align indicator names
        ])
        
        st.dataframe(styled_test_table, use_container_width=True, hide_index=True)
        st.caption("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    
    with col2:
        st.markdown("**Legend:**")
        st.markdown("- **F-Statistic**: Ratio of variances")
        st.markdown("- **P-Value**: Statistical significance")
        st.markdown("- **Higher Volatility**: Which group shows more volatility")
    
    # Test summary
    total_indicators = len(test_results)
    iceland_higher_count = test_results['Iceland_Higher_Volatility'].sum()
    sig_5pct_count = test_results['Significant_5pct'].sum()
    sig_1pct_count = test_results['Significant_1pct'].sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Iceland Higher Volatility", f"{iceland_higher_count}/{total_indicators}", f"{iceland_higher_count/total_indicators*100:.1f}%")
    with col2:
        st.metric("Significant (5%)", f"{sig_5pct_count}/{total_indicators}", f"{sig_5pct_count/total_indicators*100:.1f}%")
    with col3:
        st.metric("Significant (1%)", f"{sig_1pct_count}/{total_indicators}", f"{sig_1pct_count/total_indicators*100:.1f}%")
    
    conclusion = "Strong evidence supports" if iceland_higher_count/total_indicators > 0.6 else "Mixed evidence for"
    st.success(f"**Conclusion:** {conclusion} the hypothesis that Iceland has higher capital flow volatility.")
    
    st.markdown("---")
    
    # 4. Time Series Visualization
    st.header("4. Time Series Analysis")
    
    # Create date column
    final_data_copy = final_data.copy()
    final_data_copy['Date'] = pd.to_datetime(
        final_data_copy['YEAR'].astype(str) + '-' + 
        ((final_data_copy['QUARTER'] - 1) * 3 + 1).astype(str) + '-01'
    )
    
    # Show ALL indicators, sorted properly
    selected_indicators = sort_indicators_by_type(analysis_indicators)
    
    st.markdown(f"**Showing all {len(selected_indicators)} indicators sorted by investment type**")
    
    # Create individual time series plots for better readability and downloads
    time_series_figures = []
    
    for i, indicator in enumerate(selected_indicators):
        # Create individual plot for each indicator (very compact size)
        fig_ts, ax = plt.subplots(1, 1, figsize=(6, 2.5))
        
        clean_name = indicator.replace('_PGDP', '')
        nickname = get_nickname(clean_name)
        
        # Plot Iceland
        iceland_data = final_data_copy[final_data_copy['GROUP'] == 'Iceland']
        ax.plot(iceland_data['Date'], iceland_data[indicator], 
                color=COLORBLIND_SAFE[1], linewidth=1.5, label='Iceland')
        
        # Plot Eurozone average
        eurozone_avg = final_data_copy[final_data_copy['GROUP'] == 'Eurozone'].groupby('Date')[indicator].mean()
        ax.plot(eurozone_avg.index, eurozone_avg.values, 
                color=COLORBLIND_SAFE[0], linewidth=1.5, label='Eurozone Average')
        
        # Formatting
        f_stat = test_results[test_results['Indicator'] == clean_name]['F_Statistic'].iloc[0]
        panel_letter = chr(65 + i)  # A, B, C, etc.
        ax.set_title(f'Panel {panel_letter}: {nickname} (F-statistic: {f_stat:.2f})', 
                    fontweight='bold', fontsize=9, pad=8)
        ax.set_ylabel('% of GDP (annualized)', fontsize=8)
        ax.set_xlabel('Year', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.legend(loc='best', fontsize=8, frameon=True, fancybox=False, shadow=False)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        st.pyplot(fig_ts)
        
        # Individual download button for each time series
        buf_ts = io.BytesIO()
        fig_ts.savefig(buf_ts, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf_ts.seek(0)
        
        clean_filename = nickname.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        st.download_button(
            label=f"📥 Download {nickname} Time Series (PNG)",
            data=buf_ts.getvalue(),
            file_name=f"case_study_1_{clean_filename}_timeseries.png",
            mime="image/png",
            key=f"download_ts_{i}"
        )
        
        time_series_figures.append(fig_ts)
    
    st.markdown("---")
    
    # 5. Key Findings Summary
    st.header("5. Key Findings Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### Statistical Evidence:
        - **{iceland_higher_count/total_indicators*100:.1f}% of capital flow indicators** show higher volatility in Iceland
        - **{sig_5pct_count/total_indicators*100:.1f}% of indicators** show statistically significant differences (p<0.05)
        - **Iceland's average volatility** is {volatility_ratio:.2f} times higher than Eurozone countries
        - **Most significant differences** in portfolio investment and direct investment flows
        """)
    
    with col2:
        st.markdown("""
        ### Policy Implications:
        - Evidence supports the hypothesis that Iceland has higher capital flow volatility
        - Euro adoption could potentially reduce financial volatility for Iceland
        - Greater macroeconomic stability possible through currency union
        - Consider implementation timeline and structural adjustments needed
        """)
    
    # Download section
    st.markdown("---")
    st.header("6. Download Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Download comprehensive table
        csv = comprehensive_table.to_csv(index=True)
        st.download_button(
            label="📥 Download Summary Table (CSV)",
            data=csv,
            file_name="case_study_1_summary_statistics.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download test results
        csv = test_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Test Results (CSV)",
            data=csv,
            file_name="case_study_1_hypothesis_tests.csv",
            mime="text/csv"
        )
    
    with col3:
        # Download group statistics
        csv = group_stats.to_csv(index=False)
        st.download_button(
            label="📥 Download Group Statistics (CSV)",
            data=csv,
            file_name="case_study_1_group_statistics.csv",
            mime="text/csv"
        )
    
    with col4:
        # Generate and download HTML report  
        if st.button("📄 Generate HTML Report", type="secondary"):
            with st.spinner("Generating HTML report..."):
                html_file = generate_html_report(final_data, analysis_indicators, test_results, group_stats, boxplot_data)
                
                if html_file and Path(html_file).exists():
                    with open(html_file, "r", encoding='utf-8') as f:
                        html_data = f.read()
                    
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=html_data,
                        file_name=f"capital_flows_report_{datetime.now().strftime('%Y%m%d')}.html",
                        mime="text/html",
                        key="html_download"
                    )
                    
                    st.success("✅ HTML report generated successfully!")
                    st.info("💡 **Tip:** You can open the HTML file in any browser and print to PDF or save as PDF for a professional document.")
                    
                    # Clean up temporary file
                    try:
                        Path(html_file).unlink()
                        import shutil
                        shutil.rmtree("temp_html_reports", ignore_errors=True)
                    except:
                        pass
                else:
                    st.error("❌ Failed to generate HTML report.")

def generate_html_report(final_data, analysis_indicators, test_results, group_stats, boxplot_data):
    """Generate an HTML report that mimics the app's formatting exactly"""
    try:
        # Create temporary HTML file
        temp_dir = Path("temp_html_reports")
        temp_dir.mkdir(exist_ok=True)
        html_filename = temp_dir / f"capital_flows_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Generate plots as base64 images for embedding
        def create_plot_base64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)
            return f"data:image/png;base64,{img_base64}"
        
        # Create boxplots exactly like app
        fig1, ax1 = plt.subplots(1, 1, figsize=(4, 3))
        mean_data = boxplot_data[boxplot_data['Statistic'] == 'Mean']
        mean_iceland = mean_data[mean_data['GROUP'] == 'Iceland']['Value']
        mean_eurozone = mean_data[mean_data['GROUP'] == 'Eurozone']['Value']
        
        bp1 = ax1.boxplot([mean_eurozone, mean_iceland], labels=['Eurozone', 'Iceland'], patch_artist=True)
        bp1['boxes'][0].set_facecolor(COLORBLIND_SAFE[0])
        bp1['boxes'][1].set_facecolor(COLORBLIND_SAFE[1])
        ax1.set_title('Panel A: Distribution of Means Across All Capital Flow Indicators', 
                     fontweight='bold', fontsize=10, pad=10)
        ax1.set_ylabel('Mean (% of GDP, annualized)', fontsize=9)
        ax1.tick_params(axis='both', which='major', labelsize=8)
        ax1.text(0.02, 0.98, f'Eurozone Avg: {mean_eurozone.mean():.2f}%\nIceland Avg: {mean_iceland.mean():.2f}%', 
                transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        plt.tight_layout()
        boxplot1_img = create_plot_base64(fig1)
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(4, 3))
        std_data = boxplot_data[boxplot_data['Statistic'] == 'Standard Deviation']
        std_iceland = std_data[std_data['GROUP'] == 'Iceland']['Value']
        std_eurozone = std_data[std_data['GROUP'] == 'Eurozone']['Value']
        
        bp2 = ax2.boxplot([std_eurozone, std_iceland], labels=['Eurozone', 'Iceland'], patch_artist=True)
        bp2['boxes'][0].set_facecolor(COLORBLIND_SAFE[0])
        bp2['boxes'][1].set_facecolor(COLORBLIND_SAFE[1])
        ax2.set_title('Panel B: Distribution of Standard Deviations Across All Capital Flow Indicators', 
                     fontweight='bold', fontsize=10, pad=10)
        ax2.set_ylabel('Std Dev. (% of GDP, annualized)', fontsize=9)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        
        volatility_ratio = std_iceland.mean() / std_eurozone.mean()
        ax2.text(0.02, 0.98, f'Eurozone Avg: {std_eurozone.mean():.2f}%\nIceland Avg: {std_iceland.mean():.2f}%\nRatio: {volatility_ratio:.2f}x', 
                transform=ax2.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        plt.tight_layout()
        boxplot2_img = create_plot_base64(fig2)
        
        # Create summary statistics table exactly like app
        sorted_indicators = sort_indicators_by_type(analysis_indicators)
        table_rows = []
        for indicator in sorted_indicators:
            clean_name = indicator.replace('_PGDP', '')
            nickname = get_nickname(clean_name)
            indicator_stats = group_stats[group_stats['Indicator'] == clean_name]
            
            iceland_stats = indicator_stats[indicator_stats['Group'] == 'Iceland'].iloc[0] if len(indicator_stats[indicator_stats['Group'] == 'Iceland']) > 0 else None
            eurozone_stats = indicator_stats[indicator_stats['Group'] == 'Eurozone'].iloc[0] if len(indicator_stats[indicator_stats['Group'] == 'Eurozone']) > 0 else None
            
            if iceland_stats is not None and eurozone_stats is not None:
                cv_ratio = iceland_stats['CV_Percent']/eurozone_stats['CV_Percent'] if eurozone_stats['CV_Percent'] != 0 else float('inf')
                table_rows.append(f"""
                    <tr>
                        <td style="text-align: left; font-weight: bold;">{nickname}</td>
                        <td>{iceland_stats['Mean']:.2f}</td>
                        <td>{iceland_stats['Std_Dev']:.2f}</td>
                        <td>{iceland_stats['CV_Percent']:.1f}</td>
                        <td>{eurozone_stats['Mean']:.2f}</td>
                        <td>{eurozone_stats['Std_Dev']:.2f}</td>
                        <td>{eurozone_stats['CV_Percent']:.1f}</td>
                        <td>{cv_ratio:.2f}</td>
                    </tr>
                """)
        
        # Create hypothesis test results table exactly like app
        results_display = test_results.copy()
        results_display['Sort_Key'] = results_display['Indicator'].apply(get_investment_type_order)
        results_display = results_display.sort_values('Sort_Key')
        
        test_table_rows = []
        for _, row in results_display.iterrows():
            nickname = get_nickname(row['Indicator'])
            significance = '***' if row['P_Value'] < 0.001 else '**' if row['P_Value'] < 0.01 else '*' if row['P_Value'] < 0.05 else ''
            higher_vol = 'Iceland' if row['Iceland_Higher_Volatility'] else 'Eurozone'
            
            test_table_rows.append(f"""
                <tr>
                    <td style="text-align: left; font-weight: bold;">{nickname}</td>
                    <td>{row['F_Statistic']:.2f}</td>
                    <td>{row['P_Value']:.4f}</td>
                    <td>{significance}</td>
                    <td>{higher_vol}</td>
                </tr>
            """)
        
        # Create time series plots
        final_data_copy = final_data.copy()
        final_data_copy['Date'] = pd.to_datetime(
            final_data_copy['YEAR'].astype(str) + '-' + 
            ((final_data_copy['QUARTER'] - 1) * 3 + 1).astype(str) + '-01'
        )
        
        time_series_plots = []
        for i, indicator in enumerate(sorted_indicators):
            fig_ts, ax = plt.subplots(1, 1, figsize=(6, 2.5))
            
            clean_name = indicator.replace('_PGDP', '')
            nickname = get_nickname(clean_name)
            
            # Plot data exactly like app
            iceland_data = final_data_copy[final_data_copy['GROUP'] == 'Iceland']
            ax.plot(iceland_data['Date'], iceland_data[indicator], 
                    color=COLORBLIND_SAFE[1], linewidth=1.5, label='Iceland')
            
            eurozone_avg = final_data_copy[final_data_copy['GROUP'] == 'Eurozone'].groupby('Date')[indicator].mean()
            ax.plot(eurozone_avg.index, eurozone_avg.values, 
                    color=COLORBLIND_SAFE[0], linewidth=1.5, label='Eurozone Average')
            
            # Formatting exactly like app
            f_stat = test_results[test_results['Indicator'] == clean_name]['F_Statistic'].iloc[0]
            panel_letter = chr(65 + i)  # A, B, C, etc.
            ax.set_title(f'Panel {panel_letter}: {nickname} (F-statistic: {f_stat:.2f})', 
                        fontweight='bold', fontsize=9, pad=8)
            ax.set_ylabel('% of GDP (annualized)', fontsize=8)
            ax.set_xlabel('Year', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.legend(loc='best', fontsize=8, frameon=True, fancybox=False, shadow=False)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            
            plt.tight_layout()
            time_series_plots.append(create_plot_base64(fig_ts))
        
        # Calculate key statistics
        total_indicators = len(test_results)
        iceland_higher_count = test_results['Iceland_Higher_Volatility'].sum()
        sig_5pct_count = test_results['Significant_5pct'].sum()
        sig_1pct_count = test_results['Significant_1pct'].sum()
        
        # Generate HTML exactly like the app structure
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Capital Flow Volatility Analysis - Case Study 1</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #1f77b4; text-align: center; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; }}
                h2 {{ color: #ff7f0e; border-bottom: 2px solid #ff7f0e; padding-bottom: 5px; }}
                h3 {{ color: #2ca02c; }}
                .info-box {{ background-color: #e1f5fe; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #0288d1; }}
                .success-box {{ background-color: #e8f5e8; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #4caf50; }}
                .metric {{ display: inline-block; margin: 10px 20px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f0f0f0; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .plots {{ text-align: center; margin: 20px 0; }}
                .plot-row {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .time-series {{ margin: 15px 0; }}
                .columns {{ display: flex; gap: 30px; }}
                .column {{ flex: 1; }}
            </style>
        </head>
        <body>
            <h1>📊 Capital Flow Volatility Analysis</h1>
            <h2 style="text-align: center; color: #666;">Case Study 1: Iceland vs. Eurozone Comparison</h2>
            
            <div class="info-box">
                <strong>Research Question:</strong> Should Iceland adopt the Euro as its currency?<br>
                <strong>Hypothesis:</strong> Iceland's capital flows show more volatility than the Eurozone bloc average
            </div>
            
            <div style="margin: 20px 0;">
                <div class="metric"><strong>Observations:</strong> {final_data.shape[0]:,}</div>
                <div class="metric"><strong>Indicators:</strong> {len(analysis_indicators)}</div>
                <div class="metric"><strong>Countries:</strong> {final_data['COUNTRY'].nunique()}</div>
                <div class="metric"><strong>Time Period:</strong> {final_data['YEAR'].min()}-{final_data['YEAR'].max()}</div>
            </div>
            
            <hr>
            
            <h2>1. Summary Statistics and Boxplots</h2>
            
            <div class="plot-row">
                <img src="{boxplot1_img}" alt="Means Boxplot" style="max-width: 45%;">
                <img src="{boxplot2_img}" alt="Standard Deviations Boxplot" style="max-width: 45%;">
            </div>
            
            <div class="columns">
                <div class="column">
                    <strong>Means Across All Indicators:</strong><br>
                    • Eurozone: {mean_eurozone.mean():.2f}% (median: {mean_eurozone.median():.2f}%)<br>
                    • Iceland: {mean_iceland.mean():.2f}% (median: {mean_iceland.median():.2f}%)
                </div>
                <div class="column">
                    <strong>Standard Deviations Across All Indicators:</strong><br>
                    • Eurozone: {std_eurozone.mean():.2f}% (median: {std_eurozone.median():.2f}%)<br>
                    • Iceland: {std_iceland.mean():.2f}% (median: {std_iceland.median():.2f}%)
                </div>
            </div>
            
            <div class="info-box">
                <strong>Volatility Comparison:</strong> Iceland volatility is {volatility_ratio:.2f}x higher than Eurozone on average
            </div>
            
            <hr>
            
            <h2>2. Comprehensive Statistical Summary Table</h2>
            <p><strong>All Indicators - Iceland vs Eurozone Statistics</strong></p>
            
            <table>
                <thead>
                    <tr>
                        <th>Indicator</th>
                        <th>Iceland Mean</th>
                        <th>Iceland Std Dev</th>
                        <th>Iceland CV%</th>
                        <th>Eurozone Mean</th>
                        <th>Eurozone Std Dev</th>
                        <th>Eurozone CV%</th>
                        <th>CV Ratio (Ice/Euro)</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
            
            <div class="info-box">
                <strong>Summary:</strong> Statistics for all {len(analysis_indicators)} capital flow indicators. CV% = Coefficient of Variation (Std Dev / |Mean| × 100). Higher CV% indicates greater volatility relative to mean.
            </div>
            
            <hr>
            
            <h2>3. Hypothesis Testing Results</h2>
            <p><strong>F-Tests for Equal Variances (Iceland vs. Eurozone)</strong></p>
            
            <ul>
                <li><strong>H₀:</strong> Equal volatility</li>
                <li><strong>H₁:</strong> Different volatility</li>
                <li><strong>α = 0.05</strong></li>
            </ul>
            
            <table style="width: 80%; margin: 0 auto;">
                <thead>
                    <tr>
                        <th>Indicator</th>
                        <th>F-Statistic</th>
                        <th>P-Value</th>
                        <th>Significance</th>
                        <th>Higher Volatility</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(test_table_rows)}
                </tbody>
            </table>
            
            <p style="text-align: center; font-size: 0.9em;"><em>Significance levels: *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05</em></p>
            
            <div style="display: flex; justify-content: space-around; margin: 20px 0;">
                <div class="metric">
                    <strong>Iceland Higher Volatility:</strong><br>
                    {iceland_higher_count}/{total_indicators} ({iceland_higher_count/total_indicators*100:.1f}%)
                </div>
                <div class="metric">
                    <strong>Significant (5%):</strong><br>
                    {sig_5pct_count}/{total_indicators} ({sig_5pct_count/total_indicators*100:.1f}%)
                </div>
                <div class="metric">
                    <strong>Significant (1%):</strong><br>
                    {sig_1pct_count}/{total_indicators} ({sig_1pct_count/total_indicators*100:.1f}%)
                </div>
            </div>
            
            <div class="success-box">
                <strong>Conclusion:</strong> {'Strong evidence supports' if iceland_higher_count/total_indicators > 0.6 else 'Mixed evidence for'} the hypothesis that Iceland has higher capital flow volatility.
            </div>
            
            <hr>
            
            <h2>4. Time Series Analysis</h2>
            <p><strong>Showing all {len(analysis_indicators)} indicators sorted by investment type</strong></p>
            
            <div class="time-series">
                {''.join([f'<img src="{plot}" alt="Time Series Plot" style="max-width: 100%; margin: 10px 0;"><br>' for plot in time_series_plots])}
            </div>
            
            <hr>
            
            <h2>5. Key Findings Summary</h2>
            
            <div class="columns">
                <div class="column">
                    <h3>Statistical Evidence:</h3>
                    <ul>
                        <li><strong>{iceland_higher_count/total_indicators*100:.1f}% of capital flow indicators</strong> show higher volatility in Iceland</li>
                        <li><strong>{sig_5pct_count/total_indicators*100:.1f}% of indicators</strong> show statistically significant differences (p&lt;0.05)</li>
                        <li><strong>Iceland's average volatility</strong> is {volatility_ratio:.2f} times higher than Eurozone countries</li>
                        <li><strong>Most significant differences</strong> in portfolio investment and direct investment flows</li>
                    </ul>
                </div>
                <div class="column">
                    <h3>Policy Implications:</h3>
                    <ul>
                        <li>Evidence supports the hypothesis that Iceland has higher capital flow volatility</li>
                        <li>Euro adoption could potentially reduce financial volatility for Iceland</li>
                        <li>Greater macroeconomic stability possible through currency union</li>
                        <li>Consider implementation timeline and structural adjustments needed</li>
                    </ul>
                </div>
            </div>
            
            <hr>
            <p style="text-align: center; color: #666; font-size: 0.9em;">
                Report generated on {datetime.now().strftime('%B %d, %Y at %H:%M')} using automated analysis pipeline.
            </p>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_filename)
        
    except Exception as e:
        st.error(f"Error generating HTML report: {str(e)}")
        return None

if __name__ == "__main__":
    main()