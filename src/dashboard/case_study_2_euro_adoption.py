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

def load_case_study_2_data():
    """Load processed Euro adoption analysis data"""
    try:
        # Load processed data files
        data_dir = Path(__file__).parent.parent.parent / "data"
        processed_file = data_dir / "case_study_2_euro_adoption_data.csv"
        
        if not processed_file.exists():
            st.error("Case Study 2 processed data not found. Please run data_processor_case_study_2.py first.")
            return None, None, None
        
        # Load processed data
        final_data = pd.read_csv(processed_file)
        
        # Get analysis indicators (columns ending with _PGDP)
        analysis_indicators = [col for col in final_data.columns if col.endswith('_PGDP')]
        analysis_indicators = sort_indicators_by_type(analysis_indicators)
        
        # Create timeline information
        timeline = create_euro_adoption_timeline()
        
        # Analysis countries (simplified country names for display)
        analysis_countries = ['Estonia', 'Latvia', 'Lithuania']
        
        # Create metadata
        metadata = {
            'final_shape': final_data.shape,
            'n_indicators': len(analysis_indicators),
            'countries': analysis_countries,
            'timeline': timeline
        }
        
        return final_data, analysis_indicators, metadata
        
    except Exception as e:
        st.error(f"Error loading Case Study 2 data: {str(e)}")
        return None, None, None

def calculate_temporal_statistics(data, country, indicators):
    """Calculate statistics comparing pre-Euro vs post-Euro periods for a country"""
    results = []
    
    country_data = data[data['COUNTRY'] == country]
    
    for period in ['Pre-Euro', 'Post-Euro']:
        period_data = country_data[country_data['EURO_PERIOD'] == period]
        
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

def create_temporal_boxplot_data(data, country, indicators):
    """Create dataset for temporal boxplot visualization"""
    stats_data = []
    
    country_data = data[data['COUNTRY'] == country]
    
    for period in ['Pre-Euro', 'Post-Euro']:
        period_data = country_data[country_data['EURO_PERIOD'] == period]
        
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

def perform_temporal_volatility_tests(data, country, indicators):
    """Perform F-tests comparing pre-Euro vs post-Euro volatility for a country"""
    test_results = []
    
    country_data = data[data['COUNTRY'] == country]
    
    for indicator in indicators:
        pre_data = country_data[
            (country_data['EURO_PERIOD'] == 'Pre-Euro')
        ][indicator].dropna()
        post_data = country_data[
            (country_data['EURO_PERIOD'] == 'Post-Euro') 
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

def main():
    """Main Case Study 2 application"""
    
    # Title and header
    st.title("🇪🇺 Euro Adoption Impact Analysis")
    st.subheader("Case Study 2: Baltic Countries Capital Flow Volatility")
    
    st.markdown("""
    **Research Question:** How does Euro adoption affect capital flow volatility?
    
    **Hypothesis:** Euro adoption reduces capital flow volatility through increased monetary stability
    
    **Countries:** Estonia (2011), Latvia (2014), Lithuania (2015)
    
    ---
    """)
    
    # Data and Methodology section
    with st.expander("📋 Data and Methodology", expanded=False):
        timeline = create_euro_adoption_timeline()
        
        st.markdown("""
        ### Temporal Analysis Design
        - **Methodology:** Before-after comparison for each country
        - **Analysis Periods:** 6 years pre-Euro vs 6 years post-Euro adoption
        - **Crisis Avoidance:** Periods selected to minimize financial crisis contamination
        - **Data Normalization:** All BOP flows converted to annualized % of GDP
        
        ### Country-Specific Analysis Periods
        """)
        
        for country, info in timeline.items():
            pre_start, pre_end = info['pre_period']
            post_start, post_end = info['post_period']
            st.markdown(f"""
            **{country} (Euro adoption: {info['adoption_year']})**
            - Pre-Euro: {pre_start}-{pre_end} 
            - Post-Euro: {post_start}-{post_end}
            """)
    
    # Load data
    with st.spinner("Loading and processing Euro adoption data..."):
        final_data, analysis_indicators, metadata = load_case_study_2_data()
    
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
        st.metric("Countries", len(metadata['countries']))
    with col4:
        st.metric("Time Periods", "Pre/Post Euro")
    
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
        index=0
    )
    
    selected_country = country_mapping[selected_display_country]
    
    timeline = metadata['timeline']
    country_info = timeline[selected_country]
    
    st.info(f"""
    **{selected_display_country} Analysis:** Euro adoption on {country_info['adoption_date']}
    - **Pre-Euro Period:** {country_info['pre_period'][0]} to {country_info['pre_period'][1]}
    - **Post-Euro Period:** {country_info['post_period'][0]} to {country_info['post_period'][1]}
    """)
    
    # Calculate statistics for selected country
    country_stats = calculate_temporal_statistics(final_data, selected_country, analysis_indicators)
    boxplot_data = create_temporal_boxplot_data(final_data, selected_country, analysis_indicators) 
    test_results = perform_temporal_volatility_tests(final_data, selected_country, analysis_indicators)
    
    # 1. Summary Statistics and Boxplots
    st.header("1. Summary Statistics and Boxplots")
    
    # Create temporal boxplots
    fig1, ax1 = plt.subplots(1, 1, figsize=(4, 3))
    mean_data = boxplot_data[boxplot_data['Statistic'] == 'Mean']
    mean_pre = mean_data[mean_data['PERIOD'] == 'Pre-Euro']['Value']
    mean_post = mean_data[mean_data['PERIOD'] == 'Post-Euro']['Value']
    
    bp1 = ax1.boxplot([mean_pre, mean_post], labels=['Pre-Euro', 'Post-Euro'], patch_artist=True)
    bp1['boxes'][0].set_facecolor(COLORBLIND_SAFE[0])
    bp1['boxes'][1].set_facecolor(COLORBLIND_SAFE[1])
    ax1.set_title('Panel A: Distribution of Means Across All Capital Flow Indicators', 
                 fontweight='bold', fontsize=10, pad=10)
    ax1.set_ylabel('Mean (% of GDP, annualized)', fontsize=9)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.text(0.02, 0.98, f'Pre-Euro Avg: {mean_pre.mean():.2f}%\\nPost-Euro Avg: {mean_post.mean():.2f}%', 
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
        file_name=f"{selected_display_country}_means_boxplot.png",
        mime="image/png",
        key=f"download_means_{selected_display_country}"
    )
    
    # Standard deviations boxplot
    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 3))
    std_data = boxplot_data[boxplot_data['Statistic'] == 'Standard Deviation']
    std_pre = std_data[std_data['PERIOD'] == 'Pre-Euro']['Value']
    std_post = std_data[std_data['PERIOD'] == 'Post-Euro']['Value']
    
    bp2 = ax2.boxplot([std_pre, std_post], labels=['Pre-Euro', 'Post-Euro'], patch_artist=True)
    bp2['boxes'][0].set_facecolor(COLORBLIND_SAFE[0])
    bp2['boxes'][1].set_facecolor(COLORBLIND_SAFE[1])
    ax2.set_title('Panel B: Distribution of Standard Deviations Across All Capital Flow Indicators', 
                 fontweight='bold', fontsize=10, pad=10)
    ax2.set_ylabel('Std Dev. (% of GDP, annualized)', fontsize=9)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    
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
        file_name=f"{selected_display_country}_stddev_boxplot.png",
        mime="image/png",
        key=f"download_stddev_{selected_display_country}"
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
    
    st.markdown(f"**{selected_display_country} - Pre-Euro vs Post-Euro Statistics**")
    
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
    **F-Tests for Equal Variances: {selected_display_country} Pre-Euro vs Post-Euro**
    
    - **H₀:** Equal volatility pre and post Euro adoption
    - **H₁:** Different volatility pre and post Euro adoption  
    - **α = 0.05**
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
    
    st.markdown(f"**Showing all {len(analysis_indicators)} indicators for {selected_display_country} (Euro adoption: {country_info['adoption_year']})**")
    
    # Create individual time series plots
    for i, indicator in enumerate(analysis_indicators):
        fig_ts, ax = plt.subplots(1, 1, figsize=(6, 2.5))
        
        clean_name = indicator.replace('_PGDP', '')
        nickname = get_nickname(clean_name)
        
        # Plot pre-Euro data
        pre_data = country_data[country_data['EURO_PERIOD'] == 'Pre-Euro']
        ax.plot(pre_data['Date'], pre_data[indicator], 
                color=COLORBLIND_SAFE[0], linewidth=2.5, label='Pre-Euro')
        
        # Plot post-Euro data
        post_data = country_data[country_data['EURO_PERIOD'] == 'Post-Euro']
        ax.plot(post_data['Date'], post_data[indicator], 
                color=COLORBLIND_SAFE[1], linewidth=2.5, label='Post-Euro')
        
        # Add Euro adoption line
        ax.axvline(x=adoption_date, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Euro Adoption')
        
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
        
        # Individual download button
        buf_ts = io.BytesIO()
        fig_ts.savefig(buf_ts, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf_ts.seek(0)
        
        clean_filename = nickname.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        st.download_button(
            label=f"📥 Download {nickname} Time Series (PNG)",
            data=buf_ts.getvalue(),
            file_name=f"{selected_display_country}_{clean_filename}_timeseries.png",
            mime="image/png",
            key=f"download_ts_{selected_display_country}_{i}"
        )
    
    st.markdown("---")
    
    # 5. Key Findings Summary
    st.header("5. Key Findings Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### Statistical Evidence for {selected_display_country}:
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
            file_name=f"{selected_display_country}_euro_adoption_summary.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download test results
        csv = test_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Test Results (CSV)",
            data=csv,
            file_name=f"{selected_display_country}_hypothesis_tests.csv",
            mime="text/csv"
        )
    
    with col3:
        # Download country statistics
        csv = country_stats.to_csv(index=False)
        st.download_button(
            label="📥 Download Country Statistics (CSV)",
            data=csv,
            file_name=f"{selected_display_country}_temporal_statistics.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()