"""
Capital Flows Analysis - Case Study 2: Lithuania Euro Adoption Report

This report analyzes Lithuania's capital flow volatility before and after Euro adoption (2015).
Consolidated version with parameterized data_type and output_mode.
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
from datetime import datetime
import io
import base64

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

warnings.filterwarnings('ignore')

# Import shared CS2 functions
from cs2_shared_functions import (
    create_euro_adoption_timeline,
    create_euro_periods,
    get_country_specific_crisis_text,
    add_country_specific_crisis_shading,
    calculate_temporal_statistics,
    create_temporal_boxplot_data,
    perform_temporal_volatility_tests,
    load_case_study_2_data,
    load_overall_capital_flows_data_cs2
)

# Import centralized configuration
from dashboard_config import get_data_paths, COLORBLIND_SAFE
from config.constants import (
    get_indicator_nickname,
    get_investment_type_sort_key,
    sort_indicators_by_type
)

# Lithuania-specific configuration
COUNTRY_NAME = 'Lithuania, Republic of'
COUNTRY_SHORT = 'Lithuania'
COUNTRY_FLAG = '🇱🇹'
EURO_ADOPTION_YEAR = 2015

def get_data_configuration(data_type="full"):
    """Get data configuration for Lithuania analysis"""
    if data_type == "winsorized":
        return {
            'analysis_type': 'winsorized',
            'data_label': 'Outlier-Adjusted'
        }
    else:
        return {
            'analysis_type': 'full',
            'data_label': 'Full'
        }

def configure_ui_elements(output_mode="interactive"):
    """Configure UI elements based on output mode"""
    if output_mode == "pdf":
        return {
            'show_download_buttons': False,
            'use_expanders': False,
            'show_tabs': False
        }
    else:
        return {
            'show_download_buttons': True,
            'use_expanders': True,
            'show_tabs': True
        }

def show_lithuania_overall_analysis(data_config, ui_config, include_crisis_years=True):
    """Display Overall Capital Flows Analysis for Lithuania"""
    study_version = "Full Series" if include_crisis_years else "Crisis-Excluded"
    st.markdown(f"*Aggregate net capital flows summary - {study_version}*")

    # Load data with proper data type
    data = load_overall_capital_flows_data_cs2(include_crisis_years, data_config['analysis_type'])

    if data.empty:
        st.error("Failed to load overall capital flows data.")
        return

    # Filter for Lithuania only
    data = data[data['COUNTRY'] == COUNTRY_NAME].copy()

    # Get period column
    period_col = 'EURO_PERIOD_FULL' if include_crisis_years else 'EURO_PERIOD_CRISIS_EXCLUDED'

    # Get available indicators
    indicators = data['INDICATOR'].unique()

    # Colors for periods
    colors = {'Pre-Euro': COLORBLIND_SAFE[0], 'Post-Euro': COLORBLIND_SAFE[1]}

    # Summary Statistics
    if ui_config['use_expanders']:
        with st.expander("📊 Summary Statistics by Period", expanded=True):
            display_summary_statistics(data, indicators, period_col)
    else:
        st.subheader("📊 Summary Statistics by Period")
        display_summary_statistics(data, indicators, period_col)

    # Distribution Comparison
    st.subheader("📦 Distribution Comparison by Period")
    display_distribution_comparison(data, indicators[:4], period_col, colors)

    # Time Series Analysis
    st.subheader("📈 Time Series by Period")
    display_time_series_analysis(data, indicators[:4], period_col, colors, include_crisis_years)

def display_summary_statistics(data, indicators, period_col):
    """Display summary statistics table"""
    summary_stats = []

    for indicator in indicators:
        indicator_data = data[data['INDICATOR'] == indicator]
        for period in ['Pre-Euro', 'Post-Euro']:
            period_data = indicator_data[indicator_data[period_col] == period]['VALUE'].dropna()
            if len(period_data) > 0:
                summary_stats.append({
                    'Indicator': get_indicator_nickname(indicator),
                    'Period': period,
                    'Mean': period_data.mean(),
                    'Std Dev': period_data.std(),
                    'Median': period_data.median(),
                    'Min': period_data.min(),
                    'Max': period_data.max(),
                    'Count': len(period_data)
                })

    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)

        # Pivot for better display
        pivot_summary = summary_df.pivot_table(
            index='Indicator',
            columns='Period',
            values=['Mean', 'Std Dev', 'Median'],
            aggfunc='first'
        ).round(2)

        st.dataframe(pivot_summary, use_container_width=True)

def display_distribution_comparison(data, indicators, period_col, colors):
    """Display distribution comparison boxplots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, indicator in enumerate(indicators):
        if i >= 4:
            break

        ax = axes[i]
        indicator_data = data[data['INDICATOR'] == indicator]

        # Prepare data for boxplot
        pre_data = indicator_data[indicator_data[period_col] == 'Pre-Euro']['VALUE'].dropna()
        post_data = indicator_data[indicator_data[period_col] == 'Post-Euro']['VALUE'].dropna()

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

            ax.set_title(get_indicator_nickname(indicator), fontweight='bold', fontsize=10)
            ax.set_ylabel('% of GDP (annualized)', fontsize=9)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.tick_params(labelsize=8)

    fig.tight_layout()
    st.pyplot(fig)

def display_time_series_analysis(data, indicators, period_col, colors, include_crisis_years):
    """Display time series analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, indicator in enumerate(indicators):
        if i >= 4:
            break

        ax = axes[i]
        indicator_data = data[data['INDICATOR'] == indicator].sort_values('DATE')

        # Plot by period
        for period in ['Pre-Euro', 'Post-Euro']:
            period_data = indicator_data[indicator_data[period_col] == period]
            if len(period_data) > 0:
                ax.plot(period_data['DATE'], period_data['VALUE'],
                       color=colors[period], label=period, linewidth=2, alpha=0.8)

        # Add crisis shading if excluded
        if not include_crisis_years:
            add_country_specific_crisis_shading(ax, COUNTRY_NAME, include_labels=(i == 0))

        # Add Euro adoption line
        adoption_date = pd.to_datetime(f'{EURO_ADOPTION_YEAR}-01-01')
        ax.axvline(x=adoption_date, color='red', linestyle='--', alpha=0.7, linewidth=2,
                  label='Euro Adoption' if i == 0 else "")

        ax.set_title(get_indicator_nickname(indicator), fontweight='bold', fontsize=10)
        ax.set_ylabel('% of GDP (annualized)', fontsize=9)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)

def show_lithuania_indicator_analysis(data_config, ui_config, include_crisis_years=True):
    """Display Indicator-Level Analysis for Lithuania"""
    study_version = "Full Series" if include_crisis_years else "Crisis-Excluded"
    st.markdown(f"*Detailed analysis by capital flow type - {study_version}*")

    # Load data with proper data type
    data = load_case_study_2_data(include_crisis_years, data_config['analysis_type'])

    if data.empty:
        st.error("Failed to load indicator-level data.")
        return

    # Filter for Lithuania
    data = data[data['COUNTRY'] == COUNTRY_NAME].copy()

    # Get period column
    period_col = 'EURO_PERIOD_FULL' if include_crisis_years else 'EURO_PERIOD_CRISIS_EXCLUDED'

    # Get unique indicators
    indicators = sorted(data['INDICATOR'].unique())

    # Calculate statistics
    stats = calculate_temporal_statistics(data, COUNTRY_NAME, indicators, period_col)

    # Perform F-tests
    test_results = perform_temporal_volatility_tests(data, COUNTRY_NAME, indicators, period_col)

    # Display statistical tests
    st.subheader("📊 F-Test Results: Variance Comparison")
    display_f_test_results(test_results, stats)

    # Create temporal boxplots
    st.subheader("📦 Volatility by Indicator Type")
    display_indicator_boxplots(data, indicators, period_col)

def display_f_test_results(test_results, stats):
    """Display F-test results table"""
    results_data = []

    for indicator, test in test_results.items():
        indicator_stats = stats.get(indicator, {})

        # Determine significance
        p_value = test['p_value']
        if pd.notna(p_value):
            if p_value < 0.01:
                significance = '***'
            elif p_value < 0.05:
                significance = '**'
            elif p_value < 0.10:
                significance = '*'
            else:
                significance = ''
        else:
            significance = ''

        results_data.append({
            'Indicator': get_indicator_nickname(indicator),
            'Pre-Euro Std': indicator_stats.get('Pre-Euro', {}).get('std', np.nan),
            'Post-Euro Std': indicator_stats.get('Post-Euro', {}).get('std', np.nan),
            'F-Statistic': test['f_statistic'],
            'P-Value': p_value,
            'Significance': significance,
            'Higher Volatility': 'Pre-Euro' if test.get('f_statistic', 0) > 1 else 'Post-Euro'
        })

    results_df = pd.DataFrame(results_data)

    # Format for display
    formatted_df = results_df.copy()
    formatted_df['Pre-Euro Std'] = formatted_df['Pre-Euro Std'].round(2)
    formatted_df['Post-Euro Std'] = formatted_df['Post-Euro Std'].round(2)
    formatted_df['F-Statistic'] = formatted_df['F-Statistic'].round(3)
    formatted_df['P-Value'] = formatted_df['P-Value'].round(4)

    st.dataframe(formatted_df, use_container_width=True)

    # Summary
    significant_count = len([r for r in results_data if r['Significance'] != ''])
    pre_higher_count = len([r for r in results_data if r['Higher Volatility'] == 'Pre-Euro'])

    st.info(f"""
    **Summary:**
    - {significant_count}/{len(results_data)} indicators show significant volatility differences
    - {pre_higher_count}/{len(results_data)} indicators had higher volatility pre-Euro
    - Significance levels: *** p<0.01, ** p<0.05, * p<0.10
    """)

def display_indicator_boxplots(data, indicators, period_col):
    """Display indicator-level boxplots"""
    # Create boxplot data
    boxplot_df = create_temporal_boxplot_data(data, COUNTRY_NAME, indicators, period_col)

    if not boxplot_df.empty:
        # Create figure with subplots for different indicator types
        investment_indicators = [ind for ind in indicators if 'Investment' in ind]
        other_indicators = [ind for ind in indicators if 'Investment' not in ind]

        # Plot investment indicators
        if investment_indicators:
            st.markdown("**Investment Flows**")
            fig1, ax1 = plt.subplots(figsize=(12, 6))

            investment_data = boxplot_df[boxplot_df['Indicator'].isin(
                [get_indicator_nickname(ind) for ind in investment_indicators]
            )]

            if not investment_data.empty:
                sns.boxplot(data=investment_data, x='Indicator', y='Value', hue='Period',
                           palette={'Pre-Euro': COLORBLIND_SAFE[0], 'Post-Euro': COLORBLIND_SAFE[1]},
                           ax=ax1)
                ax1.set_xlabel('')
                ax1.set_ylabel('% of GDP (annualized)')
                ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
                ax1.legend(title='Period')
                plt.tight_layout()
                st.pyplot(fig1)

        # Plot other indicators
        if other_indicators:
            st.markdown("**Other Capital Flows**")
            fig2, ax2 = plt.subplots(figsize=(12, 6))

            other_data = boxplot_df[boxplot_df['Indicator'].isin(
                [get_indicator_nickname(ind) for ind in other_indicators]
            )]

            if not other_data.empty:
                sns.boxplot(data=other_data, x='Indicator', y='Value', hue='Period',
                           palette={'Pre-Euro': COLORBLIND_SAFE[0], 'Post-Euro': COLORBLIND_SAFE[1]},
                           ax=ax2)
                ax2.set_xlabel('')
                ax2.set_ylabel('% of GDP (annualized)')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
                ax2.legend(title='Period')
                plt.tight_layout()
                st.pyplot(fig2)

def main(data_type="full", output_mode="interactive", context="standalone"):
    """Lithuania Euro Adoption Analysis - Main Function"""

    # Get configurations
    data_config = get_data_configuration(data_type)
    ui_config = configure_ui_elements(output_mode)

    # Page configuration (only if standalone)
    if context == "standalone":
        st.set_page_config(
            page_title=f"CS2: Lithuania Euro Adoption Analysis",
            page_icon="🇱🇹",
            layout="wide"
        )

    # Title and header
    title_suffix = " (Outlier-Adjusted)" if data_type == "winsorized" else ""
    st.title(f"{COUNTRY_FLAG} Case Study 2: Lithuania Euro Adoption ({EURO_ADOPTION_YEAR}) Analysis{title_suffix}")
    st.subheader(f"Capital Flow Volatility Before and After Euro Adoption")

    # Methodology section
    if ui_config['use_expanders']:
        with st.expander("📋 Data and Methodology", expanded=False):
            display_methodology()
    else:
        st.markdown("### 📋 Data and Methodology")
        display_methodology()

    # PDF export tip
    if output_mode == "interactive":
        st.info("💡 **Tip:** You can print this page to PDF using your browser's print function for a professional document.")

    # Add PDF-specific CSS if needed
    if output_mode == "pdf":
        add_pdf_styling()

    # Full Time Period Analysis
    st.markdown("---")
    st.header("📊 Full Time Period Analysis")
    st.markdown("*Complete temporal analysis using all available data*")

    # Overall Capital Flows
    st.subheader("📈 Overall Capital Flows Analysis")
    show_lithuania_overall_analysis(data_config, ui_config, include_crisis_years=True)

    # Indicator-Level Analysis
    st.subheader("🔍 Indicator-Level Analysis")
    show_lithuania_indicator_analysis(data_config, ui_config, include_crisis_years=True)

    # Crisis-Excluded Analysis
    st.markdown("---")
    st.header("🚫 Excluding Financial Crises")
    crisis_text = get_country_specific_crisis_text(COUNTRY_NAME)
    st.markdown(f"*Analysis excluding {crisis_text}*")

    # Overall Capital Flows - Crisis Excluded
    st.subheader("📈 Overall Capital Flows Analysis")
    show_lithuania_overall_analysis(data_config, ui_config, include_crisis_years=False)

    # Indicator-Level Analysis - Crisis Excluded
    st.subheader("🔍 Indicator-Level Analysis")
    show_lithuania_indicator_analysis(data_config, ui_config, include_crisis_years=False)

def display_methodology():
    """Display methodology information"""
    st.markdown(f"""
    ### Data Sources
    - **Balance of Payments Data:** IMF, quarterly frequency (1999-2025)
    - **GDP Data:** IMF World Economic Outlook, annual frequency
    - **Country:** Lithuania, Republic of

    ### Methodology
    1. **Data Normalization:** All BOP flows converted to annualized % of GDP
    2. **Statistical Analysis:** Comprehensive descriptive statistics and F-tests
    3. **Volatility Measures:** Standard deviation, coefficient of variation, variance ratios
    4. **Temporal Comparison:** Pre-Euro vs Post-Euro period analysis

    ### Euro Adoption Timeline
    - **Euro Adoption Date:** January 1, {EURO_ADOPTION_YEAR}
    - **Pre-Euro Period:** 1999-2014 (full series)
    - **Post-Euro Period:** {EURO_ADOPTION_YEAR}-2025 (full series)
    - **Crisis Exclusion:** Global Financial Crisis (2008-2010) and COVID-19 (2020-2022)

    ### Lithuania-Specific Considerations
    - **Latest Euro Adopter:** Lithuania was the last of the three Baltic countries to adopt the Euro
    - **Longest Pre-Euro Period:** More extensive pre-Euro data available for analysis
    - **Post-Crisis Adoption:** Avoided major financial crises during Euro transition
    """)

def add_pdf_styling():
    """Add CSS styling optimized for PDF export"""
    st.markdown("""
    <style>
        @media print {
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
            }
            .stApp {
                margin: 40px;
            }
            .plot-container {
                text-align: center;
                margin: 20px 0;
            }
        }
        .stApp {
            max-width: none;
        }
        .block-container {
            padding: 2rem 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()