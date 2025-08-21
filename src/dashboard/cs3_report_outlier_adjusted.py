"""
Outlier-Adjusted Capital Flows Analysis - Case Study 3: Iceland vs Small Open Economies Report

This Streamlit application provides an exact mirror of Case Study 1 structure using winsorized data,
optimized for clean PDF export with professional formatting.

Research Focus: Iceland vs Small Open Economies - Capital Flow Volatility Comparison (1999-2025)
Using 5% symmetric winsorization for robust statistical analysis
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

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

# Configure matplotlib for PDF export optimization (matching simple_report_app.py)
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

# Colorblind-friendly econometrics palette (matching simple_report_app.py)
COLORBLIND_SAFE = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#FBAFE4']
sns.set_palette(COLORBLIND_SAFE)

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

def load_case_study_3_outlier_adjusted_data():
    """Load Case Study 3 data: Iceland vs Small Open Economies"""
    try:
        # Load the comprehensive labeled dataset
        data_dir = Path(__file__).parent.parent.parent / "updated_data" / "Clean"
        file_path = data_dir / "comprehensive_df_PGDP_labeled_winsorized.csv"
        
        if not file_path.exists():
            st.error(f"❌ Data file not found: {file_path}")
            return None, None
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Filter for Case Study 3 countries only
        cs3_data = df[df['CS3_GROUP'].notna()].copy()
        
        if len(cs3_data) == 0:
            st.error("❌ No Case Study 3 data found in dataset")
            return None, None
        
        # Get indicator columns (ending with _PGDP)
        indicator_columns = [col for col in cs3_data.columns if col.endswith('_PGDP')]
        
        # Standard analysis indicators (matching CS1)
        analysis_indicators = [
            'Assets - Direct investment, Total financial assets/liabilities_PGDP',
            'Assets - Other investment, Debt instruments_PGDP', 
            'Assets - Other investment, Debt instruments, Deposit taking corporations, except the Central Bank_PGDP',
            'Assets - Portfolio investment, Debt securities_PGDP',
            'Assets - Portfolio investment, Equity and investment fund shares_PGDP',
            'Assets - Portfolio investment, Total financial assets/liabilities_PGDP',
            'Liabilities - Direct investment, Total financial assets/liabilities_PGDP',
            'Liabilities - Other investment, Debt instruments, Deposit taking corporations, except the Central Bank_PGDP',
            'Liabilities - Portfolio investment, Debt securities_PGDP', 
            'Liabilities - Portfolio investment, Equity and investment fund shares_PGDP',
            'Liabilities - Portfolio investment, Total financial assets/liabilities_PGDP',
            'Net - Direct investment, Total financial assets/liabilities_PGDP',
            'Net - Portfolio investment, Total financial assets/liabilities_PGDP',
            'Net - Other investment, Total financial assets/liabilities_PGDP'
        ]
        
        # Filter to available indicators
        available_indicators = [ind for ind in analysis_indicators if ind in cs3_data.columns]
        
        if len(available_indicators) == 0:
            st.error("❌ No analysis indicators found in CS3 data")
            return None, None
        
        st.success(f"✅ Loaded CS3 data: {len(cs3_data)} observations, {len(available_indicators)} indicators")
        
        return cs3_data, available_indicators
        
    except Exception as e:
        st.error(f"❌ Error loading CS3 data: {str(e)}")
        return None, None

def main():
    """CS3 report app - exact mirror of CS1 structure, optimized for PDF export"""
    
    # Note: st.set_page_config() is now handled by main_app.py (matching simple_report_app.py)
    # Removing page config call to prevent margin/layout conflicts that affect PDF export
    
    st.title("🛡️ 🇮🇸 Comparative Analysis: Iceland and Small Open Economies (Outlier-Adjusted)")
    st.subheader("Capital Flow Volatility Patterns (1999-2025)")
    
    st.markdown("""
    **Research Focus:** How do capital flow volatility patterns compare between Iceland and other small open economies with similar characteristics?
    
    **Methodology:** Cross-sectional comparison of capital flow patterns between Iceland and comparable small open economies from 1999-2025.
    
    **📋 Outlier-Adjusted Analysis:** This analysis uses 5% symmetric winsorization to assess
    the robustness of statistical findings to extreme values. Data points below the 5th percentile
    are replaced with the 5th percentile value; data points above the 95th percentile are replaced
    with the 95th percentile value.
    
    **Key Hypothesis:** Iceland and other small open economies may exhibit different capital flow volatility patterns despite similar economic structures and currency regimes.
    """)
    
    # Data and Methodology section (matching simple_report_app.py format) 
    with st.expander("📋 Data and Methodology", expanded=False):
        st.markdown("""
        ### Data Sources
        - **Balance of Payments Data:** IMF, quarterly frequency (1999-2025)
        - **GDP Data:** IMF World Economic Outlook, annual frequency
        - **Countries:** Iceland vs Small Open Economies (as defined in CS3_GROUP)
        
        ### Methodology
        1. **Data Normalization:** All BOP flows converted to annualized % of GDP
        2. **Statistical Analysis:** F-tests for variance equality without directional assumptions
        3. **Volatility Measures:** Standard deviation, coefficient of variation, comparative patterns
        4. **Comparative Framework:** Neutral analysis of volatility differences between similar economies
        
        ### Time Period Coverage
        - **Full Analysis Period:** 1999-2025 (all available data)
        - **Crisis Exclusion Analysis:** Excludes Global Financial Crisis (2008-2010) and COVID-19 (2020-2022)
        """)
    
    # Add PDF export tip (matching simple_report_app.py)
    st.info("💡 **Tip:** You can print this page to PDF using your browser's print function for a professional document with proper margins.")
    
    # Add PDF-specific CSS styling (matching simple_report_app.py margins)
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
        /* General margin improvements for PDF export */
        .stApp { 
            max-width: none;
        }
        .block-container { 
            padding: 2rem 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Load data
    final_data, analysis_indicators = load_case_study_3_outlier_adjusted_data()
    
    if final_data is None or analysis_indicators is None:
        st.error("❌ Failed to load Case Study 3 data. Please check data availability.")
        return
    
    # Full Time Period Section
    st.markdown("---")
    st.header("📊 Full Time Period Analysis")
    st.markdown("*Complete temporal analysis using all available data*")
    
    # Call CS3 main analysis function (to be implemented)
    case_study_3_outlier_adjusted_main(context="standalone")
    
    # Crisis-Excluded Section
    st.markdown("---")
    st.header("🚫 Excluding Financial Crises")
    st.markdown("*Analysis excluding Global Financial Crisis (2008-2010) and COVID-19 (2020-2022) periods*")
    
    # Call CS3 crisis-excluded analysis function (to be implemented)
    case_study_3_outlier_adjusted_main_crisis_excluded(context="standalone")

def case_study_3_outlier_adjusted_main(context="standalone"):
    """CS3 main analysis function - exact replica of CS1 structure"""
    
    # Load CS3 data
    from cs3_complete_functions import (load_cs3_data, calculate_group_statistics, create_boxplot_data, 
                                        perform_volatility_tests, create_plot_base64, sort_indicators_by_type, 
                                        get_investment_type_order, create_individual_country_boxplot_data,
                                        load_overall_capital_flows_data_cs3)
    
    # Load full time period data
    with st.spinner("Loading and processing CS3 data..."):
        final_data, analysis_indicators, metadata = load_cs3_data(include_crisis_years=True)
    
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
    
    # Overall Capital Flows Analysis - Complete Implementation
    st.subheader("📈 Overall Capital Flows Analysis")
    st.markdown("*High-level summary of aggregate net capital flows before detailed disaggregated analysis*")
    
    # Load overall capital flows data
    overall_data, indicators_mapping = load_overall_capital_flows_data_cs3(include_crisis_years=True)
    
    if overall_data is not None and indicators_mapping is not None:
        # Use consistent COLORBLIND_SAFE palette
        colors = {'Iceland': COLORBLIND_SAFE[1], 'Small Open Economies': COLORBLIND_SAFE[0]}
        
        # Summary statistics
        st.subheader("📊 Summary Statistics by Group")
        
        summary_stats = []
        for clean_name, col_name in indicators_mapping.items():
            if col_name in overall_data.columns:
                for group in ['Iceland', 'Small Open Economies']:
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
        
        # Create custom HTML table for overall capital flows summary
        st.markdown("""
        <style>
        .overall-summary-table {
            width: 100% !important;
            border-collapse: collapse !important;
            table-layout: fixed !important;
            font-size: 9px !important;
            font-family: Arial, sans-serif !important;
        }
        .overall-summary-table th, .overall-summary-table td {
            border: 1px solid #ddd !important;
            padding: 3px !important;
            text-align: center !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
        .overall-summary-table th {
            background-color: #f0f0f0 !important;
            font-weight: bold !important;
            font-size: 10px !important;
        }
        .overall-summary-table th:first-child, .overall-summary-table td:first-child {
            width: 180px !important;
            max-width: 180px !important;
            text-align: left !important;
            font-weight: bold !important;
        }
        .overall-summary-table th:not(:first-child), .overall-summary-table td:not(:first-child) {
            width: 80px !important;
            max-width: 80px !important;
        }
        .overall-summary-table tr:nth-child(even) {
            background-color: #f9f9f9 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Convert pivot table to HTML
        pivot_df = pivot_summary.reset_index()
        html_table = '<table class="overall-summary-table">'
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
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Side-by-side boxplots
        st.subheader("📦 Distribution Comparison by Group")
        
        fig_overall, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (clean_name, col_name) in enumerate(indicators_mapping.items()):
            if col_name in overall_data.columns and i < 4:
                ax = axes[i]
                
                # Prepare data for boxplot
                iceland_data = overall_data[overall_data['GROUP'] == 'Iceland'][col_name].dropna()
                soe_data = overall_data[overall_data['GROUP'] == 'Small Open Economies'][col_name].dropna()
                
                # Create boxplot
                bp = ax.boxplot([iceland_data, soe_data], 
                               labels=['Iceland', 'Small Open Economies'], 
                               patch_artist=True)
                
                # Color the boxes
                bp['boxes'][0].set_facecolor(colors['Iceland'])
                bp['boxes'][1].set_facecolor(colors['Small Open Economies'])
                for box in bp['boxes']:
                    box.set_alpha(0.7)
                
                ax.set_title(clean_name, fontweight='bold', fontsize=10)
                ax.set_ylabel('% of GDP (annualized)', fontsize=9)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
                ax.tick_params(labelsize=8)
        
        fig_overall.tight_layout()
        st.pyplot(fig_overall)
        
        # Time series plots
        st.subheader("📈 Time Series by Group")
        
        # Create date column
        overall_data_ts = overall_data.copy()
        overall_data_ts['DATE'] = pd.to_datetime(
            overall_data_ts['YEAR'].astype(str) + '-Q' + overall_data_ts['QUARTER'].astype(str)
        )
        
        fig_ts, axes_ts = plt.subplots(2, 2, figsize=(15, 10))
        axes_ts = axes_ts.flatten()
        
        for i, (clean_name, col_name) in enumerate(indicators_mapping.items()):
            if col_name in overall_data.columns and i < 4:
                ax = axes_ts[i]
                
                # Plot Iceland data
                iceland_data = overall_data_ts[overall_data_ts['GROUP'] == 'Iceland'].sort_values('DATE')
                if len(iceland_data) > 0:
                    ax.plot(iceland_data['DATE'], iceland_data[col_name], 
                           color=colors['Iceland'], label='Iceland', linewidth=2, alpha=0.8)
                
                # Plot Small Open Economies average
                soe_data = overall_data_ts[overall_data_ts['GROUP'] == 'Small Open Economies']
                if len(soe_data) > 0:
                    soe_avg = soe_data.groupby('DATE')[col_name].mean().reset_index()
                    ax.plot(soe_avg['DATE'], soe_avg[col_name], 
                           color=colors['Small Open Economies'], label='SOE Average', linewidth=2, alpha=0.8)
                
                ax.set_title(clean_name, fontweight='bold', fontsize=10)
                ax.set_ylabel('% of GDP (annualized)', fontsize=9)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
                ax.legend(loc='upper right', fontsize=8)
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.3)
        
        fig_ts.tight_layout()
        st.pyplot(fig_ts)
        
        # Key insights
        st.subheader("🔍 Key Insights")
        
        # Calculate volatility comparison
        volatility_comparison = []
        for clean_name, col_name in indicators_mapping.items():
            if col_name in overall_data.columns:
                iceland_std = overall_data[overall_data['GROUP'] == 'Iceland'][col_name].std()
                soe_std = overall_data[overall_data['GROUP'] == 'Small Open Economies'][col_name].std()
                volatility_comparison.append({
                    'Indicator': clean_name,
                    'Iceland Volatility': iceland_std,
                    'SOE Volatility': soe_std,
                    'Volatility Ratio': iceland_std / soe_std if soe_std != 0 else float('inf')
                })
        
        vol_df = pd.DataFrame(volatility_comparison)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Volatility Comparison (Standard Deviation)**")
            for _, row in vol_df.iterrows():
                ratio = row['Volatility Ratio']
                if ratio > 1.5:
                    st.write(f"• **{row['Indicator']}**: Iceland {ratio:.1f}x more volatile")
                elif ratio < 0.67:
                    st.write(f"• **{row['Indicator']}**: SOE {1/ratio:.1f}x more volatile")
                else:
                    st.write(f"• **{row['Indicator']}**: Similar volatility levels")
        
        with col2:
            st.markdown("**Overall Pattern**")
            high_vol_count = sum(1 for _, row in vol_df.iterrows() if row['Volatility Ratio'] > 1.5)
            total_indicators = len(vol_df)
            
            if high_vol_count >= total_indicators * 0.75:
                st.write("🔴 **Iceland shows consistently higher volatility** across most capital flow categories")
            elif high_vol_count >= total_indicators * 0.5:
                st.write("🟡 **Mixed volatility patterns** between Iceland and Small Open Economies")
            else:
                st.write("🟢 **Similar volatility levels** between Iceland and Small Open Economies")
    
    st.markdown("---")
    
    # Indicator-level Analysis
    st.subheader("🔍 Indicator-level Analysis")
    st.markdown("*Detailed analysis by individual capital flow indicators*")
    
    # Calculate all statistics
    group_stats = calculate_group_statistics(final_data, 'GROUP', analysis_indicators)
    boxplot_data = create_boxplot_data(final_data, analysis_indicators)
    individual_country_data = create_individual_country_boxplot_data(final_data, analysis_indicators)
    test_results = perform_volatility_tests(final_data, analysis_indicators)
    
    # 1. Summary Statistics and Boxplots
    st.header("1. Summary Statistics and Boxplots")
    
    # Create side-by-side boxplots for compact layout (matching CS1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Boxplot for Means
    mean_data = boxplot_data[boxplot_data['Statistic'] == 'Mean']
    mean_iceland = mean_data[mean_data['GROUP'] == 'Iceland']['Value']
    mean_soe = mean_data[mean_data['GROUP'] == 'Small Open Economies']['Value']
    
    bp1 = ax1.boxplot([mean_iceland, mean_soe], labels=['Iceland', 'Small Open Economies'], patch_artist=True)
    bp1['boxes'][0].set_facecolor(COLORBLIND_SAFE[1])  # Iceland in orange
    bp1['boxes'][1].set_facecolor(COLORBLIND_SAFE[0])  # SOE in blue
    
    ax1.set_title('Panel A: Distribution of Means - Iceland vs SOE', 
                  fontweight='bold', fontsize=10, pad=15)
    ax1.set_ylabel('Mean (% of GDP, annualized)', fontsize=9)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax1.text(0.02, 0.98, f'Iceland Avg: {mean_iceland.mean():.2f}%\nSmall Open Economies Avg: {mean_soe.mean():.2f}%', 
            transform=ax1.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Boxplot for Standard Deviations
    std_data = boxplot_data[boxplot_data['Statistic'] == 'Standard Deviation']
    std_iceland = std_data[std_data['GROUP'] == 'Iceland']['Value']
    std_soe = std_data[std_data['GROUP'] == 'Small Open Economies']['Value']
    
    bp2 = ax2.boxplot([std_iceland, std_soe], labels=['Iceland', 'Small Open Economies'], patch_artist=True)
    bp2['boxes'][0].set_facecolor(COLORBLIND_SAFE[1])  # Iceland in orange
    bp2['boxes'][1].set_facecolor(COLORBLIND_SAFE[0])  # SOE in blue
    
    ax2.set_title('Panel B: Distribution of Volatility - Iceland vs SOE', 
                  fontweight='bold', fontsize=10, pad=15)
    ax2.set_ylabel('Std Dev. (% of GDP, annualized)', fontsize=9)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    volatility_ratio = std_iceland.mean() / std_soe.mean() if std_soe.mean() != 0 else float('inf')
    ax2.text(0.02, 0.98, f'Iceland Avg: {std_iceland.mean():.2f}%\nSmall Open Economies Avg: {std_soe.mean():.2f}%\nRatio: {volatility_ratio:.2f}x', 
            transform=ax2.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Download buttons in columns for compact layout (matching CS1)
    col1, col2 = st.columns(2)
    
    with col1:
        # Download combined figure
        buf_full = io.BytesIO()
        fig.savefig(buf_full, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf_full.seek(0)
        
        st.download_button(
            label="📥 Download Combined Boxplots (PNG)",
            data=buf_full.getvalue(),
            file_name=f"cs3_boxplots_combined_full.png",
            mime="image/png",
            key=f"download_combined_cs3_full_{context}"
        )
    
    with col2:
        # Download individual std dev plot
        fig2_ind, ax2_ind = plt.subplots(1, 1, figsize=(6, 4))
        bp2_ind = ax2_ind.boxplot([std_iceland, std_soe], labels=['Iceland', 'Small Open Economies'], patch_artist=True)
        bp2_ind['boxes'][0].set_facecolor(COLORBLIND_SAFE[1])
        bp2_ind['boxes'][1].set_facecolor(COLORBLIND_SAFE[0])
        ax2_ind.set_title('Panel B: Distribution of Std Deviations - All Indicators', 
                     fontweight='bold', fontsize=10, pad=12)
        ax2_ind.set_ylabel('Std Dev. (% of GDP, annualized)', fontsize=9)
        ax2_ind.tick_params(axis='both', which='major', labelsize=8)
        ax2_ind.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax2_ind.text(0.02, 0.98, f'Iceland Avg: {std_iceland.mean():.2f}%\nSmall Open Economies Avg: {std_soe.mean():.2f}%\nRatio: {volatility_ratio:.2f}x', 
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
            file_name=f"cs3_stddev_boxplot_full.png",
            mime="image/png",
            key=f"download_stddev_cs3_full_{context}"
        )
    
    st.info(f"**Volatility Comparison:** Iceland volatility is {volatility_ratio:.2f}x higher than Small Open Economies on average")
    
    # 1b. Individual Country Comparisons
    st.subheader("1b. Individual Country Comparisons: Iceland vs Each Small Open Economy")
    
    st.markdown("""
    **Enhanced Analysis:** Rather than comparing Iceland to Small Open Economies as an aggregate group, 
    this section compares Iceland's values to each individual small open economy separately.
    """)
    
    # Prepare data for individual country boxplots
    mean_data_individual = individual_country_data[individual_country_data['Statistic'] == 'Mean']
    std_data_individual = individual_country_data[individual_country_data['Statistic'] == 'Standard Deviation']
    
    # Calculate median values for ordering
    mean_medians = mean_data_individual.groupby('COUNTRY')['Value'].median().sort_values(ascending=False)
    std_medians = std_data_individual.groupby('COUNTRY')['Value'].median().sort_values(ascending=False)
    
    # Create side-by-side boxplots for Section 1b
    fig_1b, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 5))
    
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
            box.set_facecolor(COLORBLIND_SAFE[0])  # Blue for Small Open Economies
            box.set_alpha(0.6)
    
    ax3.set_title('Panel C: Distribution of Means - Iceland vs Individual SOEs', 
                  fontweight='bold', fontsize=10, pad=15)
    ax3.set_ylabel('Mean (% of GDP, annualized)', fontsize=9)
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add reference line for Iceland's median
    iceland_median_mean = mean_medians.get('Iceland', 0)
    ax3.axhline(y=iceland_median_mean, color=COLORBLIND_SAFE[3], linestyle='--', alpha=0.7, linewidth=1.5, 
                label=f'Iceland Median: {iceland_median_mean:.2f}%')
    ax3.legend(loc='upper right', fontsize=8)
    
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
            box.set_facecolor(COLORBLIND_SAFE[0])  # Blue for Small Open Economies
            box.set_alpha(0.6)
    
    ax4.set_title('Panel D: Distribution of Volatility - Iceland vs Individual SOEs', 
                  fontweight='bold', fontsize=10, pad=15)
    ax4.set_ylabel('Std Dev (% of GDP, annualized)', fontsize=9)
    ax4.tick_params(axis='x', rotation=45, labelsize=8)
    ax4.tick_params(axis='y', labelsize=8)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add reference line for Iceland's median
    iceland_median_std = std_medians.get('Iceland', 0)
    ax4.axhline(y=iceland_median_std, color=COLORBLIND_SAFE[3], linestyle='--', alpha=0.7, linewidth=1.5, 
                label=f'Iceland Median: {iceland_median_std:.2f}%')
    ax4.legend(loc='upper right', fontsize=8)
    
    fig_1b.tight_layout()
    st.pyplot(fig_1b)
    
    # Download button for Section 1b
    buf_1b = io.BytesIO()
    fig_1b.savefig(buf_1b, format='png', dpi=300, facecolor='white')
    buf_1b.seek(0)
    
    st.download_button(
        label="📥 Download Individual Country Comparisons (PNG)",
        data=buf_1b.getvalue(),
        file_name=f"cs3_individual_country_comparisons_full.png",
        mime="image/png",
        key=f"download_1b_cs3_full_{context}"
    )
    
    # Disaggregated Analysis (Sections 2-6) - Complete implementation
    st.markdown("---")
    st.header("🔍 Disaggregated Analysis (Sections 2-6)")
    st.markdown("*Detailed analysis by individual capital flow indicators*")
    
    # 2. Comprehensive Statistical Summary Table
    st.header("2. Comprehensive Statistical Summary Table")
    
    st.markdown("**All Indicators - Iceland vs Small Open Economies Statistics**")
    
    # Create a clean table with one row per indicator (both groups side-by-side)
    sorted_indicators = sort_indicators_by_type(analysis_indicators)
    table_data = []
    for indicator in sorted_indicators:
        clean_name = indicator.replace('_PGDP', '')
        nickname = get_nickname(clean_name)
        indicator_stats = group_stats[group_stats['Indicator'] == clean_name]
        
        # Get stats for both groups (adapted for CS3)
        iceland_stats = indicator_stats[indicator_stats['Group'] == 'Iceland'].iloc[0] if len(indicator_stats[indicator_stats['Group'] == 'Iceland']) > 0 else None
        soe_stats = indicator_stats[indicator_stats['Group'] == 'Small Open Economies'].iloc[0] if len(indicator_stats[indicator_stats['Group'] == 'Small Open Economies']) > 0 else None
        
        if iceland_stats is not None and soe_stats is not None:
            table_data.append({
                'Indicator': nickname,
                'Iceland Mean': f"{iceland_stats['Mean']:.2f}",
                'Iceland Std Dev': f"{iceland_stats['Std_Dev']:.2f}",
                'Iceland CV%': f"{iceland_stats['CV_Percent']:.1f}",
                'SOE Mean': f"{soe_stats['Mean']:.2f}",
                'SOE Std Dev': f"{soe_stats['Std_Dev']:.2f}",
                'SOE CV%': f"{soe_stats['CV_Percent']:.1f}",
                'CV Ratio (Ice/SOE)': f"{iceland_stats['CV_Percent']/soe_stats['CV_Percent']:.2f}" if soe_stats['CV_Percent'] != 0 else "∞"
            })
    
    # Create DataFrame for processing
    summary_df = pd.DataFrame(table_data)
    
    # Create custom HTML table with strict column width control
    st.markdown("""
    <style>
    .section2-table {
        width: 100% !important;
        border-collapse: collapse !important;
        table-layout: fixed !important;
        font-size: 8px !important;
        font-family: Arial, sans-serif !important;
    }
    .section2-table th, .section2-table td {
        border: 1px solid #ddd !important;
        padding: 2px !important;
        text-align: center !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    .section2-table th {
        background-color: #f0f0f0 !important;
        font-weight: bold !important;
        font-size: 9px !important;
    }
    .section2-table th:first-child, .section2-table td:first-child {
        width: 220px !important;
        max-width: 220px !important;
        text-align: left !important;
        font-weight: bold !important;
    }
    .section2-table th:not(:first-child), .section2-table td:not(:first-child) {
        width: 70px !important;
        max-width: 70px !important;
    }
    .section2-table tr:nth-child(even) {
        background-color: #f9f9f9 !important;
    }
    @media print {
        .section2-table {
            font-size: 7px !important;
        }
        .section2-table th:first-child, .section2-table td:first-child {
            width: 180px !important;
            max-width: 180px !important;
        }
        .section2-table th:not(:first-child), .section2-table td:not(:first-child) {
            width: 60px !important;
            max-width: 60px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Generate HTML table content
    html_table = '<table class="section2-table">'
    html_table += '<thead><tr>'
    for col in summary_df.columns:
        html_table += f'<th>{col}</th>'
    html_table += '</tr></thead><tbody>'
    
    for _, row in summary_df.iterrows():
        html_table += '<tr>'
        for col in summary_df.columns:
            html_table += f'<td>{row[col]}</td>'
        html_table += '</tr>'
    
    html_table += '</tbody></table>'
    
    # Display the custom HTML table
    st.markdown(html_table, unsafe_allow_html=True)
    
    st.info(f"**Summary:** Statistics for all {len(analysis_indicators)} capital flow indicators. CV% = Coefficient of Variation (Std Dev / |Mean| × 100). Higher CV% indicates greater volatility relative to mean.")
    
    # 3. Hypothesis Testing Results
    st.header("3. Hypothesis Testing Results")
    
    st.markdown("**F-Tests for Variance Equality Between Iceland and Small Open Economies** | H₀: Equal volatility patterns | H₁: Different volatility patterns | α = 0.05")
    
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
    results_display['More Volatile'] = results_display['Iceland_Higher_Volatility'].map({True: 'Iceland', False: 'Small Open Economies'})
    
    # Create formatted table
    test_table_data = []
    for _, row in results_display.iterrows():
        test_table_data.append({
            'Indicator': row['Indicator_Nick'],
            'F-Statistic': f"{row['F_Statistic']:.2f}",
            'P-Value': f"{row['P_Value']:.4f}",
            'Significance': row['Significant'],
            'More Volatile': row['More Volatile']
        })
    
    test_df = pd.DataFrame(test_table_data)
    
    # Display as static table with better formatting
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Custom HTML table for hypothesis testing
        st.markdown("""
        <style>
        .hypothesis-test-table {
            width: 100% !important;
            border-collapse: collapse !important;
            table-layout: fixed !important;
            font-size: 8px !important;
            font-family: Arial, sans-serif !important;
        }
        .hypothesis-test-table th, .hypothesis-test-table td {
            border: 1px solid #ddd !important;
            padding: 2px !important;
            text-align: center !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
        .hypothesis-test-table th {
            background-color: #e6f3ff !important;
            font-weight: bold !important;
            font-size: 9px !important;
        }
        .hypothesis-test-table th:first-child, .hypothesis-test-table td:first-child {
            width: 220px !important;
            max-width: 220px !important;
            text-align: left !important;
            font-weight: bold !important;
        }
        .hypothesis-test-table th:not(:first-child), .hypothesis-test-table td:not(:first-child) {
            width: 70px !important;
            max-width: 70px !important;
        }
        .hypothesis-test-table tr:nth-child(even) {
            background-color: #f9f9f9 !important;
        }
        @media print {
            .hypothesis-test-table {
                font-size: 7px !important;
            }
            .hypothesis-test-table th:first-child, .hypothesis-test-table td:first-child {
                width: 180px !important;
                max-width: 180px !important;
            }
            .hypothesis-test-table th:not(:first-child), .hypothesis-test-table td:not(:first-child) {
                width: 60px !important;
                max-width: 60px !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Generate HTML table content for hypothesis tests
        html_table = '<table class="hypothesis-test-table">'
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
        st.markdown("- **P-Value**: Statistical significance")
        st.markdown("- **More Volatile**: Which group exhibits greater volatility")
    
    # Test summary
    total_indicators = len(test_results)
    iceland_higher_count = test_results['Iceland_Higher_Volatility'].sum()
    sig_5pct_count = test_results['Significant_5pct'].sum()
    sig_1pct_count = test_results['Significant_1pct'].sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Iceland More Volatile", f"{iceland_higher_count}/{total_indicators}", f"{iceland_higher_count/total_indicators*100:.1f}%")
    with col2:
        st.metric("Significant (5%)", f"{sig_5pct_count}/{total_indicators}", f"{sig_5pct_count/total_indicators*100:.1f}%")
    with col3:
        st.metric("Significant (1%)", f"{sig_1pct_count}/{total_indicators}", f"{sig_1pct_count/total_indicators*100:.1f}%")
    
    if iceland_higher_count/total_indicators > 0.7:
        conclusion = "Iceland shows significantly higher volatility than"
    elif iceland_higher_count/total_indicators > 0.5:
        conclusion = "Iceland shows moderately different volatility patterns from"
    elif iceland_higher_count/total_indicators > 0.3:
        conclusion = "Iceland and Small Open Economies show similar volatility with some differences in"
    else:
        conclusion = "Small Open Economies show higher volatility than Iceland in"
    
    st.success(f"**Conclusion:** {conclusion} other small open economies across capital flow indicators.")
    
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
    
    # Create grid layout for time series - 2x2 grid per set
    n_indicators = len(selected_indicators)
    
    # Process indicators in groups of 4 for 2x2 grids
    for group_idx in range(0, n_indicators, 4):
        group_indicators = selected_indicators[group_idx:min(group_idx+4, n_indicators)]
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
            
            clean_name = indicator.replace('_PGDP', '')
            nickname = get_nickname(clean_name)
            
            # Plot Iceland
            iceland_data = final_data_copy[final_data_copy['GROUP'] == 'Iceland']
            ax.plot(iceland_data['Date'], iceland_data[indicator], 
                    color=COLORBLIND_SAFE[1], linewidth=1.5, label='Iceland')
            
            # Plot Small Open Economies average
            soe_avg = final_data_copy[final_data_copy['GROUP'] == 'Small Open Economies'].groupby('Date')[indicator].mean()
            ax.plot(soe_avg.index, soe_avg.values, 
                    color=COLORBLIND_SAFE[0], linewidth=1.5, label='Small Open Economies Average')
            
            # Formatting
            f_stat = test_results[test_results['Indicator'] == clean_name]['F_Statistic'].iloc[0]
            panel_letter = chr(65 + i)  # A, B, C, etc.
            ax.set_title(f'{panel_letter}: {nickname}\n(F-stat: {f_stat:.2f})', 
                        fontweight='bold', fontsize=9, pad=12)
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
        fig_group.savefig(buf_group, format='png', dpi=300, facecolor='white')
        buf_group.seek(0)
        
        group_letter = chr(65 + group_idx // 4)  # A, B, C for each group
        st.download_button(
            label=f"📥 Download Time Series Group {group_letter} (PNG)",
            data=buf_group.getvalue(),
            file_name=f"cs3_timeseries_group_{group_letter}.png",
            mime="image/png",
            key=f"download_ts_group_{group_idx}_cs3_full_{context}"
        )
    
    st.markdown("---")
    
    # 5. Key Findings Summary
    st.header("5. Key Findings Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### Statistical Evidence:
        - **{iceland_higher_count/total_indicators*100:.1f}% of capital flow indicators** show higher volatility in Iceland
        - **{sig_5pct_count/total_indicators*100:.1f}% of indicators** show statistically significant differences (p<0.05)
        - **Volatility ratio (Iceland/SOE):** {volatility_ratio:.2f}x
        - **Pattern analysis:** Capital flow volatility patterns {'are relatively similar' if volatility_ratio < 1.5 else 'show notable differences'} between the two groups
        """)
    
    with col2:
        st.markdown(f"""
        ### Statistical Context:
        - **Sample characteristics**: Both groups classified as small open economies
        - **Data quality**: Full quarterly time series coverage
        - **Methodological approach**: F-tests for variance equality comparison
        - **Result interpretation**: Statistical differences observed in volatility patterns across groups
        """)
    
    # Download section
    st.markdown("---")
    st.header("6. Download Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Download comprehensive table
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary Table (CSV)",
            data=csv,
            file_name=f"cs3_summary_statistics_full.csv",
            mime="text/csv",
            key=f"download_summary_csv_cs3_full_{context}"
        )
    
    with col2:
        # Download test results
        csv = test_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Test Results (CSV)",
            data=csv,
            file_name=f"cs3_hypothesis_tests_full.csv",
            mime="text/csv",
            key=f"download_tests_csv_cs3_full_{context}"
        )
    
    with col3:
        # Download group statistics
        csv = group_stats.to_csv(index=False)
        st.download_button(
            label="📥 Download Group Statistics (CSV)",
            data=csv,
            file_name=f"cs3_group_statistics_full.csv",
            mime="text/csv",
            key=f"download_group_csv_cs3_full_{context}"
        )
    
    with col4:
        # Generate HTML report placeholder
        st.info("📝 HTML Report\nGeneration available\nin main dashboard")

def case_study_3_outlier_adjusted_main_crisis_excluded(context="standalone"):
    """CS3 crisis-excluded analysis function - exact replica of CS1 structure"""
    
    # Load CS3 data with crisis exclusion
    from cs3_complete_functions import (load_cs3_data, calculate_group_statistics, create_boxplot_data, 
                                        perform_volatility_tests, create_plot_base64, sort_indicators_by_type, 
                                        get_investment_type_order, create_individual_country_boxplot_data,
                                        load_overall_capital_flows_data_cs3)
    
    # Load crisis-excluded data
    with st.spinner("Loading and processing CS3 crisis-excluded data..."):
        final_data, analysis_indicators, metadata = load_cs3_data(include_crisis_years=False)
    
    if final_data is None:
        st.stop()
    
    # Data overview
    st.success(f"✅ Crisis-excluded data loaded successfully! ({metadata['study_version']})")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Observations", f"{metadata['final_shape'][0]:,}")
    with col2:
        st.metric("Indicators", metadata['n_indicators'])
    with col3:
        st.metric("Countries", final_data['COUNTRY'].nunique())
    with col4:
        excluded_years = metadata.get('crisis_years', [])
        st.metric("Excluded Years", ', '.join(map(str, excluded_years)))
    
    st.markdown("---")
    
    # Overall Capital Flows Analysis - Crisis-Excluded
    st.subheader("📈 Overall Capital Flows Analysis (Crisis-Excluded)")
    st.markdown("*High-level summary of aggregate net capital flows excluding crisis periods*")
    
    # Load overall capital flows data
    overall_data_crisis, indicators_mapping_crisis = load_overall_capital_flows_data_cs3(include_crisis_years=False)
    
    if overall_data_crisis is not None and indicators_mapping_crisis is not None:
        # Use consistent COLORBLIND_SAFE palette
        colors = {'Iceland': COLORBLIND_SAFE[1], 'Small Open Economies': COLORBLIND_SAFE[0]}
        
        # Summary statistics
        st.subheader("📊 Summary Statistics by Group (Crisis-Excluded)")
        
        summary_stats_crisis = []
        for clean_name, col_name in indicators_mapping_crisis.items():
            if col_name in overall_data_crisis.columns:
                for group in ['Iceland', 'Small Open Economies']:
                    group_data = overall_data_crisis[overall_data_crisis['GROUP'] == group][col_name].dropna()
                    summary_stats_crisis.append({
                        'Indicator': clean_name,
                        'Group': group,
                        'Mean': group_data.mean(),
                        'Std Dev': group_data.std(),
                        'Median': group_data.median(),
                        'Count': len(group_data)
                    })
        
        summary_df_crisis = pd.DataFrame(summary_stats_crisis)
        
        # Display summary table
        pivot_summary_crisis = summary_df_crisis.pivot_table(
            index='Indicator', 
            columns='Group', 
            values=['Mean', 'Std Dev', 'Median'],
            aggfunc='first'
        ).round(2)
        
        # Convert pivot table to HTML for crisis-excluded data
        pivot_df_crisis = pivot_summary_crisis.reset_index()
        html_table_crisis = '<table class="overall-summary-table">'
        html_table_crisis += '<thead><tr>'
        for col in pivot_df_crisis.columns:
            col_name = str(col).replace('(', '').replace(')', '').replace("'", '') if isinstance(col, tuple) else str(col)
            html_table_crisis += f'<th>{col_name}</th>'
        html_table_crisis += '</tr></thead><tbody>'
        
        for _, row in pivot_df_crisis.iterrows():
            html_table_crisis += '<tr>'
            for col in pivot_df_crisis.columns:
                value = row[col]
                if pd.isna(value):
                    value = '-'
                elif isinstance(value, (int, float)):
                    value = f'{value:.2f}'
                html_table_crisis += f'<td>{value}</td>'
            html_table_crisis += '</tr>'
        
        html_table_crisis += '</tbody></table>'
        st.markdown(html_table_crisis, unsafe_allow_html=True)
        
        # Distribution Comparison by Group (Crisis-Excluded)
        st.subheader("📊 Distribution Comparison by Group (Crisis-Excluded)")
        
        # Create distribution plots for crisis-excluded data
        fig_dist, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        plot_count = 0
        for clean_name, col_name in indicators_mapping_crisis.items():
            if col_name in overall_data_crisis.columns and plot_count < 4:
                ax = axes[plot_count]
                
                # Prepare data for boxplot (EXACT CS1 implementation)
                iceland_data = overall_data_crisis[overall_data_crisis['GROUP'] == 'Iceland'][col_name].dropna()
                soe_data = overall_data_crisis[overall_data_crisis['GROUP'] == 'Small Open Economies'][col_name].dropna()
                
                # Create boxplot (EXACT CS1 implementation)
                bp = ax.boxplot([iceland_data, soe_data], 
                               labels=['Iceland', 'Small Open Economies'], 
                               patch_artist=True)
                
                # Color the boxes (EXACT CS1 styling)
                bp['boxes'][0].set_facecolor(COLORBLIND_SAFE[1])  # Iceland
                bp['boxes'][1].set_facecolor(COLORBLIND_SAFE[0])  # Small Open Economies
                
                ax.set_title(f'{clean_name}', fontweight='bold', fontsize=10, pad=12)
                ax.set_ylabel('% of GDP (annualized)', fontsize=9)
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
                ax.grid(True, alpha=0.3)
                
                plot_count += 1
        
        # Hide unused subplots
        for i in range(plot_count, 4):
            axes[i].set_visible(False)
        
        fig_dist.suptitle('Overall Capital Flows Distribution by Group (Crisis-Excluded)', 
                         fontweight='bold', fontsize=12)
        fig_dist.tight_layout()
        st.pyplot(fig_dist)
        
        # Time Series by Group (Crisis-Excluded)
        st.subheader("📈 Time Series by Group (Crisis-Excluded)")
        
        # Create date column (EXACT CS1 implementation)
        overall_data_ts = overall_data_crisis.copy()
        overall_data_ts['DATE'] = pd.to_datetime(overall_data_ts['YEAR'].astype(str) + '-Q' + overall_data_ts['QUARTER'].astype(str))
        
        fig_ts, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = {'Iceland': COLORBLIND_SAFE[1], 'Small Open Economies': COLORBLIND_SAFE[0]}
        
        plot_count = 0
        for clean_name, col_name in indicators_mapping_crisis.items():
            if col_name in overall_data_crisis.columns and plot_count < 4:
                ax = axes[plot_count]
                
                # Add shaded regions for excluded crisis periods (EXACT CS1)
                ax.axvspan(pd.Timestamp('2008-01-01'), pd.Timestamp('2010-12-31'), 
                          alpha=0.15, color='red', label='GFC (2008-2010)')
                ax.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2022-12-31'), 
                          alpha=0.15, color='orange', label='COVID-19 (2020-2022)')
                
                # Plot Iceland data with line breaks at data gaps (EXACT CS1)
                iceland_data = overall_data_ts[overall_data_ts['GROUP'] == 'Iceland'].sort_values('DATE')
                if len(iceland_data) > 0:
                    # Detect gaps in the time series and split into segments
                    iceland_segments = []
                    current_segment = []
                    
                    dates = iceland_data['DATE'].tolist()
                    for i, (_, row) in enumerate(iceland_data.iterrows()):
                        current_segment.append(row)
                        
                        # Check if there's a gap to the next date (more than 1 year)
                        if i < len(dates) - 1:
                            current_date = dates[i]
                            next_date = dates[i + 1]
                            time_gap = (next_date - current_date).days
                            
                            # If gap is more than 400 days, end current segment
                            if time_gap > 400:
                                iceland_segments.append(pd.DataFrame(current_segment))
                                current_segment = []
                    
                    # Add the final segment
                    if current_segment:
                        iceland_segments.append(pd.DataFrame(current_segment))
                    
                    # Plot each segment separately to create line breaks
                    for j, segment in enumerate(iceland_segments):
                        if len(segment) > 0:
                            label = 'Iceland' if j == 0 else None
                            ax.plot(segment['DATE'], segment[col_name], 
                                   color=colors['Iceland'], label=label, linewidth=2, alpha=0.8)
                
                # Plot Small Open Economies with line breaks at data gaps (EXACT CS1)
                soe_data = overall_data_ts[overall_data_ts['GROUP'] == 'Small Open Economies']
                if len(soe_data) > 0:
                    soe_avg = soe_data.groupby('DATE')[col_name].mean().reset_index()
                    
                    # Detect gaps and split into segments
                    soe_segments = []
                    current_segment = []
                    
                    dates = soe_avg['DATE'].tolist()
                    for i, (_, row) in enumerate(soe_avg.iterrows()):
                        current_segment.append(row)
                        
                        if i < len(dates) - 1:
                            current_date = dates[i]
                            next_date = dates[i + 1]
                            time_gap = (next_date - current_date).days
                            
                            if time_gap > 400:
                                soe_segments.append(pd.DataFrame(current_segment))
                                current_segment = []
                    
                    if current_segment:
                        soe_segments.append(pd.DataFrame(current_segment))
                    
                    # Plot each segment separately
                    for j, segment in enumerate(soe_segments):
                        if len(segment) > 0:
                            label = 'Small Open Economies' if j == 0 else None
                            ax.plot(segment['DATE'], segment[col_name], 
                                   color=colors['Small Open Economies'], label=label, linewidth=2, alpha=0.8)
                
                ax.set_title(f'{clean_name}', fontweight='bold', fontsize=10, pad=12)
                ax.set_xlabel('Time', fontsize=9)
                ax.set_ylabel('% of GDP (annualized)', fontsize=9)
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                
                plot_count += 1
        
        # Hide unused subplots
        for i in range(plot_count, 4):
            axes[i].set_visible(False)
        
        fig_ts.suptitle('Overall Capital Flows Time Series by Group (Crisis-Excluded)', 
                       fontweight='bold', fontsize=12)
        fig_ts.tight_layout()
        st.pyplot(fig_ts)
        
        # Key insights for crisis-excluded
        st.subheader("🔍 Key Insights (Crisis-Excluded)")
        
        # Calculate volatility comparison
        volatility_comparison_crisis = []
        for clean_name, col_name in indicators_mapping_crisis.items():
            if col_name in overall_data_crisis.columns:
                iceland_std = overall_data_crisis[overall_data_crisis['GROUP'] == 'Iceland'][col_name].std()
                soe_std = overall_data_crisis[overall_data_crisis['GROUP'] == 'Small Open Economies'][col_name].std()
                volatility_comparison_crisis.append({
                    'Indicator': clean_name,
                    'Iceland Volatility': iceland_std,
                    'SOE Volatility': soe_std,
                    'Volatility Ratio': iceland_std / soe_std if soe_std != 0 else float('inf')
                })
        
        vol_df_crisis = pd.DataFrame(volatility_comparison_crisis)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Volatility Comparison (Crisis-Excluded)**")
            for _, row in vol_df_crisis.iterrows():
                ratio = row['Volatility Ratio']
                if ratio > 1.5:
                    st.write(f"• **{row['Indicator']}**: Iceland {ratio:.1f}x more volatile")
                elif ratio < 0.67:
                    st.write(f"• **{row['Indicator']}**: SOE {1/ratio:.1f}x more volatile")
                else:
                    st.write(f"• **{row['Indicator']}**: Similar volatility levels")
        
        with col2:
            st.markdown("**Overall Pattern (Crisis-Excluded)**")
            high_vol_count = sum(1 for _, row in vol_df_crisis.iterrows() if row['Volatility Ratio'] > 1.5)
            total_indicators = len(vol_df_crisis)
            
            if high_vol_count >= total_indicators * 0.75:
                st.write("🔴 **Iceland shows consistently higher volatility** even excluding crisis periods")
            elif high_vol_count >= total_indicators * 0.5:
                st.write("🟡 **Mixed volatility patterns** between Iceland and Small Open Economies")
            else:
                st.write("🟢 **Similar volatility levels** when crisis periods are excluded")
    
    st.markdown("---")
    
    # Indicator-level Analysis (Crisis-Excluded)
    st.subheader("🔍 Indicator-level Analysis (Crisis-Excluded)")
    st.markdown("*Detailed analysis by individual capital flow indicators excluding crisis periods*")
    
    # Calculate all statistics
    group_stats = calculate_group_statistics(final_data, 'GROUP', analysis_indicators)
    boxplot_data = create_boxplot_data(final_data, analysis_indicators)
    individual_country_data = create_individual_country_boxplot_data(final_data, analysis_indicators)
    test_results = perform_volatility_tests(final_data, analysis_indicators)
    
    # 1. Summary Statistics and Boxplots (Crisis-Excluded)
    st.header("1. Summary Statistics and Boxplots (Crisis-Excluded)")
    
    # Create side-by-side boxplots for compact layout (matching CS1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Boxplot for Means
    mean_data = boxplot_data[boxplot_data['Statistic'] == 'Mean']
    mean_iceland = mean_data[mean_data['GROUP'] == 'Iceland']['Value']
    mean_soe = mean_data[mean_data['GROUP'] == 'Small Open Economies']['Value']
    
    bp1 = ax1.boxplot([mean_iceland, mean_soe], labels=['Iceland', 'Small Open Economies'], patch_artist=True)
    bp1['boxes'][0].set_facecolor(COLORBLIND_SAFE[1])  # Iceland in orange
    bp1['boxes'][1].set_facecolor(COLORBLIND_SAFE[0])  # SOE in blue
    
    ax1.set_title('Panel A: Distribution of Means (Crisis-Excluded)', 
                  fontweight='bold', fontsize=10, pad=12)
    ax1.set_ylabel('Mean (% of GDP, annualized)', fontsize=9)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax1.text(0.02, 0.98, f'Iceland Avg: {mean_iceland.mean():.2f}%\nSmall Open Economies Avg: {mean_soe.mean():.2f}%', 
            transform=ax1.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Boxplot for Standard Deviations
    std_data = boxplot_data[boxplot_data['Statistic'] == 'Standard Deviation']
    std_iceland = std_data[std_data['GROUP'] == 'Iceland']['Value']
    std_soe = std_data[std_data['GROUP'] == 'Small Open Economies']['Value']
    
    bp2 = ax2.boxplot([std_iceland, std_soe], labels=['Iceland', 'Small Open Economies'], patch_artist=True)
    bp2['boxes'][0].set_facecolor(COLORBLIND_SAFE[1])  # Iceland in orange
    bp2['boxes'][1].set_facecolor(COLORBLIND_SAFE[0])  # SOE in blue
    
    ax2.set_title('Panel B: Distribution of Volatility (Crisis-Excluded)', 
                  fontweight='bold', fontsize=10, pad=12)
    ax2.set_ylabel('Std Dev. (% of GDP, annualized)', fontsize=9)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    volatility_ratio = std_iceland.mean() / std_soe.mean() if std_soe.mean() != 0 else float('inf')
    ax2.text(0.02, 0.98, f'Iceland Avg: {std_iceland.mean():.2f}%\nSmall Open Economies Avg: {std_soe.mean():.2f}%\nRatio: {volatility_ratio:.2f}x', 
            transform=ax2.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Download buttons in columns for compact layout (matching CS1)
    col1, col2 = st.columns(2)
    
    with col1:
        # Download combined figure
        buf_full = io.BytesIO()
        fig.savefig(buf_full, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf_full.seek(0)
        
        st.download_button(
            label="📥 Download Combined Boxplots (PNG)",
            data=buf_full.getvalue(),
            file_name=f"cs3_boxplots_combined_crisis_excluded.png",
            mime="image/png",
            key=f"download_combined_cs3_crisis_{context}"
        )
    
    with col2:
        # Download individual std dev plot
        fig2_ind, ax2_ind = plt.subplots(1, 1, figsize=(6, 4))
        bp2_ind = ax2_ind.boxplot([std_iceland, std_soe], labels=['Iceland', 'Small Open Economies'], patch_artist=True)
        bp2_ind['boxes'][0].set_facecolor(COLORBLIND_SAFE[1])
        bp2_ind['boxes'][1].set_facecolor(COLORBLIND_SAFE[0])
        ax2_ind.set_title('Panel B: Distribution of Std Deviations - All Indicators (Crisis-Excluded)', 
                     fontweight='bold', fontsize=10, pad=12)
        ax2_ind.set_ylabel('Std Dev. (% of GDP, annualized)', fontsize=9)
        ax2_ind.tick_params(axis='both', which='major', labelsize=8)
        ax2_ind.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax2_ind.text(0.02, 0.98, f'Iceland Avg: {std_iceland.mean():.2f}%\nSmall Open Economies Avg: {std_soe.mean():.2f}%\nRatio: {volatility_ratio:.2f}x', 
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
            file_name=f"cs3_stddev_boxplot_crisis_excluded.png",
            mime="image/png",
            key=f"download_stddev_cs3_crisis_{context}"
        )
    
    st.info(f"**Volatility Comparison:** Iceland volatility is {volatility_ratio:.2f}x higher than Small Open Economies on average (excluding crisis periods)")
    
    # 1b. Individual Country Comparisons (Crisis-Excluded)
    st.subheader("1b. Individual Country Comparisons: Iceland vs Each Small Open Economy (Crisis-Excluded)")
    
    st.markdown("""
    **Enhanced Analysis:** Comparing Iceland to each individual small open economy separately,
    excluding crisis periods to focus on normal market conditions.
    """)
    
    # Prepare data for individual country boxplots (crisis-excluded)
    mean_data_individual = individual_country_data[individual_country_data['Statistic'] == 'Mean']
    std_data_individual = individual_country_data[individual_country_data['Statistic'] == 'Standard Deviation']
    
    # Calculate median values for ordering
    mean_medians = mean_data_individual.groupby('COUNTRY')['Value'].median().sort_values(ascending=False)
    std_medians = std_data_individual.groupby('COUNTRY')['Value'].median().sort_values(ascending=False)
    
    # Create side-by-side boxplots for Section 1b (crisis-excluded)
    fig_1b_crisis, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 5))
    
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
            box.set_facecolor(COLORBLIND_SAFE[0])  # Blue for Small Open Economies
            box.set_alpha(0.6)
    
    ax3.set_title('Panel C: Means - Iceland vs Individual SOEs (Crisis-Excluded)', 
                  fontweight='bold', fontsize=10, pad=15)
    ax3.set_ylabel('Mean (% of GDP, annualized)', fontsize=9)
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add reference line for Iceland's median
    iceland_median_mean = mean_medians.get('Iceland', 0)
    ax3.axhline(y=iceland_median_mean, color=COLORBLIND_SAFE[3], linestyle='--', alpha=0.7, linewidth=1.5, 
                label=f'Iceland Median: {iceland_median_mean:.2f}%')
    ax3.legend(loc='upper right', fontsize=8)
    
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
            box.set_facecolor(COLORBLIND_SAFE[0])  # Blue for Small Open Economies
            box.set_alpha(0.6)
    
    ax4.set_title('Panel D: Volatility - Iceland vs Individual SOEs (Crisis-Excluded)', 
                  fontweight='bold', fontsize=10, pad=15)
    ax4.set_ylabel('Std Dev (% of GDP, annualized)', fontsize=9)
    ax4.tick_params(axis='x', rotation=45, labelsize=8)
    ax4.tick_params(axis='y', labelsize=8)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add reference line for Iceland's median
    iceland_median_std = std_medians.get('Iceland', 0)
    ax4.axhline(y=iceland_median_std, color=COLORBLIND_SAFE[3], linestyle='--', alpha=0.7, linewidth=1.5, 
                label=f'Iceland Median: {iceland_median_std:.2f}%')
    ax4.legend(loc='upper right', fontsize=8)
    
    fig_1b_crisis.tight_layout()
    st.pyplot(fig_1b_crisis)
    
    # Download button for Section 1b (crisis-excluded)
    buf_1b_crisis = io.BytesIO()
    fig_1b_crisis.savefig(buf_1b_crisis, format='png', dpi=300, facecolor='white')
    buf_1b_crisis.seek(0)
    
    st.download_button(
        label="📥 Download Individual Country Comparisons (Crisis-Excluded) (PNG)",
        data=buf_1b_crisis.getvalue(),
        file_name=f"cs3_individual_country_comparisons_crisis_excluded.png",
        mime="image/png",
        key=f"download_1b_cs3_crisis_{context}"
    )
    
    # 2. Comprehensive Statistical Summary Table (Crisis-Excluded)
    st.header("2. Comprehensive Statistical Summary Table (Crisis-Excluded)")
    
    st.markdown("**All Indicators - Iceland vs Small Open Economies Statistics (Crisis-Excluded)**")
    
    # Create a clean table with one row per indicator (both groups side-by-side)
    sorted_indicators = sort_indicators_by_type(analysis_indicators)
    table_data = []
    for indicator in sorted_indicators:
        clean_name = indicator.replace('_PGDP', '')
        nickname = get_nickname(clean_name)
        indicator_stats = group_stats[group_stats['Indicator'] == clean_name]
        
        # Get stats for both groups (adapted for CS3)
        iceland_stats = indicator_stats[indicator_stats['Group'] == 'Iceland'].iloc[0] if len(indicator_stats[indicator_stats['Group'] == 'Iceland']) > 0 else None
        soe_stats = indicator_stats[indicator_stats['Group'] == 'Small Open Economies'].iloc[0] if len(indicator_stats[indicator_stats['Group'] == 'Small Open Economies']) > 0 else None
        
        if iceland_stats is not None and soe_stats is not None:
            table_data.append({
                'Indicator': nickname,
                'Iceland Mean': f"{iceland_stats['Mean']:.2f}",
                'Iceland Std Dev': f"{iceland_stats['Std_Dev']:.2f}",
                'Iceland CV%': f"{iceland_stats['CV_Percent']:.1f}",
                'SOE Mean': f"{soe_stats['Mean']:.2f}",
                'SOE Std Dev': f"{soe_stats['Std_Dev']:.2f}",
                'SOE CV%': f"{soe_stats['CV_Percent']:.1f}",
                'CV Ratio (Ice/SOE)': f"{iceland_stats['CV_Percent']/soe_stats['CV_Percent']:.2f}" if soe_stats['CV_Percent'] != 0 else "∞"
            })
    
    # Create DataFrame for processing
    summary_df = pd.DataFrame(table_data)
    
    # Create custom HTML table with strict column width control
    st.markdown("""
    <style>
    .section2-table {
        width: 100% !important;
        border-collapse: collapse !important;
        table-layout: fixed !important;
        font-size: 8px !important;
        font-family: Arial, sans-serif !important;
    }
    .section2-table th, .section2-table td {
        border: 1px solid #ddd !important;
        padding: 2px !important;
        text-align: center !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    .section2-table th {
        background-color: #f0f0f0 !important;
        font-weight: bold !important;
        font-size: 9px !important;
    }
    .section2-table th:first-child,
    .section2-table td:first-child {
        text-align: left !important;
        width: 25% !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create HTML table
    html_table = '<table class="section2-table">'
    html_table += '<thead><tr>'
    for col in summary_df.columns:
        html_table += f'<th>{col}</th>'
    html_table += '</tr></thead><tbody>'
    
    for _, row in summary_df.iterrows():
        html_table += '<tr>'
        for col in summary_df.columns:
            html_table += f'<td>{row[col]}</td>'
        html_table += '</tr>'
    
    html_table += '</tbody></table>'
    st.markdown(html_table, unsafe_allow_html=True)
    
    st.info(f"**Summary (Crisis-Excluded):** Statistics for all {len(analysis_indicators)} capital flow indicators. CV% = Coefficient of Variation (Std Dev / |Mean| × 100). Higher CV% indicates greater volatility relative to mean.")
    
    # 3. Hypothesis Testing Results (Crisis-Excluded)
    st.header("3. Hypothesis Testing Results (Crisis-Excluded)")
    
    st.markdown("**F-Tests for Variance Equality Between Iceland and Small Open Economies (Crisis-Excluded)** | H₀: Equal volatility patterns | H₁: Different volatility patterns | α = 0.05")
    
    # Create a clean static table for hypothesis tests
    display_results = test_results.copy()
    display_results['F_Statistic'] = display_results['F_Statistic'].round(3)
    display_results['P_Value'] = display_results['P_Value'].round(4)
    display_results['Iceland_Std'] = display_results['Iceland_Std'].round(3)
    display_results['SOE_Std'] = display_results['SOE_Std'].round(3)
    
    # Add significance indicators
    display_results['Significance'] = display_results.apply(
        lambda row: '***' if row['Significant_1pct'] else ('**' if row['Significant_5pct'] else ''), axis=1
    )
    
    # Create clean indicator names using nicknames
    display_results['Clean_Indicator'] = display_results['Indicator'].apply(get_nickname)
        
    # Select columns for display
    table_columns = ['Clean_Indicator', 'Iceland_Std', 'SOE_Std', 'F_Statistic', 'P_Value', 'Significance']
    
    # Create custom HTML table for hypothesis test results
    st.markdown("""
    <style>
    .hypothesis-results-table {
        width: 100% !important;
            border-collapse: collapse !important;
            table-layout: fixed !important;
            font-size: 9px !important;
            font-family: Arial, sans-serif !important;
        }
        .hypothesis-results-table th, .hypothesis-results-table td {
            border: 1px solid #ddd !important;
            padding: 3px !important;
            text-align: center !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
        .hypothesis-results-table th {
            background-color: #e6f3ff !important;
            font-weight: bold !important;
            font-size: 10px !important;
        }
        .hypothesis-results-table th:first-child, .hypothesis-results-table td:first-child {
            width: 250px !important;
            max-width: 250px !important;
            text-align: left !important;
            font-weight: bold !important;
        }
        .hypothesis-results-table th:not(:first-child), .hypothesis-results-table td:not(:first-child) {
            width: 80px !important;
            max-width: 80px !important;
        }
        .hypothesis-results-table tr:nth-child(even) {
            background-color: #f9f9f9 !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Generate HTML table content
    html_table = '<table class="hypothesis-results-table">'
    html_table += '<thead><tr>'
    html_table += '<th>Indicator</th><th>Iceland Std Dev</th><th>SOE Std Dev</th>'
    html_table += '<th>F-Statistic</th><th>P-Value</th><th>Sig.</th>'
    html_table += '</tr></thead><tbody>'
    
    for _, row in display_results.iterrows():
        html_table += '<tr>'
        html_table += f'<td>{row["Clean_Indicator"]}</td>'
        html_table += f'<td>{row["Iceland_Std"]:.3f}</td>'
        html_table += f'<td>{row["SOE_Std"]:.3f}</td>'
        html_table += f'<td>{row["F_Statistic"]:.3f}</td>'
        html_table += f'<td>{row["P_Value"]:.4f}</td>'
        html_table += f'<td>{row["Significance"]}</td>'
        html_table += '</tr>'
    
    html_table += '</tbody></table>'
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Download test results
    csv_buffer = io.StringIO()
    test_results.to_csv(csv_buffer, index=False)
    
    st.download_button(
            label="📥 Download Test Results (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"cs3_test_results_crisis_excluded.csv",
            mime="text/csv",
            key=f"download_tests_cs3_crisis_{context}"
        )
    
    st.markdown("---")
    
    # 4. Time Series Analysis (Crisis-Excluded)
    st.header("4. Time Series Analysis (Crisis-Excluded)")
    
    # Create date column for crisis-excluded data
    final_data_copy = final_data.copy()
    final_data_copy['Date'] = pd.to_datetime(
        final_data_copy['YEAR'].astype(str) + 'Q' + final_data_copy['QUARTER'].astype(str)
    )
    
    # Show ALL indicators, sorted properly (EXACT CS1)
    selected_indicators = sort_indicators_by_type(analysis_indicators)
    
    # Create grid layout for time series - 2x2 grid per set (EXACT CS1)
    n_indicators = len(selected_indicators)
    
    # Process indicators in groups of 4 for 2x2 grids (EXACT CS1)
    for group_idx in range(0, n_indicators, 4):
        group_indicators = selected_indicators[group_idx:min(group_idx+4, n_indicators)]
        n_in_group = len(group_indicators)
        
        # Create 2x2 subplot for this group
        fig_ts, axes = plt.subplots(2, 2, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, indicator in enumerate(group_indicators):
            ax = axes[idx]
            
            clean_name = indicator.replace('_PGDP', '')
            nickname = get_nickname(clean_name)
            
            # Add shaded regions for excluded crisis periods FIRST (EXACT CS1)
            ax.axvspan(pd.Timestamp('2008-01-01'), pd.Timestamp('2010-12-31'), 
                      alpha=0.15, color='red', label='GFC (2008-2010)' if idx == 0 else '')
            ax.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2022-12-31'), 
                      alpha=0.15, color='orange', label='COVID-19 (2020-2022)' if idx == 0 else '')
            
            # Plot data with line breaks at gaps (EXACT CS1 implementation)
            colors = {'Iceland': COLORBLIND_SAFE[1], 'Small Open Economies': COLORBLIND_SAFE[0]}
            
            # Plot Iceland data with line breaks
            iceland_data = final_data_copy[final_data_copy['GROUP'] == 'Iceland'].sort_values('Date')
            if len(iceland_data) > 0:
                # Detect gaps and split into segments
                iceland_segments = []
                current_segment = []
                
                dates = iceland_data['Date'].tolist()
                for i, (_, row) in enumerate(iceland_data.iterrows()):
                    current_segment.append(row)
                    
                    if i < len(dates) - 1:
                        current_date = dates[i]
                        next_date = dates[i + 1]
                        time_gap = (next_date - current_date).days
                        
                        if time_gap > 400:  # Gap indicating crisis period
                            iceland_segments.append(pd.DataFrame(current_segment))
                            current_segment = []
                
                if current_segment:
                    iceland_segments.append(pd.DataFrame(current_segment))
                
                # Plot each segment separately
                for j, segment in enumerate(iceland_segments):
                    if len(segment) > 0:
                        label = 'Iceland' if j == 0 else None
                        ax.plot(segment['Date'], segment[indicator], 
                               color=colors['Iceland'], label=label, linewidth=2, alpha=0.8)
            
            # Plot Small Open Economies data with line breaks
            soe_data = final_data_copy[final_data_copy['GROUP'] == 'Small Open Economies']
            if len(soe_data) > 0:
                soe_avg = soe_data.groupby('Date')[indicator].mean().reset_index()
                soe_avg = soe_avg.sort_values('Date')
                
                # Detect gaps and split into segments
                soe_segments = []
                current_segment = []
                
                dates = soe_avg['Date'].tolist()
                for i, (_, row) in enumerate(soe_avg.iterrows()):
                    current_segment.append(row)
                    
                    if i < len(dates) - 1:
                        current_date = dates[i]
                        next_date = dates[i + 1]
                        time_gap = (next_date - current_date).days
                        
                        if time_gap > 400:
                            soe_segments.append(pd.DataFrame(current_segment))
                            current_segment = []
                
                if current_segment:
                    soe_segments.append(pd.DataFrame(current_segment))
                
                # Plot each segment separately
                for j, segment in enumerate(soe_segments):
                    if len(segment) > 0:
                        label = 'Small Open Economies' if j == 0 else None
                        ax.plot(segment['Date'], segment[indicator], 
                               color=colors['Small Open Economies'], label=label, linewidth=2, alpha=0.8)
            
            ax.set_title(f'{nickname}', fontweight='bold', fontsize=9, pad=12)
            ax.set_xlabel('Time', fontsize=8)
            ax.set_ylabel('% of GDP', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:  # Only show legend on first plot
                ax.legend(fontsize=7, loc='upper right')
        
        # Hide unused subplots in this group (EXACT CS1)
        for i in range(n_in_group, 4):
            axes[i].set_visible(False)
        
        fig_ts.suptitle(f'Capital Flow Time Series - Group {group_idx//4 + 1} (Crisis-Excluded)', 
                       fontweight='bold', fontsize=12)
        fig_ts.tight_layout()
        st.pyplot(fig_ts)
    
    st.info("**Note:** Red shaded areas indicate crisis periods (2008-2010 GFC, 2020-2022 COVID-19) that are excluded from the crisis-excluded analysis. Time series shows patterns during normal market conditions.")
    
    st.markdown("---")
    
    # Calculate summary statistics for Section 5 (Crisis-Excluded)
    if test_results is not None and len(test_results) > 0:
        total_indicators = len(test_results)
        iceland_higher_count = test_results['Iceland_Higher_Volatility'].sum()
        sig_5pct_count = test_results['Significant_5pct'].sum()
        sig_1pct_count = test_results['Significant_1pct'].sum()
    else:
        total_indicators = len(analysis_indicators)
        iceland_higher_count = 0
        sig_5pct_count = 0
        sig_1pct_count = 0
    
    # 5. Key Findings Summary (Crisis-Excluded)
    st.header("5. Key Findings Summary (Crisis-Excluded)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### Statistical Evidence (Crisis-Excluded):
        - **{iceland_higher_count/total_indicators*100:.1f}% of capital flow indicators** show higher volatility in Iceland
        - **{sig_5pct_count/total_indicators*100:.1f}% of indicators** show statistically significant differences (p<0.05)
        - **Volatility ratio (Iceland/SOE):** {volatility_ratio:.2f}x (excluding crisis periods)
        - **Pattern analysis:** Non-crisis volatility patterns {'are relatively similar' if volatility_ratio < 1.5 else 'show notable differences'} between groups
        """)
    
    with col2:
        st.markdown(f"""
        ### Methodological Notes:
        - **Crisis periods excluded**: 2008-2010 (GFC) and 2020-2022 (COVID-19)
        - **Comparison basis**: Similar small open economy structures
        - **Statistical approach**: F-test for variance equality without directional assumptions
        - **Interpretation**: Focus on pattern differences rather than superiority/inferiority
        """)
    
    # Download section (Crisis-Excluded)
    st.markdown("---")
    st.header("6. Download Results (Crisis-Excluded)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Download comprehensive table (crisis-excluded)
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary Table (CSV)",
            data=csv,
            file_name=f"cs3_summary_statistics_crisis_excluded.csv",
            mime="text/csv",
            key=f"download_summary_csv_cs3_crisis_{context}"
        )
    
    with col2:
        # Download test results (crisis-excluded)
        csv = test_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Test Results (CSV)",
            data=csv,
            file_name=f"cs3_hypothesis_tests_crisis_excluded.csv",
            mime="text/csv",
            key=f"download_tests_csv_cs3_crisis_{context}"
        )
    
    with col3:
        # Download group statistics (crisis-excluded)
        csv = group_stats.to_csv(index=False)
        st.download_button(
            label="📥 Download Group Statistics (CSV)",
            data=csv,
            file_name=f"cs3_group_statistics_crisis_excluded.csv",
            mime="text/csv",
            key=f"download_group_csv_cs3_crisis_{context}"
        )
    
    with col4:
        # Generate HTML report placeholder
        st.info("📝 HTML Report\nGeneration available\nin main dashboard")


if __name__ == "__main__":
    main()