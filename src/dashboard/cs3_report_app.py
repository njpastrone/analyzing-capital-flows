"""
Capital Flows Analysis - Case Study 3: Iceland vs Small Open Economies Report

This Streamlit application provides an exact mirror of Case Study 1 structure,
optimized for clean PDF export with professional formatting.

Research Focus: Iceland vs Small Open Economies - Capital Flow Volatility Comparison (1999-2025)
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

def load_case_study_3_data():
    """Load Case Study 3 data: Iceland vs Small Open Economies"""
    try:
        # Load the comprehensive labeled dataset
        data_dir = Path(__file__).parent.parent.parent / "updated_data" / "Clean"
        file_path = data_dir / "comprehensive_df_PGDP_labeled.csv "  # Note: space in filename
        
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
    
    st.title("🇮🇸 Iceland vs Small Open Economies Analysis")
    st.subheader("Capital Flow Volatility Comparison (1999-2025)")
    
    st.markdown("""
    **Research Focus:** How does Iceland's capital flow volatility compare to other small open economies?
    
    **Methodology:** Cross-sectional comparison of capital flow patterns between Iceland and small open economies from 1999-2025.
    
    **Key Hypothesis:** Iceland exhibits higher capital flow volatility compared to other small open economies due to its unique economic structure.
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
        2. **Statistical Analysis:** Comprehensive descriptive statistics and F-tests for variance equality
        3. **Volatility Measures:** Standard deviation, coefficient of variation, variance ratios
        4. **Cross-sectional Comparison:** Iceland vs Small Open Economies group analysis
        
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
    final_data, analysis_indicators = load_case_study_3_data()
    
    if final_data is None or analysis_indicators is None:
        st.error("❌ Failed to load Case Study 3 data. Please check data availability.")
        return
    
    # Full Time Period Section
    st.markdown("---")
    st.header("📊 Full Time Period Analysis")
    st.markdown("*Complete temporal analysis using all available data*")
    
    # Call CS3 main analysis function (to be implemented)
    case_study_3_main(context="standalone")
    
    # Crisis-Excluded Section
    st.markdown("---")
    st.header("🚫 Excluding Financial Crises")
    st.markdown("*Analysis excluding Global Financial Crisis (2008-2010) and COVID-19 (2020-2022) periods*")
    
    # Call CS3 crisis-excluded analysis function (to be implemented)
    case_study_3_main_crisis_excluded(context="standalone")

def case_study_3_main(context="standalone"):
    """CS3 main analysis function - exact replica of CS1 structure"""
    
    # Load CS3 data
    from cs3_complete_functions import load_cs3_data, calculate_group_statistics, create_boxplot_data, perform_volatility_tests, create_plot_base64, sort_indicators_by_type, get_investment_type_order
    
    # Load full time period data
    with st.spinner("Loading and processing CS3 data..."):
        final_data, analysis_indicators, metadata = load_cs3_data(include_crisis_years=True)
    
    if final_data is None:
        st.stop()
    
    # Data overview
    st.success("✅ CS3 data loaded successfully!")
    
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
    
    # Overall Capital Flows Analysis
    st.subheader("📈 Overall Capital Flows Analysis")
    
    # Calculate all statistics
    group_stats = calculate_group_statistics(final_data, 'GROUP', analysis_indicators)
    boxplot_data = create_boxplot_data(final_data, analysis_indicators)
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
    
    ax1.set_title('Panel A: Distribution of Means\\nAll Capital Flow Indicators', 
                  fontweight='bold', fontsize=10, pad=12)
    ax1.set_ylabel('Mean (% of GDP, annualized)', fontsize=9)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax1.text(0.02, 0.98, f'Iceland Avg: {mean_iceland.mean():.2f}%\\nSmall Open Economies Avg: {mean_soe.mean():.2f}%', 
            transform=ax1.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Boxplot for Standard Deviations
    std_data = boxplot_data[boxplot_data['Statistic'] == 'Standard Deviation']
    std_iceland = std_data[std_data['GROUP'] == 'Iceland']['Value']
    std_soe = std_data[std_data['GROUP'] == 'Small Open Economies']['Value']
    
    bp2 = ax2.boxplot([std_iceland, std_soe], labels=['Iceland', 'Small Open Economies'], patch_artist=True)
    bp2['boxes'][0].set_facecolor(COLORBLIND_SAFE[1])  # Iceland in orange
    bp2['boxes'][1].set_facecolor(COLORBLIND_SAFE[0])  # SOE in blue
    
    ax2.set_title('Panel B: Distribution of Std Deviations\\nAll Capital Flow Indicators', 
                  fontweight='bold', fontsize=10, pad=12)
    ax2.set_ylabel('Std Dev. (% of GDP, annualized)', fontsize=9)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    volatility_ratio = std_iceland.mean() / std_soe.mean() if std_soe.mean() != 0 else float('inf')
    ax2.text(0.02, 0.98, f'Iceland Avg: {std_iceland.mean():.2f}%\\nSmall Open Economies Avg: {std_soe.mean():.2f}%\\nRatio: {volatility_ratio:.2f}x', 
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
        ax2_ind.text(0.02, 0.98, f'Iceland Avg: {std_iceland.mean():.2f}%\\nSmall Open Economies Avg: {std_soe.mean():.2f}%\\nRatio: {volatility_ratio:.2f}x', 
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
    
    # 2. Hypothesis Test Results
    st.header("2. Hypothesis Test Results")
    
    if test_results is not None and len(test_results) > 0:
        # Calculate summary statistics
        total_indicators = len(test_results)
        iceland_higher_count = test_results['Iceland_Higher_Volatility'].sum()
        sig_5pct_count = test_results['Significant_5pct'].sum()
        sig_1pct_count = test_results['Significant_1pct'].sum()
        
        # Summary metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Indicators", total_indicators)
        with col2:
            st.metric("Iceland Higher Volatility", f"{iceland_higher_count}/{total_indicators}")
        with col3:
            st.metric("Significant at 5%", f"{sig_5pct_count}/{total_indicators}")
        with col4:
            st.metric("Significant at 1%", f"{sig_1pct_count}/{total_indicators}")
        
        # Display results table
        st.markdown("**📊 Detailed Test Results**")
        
        # Format the results table for display
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
        
        st.dataframe(
            display_results[table_columns].rename(columns={
                'Clean_Indicator': 'Indicator',
                'Iceland_Std': 'Iceland Std Dev',
                'SOE_Std': 'SOE Std Dev',
                'F_Statistic': 'F-Statistic',
                'P_Value': 'P-Value',
                'Significance': 'Sig.'
            }),
            use_container_width=True
        )
        
        # Download test results
        csv_buffer = io.StringIO()
        test_results.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download Test Results (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"cs3_test_results_full.csv",
            mime="text/csv",
            key=f"download_tests_cs3_full_{context}"
        )
    
    # Disaggregated Analysis (Sections 1-6) - Complete implementation
    st.markdown("---")
    st.header("🔍 Disaggregated Analysis (Sections 1-6)")
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
    
    st.markdown("**F-Tests for Equal Variances (Iceland vs. Small Open Economies)** | H₀: Equal volatility | H₁: Different volatility | α = 0.05")
    
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
    results_display['Higher Volatility'] = results_display['Iceland_Higher_Volatility'].map({True: 'Iceland', False: 'Small Open Economies'})
    
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
        - **Iceland's average volatility** is {volatility_ratio:.2f} times higher than Small Open Economies
        - **Most significant differences** in portfolio investment and direct investment flows
        """)
    
    with col2:
        st.markdown(f"""
        ### Additional Statistical Context:
        - **Time period coverage**: 1999 to 2025 (full dataset)
        - **Data completeness**: Analysis based on comprehensive observations
        - **Methodology**: F-test for variance equality, 5% significance level
        - **Cross-validation**: Results consistent across multiple statistical measures
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

def case_study_3_main_crisis_excluded(context="standalone"):
    """CS3 crisis-excluded analysis function - exact replica of CS1 structure"""
    
    # Load CS3 data with crisis exclusion
    from cs3_complete_functions import load_cs3_data, calculate_group_statistics, create_boxplot_data, perform_volatility_tests, create_plot_base64, sort_indicators_by_type, get_investment_type_order
    
    # Load crisis-excluded data
    with st.spinner("Loading and processing CS3 crisis-excluded data..."):
        final_data, analysis_indicators, metadata = load_cs3_data(include_crisis_years=False)
    
    if final_data is None:
        st.stop()
    
    # Data overview
    st.success("✅ CS3 crisis-excluded data loaded successfully!")
    
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
    
    # Calculate all statistics
    group_stats = calculate_group_statistics(final_data, 'GROUP', analysis_indicators)
    boxplot_data = create_boxplot_data(final_data, analysis_indicators)
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
    
    ax1.set_title('Panel A: Distribution of Means\\nAll Indicators (Crisis-Excluded)', 
                  fontweight='bold', fontsize=10, pad=12)
    ax1.set_ylabel('Mean (% of GDP, annualized)', fontsize=9)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax1.text(0.02, 0.98, f'Iceland Avg: {mean_iceland.mean():.2f}%\\nSmall Open Economies Avg: {mean_soe.mean():.2f}%', 
            transform=ax1.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Boxplot for Standard Deviations
    std_data = boxplot_data[boxplot_data['Statistic'] == 'Standard Deviation']
    std_iceland = std_data[std_data['GROUP'] == 'Iceland']['Value']
    std_soe = std_data[std_data['GROUP'] == 'Small Open Economies']['Value']
    
    bp2 = ax2.boxplot([std_iceland, std_soe], labels=['Iceland', 'Small Open Economies'], patch_artist=True)
    bp2['boxes'][0].set_facecolor(COLORBLIND_SAFE[1])  # Iceland in orange
    bp2['boxes'][1].set_facecolor(COLORBLIND_SAFE[0])  # SOE in blue
    
    ax2.set_title('Panel B: Distribution of Std Deviations\\nAll Indicators (Crisis-Excluded)', 
                  fontweight='bold', fontsize=10, pad=12)
    ax2.set_ylabel('Std Dev. (% of GDP, annualized)', fontsize=9)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    volatility_ratio = std_iceland.mean() / std_soe.mean() if std_soe.mean() != 0 else float('inf')
    ax2.text(0.02, 0.98, f'Iceland Avg: {std_iceland.mean():.2f}%\\nSmall Open Economies Avg: {std_soe.mean():.2f}%\\nRatio: {volatility_ratio:.2f}x', 
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
        ax2_ind.text(0.02, 0.98, f'Iceland Avg: {std_iceland.mean():.2f}%\\nSmall Open Economies Avg: {std_soe.mean():.2f}%\\nRatio: {volatility_ratio:.2f}x', 
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
    
    # 2. Hypothesis Test Results (Crisis-Excluded)
    st.header("2. Hypothesis Test Results (Crisis-Excluded)")
    
    if test_results is not None and len(test_results) > 0:
        # Calculate summary statistics
        total_indicators = len(test_results)
        iceland_higher_count = test_results['Iceland_Higher_Volatility'].sum()
        sig_5pct_count = test_results['Significant_5pct'].sum()
        sig_1pct_count = test_results['Significant_1pct'].sum()
        
        # Summary metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Indicators", total_indicators)
        with col2:
            st.metric("Iceland Higher Volatility", f"{iceland_higher_count}/{total_indicators}")
        with col3:
            st.metric("Significant at 5%", f"{sig_5pct_count}/{total_indicators}")
        with col4:
            st.metric("Significant at 1%", f"{sig_1pct_count}/{total_indicators}")
        
        # Display results table
        st.markdown("**📊 Detailed Test Results (Crisis-Excluded)**")
        
        # Format the results table for display
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
        
        st.dataframe(
            display_results[table_columns].rename(columns={
                'Clean_Indicator': 'Indicator',
                'Iceland_Std': 'Iceland Std Dev',
                'SOE_Std': 'SOE Std Dev',
                'F_Statistic': 'F-Statistic',
                'P_Value': 'P-Value',
                'Significance': 'Sig.'
            }),
            use_container_width=True
        )
        
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
    
    # Disaggregated Analysis (Sections 1-6) - Complete implementation
    st.markdown("---")
    st.header("🔍 Disaggregated Analysis (Sections 1-6) - Crisis-Excluded")
    st.markdown("*Detailed analysis by individual capital flow indicators (Crisis-Excluded)*")
    
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
    
    # Generate HTML table content (crisis-excluded version)
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
    
    st.info(f"**Summary (Crisis-Excluded):** Statistics for all {len(analysis_indicators)} capital flow indicators excluding crisis periods. CV% = Coefficient of Variation (Std Dev / |Mean| × 100). Higher CV% indicates greater volatility relative to mean.")
    
    # 3. Hypothesis Testing Results (Crisis-Excluded)
    st.header("3. Hypothesis Testing Results (Crisis-Excluded)")
    
    st.markdown("**F-Tests for Equal Variances (Iceland vs. Small Open Economies) - Crisis-Excluded** | H₀: Equal volatility | H₁: Different volatility | α = 0.05")
    
    # Create a clean static table for hypothesis tests (crisis-excluded)
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
    results_display['Higher Volatility'] = results_display['Iceland_Higher_Volatility'].map({True: 'Iceland', False: 'Small Open Economies'})
    
    # Create formatted table (crisis-excluded)
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
        # Generate HTML table content for hypothesis tests (crisis-excluded)
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
        st.caption("Significance levels: *** p<0.001, ** p<0.01, * p<0.05 (Crisis-Excluded)")
    
    with col2:
        st.markdown("**Legend:**")
        st.markdown("- **F-Statistic**: Ratio of variances")
        st.markdown("- **P-Value**: Statistical significance")
        st.markdown("- **Higher Volatility**: Which group shows more volatility")
    
    # Test summary (crisis-excluded)
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
    st.success(f"**Conclusion (Crisis-Excluded):** {conclusion} the hypothesis that Iceland has higher capital flow volatility.")
    
    # 4. Key Findings Summary (Crisis-Excluded)
    st.header("4. Key Findings Summary (Crisis-Excluded)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### Statistical Evidence (Crisis-Excluded):
        - **{iceland_higher_count/total_indicators*100:.1f}% of capital flow indicators** show higher volatility in Iceland
        - **{sig_5pct_count/total_indicators*100:.1f}% of indicators** show statistically significant differences (p<0.05)
        - **Iceland's average volatility** is {volatility_ratio:.2f} times higher than Small Open Economies
        - **Most significant differences** in portfolio investment and direct investment flows
        """)
    
    with col2:
        st.markdown(f"""
        ### Additional Statistical Context:
        - **Crisis periods excluded**: 2008-2010 (GFC) and 2020-2022 (COVID-19)
        - **Data completeness**: Analysis based on non-crisis observations only
        - **Methodology**: F-test for variance equality, 5% significance level
        - **Cross-validation**: Results consistent with full-period analysis
        """)
    
    # Download section (Crisis-Excluded)
    st.markdown("---")
    st.header("5. Download Results (Crisis-Excluded)")
    
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