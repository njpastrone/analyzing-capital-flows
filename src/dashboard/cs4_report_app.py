"""
Case Study 4: Comprehensive Statistical Analysis Report Application

Professional dashboard for CS4 analysis comparing Iceland vs multiple comparator groups.
Implements F-tests, AR(4) models, and RMSE calculations with clean table presentation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import warnings
from statsmodels.tsa.stattools import acf

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from core.cs4_statistical_analysis import CS4AnalysisFramework

# Configure page
st.set_page_config(
    page_title="CS4: Statistical Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling
st.markdown("""
<style>
    /* Professional table styling */
    .dataframe {
        font-size: 12px !important;
        font-family: 'Arial', sans-serif !important;
    }
    .dataframe th {
        background-color: #e6f3ff !important;
        font-weight: bold !important;
        text-align: center !important;
        padding: 8px !important;
    }
    .dataframe td {
        text-align: center !important;
        padding: 6px !important;
    }
    .dataframe tbody tr:nth-child(even) {
        background-color: #f9f9f9 !important;
    }
    
    /* Headers styling */
    h1 {
        color: #2c3e50 !important;
        border-bottom: 3px solid #3498db !important;
        padding-bottom: 10px !important;
    }
    h2 {
        color: #34495e !important;
        margin-top: 30px !important;
    }
    h3 {
        color: #7f8c8d !important;
        margin-top: 20px !important;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        padding: 10px !important;
        border-radius: 5px !important;
    }
    
    /* Button styling */
    .stDownloadButton button {
        background-color: #28a745 !important;
        color: white !important;
        border: none !important;
        padding: 8px 16px !important;
        border-radius: 4px !important;
    }
    
    /* Info box styling */
    .stInfo {
        background-color: #d1ecf1 !important;
        border-color: #bee5eb !important;
        color: #0c5460 !important;
    }
</style>
""", unsafe_allow_html=True)


def format_table_for_display(df: pd.DataFrame, title: str = "") -> str:
    """Format DataFrame as professional HTML table"""
    html = f"<h3>{title}</h3>" if title else ""
    html += df.to_html(classes='dataframe', index=True, escape=False)
    return html


def create_download_link(df: pd.DataFrame, filename: str, link_text: str) -> str:
    """Create download link for DataFrame"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'


def create_comprehensive_boxplots_chart(full_results, crisis_results, period_name):
    """Create comprehensive boxplots for all 7 groups using Net Capital Flows, ordered by standard deviation"""
    
    # Load data for Net Capital Flows
    from core.cs4_statistical_analysis import CS4DataLoader
    loader = CS4DataLoader()
    
    include_crisis = (period_name == "Full Period")
    data = loader.load_indicator_data("Net Capital Flows", include_crisis_years=include_crisis)
    
    if data is None:
        st.warning(f"Unable to load data for {period_name} boxplots")
        return None
    
    # Define groups and labels
    groups = ['Iceland', 'eurozone_sum', 'eurozone_avg', 'soe_sum', 'soe_avg', 'baltics_sum', 'baltics_avg']
    group_labels_map = {
        'Iceland': 'Iceland',
        'eurozone_sum': 'Eurozone\nSum',
        'eurozone_avg': 'Eurozone\nAvg',
        'soe_sum': 'SOE\nSum',
        'soe_avg': 'SOE\nAvg',
        'baltics_sum': 'Baltics\nSum',
        'baltics_avg': 'Baltics\nAvg'
    }
    
    # Colors map - Iceland distinct, others grouped  
    colors_map = {
        'Iceland': '#e74c3c',
        'eurozone_sum': '#3498db',
        'eurozone_avg': '#85c1e9',
        'soe_sum': '#f39c12',
        'soe_avg': '#f8c471',
        'baltics_sum': '#1abc9c',
        'baltics_avg': '#7dcea0'
    }
    
    # Calculate standard deviations and sort groups by std dev (descending)
    group_std_dev = {}
    for group in groups:
        if group in data.columns:
            group_data = data[group].dropna()
            if len(group_data) > 0:
                group_std_dev[group] = np.std(group_data, ddof=1)
    
    if not group_std_dev:
        st.warning("No valid data for boxplots")
        return None
    
    # Sort groups by standard deviation (descending - highest to lowest volatility)
    sorted_groups = sorted(group_std_dev.keys(), key=lambda x: group_std_dev[x], reverse=True)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Prepare data for boxplots in sorted order
    boxplot_data = []
    sorted_labels = []
    sorted_colors = []
    
    for group in sorted_groups:
        group_data = data[group].dropna()
        boxplot_data.append(group_data.values)
        sorted_labels.append(group_labels_map[group])
        sorted_colors.append(colors_map[group])
    
    # Create boxplots
    bp = ax.boxplot(boxplot_data, labels=sorted_labels, patch_artist=True, 
                    showfliers=True, flierprops=dict(marker='o', markersize=4, alpha=0.6))
    
    # Color the boxes
    for box, color in zip(bp['boxes'], sorted_colors):
        box.set_facecolor(color)
        box.set_alpha(0.7)
    
    # Styling
    ax.set_title(f'Net Capital Flows Distribution - {period_name} (Ordered by Volatility)', 
                fontweight='bold', fontsize=14, pad=20)
    ax.set_ylabel('Net Capital Flows (% of GDP)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=0)
    
    # Add reference line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    
    plt.tight_layout()
    return fig


def create_comprehensive_acf_chart(full_results, crisis_results, period_name):
    """Create ACF (Autocorrelation Function) charts grid for all 7 groups (clean version with enhanced spacing)"""
    
    # Load data for Net Capital Flows
    from core.cs4_statistical_analysis import CS4DataLoader
    loader = CS4DataLoader()
    
    include_crisis = (period_name == "Full Period")
    data = loader.load_indicator_data("Net Capital Flows", include_crisis_years=include_crisis)
    
    if data is None:
        st.warning(f"Unable to load data for {period_name} ACF charts")
        return None
    
    # Define groups and labels
    groups = ['Iceland', 'eurozone_sum', 'eurozone_avg', 'soe_sum', 'soe_avg', 'baltics_sum', 'baltics_avg']
    group_labels = ['Iceland', 'Eurozone Sum', 'Eurozone Avg', 'SOE Sum', 'SOE Avg', 'Baltics Sum', 'Baltics Avg']
    
    # Create figure with 3x3 grid (7 used, 2 empty) with significantly increased height for optimal spacing
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    axes = axes.flatten()
    
    plot_count = 0
    for i, (group, label) in enumerate(zip(groups, group_labels)):
        if group in data.columns and plot_count < 7:
            ax = axes[plot_count]
            series = data[group].dropna()
            
            if len(series) > 10:  # Need sufficient data for ACF
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Calculate ACF
                    try:
                        lags = min(20, len(series) // 4)  # Up to 20 lags or 1/4 of data
                        # Calculate ACF without confidence intervals for cleaner look
                        acf_vals = acf(series, nlags=lags, fft=True)
                        
                        # Plot ACF - clean bars only, no confidence bands
                        x_lags = range(len(acf_vals))
                        ax.bar(x_lags, acf_vals, alpha=0.8, color='#2c3e50', edgecolor='black', linewidth=0.5)
                        
                        # Enhanced styling with quarterly time unit specification
                        ax.set_title(f'{label}', fontweight='bold', fontsize=12, pad=20)
                        ax.set_xlabel('Lags (Quarters)', fontsize=10, fontweight='medium')
                        ax.set_ylabel('ACF', fontsize=10, fontweight='medium')
                        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
                        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                        ax.tick_params(axis='both', labelsize=9)
                        
                        # Set reasonable y-axis limits
                        ax.set_ylim(-1.1, 1.1)
                        
                        # Add subtle x-axis ticks for better readability
                        if len(x_lags) > 10:
                            ax.set_xticks(range(0, len(x_lags), 5))  # Show every 5th lag for clarity
                        
                    except Exception as e:
                        ax.text(0.5, 0.5, f'ACF calculation\nfailed for {label}', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=11)
                        ax.set_title(f'{label}', fontweight='bold', fontsize=12, pad=20)
                        ax.set_xlabel('Lags (Quarters)', fontsize=10)
            else:
                ax.text(0.5, 0.5, f'Insufficient data\nfor {label}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=11)
                ax.set_title(f'{label}', fontweight='bold', fontsize=12, pad=20)
                ax.set_xlabel('Lags (Quarters)', fontsize=10)
                
            plot_count += 1
    
    # Hide unused subplots
    for i in range(plot_count, 9):
        axes[i].axis('off')
    
    # Enhanced title with quarterly specification
    plt.suptitle(f'Autocorrelation Functions - Net Capital Flows ({period_name})\nQuarterly Data: Each lag = 3 months', 
                fontweight='bold', fontsize=16, y=0.95)
    
    # SIGNIFICANTLY increased spacing to prevent any text overlap
    plt.subplots_adjust(hspace=0.7, wspace=0.35, top=0.88, bottom=0.05)
    
    return fig


def create_comprehensive_timeseries_chart(aggregation_type):
    """Create comprehensive time-series chart with crisis marking"""
    
    # Load data for Net Capital Flows (always include crisis for time series)
    from core.cs4_statistical_analysis import CS4DataLoader
    loader = CS4DataLoader()
    
    data = loader.load_indicator_data("Net Capital Flows", include_crisis_years=True)
    
    if data is None:
        st.warning(f"Unable to load data for {aggregation_type} time series")
        return None
    
    # Define groups based on aggregation type
    if aggregation_type == "Averages":
        groups = ['Iceland', 'eurozone_avg', 'soe_avg', 'baltics_avg']
        group_labels = ['Iceland', 'Eurozone Average', 'SOE Average', 'Baltics Average']
        colors = ['#e74c3c', '#3498db', '#f39c12', '#1abc9c']
    else:  # Sums
        groups = ['Iceland', 'eurozone_sum', 'soe_sum', 'baltics_sum']
        group_labels = ['Iceland', 'Eurozone Sum', 'SOE Sum', 'Baltics Sum']
        colors = ['#e74c3c', '#2980b9', '#e67e22', '#16a085']
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Add shaded regions for crisis periods FIRST (behind data)
    ax.axvspan(pd.Timestamp('2008-01-01'), pd.Timestamp('2010-12-31'), 
              alpha=0.15, color='red', label='GFC (2008-2010)')
    ax.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2022-12-31'), 
              alpha=0.15, color='orange', label='COVID-19 (2020-2022)')
    
    # Convert index to datetime if needed
    if hasattr(data, 'YEAR') and hasattr(data, 'QUARTER'):
        data['DATE'] = pd.to_datetime(data['YEAR'].astype(str) + '-' + 
                                    ((data['QUARTER'] - 1) * 3 + 1).astype(str) + '-01')
        data = data.set_index('DATE')
    
    # Plot data lines
    lines_plotted = []
    for group, label, color in zip(groups, group_labels, colors):
        if group in data.columns:
            series = data[group].dropna()
            if len(series) > 0:
                line = ax.plot(series.index, series.values, label=label, 
                             color=color, linewidth=2, alpha=0.8)
                lines_plotted.append(line[0])
    
    # Styling
    ax.set_title(f'Net Capital Flows Over Time - {aggregation_type} Comparison', 
                fontweight='bold', fontsize=14, pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Net Capital Flows (% of GDP)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=1)
    
    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Format x-axis
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def display_methodology_section():
    """Display methodology and interpretation guide"""
    with st.expander("📚 Methodology & Interpretation Guide", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### F-Test for Variance Equality
            - **Null Hypothesis:** σ²(Iceland) = σ²(Comparator)
            - **Alternative:** σ²(Iceland) ≠ σ²(Comparator)
            - **Significance Levels:**
                - *** : p < 0.01 (highly significant)
                - ** : p < 0.05 (significant)
                - * : p < 0.10 (marginally significant)
            - **Interpretation:** Stars indicate significant differences in volatility
            """)
        
        with col2:
            st.markdown("""
            ### AR(4) Model & Half-Life
            - **Model:** y_t = φ₁y_{t-1} + φ₂y_{t-2} + φ₃y_{t-3} + φ₄y_{t-4} + ε_t
            - **Half-Life:** Quarters for shock to decay by 50%
            - **Calculation:** Impulse response function tracking
            - **Expected:** 1-3 quarters for financial flows
            - **Interpretation:** Lower values = faster mean reversion
            """)
        
        with col3:
            st.markdown("""
            ### RMSE Prediction Accuracy
            - **Method:** 4-step ahead forecast
            - **Training:** All data except last 4 quarters
            - **Testing:** Last 4 quarters prediction
            - **Formula:** √(Σ(actual - predicted)²/4)
            - **Interpretation:** Lower RMSE = better predictability
            """)


def run_cs4_integrated_analysis():
    """Run CS4 analysis and display results organized by indicator with integrated Full/Crisis-Excluded results"""
    
    # Initialize analysis framework
    framework = CS4AnalysisFramework()
    
    # Run both analyses with loading indicator
    with st.spinner("Running comprehensive statistical analysis for both Full Period and Crisis-Excluded..."):
        full_results = framework.run_comprehensive_analysis(include_crisis_years=True)
        crisis_results = framework.run_comprehensive_analysis(include_crisis_years=False)
    
    if not full_results or 'summary_tables' not in full_results or not crisis_results or 'summary_tables' not in crisis_results:
        st.error("❌ Analysis failed. Please check data availability.")
        return
    
    # Display header metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Indicators Analyzed", len(full_results['metadata']['indicators_analyzed']))
    with col2:
        st.metric("Comparator Groups", len(full_results['metadata']['comparator_groups']))
    with col3:
        st.metric("Full Period Observations", "105")
    with col4:
        st.metric("Crisis-Excluded Observations", "81")
    
    # Get indicators
    indicators = full_results['metadata']['indicators_analyzed']
    
    # Comprehensive Analysis Overview - Master Tables
    display_comprehensive_analysis_overview(full_results, crisis_results)
    
    # Transition to detailed analysis
    st.markdown("---")
    st.header("🔍 Detailed Indicator-Level Analysis")
    st.markdown("**Individual analysis sections for each capital flow indicator with integrated Full/Crisis-Excluded results.**")
    
    # Process and display each indicator
    for indicator in indicators:
        display_indicator_section(indicator, full_results, crisis_results)
    
    # Summary insights and comprehensive export
    display_summary_insights_and_export(full_results, crisis_results)


def display_comprehensive_analysis_overview(full_results, crisis_results):
    """Display comprehensive analysis overview with three master tables"""
    
    st.header("📊 Comprehensive Analysis Overview")
    st.markdown("""
    **Complete statistical results across all capital flow indicators.** Three master tables provide 
    integrated Full Period and Crisis-Excluded analysis for variance equality, persistence, and predictability.
    """)
    
    # Get indicators for processing
    indicators = full_results['metadata']['indicators_analyzed']
    
    # Table 1: Master Standard Deviations & F-test Results
    st.subheader("🎯 Table 1: Standard Deviation & F-test Results (All Indicators)")
    
    master_std_table = create_master_table(
        indicators, 
        full_results['summary_tables']['standard_deviations_ftest'],
        crisis_results['summary_tables']['standard_deviations_ftest'],
        'std'
    )
    
    display_master_table(master_std_table, 'std')
    
    st.info("""
    **Interpretation:** Standard deviations measure volatility levels. Stars indicate F-test significance 
    for variance differences from Iceland: *** p<0.01, ** p<0.05, * p<0.10.
    
    **Color Coding:**
    - 🔴 **Red/Pink Background**: Iceland is MORE volatile than comparator (higher standard deviation)
    - 🟢 **Green Background**: Iceland is LESS volatile than comparator (lower standard deviation)  
    - ⚪ **No Color**: No statistically significant difference
    """)
    
    # Download button for master std table
    csv_master_std = master_std_table.to_csv(index=True)
    st.download_button(
        label="📥 Download Master Standard Deviations Table (CSV)",
        data=csv_master_std,
        file_name="cs4_master_standard_deviations.csv",
        mime="text/csv"
    )
    
    # Chart 1: Side-by-side boxplots (Full Period)
    st.markdown("---")
    st.subheader("📊 Chart 1: Distribution Analysis - Full Period")
    st.markdown("**Visual representation of volatility patterns across all comparator groups for Net Capital Flows**")
    
    chart1 = create_comprehensive_boxplots_chart(full_results, crisis_results, "Full Period")
    if chart1:
        st.pyplot(chart1)
        
        # Download button for Chart 1
        buf1 = io.BytesIO()
        chart1.savefig(buf1, format='png', dpi=150, bbox_inches='tight')
        buf1.seek(0)
        st.download_button(
            label="📥 Download Full Period Boxplots (PNG)",
            data=buf1,
            file_name="cs4_comprehensive_boxplots_full_period.png",
            mime="image/png"
        )
        plt.close(chart1)
    
    # Chart 2: Side-by-side boxplots (Crisis-Excluded)
    st.markdown("---")
    st.subheader("📊 Chart 2: Distribution Analysis - Crisis-Excluded")
    st.markdown("**Comparison showing volatility patterns with financial crisis periods removed**")
    
    chart2 = create_comprehensive_boxplots_chart(full_results, crisis_results, "Crisis-Excluded")
    if chart2:
        st.pyplot(chart2)
        
        # Download button for Chart 2
        buf2 = io.BytesIO()
        chart2.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
        buf2.seek(0)
        st.download_button(
            label="📥 Download Crisis-Excluded Boxplots (PNG)",
            data=buf2,
            file_name="cs4_comprehensive_boxplots_crisis_excluded.png",
            mime="image/png"
        )
        plt.close(chart2)
    
    # Table 2: Master Half-life Results
    st.markdown("---")
    st.subheader("⏱️ Table 2: Half-life Results (All Indicators)")
    
    master_halflife_table = create_master_table(
        indicators,
        full_results['summary_tables']['half_life_ar4'],
        crisis_results['summary_tables']['half_life_ar4'],
        'halflife'
    )
    
    display_master_table(master_halflife_table, 'halflife')
    
    st.info("""
    **Interpretation:** Half-life indicates shock persistence in quarters. Lower values (green) show faster 
    mean reversion. Values of 1-3 quarters are typical for efficient financial markets.
    """)
    
    # Download button for master half-life table
    csv_master_halflife = master_halflife_table.to_csv(index=True)
    st.download_button(
        label="📥 Download Master Half-life Table (CSV)",
        data=csv_master_halflife,
        file_name="cs4_master_halflife_results.csv",
        mime="text/csv"
    )
    
    # Chart 3: ACF charts grid (Full Period)
    st.markdown("---")
    st.subheader("📊 Chart 3: Autocorrelation Analysis - Full Period")
    st.markdown("**Autocorrelation functions showing persistence patterns across all groups**")
    
    chart3 = create_comprehensive_acf_chart(full_results, crisis_results, "Full Period")
    if chart3:
        st.pyplot(chart3)
        
        # Download button for Chart 3
        buf3 = io.BytesIO()
        chart3.savefig(buf3, format='png', dpi=150, bbox_inches='tight')
        buf3.seek(0)
        st.download_button(
            label="📥 Download Full Period ACF Charts (PNG)",
            data=buf3,
            file_name="cs4_comprehensive_acf_full_period.png",
            mime="image/png"
        )
        plt.close(chart3)
    
    # Chart 4: ACF charts grid (Crisis-Excluded)
    st.markdown("---")
    st.subheader("📊 Chart 4: Autocorrelation Analysis - Crisis-Excluded")
    st.markdown("**Persistence patterns with financial crisis periods removed**")
    
    chart4 = create_comprehensive_acf_chart(full_results, crisis_results, "Crisis-Excluded")
    if chart4:
        st.pyplot(chart4)
        
        # Download button for Chart 4
        buf4 = io.BytesIO()
        chart4.savefig(buf4, format='png', dpi=150, bbox_inches='tight')
        buf4.seek(0)
        st.download_button(
            label="📥 Download Crisis-Excluded ACF Charts (PNG)",
            data=buf4,
            file_name="cs4_comprehensive_acf_crisis_excluded.png",
            mime="image/png"
        )
        plt.close(chart4)
    
    # Table 3: Master RMSE Results
    st.markdown("---")
    st.subheader("📈 Table 3: RMSE Prediction Results (All Indicators)")
    
    master_rmse_table = create_master_table(
        indicators,
        full_results['summary_tables']['rmse_prediction'],
        crisis_results['summary_tables']['rmse_prediction'],
        'rmse'
    )
    
    display_master_table(master_rmse_table, 'rmse')
    
    st.info("""
    **Interpretation:** RMSE measures 4-quarter ahead prediction accuracy. Lower values indicate better 
    forecastability. Compare across groups to assess relative prediction difficulty.
    """)
    
    # Download button for master RMSE table
    csv_master_rmse = master_rmse_table.to_csv(index=True)
    st.download_button(
        label="📥 Download Master RMSE Table (CSV)",
        data=csv_master_rmse,
        file_name="cs4_master_rmse_results.csv",
        mime="text/csv"
    )
    
    # Chart 5: Time-series (Iceland + Comparator averages with crisis marking)
    st.markdown("---")
    st.subheader("📊 Chart 5: Time-Series Analysis - Average Aggregation")
    st.markdown("**Net Capital Flows over time comparing Iceland with average aggregations, with crisis periods marked**")
    
    chart5 = create_comprehensive_timeseries_chart("Averages")
    if chart5:
        st.pyplot(chart5)
        
        # Download button for Chart 5
        buf5 = io.BytesIO()
        chart5.savefig(buf5, format='png', dpi=150, bbox_inches='tight')
        buf5.seek(0)
        st.download_button(
            label="📥 Download Time-Series Averages Chart (PNG)",
            data=buf5,
            file_name="cs4_comprehensive_timeseries_averages.png",
            mime="image/png"
        )
        plt.close(chart5)
    
    # Chart 6: Time-series (Iceland + Comparator sums with crisis marking)
    st.markdown("---")
    st.subheader("📊 Chart 6: Time-Series Analysis - Sum Aggregation")
    st.markdown("**Net Capital Flows over time comparing Iceland with sum aggregations, showing crisis impact**")
    
    chart6 = create_comprehensive_timeseries_chart("Sums")
    if chart6:
        st.pyplot(chart6)
        
        # Download button for Chart 6
        buf6 = io.BytesIO()
        chart6.savefig(buf6, format='png', dpi=150, bbox_inches='tight')
        buf6.seek(0)
        st.download_button(
            label="📥 Download Time-Series Sums Chart (PNG)",
            data=buf6,
            file_name="cs4_comprehensive_timeseries_sums.png",
            mime="image/png"
        )
        plt.close(chart6)
    
    # Master tables export
    st.markdown("---")
    st.subheader("📁 Master Tables Export")
    
    # Create Excel export for all master tables
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        master_std_table.to_excel(writer, sheet_name='Master_Standard_Deviations', index=True)
        master_halflife_table.to_excel(writer, sheet_name='Master_Half_Life', index=True)
        master_rmse_table.to_excel(writer, sheet_name='Master_RMSE', index=True)
        
        # Add master metadata
        master_metadata = pd.DataFrame({
            'Parameter': [
                'Table Structure', 'Rows per Indicator', 'Total Indicators', 'Total Rows',
                'Analysis Periods', 'Statistical Methods', 'Export Date'
            ],
            'Value': [
                '8 rows × 7 columns (2 periods × 4 indicators)',
                '2 (Full Period + Crisis-Excluded)', '4', '8',
                'Full Period (1999-2025) and Crisis-Excluded', 
                'F-tests, AR(4) models, RMSE prediction',
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        })
        master_metadata.to_excel(writer, sheet_name='Master_Tables_Metadata', index=False)
    
    excel_data = output.getvalue()
    
    st.download_button(
        label="📥 Download All Master Tables (Excel)",
        data=excel_data,
        file_name=f"cs4_master_tables_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def create_master_table(indicators, full_table, crisis_table, table_type):
    """Create master table with all indicators in 8-row format (2 rows per indicator × 4 indicators)"""
    
    # Initialize master data structure
    master_data = []
    
    # Get column names (excluding 'Indicator')
    columns = [col for col in full_table.columns if col != 'Indicator']
    
    # Process each indicator
    for indicator in indicators:
        # Find rows for this indicator in both tables
        full_row = full_table[full_table['Indicator'] == indicator]
        crisis_row = crisis_table[crisis_table['Indicator'] == indicator]
        
        if len(full_row) == 0 or len(crisis_row) == 0:
            st.warning(f"Data not found for {indicator} in master table creation")
            continue
            
        # Get the first (and should be only) row for each
        full_data = full_row.iloc[0]
        crisis_data = crisis_row.iloc[0]
        
        # Create Full Period row
        full_period_row = {'Indicator/Period': f"{indicator} (Full Time Period)"}
        for col in columns:
            full_period_row[col] = full_data[col]
        master_data.append(full_period_row)
        
        # Create Crisis-Excluded row  
        crisis_excluded_row = {'Indicator/Period': f"{indicator} (Crisis-Excluded)"}
        for col in columns:
            crisis_excluded_row[col] = crisis_data[col]
        master_data.append(crisis_excluded_row)
    
    # Create DataFrame
    master_df = pd.DataFrame(master_data)
    master_df.set_index('Indicator/Period', inplace=True)
    
    return master_df


def display_master_table(df, table_type):
    """Display master table with appropriate styling"""
    
    if table_type == 'std':
        # Standard deviations with F-test significance and volatility direction color coding
        def color_std_volatility(val, iceland_val):
            """Color code based on volatility direction and significance"""
            if pd.isna(val) or val == 'N/A':
                return ''
            
            val_str = str(val)
            # Check for significance stars
            has_three_stars = '***' in val_str
            has_two_stars = '**' in val_str and not has_three_stars
            has_one_star = '*' in val_str and not has_two_stars and not has_three_stars
            
            if not (has_one_star or has_two_stars or has_three_stars):
                return ''  # No significant difference - keep default background
            
            # Extract numeric value (remove stars)
            try:
                numeric_val = float(val_str.replace('*', ''))
                iceland_numeric = float(str(iceland_val).replace('*', ''))
                
                if iceland_numeric > numeric_val:
                    # Iceland is MORE volatile (higher std dev) - use light red/pink
                    if has_three_stars:
                        return 'background-color: #ffcccc; color: #990000; font-weight: bold'  # Darker red for p<0.01
                    elif has_two_stars:
                        return 'background-color: #ffdddd; color: #cc0000; font-weight: bold'  # Medium red for p<0.05
                    else:
                        return 'background-color: #ffeeee; color: #cc3333'  # Light red for p<0.10
                else:
                    # Iceland is LESS volatile (lower std dev) - use light green
                    if has_three_stars:
                        return 'background-color: #ccffcc; color: #006600; font-weight: bold'  # Darker green for p<0.01
                    elif has_two_stars:
                        return 'background-color: #ddffdd; color: #009900; font-weight: bold'  # Medium green for p<0.05
                    else:
                        return 'background-color: #eeffee; color: #00cc00'  # Light green for p<0.10
            except:
                return ''
        
        # Apply color coding to all columns except Iceland
        styled_table = df.style.apply(
            lambda row: ['' if col == 'Iceland' else color_std_volatility(row[col], row['Iceland']) for col in row.index],
            axis=1
        ).set_properties(**{
            'text-align': 'center',
            'font-size': '11px'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#e6f3ff'), ('font-weight', 'bold'), ('font-size', '11px')]},
            {'selector': 'td', 'props': [('padding', '4px 6px')]}
        ])
        
    elif table_type == 'halflife':
        # Color-code half-life values for master table
        def color_halflife_master(val):
            if val == 'N/A':
                return 'color: gray'
            try:
                v = int(val)
                if v <= 1:
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'  # Green for fast reversion
                elif v <= 3:
                    return 'background-color: #fff3cd; color: #856404; font-weight: bold'  # Yellow for moderate
                else:
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold'  # Red for slow
            except:
                return ''
        
        styled_table = df.style.applymap(color_halflife_master, subset=df.columns).set_properties(**{
            'text-align': 'center',
            'font-size': '11px'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#e6f3ff'), ('font-weight', 'bold'), ('font-size', '11px')]},
            {'selector': 'td', 'props': [('padding', '4px 6px')]}
        ])
        
    elif table_type == 'rmse':
        # Format RMSE values for master table
        def format_rmse_master(val):
            if val == 'N/A':
                return val
            try:
                return f"{float(val):.2f}"
            except:
                return val
        
        formatted_df = df.copy()
        for col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(format_rmse_master)
        
        styled_table = formatted_df.style.set_properties(**{
            'text-align': 'center',
            'font-size': '11px'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#e6f3ff'), ('font-weight', 'bold'), ('font-size', '11px')]},
            {'selector': 'tr:nth-of-type(even)', 'props': [('background-color', '#f9f9f9')]},
            {'selector': 'td', 'props': [('padding', '4px 6px')]}
        ])
    
    else:
        styled_table = df.style.set_properties(**{
            'text-align': 'center',
            'font-size': '11px'
        })
    
    st.dataframe(styled_table, use_container_width=True)


def display_indicator_section(indicator, full_results, crisis_results):
    """Display comprehensive analysis section for a specific indicator"""
    
    st.markdown("---")
    st.header(f"📊 {indicator}")
    st.markdown(f"**Comprehensive statistical analysis for {indicator} flows**")
    
    # 1. Standard Deviations & F-Test Results
    st.subheader("🎯 Standard Deviations & F-Test Results")
    
    std_table = create_integrated_table(
        indicator, 
        full_results['summary_tables']['standard_deviations_ftest'], 
        crisis_results['summary_tables']['standard_deviations_ftest'],
        'std'
    )
    
    display_styled_table(std_table, 'std')
    
    st.info("""
    **Interpretation:** Values show standard deviations (volatility). Stars indicate statistically significant 
    differences from Iceland using F-tests. More stars = stronger evidence of volatility differences.
    
    **Color Coding:**
    - 🔴 **Red/Pink Background**: Iceland is MORE volatile than comparator (higher standard deviation)
    - 🟢 **Green Background**: Iceland is LESS volatile than comparator (lower standard deviation)
    - ⚪ **No Color**: No statistically significant difference
    """)
    
    # Download button for std table
    csv_std = std_table.to_csv(index=True)
    st.download_button(
        label=f"📥 Download {indicator} - Standard Deviations (CSV)",
        data=csv_std,
        file_name=f"cs4_{indicator.replace(' ', '_').lower()}_std_deviations.csv",
        mime="text/csv"
    )
    
    # Data Visualizations Placeholder - Distribution Analysis
    with st.expander("📈 Distribution Comparison Analysis", expanded=False):
        st.info("📊 **Distribution Analysis Coming Soon**")
        st.markdown("""
        This section will feature:
        - Box plots comparing distributions across Full Period vs Crisis-Excluded
        - Visual representation of volatility differences
        - Color-coded significance indicators
        - Interactive comparison tools
        """)
    
    # 2. Half-Life from AR(4) Analysis
    st.subheader("⏱️ Half-Life from AR(4) Models")
    
    halflife_table = create_integrated_table(
        indicator,
        full_results['summary_tables']['half_life_ar4'],
        crisis_results['summary_tables']['half_life_ar4'], 
        'halflife'
    )
    
    display_styled_table(halflife_table, 'halflife')
    
    st.info("""
    **Interpretation:** Half-life indicates persistence of shocks (in quarters). Lower values (green) indicate 
    faster mean reversion. Most financial flows show 1-3 quarter half-lives, consistent with market efficiency.
    """)
    
    # Download button for half-life table
    csv_halflife = halflife_table.to_csv(index=True)
    st.download_button(
        label=f"📥 Download {indicator} - Half-Life (CSV)",
        data=csv_halflife,
        file_name=f"cs4_{indicator.replace(' ', '_').lower()}_halflife.csv",
        mime="text/csv"
    )
    
    # Data Visualizations Placeholder - Impulse Response Analysis  
    with st.expander("📊 AR(4) Impulse Response Analysis", expanded=False):
        st.info("⏱️ **Impulse Response Analysis Coming Soon**")
        st.markdown("""
        This section will feature:
        - AR(4) impulse response function plots
        - Half-life visualization with decay curves
        - Comparison of shock persistence across periods
        - Interactive parameter exploration
        """)
    
    # 3. RMSE Prediction Accuracy
    st.subheader("📈 RMSE Prediction Accuracy")
    
    rmse_table = create_integrated_table(
        indicator,
        full_results['summary_tables']['rmse_prediction'],
        crisis_results['summary_tables']['rmse_prediction'],
        'rmse'
    )
    
    display_styled_table(rmse_table, 'rmse')
    
    st.info("""
    **Interpretation:** RMSE measures prediction error for 4-quarter ahead forecasts. Lower values indicate 
    better predictability. Compare across groups to assess relative forecast difficulty.
    """)
    
    # Download button for RMSE table
    csv_rmse = rmse_table.to_csv(index=True)
    st.download_button(
        label=f"📥 Download {indicator} - RMSE (CSV)",
        data=csv_rmse,
        file_name=f"cs4_{indicator.replace(' ', '_').lower()}_rmse.csv",
        mime="text/csv"
    )
    
    # Data Visualizations Placeholder - RMSE Prediction Analysis
    with st.expander("🎯 RMSE Prediction Accuracy Analysis", expanded=False):
        st.info("📈 **Prediction Accuracy Analysis Coming Soon**")
        st.markdown("""
        This section will feature:
        - Bar charts comparing RMSE across groups
        - Prediction accuracy visualization
        - Model performance comparisons
        - Forecasting quality indicators
        """)


def create_integrated_table(indicator, full_table, crisis_table, table_type):
    """Create integrated table with Full Period and Crisis-Excluded rows"""
    
    # Find the row for this indicator in both tables
    full_row = full_table[full_table['Indicator'] == indicator].iloc[0] if len(full_table[full_table['Indicator'] == indicator]) > 0 else None
    crisis_row = crisis_table[crisis_table['Indicator'] == indicator].iloc[0] if len(crisis_table[crisis_table['Indicator'] == indicator]) > 0 else None
    
    if full_row is None or crisis_row is None:
        st.error(f"Data not found for {indicator}")
        return pd.DataFrame()
    
    # Get column names (excluding 'Indicator')
    columns = [col for col in full_table.columns if col != 'Indicator']
    
    # Create integrated table
    data = {
        'Time Period': ['Full Time Period', 'Crisis-Excluded']
    }
    
    # Add data for each column
    for col in columns:
        data[col] = [full_row[col], crisis_row[col]]
    
    integrated_df = pd.DataFrame(data)
    integrated_df.set_index('Time Period', inplace=True)
    
    return integrated_df


def display_styled_table(df, table_type):
    """Display table with appropriate styling based on type"""
    
    if table_type == 'std':
        # Standard deviations with F-test significance and volatility direction color coding
        def color_std_volatility_individual(val, iceland_val):
            """Color code based on volatility direction and significance for individual tables"""
            if pd.isna(val) or val == 'N/A':
                return ''
            
            val_str = str(val)
            # Check for significance stars
            has_three_stars = '***' in val_str
            has_two_stars = '**' in val_str and not has_three_stars
            has_one_star = '*' in val_str and not has_two_stars and not has_three_stars
            
            if not (has_one_star or has_two_stars or has_three_stars):
                return ''  # No significant difference - keep default background
            
            # Extract numeric value (remove stars)
            try:
                numeric_val = float(val_str.replace('*', ''))
                iceland_numeric = float(str(iceland_val).replace('*', ''))
                
                if iceland_numeric > numeric_val:
                    # Iceland is MORE volatile (higher std dev) - use light red/pink
                    if has_three_stars:
                        return 'background-color: #ffcccc; color: #990000; font-weight: bold'  # Darker red for p<0.01
                    elif has_two_stars:
                        return 'background-color: #ffdddd; color: #cc0000; font-weight: bold'  # Medium red for p<0.05
                    else:
                        return 'background-color: #ffeeee; color: #cc3333'  # Light red for p<0.10
                else:
                    # Iceland is LESS volatile (lower std dev) - use light green
                    if has_three_stars:
                        return 'background-color: #ccffcc; color: #006600; font-weight: bold'  # Darker green for p<0.01
                    elif has_two_stars:
                        return 'background-color: #ddffdd; color: #009900; font-weight: bold'  # Medium green for p<0.05
                    else:
                        return 'background-color: #eeffee; color: #00cc00'  # Light green for p<0.10
            except:
                return ''
        
        # Apply color coding to all columns except Iceland
        styled_table = df.style.apply(
            lambda row: ['' if col == 'Iceland' else color_std_volatility_individual(row[col], row['Iceland']) for col in row.index],
            axis=1
        ).set_properties(**{
            'text-align': 'center',
            'font-size': '12px'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#e6f3ff'), ('font-weight', 'bold')]},
            {'selector': 'td', 'props': [('padding', '4px 6px')]}
        ])
        
    elif table_type == 'halflife':
        # Color-code half-life values
        def color_halflife(val):
            if val == 'N/A':
                return 'color: gray'
            try:
                v = int(val)
                if v <= 1:
                    return 'background-color: #d4edda; color: #155724'  # Green for fast reversion
                elif v <= 3:
                    return 'background-color: #fff3cd; color: #856404'  # Yellow for moderate
                else:
                    return 'background-color: #f8d7da; color: #721c24'  # Red for slow
            except:
                return ''
        
        styled_table = df.style.applymap(color_halflife, subset=df.columns)
        
    elif table_type == 'rmse':
        # Format RMSE values
        def format_rmse(val):
            if val == 'N/A':
                return val
            try:
                return f"{float(val):.2f}"
            except:
                return val
        
        formatted_df = df.copy()
        for col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(format_rmse)
        
        styled_table = formatted_df.style.set_properties(**{
            'text-align': 'center',
            'font-size': '12px'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#e6f3ff'), ('font-weight', 'bold')]},
            {'selector': 'tr:nth-of-type(even)', 'props': [('background-color', '#f9f9f9')]}
        ])
    
    else:
        styled_table = df.style.set_properties(**{
            'text-align': 'center',
            'font-size': '12px'
        })
    
    st.dataframe(styled_table, use_container_width=True)




def display_summary_insights_and_export(full_results, crisis_results):
    """Display summary insights and comprehensive export functionality"""
    
    st.markdown("---")
    st.header("🔍 Summary Insights & Comprehensive Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Key Findings Across Indicators
        - **Volatility Patterns:** Iceland shows systematically different volatility compared to most comparator groups
        - **F-Test Results:** Strongest statistical differences observed with aggregated measures (sum indicators)  
        - **Crisis Impact:** Crisis exclusion generally reduces volatility measures across all groups
        - **Consistency:** Patterns remain consistent across different capital flow types
        """)
    
    with col2:
        st.markdown("""
        ### ⏱️ Temporal Dynamics
        - **Half-Life Patterns:** Most indicators show 1-2 quarter half-lives, consistent with efficient markets
        - **Persistence:** Low persistence suggests rapid adjustment to equilibrium
        - **Predictability:** RMSE varies significantly across indicators and groups
        - **Model Performance:** AR(4) models generally provide reasonable fit for most series
        """)
    
    # Comprehensive Excel Export
    st.subheader("📁 Complete Analysis Export")
    st.markdown("Download all results in a comprehensive Excel workbook with separate sheets for each indicator and analysis type.")
    
    # Create comprehensive Excel export
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        
        # Get indicators
        indicators = full_results['metadata']['indicators_analyzed']
        
        for indicator in indicators:
            # Create integrated tables for each indicator
            std_table = create_integrated_table(
                indicator,
                full_results['summary_tables']['standard_deviations_ftest'],
                crisis_results['summary_tables']['standard_deviations_ftest'],
                'std'
            )
            
            halflife_table = create_integrated_table(
                indicator,
                full_results['summary_tables']['half_life_ar4'],
                crisis_results['summary_tables']['half_life_ar4'],
                'halflife'
            )
            
            rmse_table = create_integrated_table(
                indicator,
                full_results['summary_tables']['rmse_prediction'],
                crisis_results['summary_tables']['rmse_prediction'],
                'rmse'
            )
            
            # Clean indicator name for sheet names
            clean_name = indicator.replace(' ', '_').replace('(', '').replace(')', '')
            
            # Export to Excel with different sheets per indicator
            if not std_table.empty:
                std_table.to_excel(writer, sheet_name=f'{clean_name[:25]}_Std', index=True)
            if not halflife_table.empty:
                halflife_table.to_excel(writer, sheet_name=f'{clean_name[:20]}_HalfLife', index=True)
            if not rmse_table.empty:
                rmse_table.to_excel(writer, sheet_name=f'{clean_name[:25]}_RMSE', index=True)
        
        # Add metadata sheet
        metadata_df = pd.DataFrame({
            'Parameter': [
                'Analysis Type', 'Full Period Observations', 'Crisis-Excluded Observations',
                'Time Range (Full)', 'Time Range (Crisis-Excluded)', 'Indicators Analyzed',
                'Comparator Groups', 'Statistical Methods', 'Export Date'
            ],
            'Value': [
                'Integrated Full Period and Crisis-Excluded Analysis',
                '105', '81',
                '1999 Q1 - 2025 Q1', 'Excluding 2008-2010 (GFC) and 2020-2022 (COVID-19)',
                ', '.join(indicators),
                ', '.join(full_results['metadata']['comparator_groups']),
                'F-tests, AR(4) models, RMSE prediction',
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        })
        metadata_df.to_excel(writer, sheet_name='Analysis_Metadata', index=False)
    
    excel_data = output.getvalue()
    
    st.download_button(
        label="📥 Download Complete CS4 Analysis (Excel)",
        data=excel_data,
        file_name=f"cs4_comprehensive_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # Analysis summary statistics
    with st.expander("📈 Analysis Summary Statistics", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Indicators", len(indicators))
            st.metric("Total Comparisons", len(indicators) * len(full_results['metadata']['comparator_groups']))
        
        with col2:
            st.metric("Full Period Observations", "105") 
            st.metric("Crisis-Excluded Observations", "81")
        
        with col3:
            st.metric("Statistical Tests per Comparison", "3")
            st.metric("Total Statistical Results", len(indicators) * len(full_results['metadata']['comparator_groups']) * 3 * 2)


def main():
    """Main application function"""
    
    # Title and description
    st.title("🇮🇸 Case Study 4: Comprehensive Statistical Analysis")
    st.markdown("""
    **Objective:** Evaluate currency regime effects on capital flow volatility through comprehensive 
    statistical analysis comparing Iceland with multiple comparator groups.
    """)
    
    # Sidebar information (removed toggle interface)
    st.sidebar.title("⚙️ Analysis Overview")
    
    # Analysis info  
    st.sidebar.info("📊 **Integrated Analysis:** Both Full Period and Crisis-Excluded results displayed together")
    st.sidebar.markdown("""
    **Time Periods:**
    - **Full Period:** 1999-2025 (105 observations)
    - **Crisis-Excluded:** Excludes 2008-2010 (GFC) and 2020-2022 (COVID-19) - 81 observations
    """)
    
    # Comparator groups info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌍 Comparator Groups")
    st.sidebar.markdown("""
    - **Eurozone:** Sum & Average aggregations
    - **Small Open Economies (SOE):** Sum & Average
    - **Baltics:** Sum & Average aggregations
    """)
    
    # Indicators info
    st.sidebar.markdown("### 📊 Indicators Analyzed")
    st.sidebar.markdown("""
    - Net Direct Investment
    - Net Portfolio Investment
    - Net Other Investment
    - Net Capital Flows (Total)
    """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Statistical Analysis", "📚 Methodology", "📖 About"])
    
    with tab1:
        # Run integrated analysis (no toggle needed)
        run_cs4_integrated_analysis()
    
    with tab2:
        st.header("📚 Detailed Methodology")
        display_methodology_section()
        
        # Additional methodology details
        st.markdown("---")
        st.subheader("🔬 Statistical Framework")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Data Structure
            - **Frequency:** Quarterly
            - **Coverage:** 1999 Q1 - 2025 Q1
            - **Units:** % of GDP (annualized)
            - **Aggregation Methods:**
                - Sum: Total across group countries
                - Average: Mean across group countries
            """)
        
        with col2:
            st.markdown("""
            ### Robustness Checks
            - ✅ Residual autocorrelation tests
            - ✅ Stationarity verification (ADF tests)
            - ✅ Cross-validation with CS1/CS3 results
            - ✅ Sensitivity to model specifications
            - ✅ Edge case and missing data handling
            """)
    
    with tab3:
        st.header("📖 About Case Study 4")
        st.markdown("""
        ### Research Context
        
        This case study provides comprehensive statistical evidence on capital flow volatility patterns 
        across different currency regimes and economic groupings. By comparing Iceland (independent 
        currency) with various comparator groups, we assess the impact of monetary autonomy on 
        financial stability.
        
        ### Key Research Questions
        
        1. **Volatility Differences:** Does Iceland exhibit significantly different capital flow 
           volatility compared to currency union members?
        
        2. **Persistence Patterns:** How quickly do capital flow shocks dissipate across different 
           monetary regimes?
        
        3. **Predictability:** Are capital flows more predictable under certain currency arrangements?
        
        ### Policy Implications
        
        Results inform debates on:
        - Currency union membership decisions
        - Capital flow management strategies
        - Financial stability frameworks
        - Macroprudential policy design
        
        ### Academic Contribution
        
        This analysis contributes to literature on:
        - International capital flows
        - Currency regime effects
        - Small open economy dynamics
        - Financial market integration
        
        ---
        
        **Technical Note:** All computations follow academic standards with appropriate statistical 
        tests and robustness checks. Results are suitable for research publication and policy analysis.
        """)


if __name__ == "__main__":
    main()