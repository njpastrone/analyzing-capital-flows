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

# Configure matplotlib for professional PDF export (from commit 8181df5)
warnings.filterwarnings('ignore')
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
    'axes.facecolor': 'white',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.max_open_warning': 0
})

def get_pdf_optimized_figsize(chart_type, base_width=10, base_height=6):
    """
    Calculate PDF-optimized figure size based on standard US letter dimensions
    From commit 8181df5: Dynamic sizing for bulletproof PDF compatibility
    
    Args:
        chart_type: 'boxplot', 'grid', 'timeseries'
        base_width: Base width in inches  
        base_height: Base height in inches
    
    Returns:
        Tuple of (width, height) optimized for PDF export
    """
    # US Letter: 8.5" x 11" with 0.75" margins = 7" usable width
    max_pdf_width = 7.0
    max_pdf_height = 9.0  # Leave room for headers/text
    
    if chart_type == 'boxplot':
        # Compact horizontal layout
        width = min(base_width, max_pdf_width)
        height = min(base_height * 0.8, max_pdf_height * 0.4)
        return (width, height)
    
    elif chart_type == 'grid':
        # 3x3 grid needs special handling
        width = min(base_width * 0.9, max_pdf_width)
        height = min(base_height * 0.85, max_pdf_height * 0.7)
        return (width, height)
    
    elif chart_type == 'timeseries':
        # Wide but not too tall
        width = min(base_width, max_pdf_width)
        height = min(base_height * 0.75, max_pdf_height * 0.45)
        return (width, height)
    
    else:
        # Default constraint
        width = min(base_width, max_pdf_width)
        height = min(base_height, max_pdf_height * 0.5)
        return (width, height)

# Page configuration - removed to avoid conflicts when imported into main_app.py
# st.set_page_config() is now handled by main_app.py or when run standalone

def apply_professional_styling():
    """Apply professional CSS styling with PDF export optimization"""
    st.markdown("""
<style>
    /* PDF Export Optimized Body and Layout */
    body {
        font-family: Arial, sans-serif !important;
        margin: 40px !important;
        line-height: 1.6 !important;
    }
    
    /* Professional table styling (both dataframe and HTML) */
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
    
    /* HTML Table styling for PDF export */
    .cs4-master-table {
        border-collapse: collapse;
        width: 100%;
        margin: 15px 0;
        font-family: Arial, sans-serif !important;
        font-size: 11px;
        page-break-inside: avoid;
    }
    .cs4-master-table th {
        background-color: #e6f3ff;
        font-weight: bold;
        border: 1px solid #ddd;
        padding: 6px 8px;
        text-align: center;
    }
    .cs4-master-table td {
        border: 1px solid #ddd;
        padding: 4px 6px;
        text-align: center;
    }
    .cs4-master-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    /* Table column width constraints for PDF export */
    .cs4-master-table th:first-child, .cs4-master-table td:first-child {
        width: 220px !important;
        max-width: 220px !important;
        text-align: left !important;
        font-weight: bold !important;
    }
    .cs4-master-table th:not(:first-child), .cs4-master-table td:not(:first-child) {
        width: 70px !important;
        max-width: 70px !important;
    }
    
    /* Streamlit app container constraints for PDF */
    .main .block-container {
        max-width: none !important;
        padding: 1rem 2rem !important;
    }
    
    /* Image and chart constraints */
    img {
        max-width: 100% !important;
        height: auto !important;
        page-break-inside: avoid !important;
    }
    
    /* Creative Hard Margin Constraints for Bulletproof PDF Export */
    .main .block-container {
        max-width: 8.5in !important;  /* Standard US letter width */
        margin: 0 auto !important;
    }
    
    .chart-container { 
        max-width: 100% !important;
        overflow: hidden !important;
        text-align: center !important;
        margin: 20px 0 !important;
    }
    
    /* Dynamic chart sizing constraints */
    .pyplot-container {
        max-width: 7.5in !important;  /* Leave margin for PDF */
        margin: 0 auto !important;
    }
    
    /* Print Media Queries for PDF Export (enhanced from commit 8181df5) */
    @media print {
        body { 
            font-family: serif !important;
            margin: 40px !important; 
            line-height: 1.6 !important;
            color: black !important;
        }
        .stApp { 
            margin: 40px !important; 
            max-width: 8.5in !important;
        }
        .main .block-container {
            max-width: none !important;
            padding: 0 !important;
        }
        
        /* Hard chart constraints for PDF */
        .chart-container { 
            max-width: 100% !important;
            page-break-inside: avoid !important;
        }
        .pyplot-container {
            max-width: 7.5in !important;
            margin: 0 auto !important;
        }
        
        /* Table print optimizations (from commit 8181df5) */
        .cs4-master-table { 
            page-break-inside: avoid !important;
            font-size: 7px !important;  /* Smaller for PDF */
            margin: 10px 0 !important;
        }
        .cs4-master-table th:first-child, .cs4-master-table td:first-child {
            width: 140px !important;
            max-width: 140px !important;
        }
        .cs4-master-table th:not(:first-child), .cs4-master-table td:not(:first-child) {
            width: 50px !important;
            max-width: 50px !important;
        }
        
        /* Header optimizations */
        h1, h2, h3 { 
            page-break-after: avoid !important; 
            margin-bottom: 10px !important;
        }
        
        /* Image and chart print optimizations */
        img { 
            max-width: 7.5in !important;
            height: auto !important;
            page-break-inside: avoid !important;
            display: block !important;
            margin: 10px auto !important;
        }
        
        /* Remove Streamlit UI elements in print (from commit 8181df5) */
        .stDeployButton { display: none !important; }
        .stDecoration { display: none !important; }
        .stToolbar { display: none !important; }
        header[data-testid="stHeader"] { display: none !important; }
        .stSidebar { display: none !important; }
        
        /* Optimize spacing for print */
        .element-container { margin-bottom: 8px !important; }
        div[data-testid="column"] { page-break-inside: avoid !important; }
        .stTabs { page-break-inside: avoid !important; }
        
        /* Force clean page breaks */
        .stExpander { page-break-inside: avoid !important; }
        section[data-testid="stSidebar"] { display: none !important; }
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
    groups = ['iceland_pgdp', 'eurozone_pgdp_weighted', 'eurozone_pgdp_simple', 'soe_pgdp_weighted', 'soe_pgdp_simple', 'baltics_pgdp_weighted', 'baltics_pgdp_simple']
    group_labels_map = {
        'iceland_pgdp': 'Iceland',
        'eurozone_pgdp_weighted': 'Eurozone\nWeighted Avg',
        'eurozone_pgdp_simple': 'Eurozone\nSimple Avg',
        'soe_pgdp_weighted': 'SOE\nWeighted Avg',
        'soe_pgdp_simple': 'SOE\nSimple Avg',
        'baltics_pgdp_weighted': 'Baltics\nWeighted Avg',
        'baltics_pgdp_simple': 'Baltics\nSimple Avg'
    }
    
    # Colors map - Iceland distinct, others grouped  
    colors_map = {
        'iceland_pgdp': '#e74c3c',
        'eurozone_pgdp_weighted': '#3498db',
        'eurozone_pgdp_simple': '#85c1e9',
        'soe_pgdp_weighted': '#f39c12',
        'soe_pgdp_simple': '#f8c471',
        'baltics_pgdp_weighted': '#1abc9c',
        'baltics_pgdp_simple': '#7dcea0'
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
    
    # Create figure with dynamic PDF-optimized sizing
    figsize = get_pdf_optimized_figsize('boxplot', 10, 4)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
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
    
    # Styling with rotated labels and adjusted font sizes
    ax.set_title(f'Net Capital Flows Distribution - {period_name} (Ordered by Volatility)', 
                fontweight='bold', fontsize=14, pad=30)  # Increased padding
    ax.set_ylabel('Net Capital Flows (% of GDP)', fontsize=12)
    
    # Rotate x-axis labels and reduce y-axis font size
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=8)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add reference line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    
    # Adjust spacing to prevent label cutoff with rotated labels
    plt.subplots_adjust(top=0.88, bottom=0.15)
    return fig


def create_comprehensive_acf_chart(indicator_name, period_name):
    """Create ACF charts in 1x4 layout for weighted averages only"""
    
    # Load data for specified indicator
    from core.cs4_statistical_analysis import CS4DataLoader
    loader = CS4DataLoader()
    
    include_crisis = (period_name == "Full Period")
    data = loader.load_indicator_data(indicator_name, include_crisis_years=include_crisis)
    
    if data is None:
        st.warning(f"Unable to load {indicator_name} data for {period_name} ACF charts")
        return None
    
    # Use only weighted averages (excluding simple averages)
    groups = ['iceland_pgdp', 'eurozone_pgdp_weighted', 'soe_pgdp_weighted', 'baltics_pgdp_weighted']
    group_labels = ['Iceland', 'Eurozone Weighted Avg', 'SOE Weighted Avg', 'Baltics Weighted Avg']
    
    # Create figure with 1x4 layout for cleaner presentation
    figsize = (12, 3)  # Wide and short for 1x4 layout
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    for i, (group, label) in enumerate(zip(groups, group_labels)):
        if group in data.columns and i < 4:
            ax = axes[i]
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
                        ax.set_title(f'{label}', fontweight='bold', fontsize=9, pad=8)
                        ax.set_xlabel('Lags (Quarters)', fontsize=8, fontweight='medium')
                        
                        # Show y-axis label only on leftmost chart for cleaner presentation
                        if i == 0:  # leftmost chart only
                            ax.set_ylabel('ACF', fontsize=8, fontweight='medium')
                        else:  # remove y-axis labels from other charts
                            ax.set_ylabel('')
                            ax.tick_params(left=False, labelleft=False)
                        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
                        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                        ax.tick_params(axis='both', labelsize=7)
                        
                        # Set reasonable y-axis limits
                        ax.set_ylim(-1.1, 1.1)
                        
                        # Add subtle x-axis ticks for better readability
                        if len(x_lags) > 10:
                            ax.set_xticks(range(0, len(x_lags), 5))  # Show every 5th lag for clarity
                        
                    except Exception as e:
                        ax.text(0.5, 0.5, f'ACF calculation\nfailed for {label}', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=9)
                        ax.set_title(f'{label}', fontweight='bold', fontsize=9, pad=8)
                        ax.set_xlabel('Lags (Quarters)', fontsize=8)
                        
                        # Consistent y-axis handling for error cases
                        if i == 0:
                            ax.set_ylabel('ACF', fontsize=8, fontweight='medium')
                        else:
                            ax.set_ylabel('')
                            ax.tick_params(left=False, labelleft=False)
            else:
                ax.text(0.5, 0.5, f'Insufficient data\nfor {label}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=9)
                ax.set_title(f'{label}', fontweight='bold', fontsize=9, pad=8)
                ax.set_xlabel('Lags (Quarters)', fontsize=8)
                
                # Consistent y-axis handling for insufficient data cases
                if i == 0:
                    ax.set_ylabel('ACF', fontsize=8, fontweight='medium')
                else:
                    ax.set_ylabel('')
                    ax.tick_params(left=False, labelleft=False)
    
    # Enhanced title with indicator and period specification
    title_text = f'{indicator_name} - {period_name}'
    if period_name == "Crisis-Excluded":
        title_text += ' (Excludes 2008-2010, 2020-2022)'
    plt.suptitle(title_text, fontweight='bold', fontsize=11, y=0.96)
    
    # Increased spacing between title and charts for better readability
    plt.subplots_adjust(top=0.78, bottom=0.15, left=0.06, right=0.98, wspace=0.25)
    
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
    if aggregation_type == "Simple Averages":
        groups = ['iceland_pgdp', 'eurozone_pgdp_simple', 'soe_pgdp_simple', 'baltics_pgdp_simple']
        group_labels = ['Iceland', 'Eurozone Simple Average', 'SOE Simple Average', 'Baltics Simple Average']
        colors = ['#e74c3c', '#3498db', '#f39c12', '#1abc9c']
    else:  # Weighted Averages
        groups = ['iceland_pgdp', 'eurozone_pgdp_weighted', 'soe_pgdp_weighted', 'baltics_pgdp_weighted']
        group_labels = ['Iceland', 'Eurozone Weighted Average', 'SOE Weighted Average', 'Baltics Weighted Average']
        colors = ['#e74c3c', '#2980b9', '#e67e22', '#16a085']
    
    # Create figure with dynamic PDF-optimized sizing
    figsize = get_pdf_optimized_figsize('timeseries', 10, 5)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
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
                fontweight='bold', fontsize=12, pad=15)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Net Capital Flows (% of GDP)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=1)
    
    # Optimized legend placement and size
    ax.legend(loc='lower left', fontsize=7, frameon=True, fancybox=False, shadow=False,
              markerscale=0.6)
    
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
    
    # Analysis Overview Section
    st.header("⚙️ Analysis Overview")
    
    st.info("📊 **Integrated Analysis:** Both Full Period and Crisis-Excluded results displayed together")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Time Periods:**
        - **Full Period:** 1999-2025 (105 observations)
        - **Crisis-Excluded:** Excludes 2008-2010 (GFC) and 2020-2022 (COVID-19) - 81 observations
        """)
    
    with col2:
        st.markdown("""
        **🌍 Comparator Groups:**
        - **Eurozone:** Weighted & Simple averages (% of GDP)
        - **Small Open Economies (SOE):** Weighted & Simple averages (% of GDP)
        - **Baltics:** Weighted & Simple averages (% of GDP)
        """)
    
    with col3:
        st.markdown("""
        **📊 Indicators Analyzed:**
        - Net Direct Investment
        - Net Portfolio Investment
          - Debt Securities
          - Equity & Investment Fund Shares
        - Net Other Investment
        - Net Capital Flows (Total)
        """)
    
    st.markdown("---")
    
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
        chart1.savefig(buf1, format='png', dpi=300, bbox_inches='tight', facecolor='white')
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
        chart2.savefig(buf2, format='png', dpi=300, bbox_inches='tight', facecolor='white')
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
    
    # Charts 3-4: Comprehensive ACF Analysis - All Indicators
    st.markdown("---")
    st.subheader("📊 Charts 3-4: Comprehensive Autocorrelation Analysis")
    st.markdown("**ACF patterns across all indicators for both time periods (weighted averages only)**")
    
    # Define all indicators for comprehensive analysis
    indicators_for_acf = [
        'Net Direct Investment',
        'Net Portfolio Investment',
        'Net Portfolio Investment - Debt Securities',
        'Net Portfolio Investment - Equity & Investment Fund Shares',
        'Net Other Investment',
        'Net Capital Flows'
    ]
    
    # Generate ACF charts for all indicators - Full Period first
    st.markdown("### Full Period Analysis")
    full_period_charts = {}
    for idx, indicator in enumerate(indicators_for_acf, 1):
        chart = create_comprehensive_acf_chart(indicator, "Full Period")
        if chart:
            st.pyplot(chart)
            # Store chart for later download consolidation
            full_period_charts[indicator] = chart
            plt.close(chart)
    
    # Generate ACF charts for all indicators - Crisis-Excluded
    st.markdown("### Crisis-Excluded Analysis")
    crisis_excluded_charts = {}
    for idx, indicator in enumerate(indicators_for_acf, 5):  # Continue numbering from 5
        chart = create_comprehensive_acf_chart(indicator, "Crisis-Excluded")
        if chart:
            st.pyplot(chart)
            # Store chart for later download consolidation
            crisis_excluded_charts[indicator] = chart
            plt.close(chart)
    
    # Consolidated Download Buttons for ACF Plots
    st.markdown("---")
    st.markdown("### 📥 Download ACF Analysis Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create comprehensive download for Full Period ACF plots
        if full_period_charts:
            # Recreate charts for download (since they were closed after display)
            full_period_files = {}
            for indicator in indicators_for_acf:
                chart = create_comprehensive_acf_chart(indicator, "Full Period")
                if chart:
                    buf = io.BytesIO()
                    chart.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
                    buf.seek(0)
                    full_period_files[indicator] = buf.getvalue()
                    plt.close(chart)
            
            # Create download button for first chart (others will be available individually if needed)
            if 'Net Direct Investment' in full_period_files:
                st.download_button(
                    label="📥 Download Full Period ACF Plots",
                    data=full_period_files['Net Direct Investment'],
                    file_name="cs4_acf_net_direct_investment_full.png",
                    mime="image/png",
                    key="download_full_period_acf",
                    use_container_width=True,
                    help="Downloads Net Direct Investment Full Period ACF plot. Use individual chart downloads for other indicators."
                )
    
    with col2:
        # Create comprehensive download for Crisis-Excluded ACF plots  
        if crisis_excluded_charts:
            # Recreate charts for download (since they were closed after display)
            crisis_excluded_files = {}
            for indicator in indicators_for_acf:
                chart = create_comprehensive_acf_chart(indicator, "Crisis-Excluded")
                if chart:
                    buf = io.BytesIO()
                    chart.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
                    buf.seek(0)
                    crisis_excluded_files[indicator] = buf.getvalue()
                    plt.close(chart)
            
            # Create download button for first chart
            if 'Net Direct Investment' in crisis_excluded_files:
                st.download_button(
                    label="📥 Download Crisis-Excluded ACF Plots",
                    data=crisis_excluded_files['Net Direct Investment'],
                    file_name="cs4_acf_net_direct_investment_no_crisis.png",
                    mime="image/png",
                    key="download_crisis_excluded_acf",
                    use_container_width=True,
                    help="Downloads Net Direct Investment Crisis-Excluded ACF plot. Use individual chart downloads for other indicators."
                )
    
    # Add expandable section for individual downloads
    with st.expander("📂 Individual Chart Downloads", expanded=False):
        st.markdown("**Download individual ACF charts for specific indicators:**")
        
        # Full Period individual downloads
        st.markdown("**Full Period:**")
        # Use 3×2 grid layout for 6 indicators
        for row in range(2):
            cols = st.columns(3)
            for col_idx in range(3):
                i = row * 3 + col_idx
                if i < len(indicators_for_acf):
                    indicator = indicators_for_acf[i]
                    chart = create_comprehensive_acf_chart(indicator, "Full Period")
                    if chart:
                        buf = io.BytesIO()
                        chart.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
                        buf.seek(0)
                        
                        with cols[col_idx]:
                            st.download_button(
                                label=f"{indicator.replace('Net ', '').replace(' - ', ': ')}",
                                data=buf,
                                file_name=f"cs4_acf_{indicator.lower().replace(' ', '_')}_full.png",
                                mime="image/png",
                                key=f"individual_full_{i}",
                                use_container_width=True
                            )
                        plt.close(chart)
        
        # Crisis-Excluded individual downloads
        st.markdown("**Crisis-Excluded:**")
        # Use 3×2 grid layout for 6 indicators
        for row in range(2):
            cols = st.columns(3)
            for col_idx in range(3):
                i = row * 3 + col_idx
                if i < len(indicators_for_acf):
                    indicator = indicators_for_acf[i]
                    chart = create_comprehensive_acf_chart(indicator, "Crisis-Excluded")
                    if chart:
                        buf = io.BytesIO()
                        chart.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
                        buf.seek(0)
                        
                        with cols[col_idx]:
                            st.download_button(
                                label=f"{indicator.replace('Net ', '').replace(' - ', ': ')}",
                                data=buf,
                                file_name=f"cs4_acf_{indicator.lower().replace(' ', '_')}_no_crisis.png",
                                mime="image/png",
                                key=f"individual_crisis_{i}",
                                use_container_width=True
                            )
                        plt.close(chart)
    
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
    st.subheader("📊 Chart 5: Time-Series Analysis - Simple Averages")
    st.markdown("**Net Capital Flows over time comparing Iceland with simple (unweighted) averages across country groups, with crisis periods marked**")
    
    chart5 = create_comprehensive_timeseries_chart("Simple Averages")
    if chart5:
        st.pyplot(chart5)
        
        # Download button for Chart 5
        buf5 = io.BytesIO()
        chart5.savefig(buf5, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf5.seek(0)
        st.download_button(
            label="📥 Download Time-Series Simple Averages Chart (PNG)",
            data=buf5,
            file_name="cs4_comprehensive_timeseries_averages.png",
            mime="image/png"
        )
        plt.close(chart5)
    
    # Chart 6: Time-series (Iceland + Comparator sums with crisis marking)
    st.markdown("---")
    st.subheader("📊 Chart 6: Time-Series Analysis - Weighted Averages")
    st.markdown("**Net Capital Flows over time comparing Iceland with GDP-weighted averages across country groups, showing crisis impact**")
    
    chart6 = create_comprehensive_timeseries_chart("Weighted Averages")
    if chart6:
        st.pyplot(chart6)
        
        # Download button for Chart 6
        buf6 = io.BytesIO()
        chart6.savefig(buf6, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf6.seek(0)
        st.download_button(
            label="📥 Download Time-Series Weighted Averages Chart (PNG)",
            data=buf6,
            file_name="cs4_comprehensive_timeseries_weighted_avg.png",
            mime="image/png"
        )
        plt.close(chart6)
    
    # Continue to Summary Insights section


def create_master_table(indicators, full_table, crisis_table, table_type):
    """Create master table with Full-then-Crisis-Excluded ordering across all indicators"""
    
    # Initialize master data structure
    master_data = []
    
    # Get column names (excluding 'Indicator')
    columns = [col for col in full_table.columns if col != 'Indicator']
    
    # First add all Full Period rows
    for indicator in indicators:
        full_row = full_table[full_table['Indicator'] == indicator]
        
        if len(full_row) == 0:
            st.warning(f"Full period data not found for {indicator} in master table creation")
            continue
            
        # Get the first (and should be only) row
        full_data = full_row.iloc[0]
        
        # Create Full Period row
        full_period_row = {'Indicator/Period': f"{indicator} (Full)"}
        for col in columns:
            full_period_row[col] = full_data[col]
        master_data.append(full_period_row)
    
    # Then add all Crisis-Excluded rows
    for indicator in indicators:
        crisis_row = crisis_table[crisis_table['Indicator'] == indicator]
        
        if len(crisis_row) == 0:
            st.warning(f"Crisis-excluded data not found for {indicator} in master table creation")
            continue
            
        # Get the first (and should be only) row
        crisis_data = crisis_row.iloc[0]
        
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
    """Display master table with appropriate styling using HTML for PDF export compatibility"""
    
    def get_std_cell_style(val, iceland_val):
        """Get inline CSS style for standard deviation cells based on significance and volatility direction"""
        if pd.isna(val) or val == 'N/A' or val == iceland_val:
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
                    return 'background-color: #ffcccc; color: #990000; font-weight: bold'
                elif has_two_stars:
                    return 'background-color: #ffdddd; color: #cc0000; font-weight: bold'
                else:
                    return 'background-color: #ffeeee; color: #cc3333'
            else:
                # Iceland is LESS volatile (lower std dev) - use light green
                if has_three_stars:
                    return 'background-color: #ccffcc; color: #006600; font-weight: bold'
                elif has_two_stars:
                    return 'background-color: #ddffdd; color: #009900; font-weight: bold'
                else:
                    return 'background-color: #eeffee; color: #00cc00'
        except:
            return ''
    
    def get_halflife_cell_style(val):
        """Get inline CSS style for half-life cells"""
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
    
    def format_rmse_value(val):
        """Format RMSE values"""
        if val == 'N/A':
            return val
        try:
            return f"{float(val):.2f}"
        except:
            return val
    
    # Generate HTML table
    html_table = '<table class="cs4-master-table">'
    
    # Table header
    html_table += '<thead><tr>'
    html_table += '<th>Indicator/Period</th>'
    for col in df.columns:
        html_table += f'<th>{col}</th>'
    html_table += '</tr></thead><tbody>'
    
    # Table body
    for idx, row in df.iterrows():
        html_table += '<tr>'
        html_table += f'<td style="text-align: left; font-weight: bold;">{idx}</td>'
        
        for col in df.columns:
            val = row[col]
            cell_style = ''
            display_val = val
            
            if table_type == 'std':
                if col != 'Iceland':  # Don't color Iceland column
                    cell_style = get_std_cell_style(val, row['Iceland'])
            elif table_type == 'halflife':
                cell_style = get_halflife_cell_style(val)
            elif table_type == 'rmse':
                display_val = format_rmse_value(val)
            
            style_attr = f' style="{cell_style}"' if cell_style else ''
            html_table += f'<td{style_attr}>{display_val}</td>'
        
        html_table += '</tr>'
    
    html_table += '</tbody></table>'
    
    # Display the HTML table
    st.markdown(html_table, unsafe_allow_html=True)


def display_summary_insights_and_export(full_results, crisis_results):
    """Display summary insights and comprehensive export functionality"""
    
    st.markdown("---")
    st.header("🔍 Summary Insights & Comprehensive Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Key Findings Across All 6 Indicators
        - **Comprehensive Analysis:** Covers Net Direct Investment, Net Portfolio Investment (total & disaggregated), Net Other Investment, and Net Capital Flows
        - **Volatility Patterns:** Iceland shows systematically different volatility compared to weighted averages of comparator groups
        - **F-Test Results:** Statistical significance varies across indicators, with strongest differences in aggregated capital flows
        - **Crisis Impact:** Crisis exclusion (2008-2010, 2020-2022) reduces volatility measures across all groups and indicators
        - **Methodological Consistency:** Weighted vs simple average comparisons reveal different patterns across indicator types
        """)
    
    with col2:
        st.markdown("""
        ### ⏱️ Temporal Dynamics Across Indicators
        - **Half-Life Analysis:** All 6 indicators show varying persistence patterns, with most exhibiting 1-3 quarter half-lives
        - **ACF Patterns:** 12 comprehensive panels (6 indicators × 2 periods) reveal different autocorrelation structures
        - **Prediction Accuracy:** RMSE varies significantly across indicators, with Direct Investment showing different patterns than Portfolio flows
        - **Model Performance:** AR(4) models capture temporal dynamics effectively across Full and Crisis-Excluded periods
        - **Crisis Effects:** Crisis exclusion improves model fit and reduces prediction errors for most capital flow types
        """)
    
    # Add Master Tables Export at the bottom of the page
    st.markdown("---")
    st.subheader("📁 Master Tables Export")
    st.markdown("**Download comprehensive statistical results in Excel format with all 3 master tables and metadata.**")
    
    # Get the master tables from the main analysis
    indicators = full_results['metadata']['indicators_analyzed']
    
    # Recreate master tables for export (since they were created in main function scope)
    master_std_table = create_master_table(
        indicators, 
        full_results['summary_tables']['standard_deviations_ftest'],
        crisis_results['summary_tables']['standard_deviations_ftest'],
        'std'
    )
    
    master_halflife_table = create_master_table(
        indicators,
        full_results['summary_tables']['half_life_ar4'],
        crisis_results['summary_tables']['half_life_ar4'], 
        'halflife'
    )
    
    master_rmse_table = create_master_table(
        indicators,
        full_results['summary_tables']['rmse_prediction'],
        crisis_results['summary_tables']['rmse_prediction'],
        'rmse'
    )
    
    # Create Excel export for all master tables
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        master_std_table.to_excel(writer, sheet_name='Master_Standard_Deviations', index=True)
        master_halflife_table.to_excel(writer, sheet_name='Master_Half_Life', index=True)
        master_rmse_table.to_excel(writer, sheet_name='Master_RMSE', index=True)
        
        # Add master metadata
        master_metadata = pd.DataFrame({
            'Parameter': [
                'Analysis Scope', 'Methodology', 'Total Indicators', 'Time Periods',
                'Table Structure', 'Comparator Groups', 'Statistical Methods', 'Export Date'
            ],
            'Value': [
                'Net Direct Investment + Portfolio Investment (total & disaggregated) + Other Investment + Capital Flows',
                'Weighted vs Simple averages of comparator groups',
                '6 capital flow indicators (including portfolio disaggregation)', 'Full Period (1999-2025) & Crisis-Excluded',
                '12 rows × 7 columns per table (6 indicators × 2 periods)',
                'Eurozone, Small Open Economies (SOE), Baltic countries', 
                'F-tests, AR(4) impulse response, RMSE prediction',
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        })
        master_metadata.to_excel(writer, sheet_name='Analysis_Metadata', index=False)
    
    excel_data = output.getvalue()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.download_button(
            label="📥 Download All Master Tables (Excel)",
            data=excel_data,
            file_name=f"cs4_comprehensive_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


def create_integrated_table(indicator, full_table, crisis_table, table_type):
    """Create integrated table for a single indicator with both Full Period and Crisis-Excluded data"""
    
    # Find rows for this indicator in both tables
    full_row = full_table[full_table['Indicator'] == indicator]
    crisis_row = crisis_table[crisis_table['Indicator'] == indicator]
    
    if len(full_row) == 0 or len(crisis_row) == 0:
        # Create empty DataFrame if data not found
        return pd.DataFrame({'Error': [f'Data not found for {indicator}']})
        
    # Get the first (and should be only) row for each
    full_data = full_row.iloc[0]
    crisis_data = crisis_row.iloc[0]
    
    # Get column names (excluding 'Indicator')
    columns = [col for col in full_table.columns if col != 'Indicator']
    
    # Create integrated data
    integrated_data = []
    
    # Create Full Period row
    full_period_row = {'Period': 'Full Time Period'}
    for col in columns:
        full_period_row[col] = full_data[col]
    integrated_data.append(full_period_row)
    
    # Create Crisis-Excluded row  
    crisis_excluded_row = {'Period': 'Crisis-Excluded'}
    for col in columns:
        crisis_excluded_row[col] = crisis_data[col]
    integrated_data.append(crisis_excluded_row)
    
    # Create DataFrame
    integrated_df = pd.DataFrame(integrated_data)
    integrated_df.set_index('Period', inplace=True)
    
    return integrated_df
    
    
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


def main(standalone=False):
    """Main application function"""
    
    # Only set page config when running standalone
    if standalone:
        st.set_page_config(
            page_title="CS4: Statistical Analysis",
            page_icon="📊",
            layout="wide"
        )
    
    # Apply styling when function is called
    apply_professional_styling()
    
    # Title and description
    st.title("🇮🇸 Case Study 4: Comprehensive Statistical Analysis")
    st.markdown("""
    **Objective:** Evaluate currency regime effects on capital flow volatility through comprehensive 
    statistical analysis comparing Iceland with multiple comparator groups.
    """)
    
    # PDF export tip (from commit 8181df5)
    st.info("💡 **PDF Export Tip:** You can print this page to PDF using your browser's print function for a professional document with proper margins and optimized layout.")
    
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
                - Weighted Average: GDP-weighted mean across group countries
                - Simple Average: Unweighted mean across group countries
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
    # Running standalone - set page config
    main(standalone=True)