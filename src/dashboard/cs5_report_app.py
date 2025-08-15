"""
Case Study 5: Capital Controls and Exchange Rate Regime Analysis Report Application

Professional dashboard for CS5 analysis examining the relationship between financial openness,
capital controls, exchange rate regimes, and capital flow volatility using external data sources.
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
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Colorblind-friendly econometrics palette (matching CS1-CS4)
COLORBLIND_SAFE = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#FBAFE4']
sns.set_palette(COLORBLIND_SAFE)

# Configure matplotlib for professional PDF export
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

# Configure page
st.set_page_config(
    page_title="CS5: Capital Controls & Exchange Rate Regime Analysis",
    page_icon="🌐",
    layout="wide"
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
    
    /* Chart container constraints */
    .chart-container { 
        max-width: 100% !important;
        overflow: hidden !important;
        text-align: center !important;
        margin: 20px 0 !important;
    }
    
    /* Main container constraints */
    .main .block-container {
        max-width: 8.5in !important;
        margin: 0 auto !important;
        padding: 1rem 2rem !important;
    }
    
    /* CS4 Master Table Styling (optimized for 13 columns) */
    .cs4-master-table { 
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 20px 0 !important;
        font-size: 9px !important;  /* Smaller font for 13 columns */
        font-family: 'Arial', sans-serif !important;
        table-layout: fixed !important;  /* Fixed layout for better column control */
    }
    .cs4-master-table th {
        background-color: #e6f3ff !important;
        font-weight: bold !important;
        text-align: center !important;
        padding: 6px 4px !important;  /* Reduced padding for more columns */
        border: 1px solid #ddd !important;
        font-size: 8px !important;  /* Smaller headers */
        word-wrap: break-word !important;
    }
    .cs4-master-table td {
        text-align: center !important;
        padding: 4px 2px !important;  /* Reduced padding */
        border: 1px solid #ddd !important;
        font-size: 8px !important;  /* Smaller data font */
        word-wrap: break-word !important;
    }
    .cs4-master-table tbody tr:nth-child(even) {
        background-color: #f9f9f9 !important;
    }
    /* First column (Indicator/Period) wider */
    .cs4-master-table td:first-child {
        width: 20% !important;
        text-align: left !important;
        font-weight: bold !important;
        padding-left: 6px !important;
    }
    .cs4-master-table th:first-child {
        width: 20% !important;
        text-align: center !important;
    }
    
    /* Print Media Queries for PDF Export */
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
        .cs4-master-table { 
            page-break-inside: avoid !important;
            font-size: 6px !important;  /* Even smaller for PDF with 13 columns */
            margin: 10px 0 !important;
            table-layout: fixed !important;
        }
        .cs4-master-table th, .cs4-master-table td {
            padding: 2px 1px !important;  /* Minimal padding for PDF */
            font-size: 6px !important;
        }
    }
</style>
""", unsafe_allow_html=True)


def load_capital_controls_data():
    """Load capital controls analysis data"""
    base_path = Path(__file__).parent.parent.parent / "updated_data" / "Clean" / "CS5_Capital_Controls"
    
    # Load yearly standard deviations
    yearly_sd = pd.read_csv(base_path / "sd_yearly_flows.csv")
    yearly_sd_no_outliers = pd.read_csv(base_path / "sd_yearly_flows_no_outliers.csv")
    
    # Load country aggregate standard deviations
    country_sd = pd.read_csv(base_path / "sd_country_flows.csv")
    country_sd_no_outliers = pd.read_csv(base_path / "sd_country_flows_no_outliers.csv")
    
    return {
        'yearly_sd': yearly_sd,
        'yearly_sd_no_outliers': yearly_sd_no_outliers,
        'country_sd': country_sd,
        'country_sd_no_outliers': country_sd_no_outliers
    }


def load_regime_analysis_data():
    """Load exchange rate regime analysis data"""
    base_path = Path(__file__).parent.parent.parent / "updated_data" / "Clean" / "CS5_Regime_Analysis"
    
    indicators = {
        'Net Capital Flows': 'net_capital_flows',
        'Net Direct Investment': 'net_direct_investment',
        'Net Portfolio Investment': 'net_portfolio_investment',
        'Net Other Investment': 'net_other_investment'
    }
    
    data = {}
    for indicator_name, file_prefix in indicators.items():
        data[indicator_name] = {
            'full': pd.read_csv(base_path / f"{file_prefix}_full.csv"),
            'no_crises': pd.read_csv(base_path / f"{file_prefix}_no_crises.csv")
        }
    
    return data


def create_capital_controls_scatter(data, outliers_removed=False):
    """Create scatter plot for capital controls vs capital flow volatility (axes switched to match R charts)"""
    
    if outliers_removed:
        df = data['yearly_sd_no_outliers']
        title = "Capital Controls vs Capital Flow Volatility (Outliers Removed)"
    else:
        df = data['yearly_sd']
        title = "Capital Controls vs Capital Flow Volatility (Yearly Data)"
    
    # Remove NaN values
    df_clean = df.dropna(subset=['yearly_sd_net_capital_flows_pgdp', 'mean_overall_restrictions_index'])
    
    # Calculate correlation (same correlation, just switched axes for display)
    corr, p_value = stats.pearsonr(
        df_clean['yearly_sd_net_capital_flows_pgdp'],
        df_clean['mean_overall_restrictions_index']
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot (axes switched: volatility on X, controls on Y)
    scatter = ax.scatter(
        df_clean['yearly_sd_net_capital_flows_pgdp'],
        df_clean['mean_overall_restrictions_index'],
        alpha=0.6,
        s=50,
        color=COLORBLIND_SAFE[0],  # Use consistent color scheme
        edgecolors=COLORBLIND_SAFE[1],
        linewidth=0.5
    )
    
    # Add trend line (axes switched)
    z = np.polyfit(df_clean['yearly_sd_net_capital_flows_pgdp'], 
                   df_clean['mean_overall_restrictions_index'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df_clean['yearly_sd_net_capital_flows_pgdp'].min(),
                         df_clean['yearly_sd_net_capital_flows_pgdp'].max(), 100)
    ax.plot(x_trend, p(x_trend), color=COLORBLIND_SAFE[2], alpha=0.8, linewidth=2, label='Trend line')
    
    # Labels and title (axes switched)
    ax.set_xlabel('Capital Flow Volatility (Std Dev % of GDP)', fontsize=12)
    ax.set_ylabel('Overall Restrictions Index (Capital Controls)', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)
    
    # Add correlation info
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}\nP-value: {p_value:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Grid
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig, corr, p_value


def create_country_aggregate_scatter(data, outliers_removed=False):
    """Create scatter plot for country aggregate capital controls vs capital flow volatility (axes switched)"""
    
    if outliers_removed:
        df = data['country_sd_no_outliers']
        title = "Country Aggregate: Capital Controls vs Capital Flow Volatility (Outliers Removed)"
    else:
        df = data['country_sd']
        title = "Country Aggregate: Capital Controls vs Capital Flow Volatility"
    
    # Remove NaN values
    df_clean = df.dropna(subset=['country_sd_net_capital_flows_pgdp', 'mean_overall_restrictions_index'])
    
    # Calculate correlation (same correlation, just switched axes for display)
    corr, p_value = stats.pearsonr(
        df_clean['country_sd_net_capital_flows_pgdp'],
        df_clean['mean_overall_restrictions_index']
    )
    
    # Create interactive plotly figure (axes switched)
    fig = px.scatter(
        df_clean,
        x='country_sd_net_capital_flows_pgdp',
        y='mean_overall_restrictions_index',
        hover_data=['COUNTRY'],
        labels={
            'country_sd_net_capital_flows_pgdp': 'Capital Flow Volatility (Std Dev % of GDP)',
            'mean_overall_restrictions_index': 'Overall Restrictions Index',
            'COUNTRY': 'Country'
        },
        title=title,
        color_discrete_sequence=[COLORBLIND_SAFE[0]]  # Use consistent color scheme
    )
    
    # Add trend line (axes switched)
    fig.add_trace(
        go.Scatter(
            x=df_clean['country_sd_net_capital_flows_pgdp'],
            y=np.poly1d(np.polyfit(df_clean['country_sd_net_capital_flows_pgdp'],
                                  df_clean['mean_overall_restrictions_index'], 1))(df_clean['country_sd_net_capital_flows_pgdp']),
            mode='lines',
            name='Trend line',
            line=dict(color=COLORBLIND_SAFE[2], width=2)
        )
    )
    
    # Add correlation annotation
    fig.add_annotation(
        text=f"Correlation: {corr:.3f}<br>P-value: {p_value:.4f}",
        xref="paper", yref="paper",
        x=0.05, y=0.95,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12)
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig, corr, p_value


def create_cs5_master_table(regime_data, indicators):
    """Create master table exactly matching CS4 Table 1 structure for exchange rate regimes"""
    
    # Initialize master data structure  
    master_data = []
    
    # Process each indicator for both Full and Crisis-Excluded periods
    for indicator in indicators:
        for period_name, include_crisis in [('Full', True), ('Crisis-Excluded', False)]:
            # Get regime data for this indicator and period
            data_key = 'full' if include_crisis else 'no_crises'
            df = regime_data[indicator][data_key]
            
            # Calculate Iceland standard deviation
            iceland_data = df['iceland_pgdp'].dropna()
            iceland_std = np.std(iceland_data, ddof=1) if len(iceland_data) > 1 else np.nan
            
            # Initialize row with indicator/period
            row = {'Indicator/Period': f"{indicator} ({period_name})"}
            row['Iceland'] = f"{iceland_std:.4f}" if not np.isnan(iceland_std) else 'N/A'
            
            # Define regime groups exactly as in original data (ALL 6 REGIMES)
            regime_groups = {
                'Hard Peg Weighted Avg': 'hard_peg_pgdp_weighted',
                'Hard Peg Simple Avg': 'hard_peg_pgdp_simple', 
                'Crawling/Tight Weighted Avg': 'crawl_tight_pgdp_weighted',
                'Crawling/Tight Simple Avg': 'crawl_tight_pgdp_simple',
                'Managed Float Weighted Avg': 'managed_float_pgdp_weighted',
                'Managed Float Simple Avg': 'managed_float_pgdp_simple',
                'Free Float Weighted Avg': 'free_float_pgdp_weighted',
                'Free Float Simple Avg': 'free_float_pgdp_simple',
                'Freely Falling Weighted Avg': 'freely_falling_pgdp_weighted',
                'Freely Falling Simple Avg': 'freely_falling_pgdp_simple',
                'Dual Market Weighted Avg': 'dual_market_pgdp_weighted',
                'Dual Market Simple Avg': 'dual_market_pgdp_simple'
            }
            
            # Calculate statistics for each regime group
            for col_name, data_col in regime_groups.items():
                if data_col in df.columns:
                    series = df[data_col].dropna()
                    if len(series) > 1:
                        regime_std = np.std(series, ddof=1)
                        
                        # F-test against Iceland
                        if not np.isnan(iceland_std) and iceland_std > 0 and regime_std > 0:
                            f_stat = iceland_std**2 / regime_std**2
                            df1 = len(iceland_data) - 1
                            df2 = len(series) - 1
                            p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 
                                             1 - stats.f.cdf(f_stat, df1, df2))
                            
                            # Format value with significance stars
                            value = f"{regime_std:.4f}"
                            if p_value < 0.01:
                                value += "***"
                            elif p_value < 0.05:
                                value += "**"
                            elif p_value < 0.10:
                                value += "*"
                        else:
                            value = f"{regime_std:.4f}" if not np.isnan(regime_std) else 'N/A'
                    else:
                        value = 'N/A'
                else:
                    value = 'N/A'
                
                row[col_name] = value
            
            master_data.append(row)
    
    # Create DataFrame with exact CS4 structure
    master_df = pd.DataFrame(master_data)
    master_df.set_index('Indicator/Period', inplace=True)
    
    return master_df


def display_cs5_master_table(df):
    """Display CS5 master table with exact CS4 styling and color coding"""
    
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
    
    # Generate HTML table exactly like CS4
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
            
            if col != 'Iceland':  # Don't color Iceland column
                cell_style = get_std_cell_style(val, row['Iceland'])
            
            style_attr = f' style="{cell_style}"' if cell_style else ''
            html_table += f'<td{style_attr}>{val}</td>'
        
        html_table += '</tr>'
    
    html_table += '</tbody></table>'
    
    # Display with exact CS4 styling
    st.markdown(html_table, unsafe_allow_html=True)


def run_cs5_analysis():
    """Main function to run CS5 analysis"""
    
    st.title("🌐 Case Study 5: Capital Controls and Exchange Rate Regime Analysis")
    st.markdown("---")
    
    # Section 1: Analysis Overview
    st.header("📋 Section 1: Analysis Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 External Data Sources
        
        **Capital Controls Data:**
        - **Source:** Fernández et al. (2016) Capital Control Measures Database
        - **Metric:** Overall Restrictions Index (0-1 scale)
        - **Coverage:** Multiple countries, annual frequency
        - **Processing:** R script: "Testing Correlation - Financial Openness and Capital Flow Variation.qmd"
        
        **Exchange Rate Regime Data:**
        - **Source:** Ilzetzki, Reinhart, and Rogoff (2019) Classification
        - **Categories:** Hard Peg, Crawling/Tight, Managed Float, Free Float, Freely Falling, Dual Market
        - **Processing:** R script: "Analyzing Data by Currency Regime.qmd"
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Analytical Approach
        
        **Methodology:**
        1. **Capital Controls Analysis:** Examine correlation between financial openness and capital flow volatility
        2. **Regime Analysis:** Compare volatility across different exchange rate regimes
        3. **Statistical Testing:** Apply F-tests for variance equality (similar to CS4 methodology)
        
        **Key Questions:**
        - Do capital controls reduce capital flow volatility?
        - Which exchange rate regime provides most stability?
        - How does Iceland compare to different regime groups?
        """)
    
    st.markdown("---")
    
    # Section 2: Capital Controls Analysis
    st.header("🔒 Section 2: Capital Controls Analysis")
    st.markdown("**Objective:** Examine the relationship between capital controls (Overall Restrictions Index) and capital flow volatility")
    
    # Load capital controls data
    with st.spinner("Loading capital controls data..."):
        cc_data = load_capital_controls_data()
    
    # Yearly Analysis (Sequential presentation - no tabs)
    st.subheader("📈 Yearly Standard Deviations Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Full Dataset**")
        fig1, corr1, p1 = create_capital_controls_scatter(cc_data, outliers_removed=False)
        st.pyplot(fig1)
        st.info(f"**Correlation:** {corr1:.3f} | **P-value:** {p1:.4f}")
    
    with col2:
        st.markdown("**Outliers Removed**")
        fig2, corr2, p2 = create_capital_controls_scatter(cc_data, outliers_removed=True)
        st.pyplot(fig2)
        st.info(f"**Correlation:** {corr2:.3f} | **P-value:** {p2:.4f}")
    
    st.markdown("---")
    
    # Country Aggregate Analysis
    st.subheader("🌍 Country Aggregate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Full Dataset**")
        fig3, corr3, p3 = create_country_aggregate_scatter(cc_data, outliers_removed=False)
        st.plotly_chart(fig3, use_container_width=True)
        st.info(f"**Correlation:** {corr3:.3f} | **P-value:** {p3:.4f}")
    
    with col2:
        st.markdown("**Outliers Removed**")
        fig4, corr4, p4 = create_country_aggregate_scatter(cc_data, outliers_removed=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.info(f"**Correlation:** {corr4:.3f} | **P-value:** {p4:.4f}")
    
    # Interpretation (Updated based on actual findings)
    st.markdown("---")
    st.markdown(f"""
    ### 📊 Capital Controls Analysis Interpretation
    
    **Key Findings:**
    - **Yearly Analysis:** Correlation = {corr1:.3f} (p = {p1:.4f}) | Outliers Removed: {corr2:.3f} (p = {p2:.4f})
    - **Country Aggregate:** Correlation = {corr3:.3f} (p = {p3:.4f}) | Outliers Removed: {corr4:.3f} (p = {p4:.4f})
    
    **Statistical Interpretation:**
    - {'**Significant**' if min(p1, p2, p3, p4) < 0.05 else '**Not significant**'} relationship between capital controls and volatility at 5% level
    - {'Negative' if corr1 < 0 else 'Positive'} correlation suggests that {'higher' if corr1 > 0 else 'lower'} capital controls are associated with {'higher' if corr1 > 0 else 'lower'} volatility
    - Outlier removal {'strengthens' if abs(corr2) > abs(corr1) else 'weakens'} the relationship in yearly data
    - Country-level aggregation {'confirms' if (corr1 > 0 and corr3 > 0) or (corr1 < 0 and corr3 < 0) else 'reverses'} the yearly pattern
    
    **Policy Implications:**
    - Results suggest capital controls may {'increase' if corr1 > 0 else 'decrease'} rather than {'decrease' if corr1 > 0 else 'increase'} volatility
    - Heterogeneity across countries indicates regime-specific effectiveness
    - Endogeneity concerns: countries may implement controls in response to volatility
    """)
    
    st.markdown("---")
    
    # Section 3: Exchange Rate Regime Analysis
    st.header("💱 Section 3: Exchange Rate Regime Analysis")
    st.markdown("**Structure:** Standard deviations and F-tests by exchange rate regime (EXACT CS4 Table 1 replication)")
    
    # Load regime data
    with st.spinner("Loading exchange rate regime data..."):
        regime_data = load_regime_analysis_data()
    
    # Create master table exactly like CS4 Table 1
    st.subheader("🎯 Table 1: Standard Deviation & F-test Results (All Indicators)")
    
    indicators = ['Net Capital Flows', 'Net Direct Investment', 'Net Portfolio Investment', 'Net Other Investment']
    
    # Create and display master table (exact CS4 format)
    master_regime_table = create_cs5_master_table(regime_data, indicators)
    display_cs5_master_table(master_regime_table)
    
    # Add interpretation box (exact CS4 format)
    st.info("""
    **Interpretation:** Standard deviations measure volatility levels. Stars indicate F-test significance 
    for variance differences from Iceland: *** p<0.01, ** p<0.05, * p<0.10.
    
    **Color Coding:**
    - 🔴 **Red/Pink Background**: Iceland is MORE volatile than regime group (higher standard deviation)
    - 🟢 **Green Background**: Iceland is LESS volatile than regime group (lower standard deviation)  
    - ⚪ **No Color**: No statistically significant difference
    """)
    
    # Summary statistics
    st.markdown("---")
    st.subheader("📈 Summary Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Exchange Rate Regimes", "6")
        st.caption("Complete regime coverage: Hard Peg, Crawling/Tight, Managed Float, Free Float, Freely Falling, Dual Market")
    
    with col2:
        st.metric("Time Periods", "2")
        st.caption("Full Period & Crisis-Excluded")
    
    with col3:
        st.metric("Statistical Tests", "F-tests")
        st.caption("Variance equality testing (CS4 methodology)")
    
    # Export functionality
    st.markdown("---")
    st.subheader("📥 Export Results")
    
    # Download button for master table (exact CS4 format)
    csv_master = master_regime_table.to_csv(index=True)
    st.download_button(
        label="📥 Download Master Exchange Rate Regime Table (CSV)",
        data=csv_master,
        file_name="cs5_master_regime_analysis.csv",
        mime="text/csv"
    )


# Main execution
def main():
    """Main function with tabs"""
    tab1, tab2 = st.tabs(["📊 Analysis", "📚 Methodology"])
    
    with tab1:
        run_cs5_analysis()
    
    with tab2:
        st.header("📚 Methodology Documentation")
        st.markdown("""
        ## Data Sources and Cleaning
        
        ### Capital Controls Data
        - **Original Source:** Fernández, Klein, Rebucci, Schindler, and Uribe (2016)
        - **Database:** Capital Control Measures: A New Dataset
        - **Cleaning Process:** R script processes raw data to calculate yearly and country-level standard deviations
        - **Key Variable:** Overall Restrictions Index (0 = no controls, 1 = full controls)
        
        ### Exchange Rate Regime Classification
        - **Original Source:** Ilzetzki, Reinhart, and Rogoff (2019)
        - **Classification:** Fine classification with 6 main categories
        - **Aggregation:** Weighted and simple averages across countries in each regime
        
        ### Statistical Methodology
        
        **F-test for Variance Equality:**
        - Null Hypothesis: σ²(Iceland) = σ²(Regime Group)
        - Alternative: σ²(Iceland) ≠ σ²(Regime Group)
        - Two-tailed test with significance levels: 0.01, 0.05, 0.10
        
        **Correlation Analysis:**
        - Pearson correlation coefficient
        - P-value for statistical significance
        - Both with and without outliers for robustness
        
        ### Data Processing Pipeline
        1. **R Scripts:** Initial data cleaning and aggregation
        2. **Python Analysis:** Statistical tests and visualization
        3. **Quality Checks:** Outlier detection and removal
        4. **Validation:** Cross-verification with original sources
        """)


if __name__ == "__main__":
    main()