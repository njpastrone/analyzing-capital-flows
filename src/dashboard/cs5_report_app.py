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
    """Create scatter plot for capital controls vs capital flow volatility"""
    
    if outliers_removed:
        df = data['yearly_sd_no_outliers']
        title = "Capital Flow Volatility vs Capital Controls (Outliers Removed)"
    else:
        df = data['yearly_sd']
        title = "Capital Flow Volatility vs Capital Controls (Yearly Data)"
    
    # Remove NaN values
    df_clean = df.dropna(subset=['yearly_sd_net_capital_flows_pgdp', 'mean_overall_restrictions_index'])
    
    # Calculate correlation
    corr, p_value = stats.pearsonr(
        df_clean['mean_overall_restrictions_index'],
        df_clean['yearly_sd_net_capital_flows_pgdp']
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    scatter = ax.scatter(
        df_clean['mean_overall_restrictions_index'],
        df_clean['yearly_sd_net_capital_flows_pgdp'],
        alpha=0.6,
        s=50,
        color='#3498db',
        edgecolors='#2c3e50',
        linewidth=0.5
    )
    
    # Add trend line
    z = np.polyfit(df_clean['mean_overall_restrictions_index'], 
                   df_clean['yearly_sd_net_capital_flows_pgdp'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df_clean['mean_overall_restrictions_index'].min(),
                         df_clean['mean_overall_restrictions_index'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r-", alpha=0.8, linewidth=2, label='Trend line')
    
    # Labels and title
    ax.set_xlabel('Overall Restrictions Index (Capital Controls)', fontsize=12)
    ax.set_ylabel('Capital Flow Volatility (Std Dev % of GDP)', fontsize=12)
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
    """Create scatter plot for country aggregate capital controls vs capital flow volatility"""
    
    if outliers_removed:
        df = data['country_sd_no_outliers']
        title = "Country Aggregate: Capital Flow Volatility vs Capital Controls (Outliers Removed)"
    else:
        df = data['country_sd']
        title = "Country Aggregate: Capital Flow Volatility vs Capital Controls"
    
    # Remove NaN values
    df_clean = df.dropna(subset=['country_sd_net_capital_flows_pgdp', 'mean_overall_restrictions_index'])
    
    # Calculate correlation
    corr, p_value = stats.pearsonr(
        df_clean['mean_overall_restrictions_index'],
        df_clean['country_sd_net_capital_flows_pgdp']
    )
    
    # Create interactive plotly figure
    fig = px.scatter(
        df_clean,
        x='mean_overall_restrictions_index',
        y='country_sd_net_capital_flows_pgdp',
        hover_data=['COUNTRY'],
        labels={
            'mean_overall_restrictions_index': 'Overall Restrictions Index',
            'country_sd_net_capital_flows_pgdp': 'Capital Flow Volatility (Std Dev % of GDP)',
            'COUNTRY': 'Country'
        },
        title=title
    )
    
    # Add trend line
    fig.add_trace(
        go.Scatter(
            x=df_clean['mean_overall_restrictions_index'],
            y=np.poly1d(np.polyfit(df_clean['mean_overall_restrictions_index'],
                                  df_clean['country_sd_net_capital_flows_pgdp'], 1))(df_clean['mean_overall_restrictions_index']),
            mode='lines',
            name='Trend line',
            line=dict(color='red', width=2)
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


def calculate_regime_statistics(regime_data, indicator='Net Capital Flows', include_crisis=True):
    """Calculate standard deviations and F-test statistics for exchange rate regimes"""
    
    data_key = 'full' if include_crisis else 'no_crises'
    df = regime_data[indicator][data_key]
    
    # Define regime groups
    regime_groups = {
        'Hard Peg': ['hard_peg_pgdp_weighted', 'hard_peg_pgdp_simple'],
        'Crawling/Tight': ['crawl_tight_pgdp_weighted', 'crawl_tight_pgdp_simple'],
        'Managed Float': ['managed_float_pgdp_weighted', 'managed_float_pgdp_simple'],
        'Free Float': ['free_float_pgdp_weighted', 'free_float_pgdp_simple'],
        'Freely Falling': ['freely_falling_pgdp_weighted', 'freely_falling_pgdp_simple'],
        'Dual Market': ['dual_market_pgdp_weighted', 'dual_market_pgdp_simple']
    }
    
    # Include Iceland for comparison
    results = {}
    
    # Calculate Iceland statistics
    iceland_data = df['iceland_pgdp'].dropna()
    iceland_std = np.std(iceland_data, ddof=1)
    results['Iceland'] = {'std': iceland_std, 'n': len(iceland_data)}
    
    # Calculate statistics for each regime
    for regime_name, columns in regime_groups.items():
        regime_results = {}
        for col in columns:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) > 1:
                    std_dev = np.std(series, ddof=1)
                    # F-test against Iceland
                    f_stat = iceland_std**2 / std_dev**2 if std_dev > 0 else np.nan
                    df1 = len(iceland_data) - 1
                    df2 = len(series) - 1
                    p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 
                                     1 - stats.f.cdf(f_stat, df1, df2)) if not np.isnan(f_stat) else np.nan
                    
                    col_type = 'Weighted Avg' if 'weighted' in col else 'Simple Avg'
                    regime_results[col_type] = {
                        'std': std_dev,
                        'n': len(series),
                        'f_stat': f_stat,
                        'p_value': p_value
                    }
        if regime_results:
            results[regime_name] = regime_results
    
    return results


def create_regime_analysis_table(regime_data, indicator='Net Capital Flows'):
    """Create formatted table for exchange rate regime analysis"""
    
    # Get statistics for both full and crisis-excluded periods
    full_stats = calculate_regime_statistics(regime_data, indicator, include_crisis=True)
    crisis_stats = calculate_regime_statistics(regime_data, indicator, include_crisis=False)
    
    # Create DataFrame
    rows = []
    
    # Add header row
    for period_name, stats in [('Full Period', full_stats), ('Crisis-Excluded', crisis_stats)]:
        for regime_name in stats.keys():
            if regime_name == 'Iceland':
                row = {
                    'Regime': f'Iceland ({period_name})',
                    'Std Dev': f"{stats[regime_name]['std']:.4f}",
                    'Weighted Avg': '-',
                    'Simple Avg': '-',
                    'F-test (W)': '-',
                    'F-test (S)': '-'
                }
            else:
                regime_stats = stats[regime_name]
                row = {'Regime': f'{regime_name} ({period_name})'}
                
                # Add weighted average stats
                if 'Weighted Avg' in regime_stats:
                    w_stats = regime_stats['Weighted Avg']
                    row['Weighted Avg'] = f"{w_stats['std']:.4f}"
                    # Add significance stars
                    if w_stats['p_value'] < 0.01:
                        row['F-test (W)'] = '***'
                    elif w_stats['p_value'] < 0.05:
                        row['F-test (W)'] = '**'
                    elif w_stats['p_value'] < 0.10:
                        row['F-test (W)'] = '*'
                    else:
                        row['F-test (W)'] = ''
                else:
                    row['Weighted Avg'] = 'N/A'
                    row['F-test (W)'] = ''
                
                # Add simple average stats
                if 'Simple Avg' in regime_stats:
                    s_stats = regime_stats['Simple Avg']
                    row['Simple Avg'] = f"{s_stats['std']:.4f}"
                    # Add significance stars
                    if s_stats['p_value'] < 0.01:
                        row['F-test (S)'] = '***'
                    elif s_stats['p_value'] < 0.05:
                        row['F-test (S)'] = '**'
                    elif s_stats['p_value'] < 0.10:
                        row['F-test (S)'] = '*'
                    else:
                        row['F-test (S)'] = ''
                else:
                    row['Simple Avg'] = 'N/A'
                    row['F-test (S)'] = ''
                
                row['Std Dev'] = '-'
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


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
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Yearly Analysis", "Country Aggregate Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Yearly Standard Deviations")
            fig1, corr1, p1 = create_capital_controls_scatter(cc_data, outliers_removed=False)
            st.pyplot(fig1)
            st.info(f"**Correlation:** {corr1:.3f} | **P-value:** {p1:.4f}")
        
        with col2:
            st.subheader("📊 Yearly SD (Outliers Removed)")
            fig2, corr2, p2 = create_capital_controls_scatter(cc_data, outliers_removed=True)
            st.pyplot(fig2)
            st.info(f"**Correlation:** {corr2:.3f} | **P-value:** {p2:.4f}")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌍 Country Aggregate SD")
            fig3, corr3, p3 = create_country_aggregate_scatter(cc_data, outliers_removed=False)
            st.plotly_chart(fig3, use_container_width=True)
            st.info(f"**Correlation:** {corr3:.3f} | **P-value:** {p3:.4f}")
        
        with col2:
            st.subheader("🎯 Country Aggregate (Outliers Removed)")
            fig4, corr4, p4 = create_country_aggregate_scatter(cc_data, outliers_removed=True)
            st.plotly_chart(fig4, use_container_width=True)
            st.info(f"**Correlation:** {corr4:.3f} | **P-value:** {p4:.4f}")
    
    # Interpretation
    st.markdown("---")
    st.markdown("""
    ### 📊 Capital Controls Analysis Interpretation
    
    - **Positive Correlation:** Higher capital controls (restrictions) are associated with higher volatility
    - **Statistical Significance:** P-values indicate the strength of the relationship
    - **Country Heterogeneity:** Individual country effects visible in aggregate analysis
    - **Outlier Impact:** Removing outliers affects correlation strength
    
    **Note:** Correlation does not imply causation. Countries may implement controls in response to volatility.
    """)
    
    st.markdown("---")
    
    # Section 3: Exchange Rate Regime Analysis
    st.header("💱 Section 3: Exchange Rate Regime Analysis")
    st.markdown("**Structure:** Standard deviations and F-tests by exchange rate regime (similar to CS4 Table 1)")
    
    # Load regime data
    with st.spinner("Loading exchange rate regime data..."):
        regime_data = load_regime_analysis_data()
    
    # Indicator selection
    indicator = st.selectbox(
        "Select Indicator for Analysis:",
        ['Net Capital Flows', 'Net Direct Investment', 'Net Portfolio Investment', 'Net Other Investment']
    )
    
    # Create analysis table
    regime_table = create_regime_analysis_table(regime_data, indicator)
    
    # Display table
    st.subheader(f"📊 Table: {indicator} Volatility by Exchange Rate Regime")
    st.dataframe(regime_table, use_container_width=True)
    
    # Add interpretation box
    st.info("""
    **F-test Interpretation:**
    - *** : p < 0.01 (highly significant difference from Iceland)
    - ** : p < 0.05 (significant difference)
    - * : p < 0.10 (marginally significant)
    - Empty: No significant difference
    
    **Note:** F-test compares variance of each regime group against Iceland
    """)
    
    # Summary statistics
    st.markdown("---")
    st.subheader("📈 Summary Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Exchange Rate Regimes", "6")
        st.caption("Different regime classifications analyzed")
    
    with col2:
        st.metric("Time Periods", "2")
        st.caption("Full Period & Crisis-Excluded")
    
    with col3:
        st.metric("Statistical Tests", "F-tests")
        st.caption("Variance equality testing")
    
    # Export functionality
    st.markdown("---")
    st.subheader("📥 Export Results")
    
    # Create download button for regime table
    csv = regime_table.to_csv(index=False)
    st.download_button(
        label="📥 Download Regime Analysis Table (CSV)",
        data=csv,
        file_name=f"cs5_regime_analysis_{indicator.lower().replace(' ', '_')}.csv",
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