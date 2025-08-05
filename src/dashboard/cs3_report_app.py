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
    
    # Load data  
    final_data, analysis_indicators = load_case_study_3_data()
    
    if final_data is None or analysis_indicators is None:
        st.error("❌ Failed to load Case Study 3 data")
        return
    
    # Filter data for CS3 analysis
    cs3_data = final_data[final_data['CS3_GROUP'].notna()].copy()
    
    if len(cs3_data) == 0:
        st.error("❌ No CS3 data available")
        return
    
    # Create GROUP column for compatibility with CS1 functions
    cs3_data['GROUP'] = cs3_data['CS3_GROUP'].map({
        'Iceland': 'Iceland',
        'Comparator': 'Small Open Economies'
    })
    
    # Overall Capital Flows Analysis
    st.subheader("📈 Overall Capital Flows Analysis")
    
    try:
        # Import CS1 analysis functions and adapt them
        from simple_report_app import (
            calculate_group_statistics, 
            perform_variance_tests,
            create_plot_base64
        )
        
        # Calculate statistics (adapting CS1 methodology)
        group_stats = calculate_group_statistics(cs3_data, analysis_indicators, "GROUP")
        test_results = perform_variance_tests(cs3_data, analysis_indicators, "GROUP")
        
        # Display results using CS1 structure
        display_cs3_overall_analysis(cs3_data, group_stats, test_results, analysis_indicators, context)
        
    except ImportError:
        st.info("🚧 **CS3 Overall Analysis Implementation**\n\nCS3 overall analysis functions are being implemented to match CS1 structure.")
    
    # Disaggregated Analysis (Sections 1-6)
    st.subheader("🔍 Disaggregated Analysis")
    
    st.info("🚧 **CS3 Disaggregated Analysis Implementation**\n\nCS3 disaggregated analysis (Sections 1-6) will replicate all CS1 functionality with Iceland vs Small Open Economies grouping.")

def case_study_3_main_crisis_excluded(context="standalone"):
    """CS3 crisis-excluded analysis function - exact replica of CS1 structure"""
    
    # Load data with crisis exclusion
    final_data, analysis_indicators = load_case_study_3_data_crisis_excluded()
    
    if final_data is None or analysis_indicators is None:
        st.error("❌ Failed to load Case Study 3 crisis-excluded data")
        return
    
    st.info("🚧 **CS3 Crisis-Excluded Analysis Implementation**\n\nCS3 crisis-excluded analysis will replicate all CS1 functionality excluding Global Financial Crisis (2008-2010) and COVID-19 (2020-2022) periods.")

def load_case_study_3_data_crisis_excluded():
    """Load Case Study 3 data with crisis periods excluded"""
    
    # Load base data
    final_data, analysis_indicators = load_case_study_3_data()
    
    if final_data is None:
        return None, None
    
    # Exclude crisis periods (matching CS1 methodology)
    # Global Financial Crisis: 2008-2010
    # COVID-19: 2020-2022
    crisis_years = [2008, 2009, 2010, 2020, 2021, 2022]
    
    crisis_excluded_data = final_data[~final_data['YEAR'].isin(crisis_years)].copy()
    
    if len(crisis_excluded_data) == 0:
        st.error("❌ No data remaining after crisis exclusion")
        return None, None
    
    st.success(f"✅ Crisis-excluded CS3 data: {len(crisis_excluded_data)} observations (excluded {len(final_data) - len(crisis_excluded_data)} crisis period observations)")
    
    return crisis_excluded_data, analysis_indicators

def display_cs3_overall_analysis(data, group_stats, test_results, indicators, context):
    """Display CS3 overall analysis results - adapted from CS1 structure"""
    
    # Summary statistics display
    if group_stats is not None and len(group_stats) > 0:
        
        # Group the statistics by indicator for display
        iceland_stats = group_stats[group_stats['GROUP'] == 'Iceland']
        soe_stats = group_stats[group_stats['GROUP'] == 'Small Open Economies']
        
        st.markdown("**📊 Summary Statistics**")
        
        # Create comparison table (matching CS1 structure)
        if len(iceland_stats) > 0 and len(soe_stats) > 0:
            
            # Display key statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Iceland Indicators", len(iceland_stats))
            
            with col2:
                st.metric("Small Open Economies Indicators", len(soe_stats))
            
            with col3:
                st.metric("Total Comparisons", len(test_results) if test_results is not None else 0)
        
        # Test results summary
        if test_results is not None and len(test_results) > 0:
            
            # Calculate summary statistics (matching CS1 methodology)
            total_tests = len(test_results)
            significant_5pct = test_results['Significant_5pct'].sum() if 'Significant_5pct' in test_results.columns else 0
            significant_1pct = test_results['Significant_1pct'].sum() if 'Significant_1pct' in test_results.columns else 0
            iceland_higher = test_results['Iceland_Higher_Volatility'].sum() if 'Iceland_Higher_Volatility' in test_results.columns else 0
            
            st.markdown("**🔬 Hypothesis Test Results**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tests", total_tests)
            
            with col2:
                st.metric("Significant (5%)", f"{significant_5pct}/{total_tests}")
            
            with col3:
                st.metric("Significant (1%)", f"{significant_1pct}/{total_tests}")
            
            with col4:
                st.metric("Iceland Higher Volatility", f"{iceland_higher}/{total_tests}")
    
    else:
        st.warning("⚠️ No statistical results available for display")

if __name__ == "__main__":
    main()