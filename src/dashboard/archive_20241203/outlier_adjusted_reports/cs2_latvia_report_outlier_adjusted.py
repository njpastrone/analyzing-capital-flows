"""
Outlier-Adjusted Capital Flows Analysis - Case Study 2: Latvia Euro Adoption Report

This Streamlit application provides an exact mirror of the Latvia tab using winsorized data,
optimized for clean PDF export with professional formatting.

Research Focus: Latvia Euro Adoption Analysis - Capital Flow Volatility Before and After Euro Adoption (2014)
Using 5% symmetric winsorization for robust statistical analysis
"""

import streamlit as st
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

# Add parent directory to path for imports
# Add parent directory (main dashboard) and current directory (outlier_adjusted_reports) to path
current_file_dir = Path(__file__).parent.absolute()
parent_dir = current_file_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(current_file_dir) not in sys.path:
    sys.path.insert(0, str(current_file_dir))

warnings.filterwarnings('ignore')

# Configure matplotlib for PDF export optimization (matching cs1_report_app.py)
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

def main():
    """Latvia report app - exact mirror of main dashboard Latvia tab, optimized for PDF export"""
    
    # Note: st.set_page_config() is now handled by main_app.py (matching cs1_report_app.py)
    # Removing page config call to prevent margin/layout conflicts that affect PDF export
    
    # Mirror the exact content from show_case_study_2_latvia_restructured()
    st.title("🛡️ 🇱🇻 Latvia Euro Adoption Analysis (Outlier-Adjusted)")
    st.subheader("Capital Flow Volatility Before and After Euro Adoption (2014)")
    
    st.markdown("""
    **Research Focus:** How did Euro adoption affect Latvia's capital flow volatility?
    
    **Methodology:** Temporal comparison of capital flow patterns before (2007-2012) and after (2015-2020) Euro adoption.
    
    **📋 Outlier-Adjusted Analysis:** This analysis uses 5% symmetric winsorization to assess
    the robustness of statistical findings to extreme values. Data points below the 5th percentile
    are replaced with the 5th percentile value; data points above the 95th percentile are replaced
    with the 95th percentile value.
    
    **Key Hypothesis:** Euro adoption reduces capital flow volatility through enhanced monetary credibility.
    """)
    
    # Data and Methodology section (matching cs1_report_app.py format)
    with st.expander("📋 Data and Methodology", expanded=False):
        st.markdown("""
        ### Data Sources
        - **Balance of Payments Data:** IMF, quarterly frequency (1999-2025)
        - **GDP Data:** IMF World Economic Outlook, annual frequency
        - **Country:** Latvia, Republic of
        
        ### Methodology
        1. **Data Normalization:** All BOP flows converted to annualized % of GDP
        2. **Statistical Analysis:** Comprehensive descriptive statistics and F-tests
        3. **Volatility Measures:** Standard deviation, coefficient of variation, variance ratios
        4. **Temporal Comparison:** Pre-Euro vs Post-Euro period analysis
        
        ### Euro Adoption Timeline
        - **Euro Adoption Date:** January 1, 2014
        - **Pre-Euro Period:** 1999-2013 (full series) 
        - **Post-Euro Period:** 2014-2025 (full series)
        - **Crisis Exclusion:** Global Financial Crisis (2008-2010), Latvian Banking Crisis (2011-2012), and COVID-19 (2020-2022)
        """)
    
    
    # Add PDF-specific CSS styling (matching cs1_report_app.py margins)
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
    
    # Full Time Period Section
    st.markdown("---")
    st.header("📊 Full Time Period Analysis")
    st.markdown("*Complete temporal analysis using all available data*")
    
    # Overall Capital Flows Analysis
    st.subheader("📈 Overall Capital Flows Analysis")
    show_latvia_overall_analysis(include_crisis_years=True)
    
    # Indicator-Level Analysis  
    st.subheader("🔍 Indicator-Level Analysis")
    show_latvia_indicator_analysis(include_crisis_years=True)
    
    # Crisis-Excluded Section
    st.markdown("---")
    st.header("🚫 Excluding Financial Crises")
    st.markdown("*Analysis excluding Global Financial Crisis (2008-2010), Latvian Banking Crisis (2011-2012), and COVID-19 (2020-2022) periods*")
    
    # Overall Capital Flows Analysis - Crisis Excluded
    st.subheader("📈 Overall Capital Flows Analysis")
    show_latvia_overall_analysis(include_crisis_years=False)
    
    # Indicator-Level Analysis - Crisis Excluded
    st.subheader("🔍 Indicator-Level Analysis") 
    show_latvia_indicator_analysis(include_crisis_years=False)

def show_latvia_overall_analysis(include_crisis_years=True):
    """Show Latvia overall capital flows analysis with PDF-optimized formatting"""
    try:
        from case_study_2_euro_adoption_outlier_adjusted import show_overall_capital_flows_analysis_cs2
        import matplotlib.pyplot as plt
        
        # Apply PDF optimization context with figure size constraints
        with plt.style.context('default'):
            # Store original settings
            original_rcParams = plt.rcParams.copy()
            
            # Apply PDF-optimized settings matching cs1_report_app.py
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
                'savefig.facecolor': 'white',
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'figure.max_open_warning': 0
            })
            
            # Monkey-patch plt.subplots to enforce PDF-friendly figure sizes
            original_subplots = plt.subplots
            def pdf_optimized_subplots(*args, **kwargs):
                if 'figsize' in kwargs:
                    width, height = kwargs['figsize']
                    # Constrain to PDF-friendly sizes (matching cs1_report_app.py max)
                    if width > 16:
                        kwargs['figsize'] = (16, min(height, 12))
                    elif height > 12:
                        kwargs['figsize'] = (width, 12)
                return original_subplots(*args, **kwargs)
            
            plt.subplots = pdf_optimized_subplots
            
            # Monkey-patch axis set_title to shorten long titles for PDF
            import matplotlib.axes
            original_set_title = matplotlib.axes.Axes.set_title
            def pdf_optimized_set_title(self, label, *args, **kwargs):
                # Shorten common long title patterns
                if isinstance(label, str):
                    label = label.replace('Across All Capital Flow Indicators', 'All Indicators')
                    label = label.replace('Distribution of Standard Deviations', 'Distribution of Std Deviations')
                    label = label.replace('Individual Eurozone Countries', 'Eurozone Countries')
                    label = label.replace('(Ordered by Descending Median Value)', '(By Descending Median)')
                    # Limit title length for PDF compatibility
                    if len(label) > 80:
                        lines = label.split('\n')
                        if len(lines) > 1:
                            lines = [line[:60] + '...' if len(line) > 60 else line for line in lines]
                            label = '\n'.join(lines)
                        else:
                            label = label[:80] + '...'
                return original_set_title(self, label, *args, **kwargs)
            
            matplotlib.axes.Axes.set_title = pdf_optimized_set_title
            
            try:
                show_overall_capital_flows_analysis_cs2('Latvia, Republic of', 'Latvia', include_crisis_years)
            finally:
                # Restore original settings and subplots function
                plt.subplots = original_subplots
                plt.rcParams.update(original_rcParams)
                
    except Exception as e:
        st.error(f"Error displaying Latvia overall analysis: {str(e)}")

def show_latvia_indicator_analysis(include_crisis_years=True):
    """Show Latvia indicator-level analysis with PDF-optimized formatting"""
    try:
        from case_study_2_euro_adoption_outlier_adjusted import show_indicator_level_analysis_cs2
        import matplotlib.pyplot as plt
        
        # Apply PDF optimization context with figure size constraints
        with plt.style.context('default'):
            # Store original settings
            original_rcParams = plt.rcParams.copy()
            
            # Apply PDF-optimized settings matching cs1_report_app.py
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
                'savefig.facecolor': 'white',
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'figure.max_open_warning': 0
            })
            
            # Monkey-patch plt.subplots to enforce PDF-friendly figure sizes
            original_subplots = plt.subplots
            def pdf_optimized_subplots(*args, **kwargs):
                if 'figsize' in kwargs:
                    width, height = kwargs['figsize']
                    # Constrain to PDF-friendly sizes (matching cs1_report_app.py max)
                    if width > 16:
                        kwargs['figsize'] = (16, min(height, 12))
                    elif height > 12:
                        kwargs['figsize'] = (width, 12)
                return original_subplots(*args, **kwargs)
            
            plt.subplots = pdf_optimized_subplots
            
            # Monkey-patch axis set_title to shorten long titles for PDF
            import matplotlib.axes
            original_set_title = matplotlib.axes.Axes.set_title
            def pdf_optimized_set_title(self, label, *args, **kwargs):
                # Shorten common long title patterns
                if isinstance(label, str):
                    label = label.replace('Across All Capital Flow Indicators', 'All Indicators')
                    label = label.replace('Distribution of Standard Deviations', 'Distribution of Std Deviations')
                    label = label.replace('Individual Eurozone Countries', 'Eurozone Countries')
                    label = label.replace('(Ordered by Descending Median Value)', '(By Descending Median)')
                    # Limit title length for PDF compatibility
                    if len(label) > 80:
                        lines = label.split('\n')
                        if len(lines) > 1:
                            lines = [line[:60] + '...' if len(line) > 60 else line for line in lines]
                            label = '\n'.join(lines)
                        else:
                            label = label[:80] + '...'
                return original_set_title(self, label, *args, **kwargs)
            
            matplotlib.axes.Axes.set_title = pdf_optimized_set_title
            
            try:
                show_indicator_level_analysis_cs2('Latvia, Republic of', include_crisis_years)
            finally:
                # Restore original settings and subplots function
                plt.subplots = original_subplots
                plt.rcParams.update(original_rcParams)
                
    except Exception as e:
        st.error(f"Error displaying Latvia indicator analysis: {str(e)}")

if __name__ == "__main__":
    main()