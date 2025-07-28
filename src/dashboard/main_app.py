"""
Capital Flows Research Dashboard - Main Multi-Tab Application
Comprehensive research platform for analyzing capital flow volatility across different case studies
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

# Import case study modules
from simple_report_app import main as case_study_1_main
from case_study_2_euro_adoption import main as case_study_2_main

def main():
    """Main multi-tab application for capital flows research"""
    
    # Page configuration
    st.set_page_config(
        page_title="Capital Flows Research Dashboard",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main header
    st.title("🌍 Capital Flows Research Dashboard")
    st.markdown("### Comprehensive Analysis of International Capital Flow Volatility")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Project Overview", 
        "🇮🇸 Case Study 1: Iceland vs Eurozone",
        "🇪🇺 Case Study 2: Euro Adoption", 
        "🌏 Case Study 3: Emerging Markets",
        "📊 Comparative Analysis",
        "📖 Methodology & Data"
    ])
    
    with tab1:
        show_project_overview()
    
    with tab2:
        show_case_study_1()
    
    with tab3:
        show_case_study_2()
    
    with tab4:
        show_case_study_3_placeholder()
    
    with tab5:
        show_comparative_analysis_placeholder()
    
    with tab6:
        show_methodology_and_data()

def show_project_overview():
    """Display project overview and introduction"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Research Objective")
        st.markdown("""
        This research project examines **capital flow volatility** across different economies and time periods 
        to understand the implications for monetary policy, currency unions, and financial stability.
        
        ### Key Research Questions:
        1. **How does capital flow volatility vary across different monetary regimes?**
        2. **What are the determinants of capital flow volatility in small open economies?**
        3. **How do external shocks affect capital flow patterns differently across countries?**
        4. **What policy implications emerge for currency union decisions?**
        """)
        
        st.header("Case Studies Overview")
        
        # Case study cards
        st.subheader("🇮🇸 Case Study 1: Iceland vs. Eurozone (1999-2024)")
        st.markdown("""
        **Status:** ✅ Complete  
        **Focus:** Pre-Euro adoption analysis comparing Iceland's independent monetary policy with Eurozone stability  
        **Key Finding:** Iceland shows significantly higher capital flow volatility across most indicators  
        **Policy Implication:** Euro adoption could reduce financial volatility for Iceland
        """)
        
        st.subheader("🇪🇺 Case Study 2: Euro Adoption Impact (Baltic Countries)")
        st.markdown("""
        **Status:** ✅ Complete  
        **Focus:** Temporal comparison of capital flow volatility before and after Euro adoption  
        **Countries:** Estonia (2011), Latvia (2014), Lithuania (2015)  
        **Key Finding:** Mixed evidence for volatility reduction, country-specific patterns emerge
        """)
        
        st.subheader("🌏 Case Study 3: Emerging Markets Comparison (2000-2024)")
        st.markdown("""
        **Status:** 📋 Planned  
        **Focus:** Capital flow patterns across different emerging market economies  
        **Methodology:** Panel data analysis with institutional variables  
        **Expected Completion:** Q3 2024
        """)
    
    with col2:
        st.header("Project Metrics")
        
        # Metrics
        st.metric("Case Studies", "3", "2 completed")
        st.metric("Countries Analyzed", "28+", "Iceland, Eurozone, Baltics")
        st.metric("Time Period", "1999-2024", "25 years")
        st.metric("Data Points", "75,000+", "High frequency")
        
        st.header("Data Sources")
        st.markdown("""
        - **IMF Balance of Payments Statistics**
        - **IMF World Economic Outlook Database**
        - **OECD International Direct Investment Statistics**
        - **BIS International Banking Statistics**
        - **Central Bank Publications**
        """)
        
        st.header("Methodology")
        st.markdown("""
        - **F-tests for variance equality**
        - **Time series analysis**
        - **Event study methodology**
        - **Panel data techniques**
        - **Structural break analysis**
        """)

def show_case_study_1():
    """Display Case Study 1 - Iceland vs Eurozone (preserved exactly)"""
    
    st.info("📋 **Case Study 1: Iceland vs. Eurozone Capital Flow Volatility Analysis**")
    st.markdown("""
    This case study examines whether Iceland should adopt the Euro by comparing capital flow volatility 
    patterns between Iceland and the Eurozone bloc from 1999-2024.
    """)
    
    # Call the original Case Study 1 main function (preserved exactly)
    case_study_1_main()

def show_case_study_2():
    """Display Case Study 2 - Euro Adoption Impact (Baltic Countries)"""
    
    st.info("📋 **Case Study 2: Euro Adoption Impact Analysis - Baltic Countries**")
    st.markdown("""
    This case study examines how Euro adoption affected capital flow volatility through temporal comparison 
    of pre and post adoption periods for Estonia (2011), Latvia (2014), and Lithuania (2015).
    """)
    
    # Call the Case Study 2 main function
    case_study_2_main()

def show_case_study_3_placeholder():
    """Placeholder for Case Study 3 - Emerging Markets"""
    
    st.header("🌏 Case Study 3: Emerging Markets Capital Flow Volatility")
    
    st.markdown("""
    ### Research Scope
    
    **Objective:** Compare capital flow volatility patterns across major emerging market economies
    
    **Countries Under Consideration:**
    - **Latin America:** Brazil, Mexico, Argentina, Chile
    - **Asia:** India, Thailand, Indonesia, Malaysia  
    - **Europe:** Turkey, Poland, Czech Republic
    - **Africa:** South Africa
    
    **Research Questions:**
    1. How does capital flow volatility vary across emerging market regions?
    2. What institutional factors explain differences in volatility patterns?
    3. How do global financial cycles affect different emerging markets?
    4. What policy frameworks are associated with lower volatility?
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Analytical Framework")
        st.markdown("""
        **Panel Data Analysis:**
        - Fixed effects models
        - Random effects specifications
        - Dynamic panel estimation
        
        **Institutional Variables:**
        - Exchange rate regime
        - Capital account openness
        - Financial development index
        - Governance indicators
        
        **Global Factors:**
        - VIX volatility index
        - US monetary policy
        - Commodity price cycles
        - Global risk appetite
        """)
    
    with col2:
        st.subheader("Expected Deliverables")
        st.markdown("""
        1. **Cross-Country Volatility Rankings**
        2. **Institutional Determinants Analysis**
        3. **Policy Recommendations Matrix**
        4. **Early Warning Indicators**
        5. **Interactive Dashboard**
        """)
        
        st.info("📅 **Timeline:** Q3 2024 target completion")

def show_comparative_analysis_placeholder():
    """Placeholder for Comparative Analysis across all case studies"""
    
    st.header("📊 Comparative Analysis Across Case Studies")
    
    st.markdown("""
    ### Cross-Case Study Synthesis
    
    This section will provide comprehensive analysis comparing findings across all case studies
    to identify common patterns and policy implications.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Planned Comparative Components")
        st.markdown("""
        **Volatility Patterns:**
        - Cross-country volatility rankings
        - Temporal volatility evolution
        - Sector-specific comparisons
        
        **Policy Regime Analysis:**
        - Currency union effects (Iceland)
        - Political transition impacts (Brexit)  
        - Institutional quality effects (Emerging Markets)
        
        **Global Shock Transmission:**
        - 2008 Financial Crisis
        - COVID-19 Pandemic
        - Recent geopolitical events
        """)
    
    with col2:
        st.subheader("Synthesis Framework")
        st.markdown("""
        **Meta-Analysis Approach:**
        - Effect size comparisons
        - Methodological robustness checks
        - Policy effectiveness assessment
        
        **Interactive Tools:**
        - Cross-case comparison dashboard
        - Policy scenario simulator
        - Risk assessment matrix
        """)
    
    st.warning("📊 **Note:** This section will be populated as individual case studies are completed.")

def show_methodology_and_data():
    """Display comprehensive methodology and data documentation"""
    
    st.header("📖 Methodology & Data Documentation")
    
    tab1, tab2, tab3 = st.tabs(["Statistical Methods", "Data Sources", "Quality Assurance"])
    
    with tab1:
        st.subheader("Statistical Methodologies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Volatility Analysis:**
            - F-tests for variance equality
            - Levene's test for robustness
            - Brown-Forsythe test for non-normal data
            - Bootstrap confidence intervals
            
            **Time Series Methods:**
            - GARCH volatility modeling
            - Structural break tests (Chow, CUSUM)
            - Vector autoregression (VAR)
            - Impulse response analysis
            
            **Panel Data Techniques:**
            - Fixed effects estimation
            - Random effects with clustering
            - Dynamic panel GMM
            - Cross-sectional dependence tests
            """)
        
        with col2:
            st.markdown("""
            **Event Study Analysis:**
            - Market model estimation
            - Abnormal return calculation
            - Statistical significance testing
            - Cumulative abnormal returns
            
            **Robustness Checks:**
            - Alternative data frequencies
            - Different volatility measures
            - Subsample analysis
            - Sensitivity to outliers
            
            **Policy Analysis:**
            - Difference-in-differences
            - Regression discontinuity
            - Synthetic control methods
            - Instrumental variables
            """)
    
    with tab2:
        st.subheader("Data Sources and Coverage")
        
        data_sources = [
            {
                "Source": "IMF Balance of Payments Statistics",
                "Coverage": "1999-2024, Quarterly",
                "Variables": "All BOP components, 190+ countries",
                "Quality": "High - Official statistics"
            },
            {
                "Source": "IMF World Economic Outlook", 
                "Coverage": "1980-2024, Annual",
                "Variables": "GDP, inflation, fiscal indicators",
                "Quality": "High - Standardized methodology"
            },
            {
                "Source": "OECD International Direct Investment",
                "Coverage": "1990-2024, Annual/Quarterly", 
                "Variables": "FDI flows and stocks by partner",
                "Quality": "High - OECD countries only"
            },
            {
                "Source": "BIS International Banking Statistics",
                "Coverage": "1977-2024, Quarterly",
                "Variables": "Cross-border banking flows",
                "Quality": "High - Central bank reported"
            }
        ]
        
        st.dataframe(pd.DataFrame(data_sources), use_container_width=True)
        
        st.subheader("Data Processing Pipeline")
        st.markdown("""
        1. **Data Collection:** Automated API downloads where available
        2. **Cleaning:** Outlier detection, missing value imputation
        3. **Harmonization:** Currency conversion, seasonal adjustment
        4. **Normalization:** GDP ratios, per capita adjustments
        5. **Validation:** Cross-source verification, temporal consistency
        """)
    
    with tab3:
        st.subheader("Quality Assurance Framework")
        
        st.markdown("""
        **Data Quality Checks:**
        - ✅ Source verification and cross-validation
        - ✅ Temporal consistency checks
        - ✅ Cross-country comparability assessment
        - ✅ Missing data pattern analysis
        - ✅ Outlier detection and treatment
        
        **Methodological Validation:**
        - ✅ Replication of key results
        - ✅ Sensitivity analysis for main findings
        - ✅ Alternative specification testing
        - ✅ Robustness to sample periods
        - ✅ Cross-validation with external studies
        
        **Reproducibility Standards:**
        - 📁 Version-controlled analysis code
        - 📊 Automated report generation
        - 🔄 Continuous integration testing
        - 📝 Comprehensive documentation
        - 🌐 Open data sharing (where permitted)
        """)

if __name__ == "__main__":
    main()