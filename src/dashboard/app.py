"""
Capital Flows Research Dashboard - Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import PAGE_CONFIG, DATA_DIR, OUTPUT_DIR
from dashboard.templates.case_study_template import VolatilityAnalysisTemplate


def main():
    """Main application function"""
    
    # Configure page
    st.set_page_config(**PAGE_CONFIG)
    
    # Main title
    st.title("📊 Capital Flows Research Dashboard")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    
    # Case study selection
    case_study_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Volatility Analysis", "Mean Comparison", "Custom Analysis"]
    )
    
    if case_study_type == "Volatility Analysis":
        run_volatility_analysis()
    elif case_study_type == "Mean Comparison":
        st.info("Mean comparison analysis coming soon!")
    else:
        st.info("Custom analysis templates coming soon!")


def run_volatility_analysis():
    """Run volatility analysis case study"""
    
    st.header("Capital Flow Volatility Analysis")
    st.markdown("Compare capital flow volatility between different country groups")
    
    # Initialize session state
    if 'volatility_template' not in st.session_state:
        st.session_state.volatility_template = VolatilityAnalysisTemplate()
    
    template = st.session_state.volatility_template
    
    # Data Loading Section
    st.subheader("1. Data Loading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**BOP Data File**")
        bop_file = st.file_uploader(
            "Upload Balance of Payments CSV",
            type=['csv'],
            key="bop_upload",
            help="IMF Balance of Payments data in CSV format"
        )
        
        # Default file path option
        default_bop = st.checkbox("Use default BOP file (Case Study 1)")
        if default_bop:
            default_bop_path = DATA_DIR / "case_study_1_data_july_24_2025.csv"
            if default_bop_path.exists():
                bop_file = str(default_bop_path)
                st.success(f"Using: {default_bop_path.name}")
            else:
                st.error("Default BOP file not found")
    
    with col2:
        st.markdown("**GDP Data File**")
        gdp_file = st.file_uploader(
            "Upload GDP CSV",
            type=['csv'],
            key="gdp_upload",
            help="IMF GDP data in CSV format"
        )
        
        # Default file path option
        default_gdp = st.checkbox("Use default GDP file (Case Study 1)")
        if default_gdp:
            default_gdp_path = DATA_DIR / "dataset_2025-07-24T18_28_31.898465539Z_DEFAULT_INTEGRATION_IMF.RES_WEO_6.0.0.csv"
            if default_gdp_path.exists():
                gdp_file = str(default_gdp_path)
                st.success(f"Using: {default_gdp_path.name}")
            else:
                st.error("Default GDP file not found")
    
    # Load data button
    if st.button("Load Data", type="primary"):
        if bop_file and gdp_file:
            with st.spinner("Loading data..."):
                # Save uploaded files temporarily if they're file objects
                bop_path = save_uploaded_file(bop_file) if hasattr(bop_file, 'read') else bop_file
                gdp_path = save_uploaded_file(gdp_file) if hasattr(gdp_file, 'read') else gdp_file
                
                success = template.load_data(bop_path, gdp_path)
                
                if success:
                    st.success("✅ Data loaded successfully!")
                    
                    # Show data summary
                    with st.expander("Data Summary"):
                        metadata = template.metadata
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("BOP Data Shape", f"{metadata['bop_shape'][0]} × {metadata['bop_shape'][1]}")
                        with col2:
                            st.metric("GDP Data Shape", f"{metadata['gdp_shape'][0]} × {metadata['gdp_shape'][1]}")
                else:
                    st.error("❌ Failed to load data. Check file formats and content.")
        else:
            st.warning("Please upload both BOP and GDP files.")
    
    # Data Processing Section
    if template.metadata.get('data_loaded', False):
        st.subheader("2. Data Processing & Group Configuration")
        
        # Group definitions
        st.markdown("**Define Country Groups for Comparison**")
        
        # Predefined group configurations
        group_preset = st.selectbox(
            "Choose a preset or create custom groups",
            ["Custom", "Iceland vs Eurozone", "Ireland vs Eurozone", "Small vs Large Economies"]
        )
        
        if group_preset == "Iceland vs Eurozone":
            group_definitions = {
                "Iceland": ["Iceland"],
                "Eurozone": ["Austria", "Belgium", "Finland", "France", "Germany", 
                           "Ireland", "Italy", "Netherlands", "Portugal", "Spain"]
            }
        elif group_preset == "Ireland vs Eurozone":
            group_definitions = {
                "Ireland": ["Ireland"],
                "Eurozone": ["Austria", "Belgium", "Finland", "France", "Germany", 
                           "Iceland", "Italy", "Netherlands", "Portugal", "Spain"]
            }
        else:
            # Custom group definition
            group_definitions = {}
            
            num_groups = st.number_input("Number of groups", min_value=2, max_value=5, value=2)
            
            for i in range(num_groups):
                group_name = st.text_input(f"Group {i+1} name", value=f"Group_{i+1}")
                countries_text = st.text_area(
                    f"Countries in {group_name} (one per line)",
                    height=100,
                    key=f"countries_{i}"
                )
                
                if countries_text.strip():
                    countries = [country.strip() for country in countries_text.split('\\n') if country.strip()]
                    group_definitions[group_name] = countries
        
        # Display group configuration
        if group_definitions:
            st.markdown("**Current Group Configuration:**")
            for group_name, countries in group_definitions.items():
                st.write(f"- **{group_name}**: {', '.join(countries)}")
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            remove_luxembourg = st.checkbox("Remove Luxembourg (financial center)", value=True)
        with col2:
            significance_level = st.selectbox("Significance Level", [0.001, 0.01, 0.05, 0.1], index=2)
        
        # Process data button
        if st.button("Process Data", type="primary"):
            if group_definitions:
                with st.spinner("Processing data..."):
                    success = template.process_data(
                        group_definitions=group_definitions,
                        remove_luxembourg=remove_luxembourg
                    )
                    
                    if success:
                        st.success("✅ Data processed successfully!")
                        
                        # Show processing summary
                        with st.expander("Processing Summary"):
                            metadata = template.metadata
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Final Dataset Shape", f"{metadata['final_shape'][0]} × {metadata['final_shape'][1]}")
                            with col2:
                                st.metric("Analysis Indicators", metadata['n_indicators'])
                            with col3:
                                st.metric("Groups", len(metadata['groups']))
                    else:
                        st.error("❌ Failed to process data.")
            else:
                st.warning("Please define at least one group.")
    
    # Analysis Section
    if template.metadata.get('processing_completed', False):
        st.subheader("3. Volatility Analysis")
        
        # Analysis parameters
        col1, col2 = st.columns(2)
        
        with col1:
            available_groups = list(template.group_definitions.keys())
            group1 = st.selectbox("Primary Group (Group 1)", available_groups, index=0)
        
        with col2:
            group2 = st.selectbox("Comparison Group (Group 2)", available_groups, index=1 if len(available_groups) > 1 else 0)
        
        # Run analysis button
        if st.button("Run Volatility Analysis", type="primary"):
            with st.spinner("Running statistical analysis..."):
                results = template.run_analysis(
                    group1=group1,
                    group2=group2,
                    significance_level=significance_level
                )
                
                if results:
                    st.success("✅ Analysis completed!")
                    
                    # Display analysis summary
                    st.markdown("### Analysis Summary")
                    summary_text = template.get_analysis_summary()
                    st.markdown(summary_text)
                    
                    # Show detailed results
                    with st.expander("Detailed Statistical Results"):
                        if 'volatility_tests' in results:
                            st.dataframe(results['volatility_tests'])
                else:
                    st.error("❌ Analysis failed.")
    
    # Visualization Section
    if template.metadata.get('analysis_completed', False):
        st.subheader("4. Visualizations")
        
        # Visualization options
        viz_options = st.multiselect(
            "Select visualizations to generate",
            ["Statistical Boxplots", "Time Series Plots", "Interactive Charts"],
            default=["Statistical Boxplots", "Time Series Plots"]
        )
        
        # Generate visualizations button
        if st.button("Generate Visualizations", type="primary"):
            with st.spinner("Creating visualizations..."):
                visualizations = template.generate_visualizations(
                    output_dir=str(OUTPUT_DIR),
                    save_plots=True
                )
                
                if visualizations:
                    st.success("✅ Visualizations generated!")
                    
                    # Display visualizations
                    if "Statistical Boxplots" in viz_options and 'boxplots' in visualizations:
                        st.markdown("#### Statistical Comparison - Boxplots")
                        st.pyplot(visualizations['boxplots'])
                    
                    if "Time Series Plots" in viz_options and 'time_series' in visualizations:
                        st.markdown("#### Time Series Analysis")
                        st.pyplot(visualizations['time_series'])
                    
                    if "Interactive Charts" in viz_options:
                        if 'interactive_boxplots' in visualizations:
                            st.markdown("#### Interactive Statistical Comparison")
                            st.plotly_chart(visualizations['interactive_boxplots'], use_container_width=True)
                        
                        if 'interactive_time_series' in visualizations:
                            st.markdown("#### Interactive Time Series")
                            st.plotly_chart(visualizations['interactive_time_series'], use_container_width=True)
                else:
                    st.error("❌ Visualization generation failed.")
    
    # Export Section
    if template.metadata.get('visualizations_completed', False):
        st.subheader("5. Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_formats = st.multiselect(
                "Export formats",
                ["CSV", "Excel", "PDF Report"],
                default=["CSV"]
            )
        
        with col2:
            if st.button("Export Results", type="secondary"):
                with st.spinner("Exporting results..."):
                    # Convert format names to lowercase
                    formats = [fmt.lower().replace(' report', '') for fmt in export_formats]
                    
                    exported_files = template.export_results(
                        output_dir=str(OUTPUT_DIR),
                        formats=formats
                    )
                    
                    if exported_files:
                        st.success(f"✅ Exported {len(exported_files)} files!")
                        
                        # Show download links
                        for file_path in exported_files:
                            filename = Path(file_path).name
                            with open(file_path, 'rb') as f:
                                st.download_button(
                                    label=f"📥 Download {filename}",
                                    data=f.read(),
                                    file_name=filename,
                                    mime="application/octet-stream"
                                )
                    else:
                        st.error("❌ Export failed.")


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary location"""
    temp_dir = OUTPUT_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    file_path = temp_dir / uploaded_file.name
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)


if __name__ == "__main__":
    main()