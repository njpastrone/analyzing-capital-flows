"""
Centralized Professional Styling Module

This module provides CSS styling functions for all case study reports,
eliminating ~1,000 lines of duplicate code across 8 files.

Phase 3a: Foundation utilities ✅
"""

import streamlit as st


def get_professional_base_css():
    """
    Get base professional CSS styling common to all case studies.

    Returns
    -------
    str
        CSS styles for general elements (dataframes, charts, headers, buttons, etc.)
    """
    return """
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

        /* Image and chart constraints */
        img {
            max-width: 100% !important;
            height: auto !important;
            page-break-inside: avoid !important;
        }

        .chart-container {
            max-width: 100% !important;
            overflow: hidden !important;
            text-align: center !important;
            margin: 20px 0 !important;
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

            /* Remove Streamlit UI elements in print */
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
    """


def get_cs4_specific_css():
    """
    Get CS4-specific CSS styling for master tables.

    CS4 master tables require special handling for F-test results display:
    - Optimized for wide tables with many comparator columns
    - Font sizing for PDF export compatibility
    - Column width constraints for proper rendering

    Returns
    -------
    str
        CSS styles specific to CS4 master tables
    """
    return """
    <style>
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

        /* Table print optimizations */
        @media print {
            .cs4-master-table {
                page-break-inside: avoid !important;
                font-size: 7px !important;
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
        }
    </style>
    """


def get_cs5_specific_css():
    """
    Get CS5-specific CSS styling for wide tables.

    CS5 master tables require special handling for 13-column layout:
    - Capital controls and exchange rate regime analysis
    - Weighted and simple averages for multiple comparator groups
    - Very compact font sizing to fit 13 columns

    Returns
    -------
    str
        CSS styles specific to CS5 master tables (13-column layout)
    """
    return """
    <style>
        /* CS5 Master Table Styling (optimized for 13 columns - weighted & simple averages) */
        .cs4-master-table {
            width: 100% !important;
            border-collapse: collapse !important;
            margin: 20px 0 !important;
            font-size: 9px !important;
            font-family: 'Arial', sans-serif !important;
            table-layout: fixed !important;
        }
        .cs4-master-table th {
            background-color: #e6f3ff !important;
            font-weight: bold !important;
            text-align: center !important;
            padding: 6px 3px !important;
            border: 1px solid #ddd !important;
            font-size: 8px !important;
            word-wrap: break-word !important;
        }
        .cs4-master-table td {
            text-align: center !important;
            padding: 4px 2px !important;
            border: 1px solid #ddd !important;
            font-size: 8px !important;
            word-wrap: break-word !important;
        }
        .cs4-master-table tbody tr:nth-child(even) {
            background-color: #f9f9f9 !important;
        }

        /* First column (Indicator/Period) optimized for 13-column layout */
        .cs4-master-table td:first-child {
            width: 18% !important;
            text-align: left !important;
            font-weight: bold !important;
            padding-left: 6px !important;
            font-size: 8px !important;
            white-space: nowrap !important;
            word-wrap: normal !important;
            word-break: normal !important;
            overflow-wrap: normal !important;
            min-width: 18% !important;
        }
        .cs4-master-table th:first-child {
            width: 18% !important;
            text-align: center !important;
            font-size: 8px !important;
            white-space: nowrap !important;
            word-wrap: normal !important;
        }

        /* Data columns optimized for 13-column display */
        .cs4-master-table td:not(:first-child), .cs4-master-table th:not(:first-child) {
            width: 6.8% !important;
            min-width: 6.8% !important;
            max-width: 6.8% !important;
        }

        /* Print Media Queries for PDF Export */
        @media print {
            .cs4-master-table {
                page-break-inside: avoid !important;
                font-size: 7px !important;
                margin: 10px 0 !important;
                table-layout: fixed !important;
            }
            .cs4-master-table th, .cs4-master-table td {
                padding: 3px 1px !important;
                font-size: 7px !important;
            }

            /* First column width for PDF */
            .cs4-master-table td:first-child, .cs4-master-table th:first-child {
                width: 18% !important;
                font-size: 7px !important;
                white-space: nowrap !important;
                word-wrap: normal !important;
                min-width: 18% !important;
            }

            /* Data columns for PDF */
            .cs4-master-table td:not(:first-child), .cs4-master-table th:not(:first-child) {
                width: 6.8% !important;
                min-width: 6.8% !important;
                max-width: 6.8% !important;
            }
        }
    </style>
    """


def apply_professional_styling(case_study='cs4'):
    """
    Apply professional CSS styling with case study-specific customizations.

    This function centralizes all styling logic previously duplicated across:
    - 4 CS4 report files (full, outlier_adjusted, pdf, pdf_outlier_adjusted)
    - 4 CS5 report files (full, outlier_adjusted, pdf, pdf_outlier_adjusted)

    Previously duplicated in 8 files with 1,127 total lines.

    Parameters
    ----------
    case_study : str, default='cs4'
        The case study requiring styling. Options:
        - 'cs4': CS4 Statistical Analysis (standard width master tables)
        - 'cs5': CS5 Capital Controls & Regimes (13-column master tables)
        - 'base': Base styling only (no case study-specific tables)

    Notes
    -----
    - Base CSS includes dataframes, charts, headers, print media queries
    - CS4 CSS adds master table styling for F-test results
    - CS5 CSS adds 13-column table styling for comprehensive comparisons
    - All styles include PDF export optimization

    Examples
    --------
    >>> # In CS4 report files
    >>> apply_professional_styling('cs4')

    >>> # In CS5 report files
    >>> apply_professional_styling('cs5')
    """
    # Start with base CSS
    css = get_professional_base_css()

    # Add case study-specific CSS
    if case_study.lower() == 'cs4':
        css += get_cs4_specific_css()
    elif case_study.lower() == 'cs5':
        css += get_cs5_specific_css()
    elif case_study.lower() != 'base':
        # Unknown case study - warn but continue with base CSS
        st.warning(f"Unknown case_study '{case_study}'. Using base styling only.")

    # Apply the CSS
    st.markdown(css, unsafe_allow_html=True)


__all__ = [
    'get_professional_base_css',
    'get_cs4_specific_css',
    'get_cs5_specific_css',
    'apply_professional_styling'
]
