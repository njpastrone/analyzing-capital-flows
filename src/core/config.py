"""
Configuration settings for Capital Flows Research Dashboard
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CASE_STUDIES_DIR = PROJECT_ROOT / "src" / "case_studies"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# Analysis settings
DEFAULT_SIGNIFICANCE_LEVEL = 0.05
DEFAULT_PLOT_STYLE = 'seaborn-v0_8-whitegrid'
DEFAULT_COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "Capital Flows Research Dashboard",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Statistical test settings
STATISTICAL_TESTS = {
    'f_test': {
        'name': 'F-Test for Equal Variances',
        'description': 'Tests whether two groups have equal variances',
        'alpha_levels': [0.001, 0.01, 0.05, 0.1]
    },
    't_test': {
        'name': 'T-Test for Equal Means',
        'description': 'Tests whether two groups have equal means',
        'alpha_levels': [0.001, 0.01, 0.05, 0.1]
    }
}

# Export formats
EXPORT_FORMATS = {
    'csv': 'CSV',
    'excel': 'Excel (XLSX)',
    'pdf': 'PDF Report',
    'png': 'PNG Images',
    'html': 'HTML Report'
}

# Volatility measures
VOLATILITY_MEASURES = [
    'Standard Deviation',
    'Coefficient of Variation',
    'Variance',
    'Range',
    'Interquartile Range'
]