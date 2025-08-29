"""
Dashboard Configuration - Centralized constants and styling for all case study reports
"""

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

# ============================================================================
# VISUAL STYLING CONSTANTS
# ============================================================================

# Professional colorblind-safe palette for econometrics
COLORBLIND_SAFE = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#FBAFE4']

# Chart sizing constants (optimized for PDF export)
PDF_CHART_SIZES = {
    'boxplot': (10, 6),
    'timeseries': (12, 6),
    'scatter': (10, 6),
    'grid': (12, 8),
    'table': (10, 4)
}

# Standard margins for US Letter format (8.5" x 11")
PDF_MARGINS = {
    'max_width': 7.0,    # 8.5" - 2*0.75" margins
    'max_height': 9.0    # 11" - 2*1.0" margins
}

# ============================================================================
# STATISTICAL CONSTANTS
# ============================================================================

# Crisis periods for exclusion analysis
CRISIS_PERIODS = {
    'gfc': (2008, 2010),           # Global Financial Crisis
    'covid': (2020, 2022),         # COVID-19 pandemic
    'eurozone_crisis': (2011, 2013)  # Eurozone Crisis
}

# Standard significance levels
SIGNIFICANCE_LEVELS = {
    0.01: '***',
    0.05: '**', 
    0.10: '*'
}

# ============================================================================
# DATA PATHS
# ============================================================================

def get_data_paths():
    """Get standardized data paths for all applications"""
    base_path = Path(__file__).parent.parent.parent
    return {
        'clean_data': base_path / "updated_data" / "Clean",
        'master_dataset': base_path / "updated_data" / "Clean" / "comprehensive_df_PGDP_labeled.csv",
        'cs4_data': base_path / "updated_data" / "Clean" / "CS4_Statistical_Modeling",
        'cs5_controls': base_path / "updated_data" / "Clean" / "CS5_Capital_Controls",
        'cs5_regimes': base_path / "updated_data" / "Clean" / "CS5_Regime_Analysis",
        'output': base_path / "output"
    }

# ============================================================================
# MATPLOTLIB CONFIGURATION
# ============================================================================

def configure_matplotlib():
    """Apply consistent matplotlib configuration across all apps"""
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
    
    # Set colorblind-safe palette
    sns.set_palette(COLORBLIND_SAFE)

# ============================================================================
# STREAMLIT STYLING
# ============================================================================

def get_professional_css():
    """Get standardized CSS styling for all Streamlit apps"""
    return """
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
        
        
        /* Professional table styling (HTML tables) */
        .professional-table { 
            width: 100% !important;
            border-collapse: collapse !important;
            margin: 20px 0 !important;
            font-size: 11px !important;
            font-family: 'Arial', sans-serif !important;
        }
        .professional-table th {
            background-color: #e6f3ff !important;
            font-weight: bold !important;
            text-align: center !important;
            padding: 8px !important;
            border: 1px solid #ddd !important;
        }
        .professional-table td {
            text-align: center !important;
            padding: 6px !important;
            border: 1px solid #ddd !important;
        }
        .professional-table tbody tr:nth-child(even) {
            background-color: #f9f9f9 !important;
        }
        .professional-table td:first-child {
            text-align: left !important;
            font-weight: bold !important;
            padding-left: 10px !important;
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
            .professional-table { 
                page-break-inside: avoid !important;
                font-size: 9px !important;
                margin: 10px 0 !important;
            }
            .professional-table th, .professional-table td {
                padding: 4px !important;
                font-size: 9px !important;
            }
        }
    </style>
    """

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_pdf_figsize(chart_type='default', base_width=10, base_height=6):
    """Calculate PDF-optimized figure size"""
    max_width, max_height = PDF_MARGINS['max_width'], PDF_MARGINS['max_height']
    
    if chart_type in PDF_CHART_SIZES:
        width, height = PDF_CHART_SIZES[chart_type]
    else:
        width, height = base_width, base_height
    
    # Scale down if necessary
    if width > max_width:
        height = height * (max_width / width)
        width = max_width
    if height > max_height:
        width = width * (max_height / height)
        height = max_height
        
    return (width, height)

def format_significance(p_value):
    """Format p-value with significance stars"""
    for alpha, symbol in SIGNIFICANCE_LEVELS.items():
        if p_value < alpha:
            return symbol
    return ''

# ============================================================================
# INITIALIZATION
# ============================================================================

# Configure matplotlib when module is imported
configure_matplotlib()