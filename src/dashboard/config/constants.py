"""
Centralized Constants - Single source of truth for all magic strings and configuration values

This module consolidates constants that were previously duplicated across multiple files.
Import from here to ensure consistency across all case study reports.
"""

# ============================================================================
# CRISIS PERIODS
# ============================================================================

# Crisis years for exclusion analysis
# Previously defined in 8+ files as: crisis_years = [2008, 2009, 2010, 2020, 2021, 2022]
CRISIS_YEARS_LIST = [2008, 2009, 2010, 2020, 2021, 2022]

# Detailed crisis period definitions (already in dashboard_config.py, included here for completeness)
CRISIS_PERIODS_DETAILED = {
    'gfc': (2008, 2010),           # Global Financial Crisis
    'covid': (2020, 2022),         # COVID-19 pandemic
    'eurozone_crisis': (2011, 2013)  # Eurozone Crisis
}

# ============================================================================
# INDICATOR MAPPINGS
# ============================================================================

# Readable nicknames for Balance of Payments indicators
# Previously duplicated in 12+ files (cs1_report_app.py, cs1_report_outlier_adjusted.py, etc.)
INDICATOR_NICKNAMES = {
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

# ============================================================================
# COUNTRY GROUPINGS
# ============================================================================

# Case Study 2: Baltic countries and their Euro adoption dates
BALTIC_COUNTRIES = {
    'Estonia': {
        'euro_adoption_year': 2011,
        'display_name': 'Estonia'
    },
    'Latvia': {
        'euro_adoption_year': 2014,
        'display_name': 'Latvia'
    },
    'Lithuania': {
        'euro_adoption_year': 2015,
        'display_name': 'Lithuania'
    }
}

# Case Study 3: Small Open Economies compared to Iceland
# Note: Bermuda excluded due to missing GDP data (see CLAUDE.md)
SMALL_OPEN_ECONOMIES = [
    'Aruba',
    'Bahamas',
    'Brunei Darussalam',
    'Malta',
    'Mauritius',
    'Seychelles'
]

# ============================================================================
# INVESTMENT TYPE CLASSIFICATIONS
# ============================================================================

# Investment type ordering for sorting indicators
INVESTMENT_TYPE_ORDER = {
    'Direct investment': 0,
    'Portfolio investment': 1,
    'Other investment': 2,
    'Financial derivatives': 3,
    'Reserve assets': 4
}

# Disaggregation level ordering
DISAGGREGATION_ORDER = {
    'Total financial assets/liabilities': 0,  # Total comes first
    'Debt securities': 1,
    'Debt instruments': 1,
    'Deposit taking corporations': 2,  # More specific
    'Equity and investment fund shares': 3
}

# Accounting entry ordering
ACCOUNTING_ENTRY_ORDER = {
    'Assets': 0,
    'Liabilities': 1,
    'Net': 2
}

# ============================================================================
# DATA QUALITY
# ============================================================================

# Indicators to exclude from certain analyses
# (e.g., financial derivatives, which have different volatility characteristics)
EXCLUDED_INDICATORS = [
    'Net (net acquisition of financial assets less net incurrence of liabilities) - Financial derivatives (other than reserves) and employee stock options_PGDP',
    'Net (net acquisition of financial assets less net incurrence of liabilities) - Financial account balance, excluding reserves and related items_PGDP'
]

# ============================================================================
# DISPLAY CONSTANTS
# ============================================================================

# Maximum display length for indicator names in tables
MAX_INDICATOR_DISPLAY_LENGTH = 35

# Truncation suffix
TRUNCATION_SUFFIX = '...'

# ============================================================================
# CASE STUDY METADATA
# ============================================================================

CASE_STUDY_INFO = {
    'cs1': {
        'title': 'Case Study 1: Iceland vs Eurozone (1999-2024)',
        'description': 'Cross-sectional volatility comparison',
        'methodology': 'F-tests for variance equality',
        'period': (1999, 2024)
    },
    'cs2': {
        'title': 'Case Study 2: Baltic Euro Adoption',
        'description': 'Before/after temporal analysis',
        'methodology': 'Variance comparison pre/post Euro',
        'countries': list(BALTIC_COUNTRIES.keys())
    },
    'cs3': {
        'title': 'Case Study 3: Small Open Economies',
        'description': 'Iceland vs comparable small economies',
        'methodology': 'Multi-country variance analysis',
        'countries': SMALL_OPEN_ECONOMIES
    },
    'cs4': {
        'title': 'Case Study 4: Statistical Analysis Framework',
        'description': 'Advanced time series analysis',
        'methodology': 'F-tests, AR(4), RMSE predictions'
    },
    'cs5': {
        'title': 'Case Study 5: Capital Controls & Exchange Rate Regimes',
        'description': 'Policy regime effects analysis',
        'methodology': 'Correlation and regime comparison'
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_indicator_nickname(indicator_name: str, max_length: int = MAX_INDICATOR_DISPLAY_LENGTH) -> str:
    """
    Get readable nickname for indicator with optional truncation.

    Args:
        indicator_name: Full IMF indicator name
        max_length: Maximum display length (default: 35 characters)

    Returns:
        Shortened, readable indicator name
    """
    nickname = INDICATOR_NICKNAMES.get(
        indicator_name,
        indicator_name[:25] + TRUNCATION_SUFFIX if len(indicator_name) > 25 else indicator_name
    )

    # Truncate for table display while maintaining readability
    if len(nickname) > max_length:
        return nickname[:max_length] + TRUNCATION_SUFFIX
    return nickname


def get_investment_type_sort_key(indicator_name: str) -> tuple:
    """
    Extract sorting key for indicators: Type -> Disaggregation -> Accounting Entry.

    Args:
        indicator_name: Full indicator name

    Returns:
        Tuple of (investment_type_order, disaggregation_order, accounting_entry_order)
    """
    # Investment type
    inv_type = 9  # Default: unknown
    for inv_name, order in INVESTMENT_TYPE_ORDER.items():
        if inv_name in indicator_name:
            inv_type = order
            break

    # Disaggregation level
    disagg = 9  # Default: no disaggregation
    for disagg_name, order in DISAGGREGATION_ORDER.items():
        if disagg_name in indicator_name:
            disagg = order
            break

    # Accounting entry
    acc_entry = 9  # Default: unknown
    for entry_type, order in ACCOUNTING_ENTRY_ORDER.items():
        if indicator_name.startswith(entry_type):
            acc_entry = order
            break

    return (inv_type, disagg, acc_entry)


def is_crisis_year(year: int) -> bool:
    """
    Check if a given year is a crisis year.

    Args:
        year: Year to check

    Returns:
        True if year is in CRISIS_YEARS_LIST, False otherwise
    """
    return year in CRISIS_YEARS_LIST


def sort_indicators_by_type(indicators: list) -> list:
    """
    Sort indicators by investment type, disaggregation, then accounting entry.

    Handles indicators with or without _PGDP suffix. Uses get_investment_type_sort_key()
    for the actual sorting logic.

    Previously duplicated in 7 files: cs1_report_app.py, cs1_report_outlier_adjusted.py,
    cs1_report_app_pdf.py, cs1_report_outlier_adjusted_pdf.py, and CS2 versions.

    Args:
        indicators: List of indicator names (with or without _PGDP suffix)

    Returns:
        Sorted list of indicators in the same format as input

    Examples:
        >>> indicators = ['Net - Direct investment_PGDP', 'Assets - Portfolio (Total)_PGDP']
        >>> sort_indicators_by_type(indicators)
        ['Assets - Portfolio (Total)_PGDP', 'Net - Direct investment_PGDP']
    """
    # Convert to clean names if they have _PGDP suffix
    clean_indicators = [
        ind.replace('_PGDP', '') if ind.endswith('_PGDP') else ind
        for ind in indicators
    ]

    # Sort using the centralized sorting key function
    sorted_clean = sorted(clean_indicators, key=get_investment_type_sort_key)

    # Convert back to original format if needed
    if any(ind.endswith('_PGDP') for ind in indicators):
        return [ind + '_PGDP' for ind in sorted_clean]
    else:
        return sorted_clean


# ============================================================================
# NOTES FOR DEVELOPERS
# ============================================================================

"""
MIGRATION GUIDE:

Old code pattern:
    def create_indicator_nicknames():
        return {...}

    nicknames = create_indicator_nicknames()

New code pattern:
    from dashboard.config.constants import INDICATOR_NICKNAMES, get_indicator_nickname

    nickname = get_indicator_nickname(indicator_name)

Old code pattern:
    crisis_years = [2008, 2009, 2010, 2020, 2021, 2022]

New code pattern:
    from dashboard.config.constants import CRISIS_YEARS_LIST, is_crisis_year

    if is_crisis_year(year):
        # exclude from analysis

This centralization eliminates ~2,000 lines of duplicated code across the codebase.
"""
