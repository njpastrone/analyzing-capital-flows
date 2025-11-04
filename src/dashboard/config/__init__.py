"""
Dashboard Configuration Package

Centralized configuration, constants, and styling for all case study reports.
"""

# Re-export commonly used items for convenience
from .constants import (
    CRISIS_YEARS_LIST,
    INDICATOR_NICKNAMES,
    BALTIC_COUNTRIES,
    SMALL_OPEN_ECONOMIES,
    CASE_STUDY_INFO,
    get_indicator_nickname,
    get_investment_type_sort_key,
    is_crisis_year
)

__all__ = [
    'CRISIS_YEARS_LIST',
    'INDICATOR_NICKNAMES',
    'BALTIC_COUNTRIES',
    'SMALL_OPEN_ECONOMIES',
    'CASE_STUDY_INFO',
    'get_indicator_nickname',
    'get_investment_type_sort_key',
    'is_crisis_year'
]
