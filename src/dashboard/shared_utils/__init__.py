"""
Shared Utilities Module

This module provides centralized utilities for all report files,
reducing duplication and improving maintainability.

Modules:
- data_loading: Generic data loading functions (Phase 2 ✅)
- styling: Professional CSS styling (Phase 3a ✅)
- (Phase 3b) helpers: Shared helper functions
- (Phase 4) analysis_config: Analysis type configuration wrapper
- (Phase 5) output_formatting: Format-specific output utilities

Phase 1: Module structure established ✅
Phase 2: Data loading utilities complete ✅
Phase 3a: Foundation utilities (styling, report generation, PDF) 🚧
"""

# Phase 2: Data loading utilities
from .data_loading import (
    load_case_study_data,
    load_overall_capital_flows_data
)

# Phase 3a: Styling utilities
from .styling import (
    apply_professional_styling,
    get_professional_base_css,
    get_cs4_specific_css,
    get_cs5_specific_css
)

__version__ = "0.3.0-phase3a"
__all__ = [
    # Data loading
    'load_case_study_data',
    'load_overall_capital_flows_data',
    # Styling
    'apply_professional_styling',
    'get_professional_base_css',
    'get_cs4_specific_css',
    'get_cs5_specific_css'
]
