"""
Shared Utilities Module

This module provides centralized utilities for all report files,
reducing duplication and improving maintainability.

Modules:
- data_loading: Generic data loading functions (Phase 2 ✅)
- (Phase 3) helpers: Shared helper functions
- (Phase 4) analysis_config: Analysis type configuration wrapper
- (Phase 5) output_formatting: Format-specific output utilities

Phase 1: Module structure established ✅
Phase 2: Data loading utilities complete ✅
"""

# Phase 2: Data loading utilities
from .data_loading import (
    load_case_study_data,
    load_overall_capital_flows_data
)

__version__ = "0.2.0-phase2"
__all__ = [
    'load_case_study_data',
    'load_overall_capital_flows_data'
]
