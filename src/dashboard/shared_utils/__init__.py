"""
Shared Utilities Module

This module provides centralized utilities for all report files,
reducing duplication and improving maintainability.

Modules:
- (Phase 2) data_loading: Generic data loading functions
- (Phase 3) helpers: Shared helper functions
- (Phase 4) analysis_config: Analysis type configuration wrapper
- (Phase 5) output_formatting: Format-specific output utilities

Phase 1: Module structure established
"""

# Phase 1: Re-export commonly used items from centralized config
# This makes imports cleaner: from shared_utils import CRISIS_YEARS_LIST
# Instead of: from config.constants import CRISIS_YEARS_LIST

# Will be populated in future phases
__version__ = "0.1.0-phase1"
