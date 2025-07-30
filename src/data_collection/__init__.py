"""
Data Collection Module for IMF Balance of Payments Data
=======================================================

This module handles fetching missing BOP data series from IMF API
to complete the Financial Account components for comprehensive 
"Overall Capital Flows" calculation.

Components:
- imf_bop_fetcher.py: Main data fetching class
- data_processor.py: Process and integrate with existing data
- validator.py: Data quality validation
"""

from .imf_bop_fetcher import IMFBOPFetcher

__all__ = ['IMFBOPFetcher']