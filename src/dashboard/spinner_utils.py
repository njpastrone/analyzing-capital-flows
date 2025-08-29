"""
Spinner Utilities for Streamlit Dashboard
Provides consistent, informative loading feedback throughout the application
"""

import streamlit as st
import time
from contextlib import contextmanager, nullcontext
from typing import Optional, List, Tuple, Callable
from pathlib import Path


# Icon mapping for different operation types
OPERATION_ICONS = {
    'load': '📂',
    'calculate': '🔢',
    'visualize': '📈',
    'process': '⚙️',
    'export': '💾',
    'analyze': '🔍',
    'validate': '✅',
    'compare': '🔊',
    'filter': '🔎',
    'aggregate': '📊',
    'statistical': '⚡',
    'temporal': '📅',
    'spatial': '🗺️',
    'merge': '🔄',
    'clean': '🧹'
}


def smart_spinner(operation_type: str, details: str = "", estimated_time: Optional[str] = None):
    """
    Context-aware spinner with consistent messaging and icons
    
    Args:
        operation_type: Type of operation (load, calculate, visualize, etc.)
        details: Specific details about the operation
        estimated_time: Optional estimated time (e.g., "5-10s")
    
    Returns:
        Streamlit spinner context manager
    
    Example:
        with smart_spinner('load', 'Iceland capital flows data (1999-2024)', '5-10s'):
            data = load_iceland_data()
    """
    icon = OPERATION_ICONS.get(operation_type, '🔄')
    message = f"{icon} {details}" if details else f"{icon} Processing..."
    
    if estimated_time:
        message += f" (Est. {estimated_time})"
    
    return st.spinner(message)


@contextmanager
def conditional_spinner(message: str, condition: bool = True, threshold: float = 1.0):
    """
    Show spinner conditionally based on operation characteristics
    
    Args:
        message: Spinner message to display
        condition: Whether to show spinner at all
        threshold: Minimum duration to show spinner (seconds)
    
    Example:
        with conditional_spinner("Processing data...", len(data) > 1000):
            process_large_dataset(data)
    """
    if condition:
        with st.spinner(message):
            yield
    else:
        yield


def progress_spinner(steps: List[Tuple[str, Callable]], show_steps: bool = True):
    """
    Execute multiple steps with progress feedback
    
    Args:
        steps: List of (step_name, step_function) tuples
        show_steps: Whether to show step counter
    
    Example:
        steps = [
            ("Loading data", load_data),
            ("Processing", process_data),
            ("Generating charts", create_visualizations)
        ]
        results = progress_spinner(steps)
    """
    results = []
    progress_text = st.empty() if show_steps else None
    
    for i, (step_name, step_func) in enumerate(steps, 1):
        if show_steps and progress_text:
            progress_text.text(f"Step {i}/{len(steps)}: {step_name}")
        
        with st.spinner(step_name):
            result = step_func()
            results.append(result)
    
    if progress_text:
        progress_text.empty()
    
    return results


def nested_spinner(main_message: str, sub_operations: List[Tuple[str, Callable]]):
    """
    Handle nested operations with main and sub-operation feedback
    
    Args:
        main_message: Main operation message
        sub_operations: List of (sub_message, sub_function) tuples
    
    Example:
        nested_spinner(
            "📊 Running Case Study 4 Analysis...",
            [
                ("📂 Loading datasets...", load_cs4_data),
                ("🔢 Calculating F-tests...", calculate_ftests),
                ("📈 Generating models...", generate_models)
            ]
        )
    """
    with st.spinner(main_message):
        status_placeholder = st.empty()
        results = []
        
        for sub_message, sub_func in sub_operations:
            status_placeholder.text(sub_message)
            result = sub_func()
            results.append(result)
        
        status_placeholder.empty()
        return results


# Predefined spinner messages for common operations
class SpinnerMessages:
    """Standard spinner messages for consistency across the application"""
    
    # Data loading messages
    LOAD_COMPREHENSIVE = "📂 Loading comprehensive dataset: {filename}"
    LOAD_WINSORIZED = "📂 Loading outlier-adjusted (winsorized) data..."
    LOAD_BOP = "📂 Loading Balance of Payments data ({countries}, {years})..."
    LOAD_GDP = "📂 Loading GDP normalization data..."
    LOAD_CONTROLS = "📂 Loading capital controls dataset (Fernández et al.)..."
    LOAD_REGIMES = "📂 Loading exchange rate regime classifications..."
    
    # Processing messages
    PROCESS_NORMALIZATION = "⚙️ Normalizing flows to % of GDP..."
    PROCESS_FILTERING = "⚙️ Filtering data: {criteria}..."
    PROCESS_AGGREGATION = "⚙️ Aggregating {frequency} data..."
    PROCESS_WINSORIZATION = "⚙️ Applying {level}% winsorization..."
    PROCESS_CRISIS_EXCLUSION = "⚙️ Excluding crisis periods ({periods})..."
    
    # Calculation messages
    CALC_VOLATILITY = "🔢 Calculating volatility metrics..."
    CALC_FTEST = "🔢 Running F-test for variance equality..."
    CALC_AR_MODEL = "🔢 Fitting AR({order}) model..."
    CALC_RMSE = "🔢 Computing RMSE for predictions..."
    CALC_CORRELATION = "🔢 Calculating correlation coefficients..."
    CALC_STATISTICS = "🔢 Computing descriptive statistics..."
    
    # Visualization messages
    VIZ_TIMESERIES = "📈 Generating time series plots..."
    VIZ_BOXPLOT = "📈 Creating comparative boxplots..."
    VIZ_SCATTER = "📈 Plotting scatter diagrams..."
    VIZ_HEATMAP = "📈 Building correlation heatmap..."
    VIZ_TABLE = "📈 Formatting statistical tables..."
    
    # Analysis messages
    ANALYZE_TEMPORAL = "🔍 Analyzing temporal patterns ({period})..."
    ANALYZE_CROSS_SECTIONAL = "🔍 Comparing cross-sectional volatility..."
    ANALYZE_REGIME = "🔍 Evaluating regime-specific effects..."
    ANALYZE_ROBUST = "🔍 Running robustness checks..."
    
    # Export messages
    EXPORT_CSV = "💾 Preparing CSV export..."
    EXPORT_EXCEL = "💾 Generating Excel workbook..."
    EXPORT_PDF = "💾 Creating PDF report..."
    EXPORT_HTML = "💾 Building HTML report..."
    
    @staticmethod
    def format(template: str, **kwargs) -> str:
        """Format a template message with provided values"""
        return template.format(**kwargs)


def get_case_study_spinner(cs_number: int, operation: str = "Loading") -> str:
    """
    Get standardized spinner message for case study operations
    
    Args:
        cs_number: Case study number (1-5)
        operation: Type of operation being performed
    
    Returns:
        Formatted spinner message
    """
    cs_descriptions = {
        1: "Iceland vs Eurozone Analysis",
        2: "Baltic Euro Adoption Study",
        3: "Small Open Economies Comparison",
        4: "Statistical Analysis Framework",
        5: "Capital Controls & Exchange Rates"
    }
    
    description = cs_descriptions.get(cs_number, f"Case Study {cs_number}")
    return f"📊 {operation} {description}..."


def data_operation_spinner(operation: str, dataset_name: str, 
                          record_count: Optional[int] = None,
                          time_range: Optional[str] = None) -> str:
    """
    Create informative spinner for data operations
    
    Args:
        operation: Type of operation (Loading, Processing, etc.)
        dataset_name: Name of the dataset
        record_count: Optional number of records
        time_range: Optional time range (e.g., "1999-2024")
    
    Returns:
        Formatted spinner message
    """
    message = f"📂 {operation} {dataset_name}"
    
    details = []
    if record_count:
        details.append(f"{record_count:,} records")
    if time_range:
        details.append(time_range)
    
    if details:
        message += f" ({', '.join(details)})"
    
    return message + "..."


def statistical_operation_spinner(test_type: str, indicators: Optional[List[str]] = None) -> str:
    """
    Create spinner for statistical operations
    
    Args:
        test_type: Type of statistical test
        indicators: Optional list of indicators being tested
    
    Returns:
        Formatted spinner message
    """
    message = f"🔢 Running {test_type}"
    
    if indicators:
        if len(indicators) <= 3:
            message += f" for {', '.join(indicators)}"
        else:
            message += f" for {len(indicators)} indicators"
    
    return message + "..."


# Quick access functions for common operations
def loading_spinner(details: str, estimated_time: Optional[str] = None):
    """Quick spinner for loading operations"""
    return smart_spinner('load', details, estimated_time)


def calculating_spinner(details: str, estimated_time: Optional[str] = None):
    """Quick spinner for calculations"""
    return smart_spinner('calculate', details, estimated_time)


def processing_spinner(details: str, estimated_time: Optional[str] = None):
    """Quick spinner for processing operations"""
    return smart_spinner('process', details, estimated_time)


def visualizing_spinner(details: str, estimated_time: Optional[str] = None):
    """Quick spinner for visualization generation"""
    return smart_spinner('visualize', details, estimated_time)