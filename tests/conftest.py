"""
Shared test fixtures and utilities for Capital Flows Research Project tests
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import warnings

# Add src/dashboard to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "dashboard"))

# Suppress warnings during testing
warnings.filterwarnings('ignore')

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory path"""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session") 
def data_dir(project_root):
    """Return the main data directory path"""
    return project_root / "updated_data" / "Clean"

@pytest.fixture(scope="session")
def comprehensive_data_file(data_dir):
    """Return the comprehensive dataset file path"""
    return data_dir / "comprehensive_df_PGDP_labeled.csv"

@pytest.fixture(scope="session")
def cs4_data_dir(data_dir):
    """Return the CS4 specialized data directory"""
    return data_dir / "CS4_Statistical_Modeling"

@pytest.fixture(scope="session") 
def cs5_controls_dir(data_dir):
    """Return the CS5 capital controls data directory"""
    return data_dir / "CS5_Capital_Controls"

@pytest.fixture(scope="session")
def cs5_regime_dir(data_dir):
    """Return the CS5 regime analysis data directory"""
    return data_dir / "CS5_Regime_Analysis"

@pytest.fixture(scope="session")
def comprehensive_dataframe(comprehensive_data_file):
    """Load and return the comprehensive dataset once per test session"""
    if not comprehensive_data_file.exists():
        pytest.skip(f"Comprehensive data file not found: {comprehensive_data_file}")
    return pd.read_csv(comprehensive_data_file)

@pytest.fixture
def expected_cs1_shape():
    """Expected shape for CS1 data (Iceland vs Eurozone)"""
    return (1093, 25)  # Based on actual data structure

@pytest.fixture
def expected_cs2_shape():
    """Expected shape for CS2 data (Baltic Euro adoption)"""
    return (315, 25)  # 3 countries * 105 observations

@pytest.fixture
def expected_cs3_shape():
    """Expected shape for CS3 data (Iceland vs SOEs)"""
    return (763, 25)  # Iceland + 7 SOE countries

@pytest.fixture
def expected_indicators_count():
    """Expected number of capital flow indicators"""
    return 14  # Standard indicator count after filtering

@pytest.fixture
def expected_cs1_countries():
    """Expected countries in CS1"""
    return [
        'Austria', 'Belgium', 'Finland', 'France', 'Germany',
        'Iceland', 'Ireland', 'Italy', 'Netherlands, The', 
        'Portugal', 'Spain'
    ]

@pytest.fixture
def expected_cs2_countries():
    """Expected countries in CS2"""
    return [
        'Estonia, Republic of', 
        'Latvia, Republic of',
        'Lithuania, Republic of'
    ]

@pytest.fixture
def crisis_years():
    """Crisis years that should be excluded in crisis-excluded analysis"""
    return [2008, 2009, 2010, 2020, 2021, 2022]

class TestDataQuality:
    """Utility class for common data quality checks"""
    
    @staticmethod
    def check_no_trailing_spaces(file_path):
        """Check that file path has no trailing spaces"""
        return not str(file_path).endswith(' ')
    
    @staticmethod
    def check_required_columns(df, required_cols):
        """Check that DataFrame has all required columns"""
        missing = set(required_cols) - set(df.columns)
        return len(missing) == 0, missing
    
    @staticmethod
    def check_group_integrity(df, group_col, expected_groups):
        """Check that group column has expected unique values"""
        actual_groups = set(df[group_col].dropna().unique())
        return actual_groups == set(expected_groups)
    
    @staticmethod
    def check_indicator_columns(df):
        """Return columns that end with _PGDP (indicator columns)"""
        return [col for col in df.columns if col.endswith('_PGDP')]