"""
Case Study 4 Specific Functionality Tests - Statistical Analysis

Tests CS4 specialized data loading and advanced statistical modeling
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from conftest import TestDataQuality

class TestCS4DataFileAccess:
    """Test CS4 specialized data file access"""
    
    def test_cs4_directory_exists(self, cs4_data_dir):
        """Test that CS4 specialized data directory exists"""
        assert cs4_data_dir.exists(), f"CS4 data directory not found: {cs4_data_dir}"
        assert cs4_data_dir.is_dir(), f"CS4 path is not a directory: {cs4_data_dir}"
    
    def test_cs4_core_files_exist(self, cs4_data_dir):
        """Test that CS4 core indicator files exist"""
        core_indicators = [
            'net_capital_flows',
            'net_direct_investment', 
            'net_portfolio_investment',
            'net_other_investment'
        ]
        
        versions = ['_full.csv', '_no_crises.csv']
        
        for indicator in core_indicators:
            for version in versions:
                filename = indicator + version
                file_path = cs4_data_dir / filename
                assert file_path.exists(), f"CS4 core file missing: {file_path}"
                assert file_path.stat().st_size > 0, f"CS4 file is empty: {file_path}"
    
    def test_cs4_portfolio_disaggregation_files(self, cs4_data_dir):
        """Test that CS4 portfolio investment disaggregation files exist"""
        portfolio_disaggregations = [
            'net_debt_portfolio_investment',
            'net_equity_portfolio_investment'
        ]
        
        versions = ['_full.csv', '_no_crises.csv']
        
        for disaggregation in portfolio_disaggregations:
            for version in versions:
                filename = disaggregation + version
                file_path = cs4_data_dir / filename
                assert file_path.exists(), f"CS4 portfolio disaggregation file missing: {file_path}"

class TestCS4DataLoading:
    """Test CS4 data loading functionality"""
    
    def test_cs4_file_loading(self, cs4_data_dir):
        """Test that CS4 files load without errors"""
        test_file = cs4_data_dir / "net_capital_flows_full.csv"
        
        try:
            df = pd.read_csv(test_file)
            assert isinstance(df, pd.DataFrame), "Failed to load as DataFrame"
            assert len(df) > 0, "Loaded DataFrame is empty"
            assert len(df.columns) > 0, "Loaded DataFrame has no columns"
        except Exception as e:
            pytest.fail(f"Failed to load CS4 test file: {e}")
    
    def test_cs4_data_structure_consistency(self, cs4_data_dir):
        """Test that CS4 files have consistent structure"""
        core_files = [
            "net_capital_flows_full.csv",
            "net_direct_investment_full.csv",
            "net_portfolio_investment_full.csv"
        ]
        
        expected_columns = None
        
        for filename in core_files:
            file_path = cs4_data_dir / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                if expected_columns is None:
                    expected_columns = set(df.columns)
                else:
                    # All files should have similar column structure
                    current_columns = set(df.columns)
                    # Allow for some variation but core columns should be consistent
                    common_columns = expected_columns & current_columns
                    assert len(common_columns) >= 5, \
                        f"Inconsistent structure in {filename}: only {len(common_columns)} common columns"
    
    def test_cs4_full_vs_no_crises_comparison(self, cs4_data_dir):
        """Test that full and no_crises versions are consistent"""
        indicators = ['net_capital_flows', 'net_direct_investment']
        
        for indicator in indicators:
            full_file = cs4_data_dir / f"{indicator}_full.csv"
            no_crises_file = cs4_data_dir / f"{indicator}_no_crises.csv"
            
            if full_file.exists() and no_crises_file.exists():
                df_full = pd.read_csv(full_file)
                df_no_crises = pd.read_csv(no_crises_file)
                
                # No crises version should have fewer or equal rows
                assert len(df_no_crises) <= len(df_full), \
                    f"No crises version has more data than full version for {indicator}"
                
                # Columns should be identical
                assert set(df_full.columns) == set(df_no_crises.columns), \
                    f"Column mismatch between full and no_crises versions for {indicator}"

class TestCS4StatisticalModelingData:
    """Test CS4 data suitability for statistical modeling"""
    
    def test_cs4_time_series_structure(self, cs4_data_dir):
        """Test that CS4 data has proper time series structure"""
        test_file = cs4_data_dir / "net_capital_flows_full.csv"
        
        if test_file.exists():
            df = pd.read_csv(test_file)
            
            # Should have time-related columns
            time_columns = ['YEAR', 'QUARTER', 'Date']
            available_time_cols = [col for col in time_columns if col in df.columns]
            assert len(available_time_cols) >= 1, f"No time columns found in CS4 data: {df.columns.tolist()}"
            
            # If YEAR column exists, check reasonable range
            if 'YEAR' in df.columns:
                years = df['YEAR'].dropna().unique()
                assert min(years) >= 1990, f"CS4 data starts too early: {min(years)}"
                assert max(years) <= 2030, f"CS4 data extends too far: {max(years)}"
    
    def test_cs4_country_coverage(self, cs4_data_dir):
        """Test that CS4 has adequate country coverage"""
        test_file = cs4_data_dir / "net_capital_flows_full.csv"
        
        if test_file.exists():
            df = pd.read_csv(test_file)
            
            if 'COUNTRY' in df.columns:
                countries = df['COUNTRY'].unique()
                
                # Should have multiple countries for comparative analysis
                assert len(countries) >= 5, f"Insufficient country coverage: {len(countries)} countries"
                
                # Should include key comparator countries
                key_countries = ['Iceland']
                for country in key_countries:
                    country_matches = [c for c in countries if 'Iceland' in c]
                    assert len(country_matches) > 0, f"Key country Iceland not found in CS4 data"
    
    def test_cs4_indicator_data_quality(self, cs4_data_dir):
        """Test CS4 indicator data quality for modeling"""
        test_files = [
            "net_capital_flows_full.csv",
            "net_direct_investment_full.csv"
        ]
        
        for filename in test_files:
            file_path = cs4_data_dir / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                # Find numeric columns that are likely indicators
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                indicator_cols = [col for col in numeric_cols if col not in ['YEAR', 'QUARTER']]
                
                # Test first few indicator columns
                for col in indicator_cols[:3]:
                    indicator_data = df[col].dropna()
                    
                    if len(indicator_data) > 0:
                        # Check for reasonable statistical properties
                        assert pd.api.types.is_numeric_dtype(indicator_data), \
                            f"Indicator {col} should be numeric in {filename}"
                        
                        # Check for extreme outliers that might break statistical models
                        if len(indicator_data) > 1:
                            q99 = indicator_data.quantile(0.99)
                            q01 = indicator_data.quantile(0.01)
                            
                            # Statistical models can be sensitive to extreme outliers
                            assert abs(q99) < 50000, \
                                f"Extreme outlier in {col} ({filename}): 99th percentile = {q99}"
                            assert abs(q01) < 50000, \
                                f"Extreme outlier in {col} ({filename}): 1st percentile = {q01}"

class TestCS4AdvancedAnalysisCapability:
    """Test CS4 data capability for advanced statistical analysis"""
    
    def test_cs4_variance_modeling_setup(self, cs4_data_dir):
        """Test that CS4 data supports variance modeling"""
        test_file = cs4_data_dir / "net_capital_flows_full.csv"
        
        if test_file.exists():
            df = pd.read_csv(test_file)
            
            # For variance modeling, need sufficient data points per group
            if 'COUNTRY' in df.columns:
                countries = df['COUNTRY'].unique()
                
                for country in countries[:3]:  # Test first 3 countries
                    country_data = df[df['COUNTRY'] == country]
                    
                    # Should have sufficient observations for variance estimation
                    assert len(country_data) >= 20, \
                        f"Insufficient data for variance modeling for {country}: {len(country_data)} observations"
                    
                    # Should span multiple years for temporal variance analysis
                    if 'YEAR' in country_data.columns:
                        year_span = country_data['YEAR'].max() - country_data['YEAR'].min()
                        assert year_span >= 10, \
                            f"Insufficient time span for variance modeling for {country}: {year_span} years"
    
    def test_cs4_crisis_period_identification(self, cs4_data_dir):
        """Test that CS4 can identify and handle crisis periods"""
        full_file = cs4_data_dir / "net_capital_flows_full.csv"
        no_crises_file = cs4_data_dir / "net_capital_flows_no_crises.csv"
        
        if full_file.exists() and no_crises_file.exists():
            df_full = pd.read_csv(full_file)
            df_no_crises = pd.read_csv(no_crises_file)
            
            if 'YEAR' in df_full.columns and 'YEAR' in df_no_crises.columns:
                # Crisis periods should be identifiable by data exclusion
                full_years = set(df_full['YEAR'].unique())
                no_crises_years = set(df_no_crises['YEAR'].unique())
                
                excluded_years = full_years - no_crises_years
                
                # Should have excluded some crisis years
                if len(excluded_years) > 0:
                    # Common crisis years
                    expected_crisis_years = {2008, 2009, 2010, 2020, 2021, 2022}
                    crisis_years_excluded = excluded_years & expected_crisis_years
                    
                    assert len(crisis_years_excluded) > 0, \
                        f"No expected crisis years excluded: {excluded_years}"
    
    def test_cs4_portfolio_disaggregation_analysis(self, cs4_data_dir):
        """Test CS4 portfolio investment disaggregation capability"""
        portfolio_files = [
            "net_portfolio_investment_full.csv",
            "net_debt_portfolio_investment_full.csv", 
            "net_equity_portfolio_investment_full.csv"
        ]
        
        existing_files = []
        for filename in portfolio_files:
            file_path = cs4_data_dir / filename
            if file_path.exists():
                existing_files.append(file_path)
        
        assert len(existing_files) >= 2, \
            f"Insufficient portfolio investment files for disaggregation analysis: {len(existing_files)}"
        
        # Test that disaggregated components are consistent with totals
        if len(existing_files) >= 3:
            # Load the files
            dfs = {}
            for file_path in existing_files:
                key = file_path.stem.replace('_full', '')
                dfs[key] = pd.read_csv(file_path)
            
            # Basic consistency check: disaggregated components should have similar structure
            if 'net_portfolio_investment' in dfs and 'net_debt_portfolio_investment' in dfs:
                total_df = dfs['net_portfolio_investment']
                debt_df = dfs['net_debt_portfolio_investment']
                
                # Should have compatible column structures
                common_cols = set(total_df.columns) & set(debt_df.columns)
                assert len(common_cols) >= 5, \
                    "Portfolio total and debt components have incompatible structures"

class TestCS4DataIntegrity:
    """Test CS4 data integrity and consistency"""
    
    def test_cs4_file_naming_consistency(self, cs4_data_dir):
        """Test that CS4 files follow consistent naming conventions"""
        files = list(cs4_data_dir.glob("*.csv"))
        
        # Check for consistent naming patterns
        expected_patterns = ['_full.csv', '_no_crises.csv']
        
        pattern_files = {}
        for pattern in expected_patterns:
            pattern_files[pattern] = [f for f in files if str(f).endswith(pattern)]
        
        # Should have files for both patterns
        for pattern, file_list in pattern_files.items():
            assert len(file_list) > 0, f"No files found with pattern {pattern}"
        
        # Should have similar numbers of full and no_crises files
        full_count = len(pattern_files['_full.csv'])
        no_crises_count = len(pattern_files['_no_crises.csv'])
        
        # Allow for some variation but should be roughly balanced
        ratio = max(full_count, no_crises_count) / min(full_count, no_crises_count)
        assert ratio <= 2.0, f"Imbalanced full vs no_crises files: {full_count} vs {no_crises_count}"
    
    def test_cs4_data_completeness(self, cs4_data_dir):
        """Test that CS4 data files are complete and not corrupted"""
        test_files = [
            "net_capital_flows_full.csv",
            "net_direct_investment_full.csv"
        ]
        
        for filename in test_files:
            file_path = cs4_data_dir / filename
            if file_path.exists():
                # Test file can be fully read
                try:
                    df = pd.read_csv(file_path)
                    
                    # Should not be empty
                    assert len(df) > 0, f"CS4 file {filename} is empty"
                    assert len(df.columns) > 0, f"CS4 file {filename} has no columns"
                    
                    # Should not have all missing values
                    non_null_cols = df.count().sum()
                    assert non_null_cols > 0, f"CS4 file {filename} has no non-null values"
                    
                except Exception as e:
                    pytest.fail(f"CS4 file {filename} appears corrupted: {e}")
    
    def test_cs4_no_trailing_spaces_in_filenames(self, cs4_data_dir):
        """Test that CS4 files don't have trailing spaces (prevent pipeline breaks)"""
        files = list(cs4_data_dir.glob("*"))
        
        for file_path in files:
            filename = file_path.name
            assert TestDataQuality.check_no_trailing_spaces(filename), \
                f"CS4 filename has trailing spaces: '{filename}'"