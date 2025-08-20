"""
Case Study 5 Specific Functionality Tests - Capital Controls and Exchange Rate Regimes

Tests CS5 data loading and external data integration
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from conftest import TestDataQuality

class TestCS5DataDirectories:
    """Test CS5 data directory structure and access"""
    
    def test_cs5_controls_directory_exists(self, cs5_controls_dir):
        """Test that CS5 capital controls directory exists"""
        assert cs5_controls_dir.exists(), f"CS5 controls directory not found: {cs5_controls_dir}"
        assert cs5_controls_dir.is_dir(), f"CS5 controls path is not a directory: {cs5_controls_dir}"
    
    def test_cs5_regime_directory_exists(self, cs5_regime_dir):
        """Test that CS5 regime analysis directory exists"""
        assert cs5_regime_dir.exists(), f"CS5 regime directory not found: {cs5_regime_dir}"
        assert cs5_regime_dir.is_dir(), f"CS5 regime path is not a directory: {cs5_regime_dir}"
    
    def test_cs5_controls_files_exist(self, cs5_controls_dir):
        """Test that CS5 capital controls files exist"""
        expected_files = [
            'sd_yearly_flows.csv',
            'sd_yearly_flows_no_outliers.csv',
            'sd_country_flows.csv',
            'sd_country_flows_no_outliers.csv'
        ]
        
        for filename in expected_files:
            file_path = cs5_controls_dir / filename
            assert file_path.exists(), f"CS5 controls file missing: {file_path}"
            assert file_path.stat().st_size > 0, f"CS5 controls file is empty: {file_path}"
    
    def test_cs5_regime_files_exist(self, cs5_regime_dir):
        """Test that CS5 exchange rate regime files exist"""
        # Expected regime analysis files
        indicators = ['net_capital_flows', 'net_direct_investment', 
                     'net_portfolio_investment', 'net_other_investment']
        versions = ['_full.csv', '_no_crises.csv']
        
        expected_file_count = len(indicators) * len(versions)  # 4 * 2 = 8
        
        actual_files = list(cs5_regime_dir.glob("*.csv"))
        assert len(actual_files) >= expected_file_count, \
            f"Expected at least {expected_file_count} CS5 regime files, found {len(actual_files)}"
        
        # Check for specific core files
        core_files = ['net_capital_flows_full.csv', 'net_capital_flows_no_crises.csv']
        for filename in core_files:
            file_path = cs5_regime_dir / filename
            assert file_path.exists(), f"CS5 regime core file missing: {file_path}"

class TestCS5DataLoading:
    """Test CS5 data loading functionality"""
    
    def test_cs5_controls_data_loading(self, cs5_controls_dir):
        """Test that CS5 capital controls data loads correctly"""
        test_file = cs5_controls_dir / "sd_yearly_flows.csv"
        
        try:
            df = pd.read_csv(test_file)
            assert isinstance(df, pd.DataFrame), "Failed to load CS5 controls data as DataFrame"
            assert len(df) > 0, "CS5 controls DataFrame is empty"
            assert len(df.columns) > 0, "CS5 controls DataFrame has no columns"
        except Exception as e:
            pytest.fail(f"Failed to load CS5 controls data: {e}")
    
    def test_cs5_regime_data_loading(self, cs5_regime_dir):
        """Test that CS5 regime analysis data loads correctly"""
        test_file = cs5_regime_dir / "net_capital_flows_full.csv"
        
        try:
            df = pd.read_csv(test_file)
            assert isinstance(df, pd.DataFrame), "Failed to load CS5 regime data as DataFrame"
            assert len(df) > 0, "CS5 regime DataFrame is empty"
            assert len(df.columns) > 0, "CS5 regime DataFrame has no columns"
        except Exception as e:
            pytest.fail(f"Failed to load CS5 regime data: {e}")
    
    def test_cs5_outlier_vs_regular_versions(self, cs5_controls_dir):
        """Test consistency between regular and no_outliers versions"""
        regular_file = cs5_controls_dir / "sd_yearly_flows.csv"
        no_outliers_file = cs5_controls_dir / "sd_yearly_flows_no_outliers.csv"
        
        if regular_file.exists() and no_outliers_file.exists():
            df_regular = pd.read_csv(regular_file)
            df_no_outliers = pd.read_csv(no_outliers_file)
            
            # No outliers version should have fewer or equal rows
            assert len(df_no_outliers) <= len(df_regular), \
                "No outliers version has more data than regular version"
            
            # Columns should be identical
            assert set(df_regular.columns) == set(df_no_outliers.columns), \
                "Column mismatch between regular and no_outliers versions"

class TestCS5ExternalDataIntegration:
    """Test CS5 integration with external data sources"""
    
    def test_cs5_capital_controls_data_structure(self, cs5_controls_dir):
        """Test that CS5 capital controls data has expected structure for correlation analysis"""
        test_file = cs5_controls_dir / "sd_yearly_flows.csv"
        
        if test_file.exists():
            df = pd.read_csv(test_file)
            
            # Should have time-related structure for correlation analysis
            likely_time_cols = [col for col in df.columns if any(time_word in col.lower() 
                              for time_word in ['year', 'time', 'date', 'period'])]
            assert len(likely_time_cols) > 0, \
                f"No time-related columns found in CS5 controls data: {df.columns.tolist()}"
            
            # Should have country or entity identifiers
            likely_country_cols = [col for col in df.columns if any(country_word in col.lower()
                                 for country_word in ['country', 'economy', 'iso', 'code'])]
            assert len(likely_country_cols) > 0, \
                f"No country identifier columns found in CS5 controls data: {df.columns.tolist()}"
    
    def test_cs5_regime_classification_coverage(self, cs5_regime_dir):
        """Test that CS5 has adequate coverage for exchange rate regime analysis"""
        test_file = cs5_regime_dir / "net_capital_flows_full.csv"
        
        if test_file.exists():
            df = pd.read_csv(test_file)
            
            # Should have country coverage for regime analysis
            if 'COUNTRY' in df.columns:
                countries = df['COUNTRY'].unique()
                assert len(countries) >= 5, \
                    f"Insufficient country coverage for regime analysis: {len(countries)} countries"
                
                # Should include Iceland as a key case
                iceland_present = any('Iceland' in str(country) for country in countries)
                assert iceland_present, "Iceland should be present in CS5 regime analysis"
            
            # Should have time coverage for regime comparison
            if 'YEAR' in df.columns:
                years = df['YEAR'].unique()
                year_span = max(years) - min(years)
                assert year_span >= 15, \
                    f"Insufficient time span for regime analysis: {year_span} years"

class TestCS5CorrelationAnalysisCapability:
    """Test CS5 data suitability for correlation analysis"""
    
    def test_cs5_correlation_data_structure(self, cs5_controls_dir):
        """Test that CS5 data supports correlation analysis"""
        test_files = ['sd_yearly_flows.csv', 'sd_country_flows.csv']
        
        for filename in test_files:
            file_path = cs5_controls_dir / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                # Should have numeric columns suitable for correlation
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                assert len(numeric_cols) >= 2, \
                    f"Insufficient numeric columns for correlation in {filename}: {len(numeric_cols)}"
                
                # Test that we have reasonable variation in numeric columns
                for col in numeric_cols[:3]:  # Test first 3 numeric columns
                    col_data = df[col].dropna()
                    if len(col_data) > 1:
                        variance = col_data.var()
                        # Should have some variation for meaningful correlation
                        if variance == 0:
                            unique_values = col_data.nunique()
                            # Allow for legitimately constant variables
                            assert unique_values <= 2, \
                                f"Column {col} in {filename} has suspicious zero variance"
    
    def test_cs5_time_series_alignment(self, cs5_controls_dir, cs5_regime_dir):
        """Test that CS5 controls and regime data can be temporally aligned"""
        controls_file = cs5_controls_dir / "sd_yearly_flows.csv"
        regime_file = cs5_regime_dir / "net_capital_flows_full.csv"
        
        if controls_file.exists() and regime_file.exists():
            df_controls = pd.read_csv(controls_file)
            df_regime = pd.read_csv(regime_file)
            
            # Try to find overlapping time periods
            controls_time_cols = [col for col in df_controls.columns 
                                if 'year' in col.lower() or 'time' in col.lower()]
            regime_time_cols = [col for col in df_regime.columns 
                              if 'year' in col.lower() or 'YEAR' in col]
            
            if len(controls_time_cols) > 0 and len(regime_time_cols) > 0:
                # Check for temporal overlap potential
                controls_time_col = controls_time_cols[0]
                regime_time_col = regime_time_cols[0]
                
                controls_years = set(df_controls[controls_time_col].dropna())
                regime_years = set(df_regime[regime_time_col].dropna())
                
                # Should have some overlapping years for correlation analysis
                overlap = controls_years & regime_years
                if len(overlap) == 0:
                    # If no direct overlap, check if ranges are reasonable
                    controls_range = (min(controls_years), max(controls_years)) if controls_years else (0, 0)
                    regime_range = (min(regime_years), max(regime_years)) if regime_years else (0, 0)
                    
                    # Should have reasonable time coverage
                    assert controls_range[1] >= 2000, \
                        f"CS5 controls data coverage too old: {controls_range}"
                    assert regime_range[1] >= 2010, \
                        f"CS5 regime data coverage too old: {regime_range}"

class TestCS5AnalysisMethodology:
    """Test CS5 methodology and analysis approach"""
    
    def test_cs5_volatility_measures(self, cs5_controls_dir):
        """Test that CS5 volatility measures are properly constructed"""
        sd_files = ['sd_yearly_flows.csv', 'sd_country_flows.csv']
        
        for filename in sd_files:
            file_path = cs5_controls_dir / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                # Standard deviation files should have positive values (or zero)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols[:3]:  # Test first 3 numeric columns
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        # Standard deviations should not be negative
                        negative_count = (col_data < 0).sum()
                        assert negative_count == 0, \
                            f"Found negative standard deviations in {col} ({filename}): {negative_count} values"
                        
                        # Should have reasonable range for economic data
                        max_val = col_data.max()
                        assert max_val < 10000, \
                            f"Extreme volatility measure in {col} ({filename}): {max_val}"
    
    def test_cs5_regime_classification_consistency(self, cs5_regime_dir):
        """Test that CS5 regime classification is consistent across indicators"""
        indicators = ['net_capital_flows', 'net_direct_investment']
        
        regime_data = {}
        for indicator in indicators:
            file_path = cs5_regime_dir / f"{indicator}_full.csv"
            if file_path.exists():
                regime_data[indicator] = pd.read_csv(file_path)
        
        if len(regime_data) >= 2:
            # Should have consistent country and time coverage across indicators
            indicator_keys = list(regime_data.keys())
            df1 = regime_data[indicator_keys[0]]
            df2 = regime_data[indicator_keys[1]]
            
            # Check for consistent structure
            if 'COUNTRY' in df1.columns and 'COUNTRY' in df2.columns:
                countries1 = set(df1['COUNTRY'].unique())
                countries2 = set(df2['COUNTRY'].unique())
                
                # Should have substantial overlap in country coverage
                overlap = countries1 & countries2
                union = countries1 | countries2
                overlap_ratio = len(overlap) / len(union)
                
                assert overlap_ratio >= 0.7, \
                    f"Inconsistent country coverage across CS5 regime indicators: {overlap_ratio:.2f}"

class TestCS5DataIntegrity:
    """Test CS5 data integrity and consistency"""
    
    def test_cs5_file_naming_consistency(self, cs5_controls_dir, cs5_regime_dir):
        """Test that CS5 files follow consistent naming conventions"""
        # Test controls directory naming
        controls_files = list(cs5_controls_dir.glob("*.csv"))
        for file_path in controls_files:
            filename = file_path.name
            assert TestDataQuality.check_no_trailing_spaces(filename), \
                f"CS5 controls filename has trailing spaces: '{filename}'"
            assert filename.endswith('.csv'), f"CS5 controls file should be CSV: {filename}"
        
        # Test regime directory naming
        regime_files = list(cs5_regime_dir.glob("*.csv"))
        for file_path in regime_files:
            filename = file_path.name
            assert TestDataQuality.check_no_trailing_spaces(filename), \
                f"CS5 regime filename has trailing spaces: '{filename}'"
            assert filename.endswith('.csv'), f"CS5 regime file should be CSV: {filename}"
    
    def test_cs5_data_quality_controls(self, cs5_controls_dir):
        """Test CS5 data quality and reasonable value ranges"""
        test_file = cs5_controls_dir / "sd_yearly_flows.csv"
        
        if test_file.exists():
            df = pd.read_csv(test_file)
            
            # Test for data completeness
            total_cells = df.shape[0] * df.shape[1]
            null_cells = df.isnull().sum().sum()
            null_ratio = null_cells / total_cells
            
            # Should not be mostly empty
            assert null_ratio < 0.8, f"CS5 controls data is mostly empty: {null_ratio:.2f} null ratio"
            
            # Test numeric columns for quality
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:3]:  # Test first 3 numeric columns
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # Check for infinite or extremely large values
                    inf_count = np.isinf(col_data).sum()
                    assert inf_count == 0, f"Found infinite values in {col}"
                    
                    # Check for NaN (should be handled by dropna, but double-check)
                    nan_count = np.isnan(col_data).sum()
                    assert nan_count == 0, f"Found NaN values after dropna in {col}"
    
    def test_cs5_external_data_integration_feasibility(self, cs5_controls_dir):
        """Test that CS5 data structure supports external data integration"""
        test_file = cs5_controls_dir / "sd_yearly_flows.csv"
        
        if test_file.exists():
            df = pd.read_csv(test_file)
            
            # Should have identifiers suitable for merging with external data
            potential_merge_cols = []
            
            # Check for country identifiers
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['country', 'iso', 'code', 'economy']):
                    potential_merge_cols.append(col)
            
            # Check for time identifiers
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['year', 'time', 'date', 'period']):
                    potential_merge_cols.append(col)
            
            assert len(potential_merge_cols) >= 2, \
                f"Insufficient merge keys for external data integration: {potential_merge_cols}"