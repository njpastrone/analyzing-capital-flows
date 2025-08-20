"""
Data Loading Verification Tests for Capital Flows Research Project

Tests to prevent data pipeline breaks and ensure file access reliability
"""

import pytest
import pandas as pd
from pathlib import Path
from conftest import TestDataQuality

class TestDataFileAccess:
    """Test data file existence and accessibility"""
    
    def test_comprehensive_dataset_exists(self, comprehensive_data_file):
        """Test that comprehensive dataset file exists and is accessible"""
        assert comprehensive_data_file.exists(), f"Comprehensive dataset not found: {comprehensive_data_file}"
        assert comprehensive_data_file.is_file(), f"Path is not a file: {comprehensive_data_file}"
        assert comprehensive_data_file.stat().st_size > 0, "Comprehensive dataset is empty"
    
    def test_no_trailing_spaces_in_filename(self, comprehensive_data_file):
        """Test that comprehensive dataset filename has no trailing spaces"""
        assert TestDataQuality.check_no_trailing_spaces(comprehensive_data_file.name), \
            f"Filename has trailing spaces: '{comprehensive_data_file.name}'"
    
    def test_cs4_specialized_files_exist(self, cs4_data_dir):
        """Test that CS4 specialized data files exist"""
        assert cs4_data_dir.exists(), f"CS4 data directory not found: {cs4_data_dir}"
        
        # Expected CS4 files
        expected_files = [
            'net_capital_flows_full.csv',
            'net_capital_flows_no_crises.csv',
            'net_direct_investment_full.csv',
            'net_direct_investment_no_crises.csv',
            'net_portfolio_investment_full.csv',
            'net_portfolio_investment_no_crises.csv',
            'net_other_investment_full.csv',
            'net_other_investment_no_crises.csv'
        ]
        
        for file_name in expected_files:
            file_path = cs4_data_dir / file_name
            assert file_path.exists(), f"CS4 file missing: {file_path}"
    
    def test_cs5_data_directories_exist(self, cs5_controls_dir, cs5_regime_dir):
        """Test that CS5 data directories and files exist"""
        # Test CS5 Capital Controls directory
        assert cs5_controls_dir.exists(), f"CS5 controls directory not found: {cs5_controls_dir}"
        
        controls_files = list(cs5_controls_dir.glob('*.csv'))
        assert len(controls_files) >= 4, f"Expected at least 4 CS5 control files, found {len(controls_files)}"
        
        # Test CS5 Regime Analysis directory
        assert cs5_regime_dir.exists(), f"CS5 regime directory not found: {cs5_regime_dir}"
        
        regime_files = list(cs5_regime_dir.glob('*.csv'))
        assert len(regime_files) >= 8, f"Expected at least 8 CS5 regime files, found {len(regime_files)}"

class TestDataLoading:
    """Test data loading functionality"""
    
    def test_comprehensive_dataset_loads(self, comprehensive_data_file):
        """Test that comprehensive dataset loads without errors"""
        try:
            df = pd.read_csv(comprehensive_data_file)
            assert isinstance(df, pd.DataFrame), "Failed to load as DataFrame"
            assert len(df) > 0, "Loaded DataFrame is empty"
            assert len(df.columns) > 0, "Loaded DataFrame has no columns"
        except Exception as e:
            pytest.fail(f"Failed to load comprehensive dataset: {e}")
    
    def test_comprehensive_dataset_structure(self, comprehensive_dataframe):
        """Test that comprehensive dataset has expected structure"""
        df = comprehensive_dataframe
        
        # Check minimum expected columns
        required_columns = [
            'COUNTRY', 'YEAR', 'QUARTER', 'UNIT',
            'CS1_GROUP', 'CS2_GROUP', 'CS3_GROUP'
        ]
        
        has_required, missing = TestDataQuality.check_required_columns(df, required_columns)
        assert has_required, f"Missing required columns: {missing}"
        
        # Check for indicator columns (_PGDP suffix)
        indicator_cols = TestDataQuality.check_indicator_columns(df)
        assert len(indicator_cols) > 0, "No indicator columns found (ending with _PGDP)"
    
    def test_data_types_consistency(self, comprehensive_dataframe):
        """Test that data types are consistent"""
        df = comprehensive_dataframe
        
        # Check that YEAR is numeric
        assert pd.api.types.is_numeric_dtype(df['YEAR']), "YEAR column should be numeric"
        
        # Check that QUARTER is numeric
        assert pd.api.types.is_numeric_dtype(df['QUARTER']), "QUARTER column should be numeric"
        
        # Check that indicator columns are numeric
        indicator_cols = TestDataQuality.check_indicator_columns(df)
        for col in indicator_cols[:5]:  # Check first 5 indicators
            assert pd.api.types.is_numeric_dtype(df[col]), f"Indicator column {col} should be numeric"

class TestPathHandling:
    """Test path handling across different environments"""
    
    def test_absolute_paths_resolve(self, project_root):
        """Test that absolute paths resolve correctly"""
        data_path = project_root / "updated_data" / "Clean"
        assert data_path.is_absolute(), "Data path should be absolute"
        assert data_path.exists(), f"Data path should exist: {data_path}"
    
    def test_relative_path_construction(self, project_root):
        """Test that relative paths can be constructed correctly"""
        # Test path construction from different starting points
        dashboard_dir = project_root / "src" / "dashboard"
        if dashboard_dir.exists():
            # Simulate path construction from dashboard directory
            data_path = dashboard_dir.parent.parent / "updated_data" / "Clean"
            assert data_path.exists(), f"Relative path construction failed: {data_path}"
    
    def test_cross_platform_path_separators(self, comprehensive_data_file):
        """Test that paths work with different separators"""
        # Convert to string and back to Path to test separator handling
        path_str = str(comprehensive_data_file)
        reconstructed_path = Path(path_str)
        assert reconstructed_path.exists(), "Path separator conversion failed"

class TestFileEncodingAndFormat:
    """Test file encoding and format consistency"""
    
    def test_csv_format_consistency(self, comprehensive_data_file):
        """Test that CSV file has consistent format"""
        try:
            # Test different encoding options
            df_utf8 = pd.read_csv(comprehensive_data_file, encoding='utf-8')
            assert len(df_utf8) > 0, "UTF-8 encoding failed to load data"
        except UnicodeDecodeError:
            pytest.fail("CSV file has encoding issues")
    
    def test_no_corrupted_data_markers(self, comprehensive_dataframe):
        """Test that data doesn't contain corruption markers"""
        df = comprehensive_dataframe
        
        # Check for common corruption markers
        corruption_markers = ['�', 'NULL', 'NaN', '#N/A']
        
        for col in df.select_dtypes(include=['object']).columns[:5]:  # Check first 5 string columns
            for marker in corruption_markers:
                corrupt_count = df[col].astype(str).str.contains(marker, na=False).sum()
                if corrupt_count > 0:
                    pytest.fail(f"Found {corrupt_count} corruption markers '{marker}' in column {col}")

class TestErrorHandling:
    """Test graceful error handling"""
    
    def test_missing_file_handling(self, data_dir):
        """Test that missing files are handled gracefully"""
        fake_file = data_dir / "nonexistent_file.csv"
        assert not fake_file.exists(), "Test file should not exist"
        
        # Test that we can detect missing files without crashing
        try:
            if fake_file.exists():
                df = pd.read_csv(fake_file)
            else:
                # This is the expected path - file doesn't exist
                assert True
        except Exception as e:
            # If an exception occurs, it should be a file not found error
            assert "FileNotFoundError" in str(type(e)) or "No such file" in str(e)
    
    def test_empty_file_handling(self, data_dir):
        """Test handling of empty or minimal files"""
        # Create a temporary minimal CSV file
        temp_file = data_dir / "temp_test_file.csv"
        try:
            temp_file.write_text("column1\nvalue1\n")
            df = pd.read_csv(temp_file)
            assert len(df) == 1, "Should load minimal file correctly"
            assert 'column1' in df.columns, "Should parse header correctly"
        finally:
            if temp_file.exists():
                temp_file.unlink()  # Clean up