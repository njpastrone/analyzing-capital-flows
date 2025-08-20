"""
Case Study 1 Specific Functionality Tests - Iceland vs Eurozone

Tests CS1 data loading, filtering, and statistical analysis
"""

import pytest
import pandas as pd
import numpy as np
from conftest import TestDataQuality

# Import CS1 functions
try:
    from simple_report_app import load_default_data, calculate_group_statistics, perform_volatility_tests
    CS1_IMPORTS_AVAILABLE = True
except ImportError as e:
    CS1_IMPORTS_AVAILABLE = False
    import_error = str(e)

class TestCS1DataLoading:
    """Test CS1 data loading functionality"""
    
    @pytest.mark.skipif(not CS1_IMPORTS_AVAILABLE, reason=f"CS1 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs1_full_period_loading(self):
        """Test CS1 loads full period data correctly"""
        final_data, analysis_indicators, metadata = load_default_data(include_crisis_years=True)
        
        assert final_data is not None, "CS1 full period data loading failed"
        assert analysis_indicators is not None, "CS1 indicators loading failed"
        assert metadata is not None, "CS1 metadata loading failed"
        
        # Check metadata
        assert metadata['study_version'] == "Full Time Period", "Incorrect study version"
        assert metadata['include_crisis_years'] is True, "Crisis years flag incorrect"
        
        # Check data structure
        expected_shape = (1093, 25)
        assert final_data.shape == expected_shape, f"Expected shape {expected_shape}, got {final_data.shape}"
        
        # Check indicators count
        assert len(analysis_indicators) == 14, f"Expected 14 indicators, got {len(analysis_indicators)}"
    
    @pytest.mark.skipif(not CS1_IMPORTS_AVAILABLE, reason=f"CS1 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs1_crisis_excluded_loading(self):
        """Test CS1 loads crisis-excluded data correctly"""
        final_data, analysis_indicators, metadata = load_default_data(include_crisis_years=False)
        
        assert final_data is not None, "CS1 crisis-excluded data loading failed"
        
        # Check metadata
        assert metadata['study_version'] == "Crisis-Excluded", "Incorrect study version"
        assert metadata['include_crisis_years'] is False, "Crisis years flag incorrect"
        
        # Check that crisis years are excluded
        crisis_years = [2008, 2009, 2010, 2020, 2021, 2022]
        actual_years = set(final_data['YEAR'].unique())
        crisis_years_present = set(crisis_years) & actual_years
        assert len(crisis_years_present) == 0, f"Crisis years still present: {crisis_years_present}"
        
        # Should have fewer observations than full period
        assert len(final_data) < 1093, "Crisis-excluded should have fewer observations"

class TestCS1GroupFiltering:
    """Test CS1 country group filtering"""
    
    @pytest.mark.skipif(not CS1_IMPORTS_AVAILABLE, reason=f"CS1 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs1_group_integrity(self, expected_cs1_countries):
        """Test that CS1 groups are correctly formed"""
        final_data, _, _ = load_default_data(include_crisis_years=True)
        
        # Check CS1_GROUP column exists
        assert 'CS1_GROUP' in final_data.columns, "CS1_GROUP column missing"
        
        # Check group values
        cs1_groups = final_data['CS1_GROUP'].unique()
        expected_groups = ['Iceland', 'Eurozone']
        assert set(cs1_groups) == set(expected_groups), f"Expected {expected_groups}, got {list(cs1_groups)}"
        
        # Check Iceland group
        iceland_data = final_data[final_data['CS1_GROUP'] == 'Iceland']
        iceland_countries = iceland_data['COUNTRY'].unique()
        assert list(iceland_countries) == ['Iceland'], f"Iceland group should only contain Iceland, got {list(iceland_countries)}"
        
        # Check Eurozone group  
        eurozone_data = final_data[final_data['CS1_GROUP'] == 'Eurozone']
        eurozone_countries = set(eurozone_data['COUNTRY'].unique())
        
        # Should contain main Eurozone countries (excluding Luxembourg)
        expected_eurozone = set(expected_cs1_countries) - {'Iceland'}
        assert eurozone_countries.issubset(set(expected_cs1_countries)), \
            f"Unexpected countries in Eurozone group: {eurozone_countries - set(expected_cs1_countries)}"
    
    @pytest.mark.skipif(not CS1_IMPORTS_AVAILABLE, reason=f"CS1 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs1_observations_per_group(self):
        """Test expected number of observations per group"""
        final_data, _, _ = load_default_data(include_crisis_years=True)
        
        # Check Iceland observations
        iceland_count = len(final_data[final_data['CS1_GROUP'] == 'Iceland'])
        assert iceland_count == 105, f"Expected 105 Iceland observations, got {iceland_count}"
        
        # Check Eurozone observations  
        eurozone_count = len(final_data[final_data['CS1_GROUP'] == 'Eurozone'])
        assert eurozone_count == 988, f"Expected 988 Eurozone observations, got {eurozone_count}"
        
        # Total should match expected
        total_count = iceland_count + eurozone_count
        assert total_count == 1093, f"Total observations should be 1093, got {total_count}"

class TestCS1StatisticalFunctions:
    """Test CS1 statistical analysis functions"""
    
    @pytest.mark.skipif(not CS1_IMPORTS_AVAILABLE, reason=f"CS1 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs1_summary_statistics(self):
        """Test CS1 summary statistics calculation"""
        final_data, analysis_indicators, _ = load_default_data(include_crisis_years=True)
        
        # Calculate summary statistics
        summary_stats = calculate_group_statistics(final_data, 'CS1_GROUP', analysis_indicators)
        
        assert len(summary_stats) > 0, "Summary statistics calculation failed"
        
        # Should have stats for both groups - handle DataFrame format
        if isinstance(summary_stats, pd.DataFrame):
            if 'Group' in summary_stats.columns:
                groups_in_stats = set(summary_stats['Group'].unique())
                expected_groups = {'Iceland', 'Eurozone'}
                assert groups_in_stats == expected_groups, f"Expected groups {expected_groups}, got {groups_in_stats}"
            
            # Check that we have stats for multiple indicators
            if 'Indicator' in summary_stats.columns:
                indicators_in_stats = set(summary_stats['Indicator'].unique())
                assert len(indicators_in_stats) >= 10, f"Expected stats for at least 10 indicators, got {len(indicators_in_stats)}"
            else:
                # Accept that we have summary statistics in some format
                assert len(summary_stats) >= 20, f"Expected comprehensive statistics, got {len(summary_stats)} rows"
    
    @pytest.mark.skipif(not CS1_IMPORTS_AVAILABLE, reason=f"CS1 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs1_volatility_tests(self):
        """Test CS1 volatility (F-test) calculations"""
        final_data, analysis_indicators, _ = load_default_data(include_crisis_years=True)
        
        # Perform volatility tests
        f_test_results = perform_volatility_tests(final_data, analysis_indicators)
        
        assert len(f_test_results) > 0, "F-test calculations failed"
        assert len(f_test_results) == len(analysis_indicators), \
            f"Expected {len(analysis_indicators)} F-tests, got {len(f_test_results)}"
        
        # Check that results contain expected fields - handle DataFrame format
        if len(f_test_results) > 0:
            if isinstance(f_test_results, pd.DataFrame):
                # Check for expected columns in DataFrame
                expected_fields = ['Indicator', 'F_Statistic', 'P_Value']
                available_fields = set(f_test_results.columns)
                
                # Should have some key statistical fields
                key_fields_present = [field for field in expected_fields if field in available_fields]
                assert len(key_fields_present) >= 2, f"F-test DataFrame missing key fields. Available: {list(available_fields)}"
                
                # Should have results for each indicator
                if 'Indicator' in f_test_results.columns:
                    unique_indicators = len(f_test_results['Indicator'].unique())
                    assert unique_indicators >= 10, f"Expected F-tests for multiple indicators, got {unique_indicators}"
            else:
                # Handle list of dictionaries format
                first_result = f_test_results[0]
                expected_fields = ['Indicator', 'Iceland_Variance', 'Eurozone_Variance', 'F_Statistic']
                
                if isinstance(first_result, dict):
                    result_fields = set(first_result.keys())
                    missing_fields = set(expected_fields) - result_fields
                    assert len(missing_fields) <= 1, f"F-test results missing expected fields: {missing_fields}"

class TestCS1DataIntegrity:
    """Test CS1 data integrity and consistency"""
    
    @pytest.mark.skipif(not CS1_IMPORTS_AVAILABLE, reason=f"CS1 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs1_indicator_data_quality(self):
        """Test that CS1 indicator data has reasonable values"""
        final_data, analysis_indicators, _ = load_default_data(include_crisis_years=True)
        
        # Check first few indicators for data quality
        for indicator in analysis_indicators[:3]:
            indicator_data = final_data[indicator].dropna()
            
            # Should have numeric data
            assert pd.api.types.is_numeric_dtype(indicator_data), f"Indicator {indicator} should be numeric"
            
            # Should not be all zeros (unless it's a legitimate zero-flow indicator)
            non_zero_count = (indicator_data != 0).sum()
            assert non_zero_count > 0, f"Indicator {indicator} appears to be all zeros"
            
            # Check for extreme outliers (basic sanity check)
            if len(indicator_data) > 0:
                q99 = indicator_data.quantile(0.99)
                q01 = indicator_data.quantile(0.01)
                # Values shouldn't be astronomically large (> 10000% of GDP)
                assert abs(q99) < 10000, f"Indicator {indicator} has extreme outliers: 99th percentile = {q99}"
                assert abs(q01) < 10000, f"Indicator {indicator} has extreme outliers: 1st percentile = {q01}"
    
    @pytest.mark.skipif(not CS1_IMPORTS_AVAILABLE, reason=f"CS1 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs1_time_series_consistency(self):
        """Test that CS1 time series data is consistent"""
        final_data, _, _ = load_default_data(include_crisis_years=True)
        
        # Check year range
        min_year = final_data['YEAR'].min()
        max_year = final_data['YEAR'].max()
        
        assert min_year >= 1999, f"Data starts too early: {min_year}"
        assert max_year <= 2025, f"Data extends too far: {max_year}"
        assert max_year >= 2020, f"Data doesn't include recent years: {max_year}"
        
        # Check quarter values
        quarters = set(final_data['QUARTER'].unique())
        expected_quarters = {1, 2, 3, 4}
        assert quarters == expected_quarters, f"Expected quarters {expected_quarters}, got {quarters}"
    
    @pytest.mark.skipif(not CS1_IMPORTS_AVAILABLE, reason=f"CS1 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs1_luxembourg_exclusion(self):
        """Test that Luxembourg is excluded from CS1 analysis"""
        final_data, _, _ = load_default_data(include_crisis_years=True)
        
        countries = set(final_data['COUNTRY'].unique())
        assert 'Luxembourg' not in countries, "Luxembourg should be excluded from CS1 analysis"
        
        # Check that we still have the main Eurozone countries
        expected_eurozone_countries = [
            'Austria', 'Belgium', 'Finland', 'France', 'Germany',
            'Ireland', 'Italy', 'Netherlands, The', 'Portugal', 'Spain'
        ]
        
        for country in expected_eurozone_countries:
            assert country in countries, f"Expected Eurozone country {country} missing from CS1"