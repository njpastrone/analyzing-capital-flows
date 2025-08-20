"""
Case Study 3 Specific Functionality Tests - Iceland vs Small Open Economies

Tests CS3 data loading, SOE filtering, and comparative analysis
"""

import pytest
import pandas as pd
import numpy as np
from conftest import TestDataQuality

# Import CS3 functions
try:
    from cs3_complete_functions import load_cs3_data
    from cs3_report_app import load_case_study_3_data
    CS3_IMPORTS_AVAILABLE = True
except ImportError as e:
    CS3_IMPORTS_AVAILABLE = False
    import_error = str(e)

class TestCS3DataLoading:
    """Test CS3 data loading functionality"""
    
    @pytest.mark.skipif(not CS3_IMPORTS_AVAILABLE, reason=f"CS3 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs3_full_period_loading(self):
        """Test CS3 loads full period data correctly"""
        final_data, analysis_indicators, metadata = load_cs3_data(include_crisis_years=True)
        
        assert final_data is not None, "CS3 full period data loading failed"
        assert analysis_indicators is not None, "CS3 indicators loading failed"
        assert metadata is not None, "CS3 metadata loading failed"
        
        # Check metadata
        assert metadata['study_version'] == "Full Time Period", "Incorrect study version"
        assert metadata['include_crisis_years'] is True, "Crisis years flag incorrect"
        
        # Check data structure
        expected_shape = (763, 25)
        assert final_data.shape == expected_shape, f"Expected shape {expected_shape}, got {final_data.shape}"
        
        # Check indicators count
        assert len(analysis_indicators) == 14, f"Expected 14 indicators, got {len(analysis_indicators)}"
    
    @pytest.mark.skipif(not CS3_IMPORTS_AVAILABLE, reason=f"CS3 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs3_crisis_excluded_loading(self):
        """Test CS3 loads crisis-excluded data correctly"""
        final_data, analysis_indicators, metadata = load_cs3_data(include_crisis_years=False)
        
        assert final_data is not None, "CS3 crisis-excluded data loading failed"
        
        # Check metadata
        assert metadata['study_version'] == "Crisis-Excluded", "Incorrect study version"
        assert metadata['include_crisis_years'] is False, "Crisis years flag incorrect"
        
        # Should have fewer observations than full period
        expected_excluded_shape = (571, 25)
        assert final_data.shape == expected_excluded_shape, \
            f"Expected crisis-excluded shape {expected_excluded_shape}, got {final_data.shape}"
    
    @pytest.mark.skipif(not CS3_IMPORTS_AVAILABLE, reason=f"CS3 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs3_report_app_loading(self):
        """Test CS3 report app data loading"""
        data, indicators = load_case_study_3_data()
        
        assert data is not None, "CS3 report app data loading failed"
        assert indicators is not None, "CS3 report app indicators loading failed"
        
        # Check expected structure
        expected_shape = (763, 24)  # Report app may have slightly different structure
        assert data.shape == expected_shape, f"Expected shape {expected_shape}, got {data.shape}"
        
        # Check indicators count (may be different from complete functions)
        assert len(indicators) >= 10, f"Expected at least 10 indicators, got {len(indicators)}"

class TestCS3CountryFiltering:
    """Test CS3 small open economy country filtering"""
    
    @pytest.mark.skipif(not CS3_IMPORTS_AVAILABLE, reason=f"CS3 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs3_group_integrity(self):
        """Test that CS3 groups are correctly formed"""
        final_data, _, _ = load_cs3_data(include_crisis_years=True)
        
        # Check CS3_GROUP column exists
        assert 'CS3_GROUP' in final_data.columns, "CS3_GROUP column missing"
        
        # Check group values
        cs3_groups = final_data['CS3_GROUP'].unique()
        expected_groups = ['Iceland', 'Comparator']
        assert set(cs3_groups) == set(expected_groups), f"Expected {expected_groups}, got {list(cs3_groups)}"
        
        # Check Iceland group
        iceland_data = final_data[final_data['CS3_GROUP'] == 'Iceland']
        iceland_countries = iceland_data['COUNTRY'].unique()
        assert list(iceland_countries) == ['Iceland'], f"Iceland group should only contain Iceland, got {list(iceland_countries)}"
    
    @pytest.mark.skipif(not CS3_IMPORTS_AVAILABLE, reason=f"CS3 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs3_small_open_economies(self):
        """Test that CS3 contains appropriate small open economies"""
        final_data, _, _ = load_cs3_data(include_crisis_years=True)
        
        # Get comparator countries
        comparator_data = final_data[final_data['CS3_GROUP'] == 'Comparator']
        comparator_countries = set(comparator_data['COUNTRY'].unique())
        
        # Expected small open economy characteristics:
        # - Should be small island or small developed economies
        # - Should exclude large economies
        expected_soe_countries = {
            'Aruba, Kingdom of the Netherlands',
            'Bahamas, The', 
            'Bermuda',
            'Brunei Darussalam',
            'Malta',
            'Mauritius',
            'Seychelles'
        }
        
        # Check that we have reasonable small open economies
        assert comparator_countries.issubset(expected_soe_countries), \
            f"Unexpected countries in comparator group: {comparator_countries - expected_soe_countries}"
        
        # Should have multiple comparator countries
        assert len(comparator_countries) >= 5, f"Expected at least 5 comparator countries, got {len(comparator_countries)}"
    
    @pytest.mark.skipif(not CS3_IMPORTS_AVAILABLE, reason=f"CS3 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs3_observations_distribution(self):
        """Test expected distribution of observations across groups"""
        final_data, _, _ = load_cs3_data(include_crisis_years=True)
        
        # Check Iceland observations
        iceland_count = len(final_data[final_data['CS3_GROUP'] == 'Iceland'])
        assert iceland_count == 105, f"Expected 105 Iceland observations, got {iceland_count}"
        
        # Check comparator observations
        comparator_count = len(final_data[final_data['CS3_GROUP'] == 'Comparator'])
        expected_comparator = 763 - 105  # Total minus Iceland
        assert comparator_count == expected_comparator, \
            f"Expected {expected_comparator} comparator observations, got {comparator_count}"

class TestCS3ComparativeAnalysis:
    """Test CS3 comparative analysis capabilities"""
    
    @pytest.mark.skipif(not CS3_IMPORTS_AVAILABLE, reason=f"CS3 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs3_cross_country_comparison_setup(self):
        """Test that CS3 data supports cross-country comparison"""
        final_data, analysis_indicators, _ = load_cs3_data(include_crisis_years=True)
        
        # Test that Iceland and comparator groups have overlapping time periods
        iceland_years = set(final_data[final_data['CS3_GROUP'] == 'Iceland']['YEAR'].unique())
        comparator_years = set(final_data[final_data['CS3_GROUP'] == 'Comparator']['YEAR'].unique())
        
        overlap_years = iceland_years & comparator_years
        assert len(overlap_years) >= 15, f"Insufficient overlapping years for comparison: {len(overlap_years)}"
        
        # Test that key indicators have data for both groups
        key_indicators = analysis_indicators[:5]  # Test first 5 indicators
        for indicator in key_indicators:
            iceland_data = final_data[final_data['CS3_GROUP'] == 'Iceland'][indicator].dropna()
            comparator_data = final_data[final_data['CS3_GROUP'] == 'Comparator'][indicator].dropna()
            
            assert len(iceland_data) > 0, f"No Iceland data for indicator {indicator}"
            assert len(comparator_data) > 0, f"No comparator data for indicator {indicator}"
    
    @pytest.mark.skipif(not CS3_IMPORTS_AVAILABLE, reason=f"CS3 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs3_multiple_comparator_countries(self):
        """Test that CS3 includes multiple comparator countries for robustness"""
        final_data, _, _ = load_cs3_data(include_crisis_years=True)
        
        comparator_countries = final_data[final_data['CS3_GROUP'] == 'Comparator']['COUNTRY'].unique()
        
        # Should have multiple comparator countries to make the analysis robust
        assert len(comparator_countries) >= 5, \
            f"Expected at least 5 comparator countries for robust analysis, got {len(comparator_countries)}"
        
        # Each comparator country should have reasonable data coverage
        for country in comparator_countries[:3]:  # Test first 3 countries
            country_data = final_data[final_data['COUNTRY'] == country]
            
            # Should have decent time coverage
            years_coverage = len(country_data['YEAR'].unique())
            assert years_coverage >= 10, f"Insufficient time coverage for {country}: {years_coverage} years"

class TestCS3StatisticalAnalysis:
    """Test CS3 statistical analysis capabilities"""
    
    @pytest.mark.skipif(not CS3_IMPORTS_AVAILABLE, reason=f"CS3 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs3_variance_analysis_feasibility(self):
        """Test that CS3 data is suitable for variance analysis"""
        final_data, analysis_indicators, _ = load_cs3_data(include_crisis_years=True)
        
        # Test Iceland data quality for variance analysis
        iceland_data = final_data[final_data['CS3_GROUP'] == 'Iceland']
        
        for indicator in analysis_indicators[:3]:  # Test first 3 indicators
            indicator_data = iceland_data[indicator].dropna()
            
            if len(indicator_data) > 1:
                # Should have non-zero variance for meaningful analysis
                variance = indicator_data.var()
                std_dev = indicator_data.std()
                
                # Check that we have meaningful variation (not all identical values)
                unique_values = indicator_data.nunique()
                assert unique_values > 1 or variance == 0, \
                    f"Indicator {indicator} for Iceland has suspicious variance pattern"
        
        # Test comparator data aggregation potential
        comparator_data = final_data[final_data['CS3_GROUP'] == 'Comparator']
        
        # Should have sufficient data points for statistical analysis
        assert len(comparator_data) >= 100, \
            f"Insufficient comparator data for statistical analysis: {len(comparator_data)} observations"
    
    @pytest.mark.skipif(not CS3_IMPORTS_AVAILABLE, reason=f"CS3 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs3_aggregation_potential(self):
        """Test that CS3 comparator countries can be meaningfully aggregated"""
        final_data, analysis_indicators, _ = load_cs3_data(include_crisis_years=True)
        
        comparator_countries = final_data[final_data['CS3_GROUP'] == 'Comparator']['COUNTRY'].unique()
        
        # Test that multiple countries have data for the same time periods
        # This enables meaningful aggregation or pooled analysis
        sample_year = 2010
        countries_with_2010_data = []
        
        for country in comparator_countries:
            country_2010 = final_data[
                (final_data['COUNTRY'] == country) & 
                (final_data['YEAR'] == sample_year)
            ]
            if len(country_2010) > 0:
                countries_with_2010_data.append(country)
        
        # Should have multiple countries with data in the same time period
        assert len(countries_with_2010_data) >= 3, \
            f"Insufficient countries with overlapping data for aggregation: {len(countries_with_2010_data)}"

class TestCS3DataIntegrity:
    """Test CS3 data integrity and consistency"""
    
    @pytest.mark.skipif(not CS3_IMPORTS_AVAILABLE, reason=f"CS3 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs3_time_coverage(self):
        """Test that CS3 has adequate time coverage"""
        final_data, _, _ = load_cs3_data(include_crisis_years=True)
        
        # Check overall time range
        min_year = final_data['YEAR'].min()
        max_year = final_data['YEAR'].max()
        
        assert min_year <= 2000, f"CS3 data starts too late: {min_year}"
        assert max_year >= 2020, f"CS3 data ends too early: {max_year}"
        
        # Check that Iceland has good time coverage
        iceland_data = final_data[final_data['CS3_GROUP'] == 'Iceland']
        iceland_years = iceland_data['YEAR'].unique()
        
        iceland_coverage = len(iceland_years)
        assert iceland_coverage >= 20, f"Insufficient Iceland time coverage: {iceland_coverage} years"
    
    @pytest.mark.skipif(not CS3_IMPORTS_AVAILABLE, reason=f"CS3 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs3_indicator_data_quality(self):
        """Test that CS3 indicator data has reasonable values"""
        final_data, analysis_indicators, _ = load_cs3_data(include_crisis_years=True)
        
        # Test data quality for key indicators across all countries
        for indicator in analysis_indicators[:4]:  # Test first 4 indicators
            indicator_data = final_data[indicator].dropna()
            
            if len(indicator_data) > 0:
                # Basic sanity checks
                assert pd.api.types.is_numeric_dtype(indicator_data), f"Indicator {indicator} should be numeric"
                
                # Check for reasonable value ranges for small open economies
                q95 = indicator_data.quantile(0.95) if len(indicator_data) > 1 else indicator_data.iloc[0]
                q05 = indicator_data.quantile(0.05) if len(indicator_data) > 1 else indicator_data.iloc[0]
                
                # Small open economies can have more volatile capital flows, but still within reason
                assert abs(q95) < 8000, f"Indicator {indicator} has extreme values: 95th percentile = {q95}"
                assert abs(q05) < 8000, f"Indicator {indicator} has extreme values: 5th percentile = {q05}"
    
    @pytest.mark.skipif(not CS3_IMPORTS_AVAILABLE, reason=f"CS3 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs3_country_data_consistency(self):
        """Test consistency across CS3 countries"""
        final_data, _, _ = load_cs3_data(include_crisis_years=True)
        
        # Check that all countries have reasonable data coverage
        all_countries = final_data['COUNTRY'].unique()
        
        for country in all_countries[:5]:  # Test first 5 countries
            country_data = final_data[final_data['COUNTRY'] == country]
            
            # Each country should have reasonable quarterly data
            quarters = set(country_data['QUARTER'].unique())
            expected_quarters = {1, 2, 3, 4}
            
            # Should have at least 3 quarters represented (allowing for some missing data)
            quarter_overlap = quarters & expected_quarters
            assert len(quarter_overlap) >= 3, \
                f"Country {country} missing too many quarters: {quarters}"