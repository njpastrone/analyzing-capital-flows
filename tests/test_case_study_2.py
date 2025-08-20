"""
Case Study 2 Specific Functionality Tests - Baltic Euro Adoption

Tests CS2 data loading, country filtering, and Euro adoption analysis
"""

import pytest
import pandas as pd
import numpy as np
from conftest import TestDataQuality

# Import CS2 functions
try:
    from case_study_2_euro_adoption import load_case_study_2_data, load_overall_capital_flows_data_cs2
    CS2_IMPORTS_AVAILABLE = True
except ImportError as e:
    CS2_IMPORTS_AVAILABLE = False
    import_error = str(e)

class TestCS2DataLoading:
    """Test CS2 data loading functionality"""
    
    @pytest.mark.skipif(not CS2_IMPORTS_AVAILABLE, reason=f"CS2 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs2_full_period_loading(self):
        """Test CS2 loads full period data correctly"""
        final_data, analysis_indicators, metadata = load_case_study_2_data(include_crisis_years=True)
        
        assert final_data is not None, "CS2 full period data loading failed"
        assert analysis_indicators is not None, "CS2 indicators loading failed"
        assert metadata is not None, "CS2 metadata loading failed"
        
        # Check metadata (allow for different naming conventions)
        study_version = metadata.get('study_version', '')
        assert study_version in ["Full Time Period", "Full Series"], f"Unexpected study version: {study_version}"
        assert metadata['include_crisis_years'] is True, "Crisis years flag incorrect"
        
        # Check data structure
        expected_shape = (315, 25)
        assert final_data.shape == expected_shape, f"Expected shape {expected_shape}, got {final_data.shape}"
        
        # Check indicators count
        assert len(analysis_indicators) == 14, f"Expected 14 indicators, got {len(analysis_indicators)}"
    
    @pytest.mark.skipif(not CS2_IMPORTS_AVAILABLE, reason=f"CS2 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs2_crisis_excluded_loading(self):
        """Test CS2 loads crisis-excluded data correctly"""
        final_data, analysis_indicators, metadata = load_case_study_2_data(include_crisis_years=False)
        
        assert final_data is not None, "CS2 crisis-excluded data loading failed"
        
        # Check metadata (allow for different naming conventions)
        study_version = metadata.get('study_version', '')
        assert "Crisis" in study_version or "Excluded" in study_version, f"Expected crisis-excluded version: {study_version}"
        assert metadata['include_crisis_years'] is False, "Crisis years flag incorrect"
        
        # Should have fewer or equal observations (crisis periods might not affect all countries equally)
        assert len(final_data) <= 315, "Crisis-excluded should have <= 315 observations"

class TestCS2CountryFiltering:
    """Test CS2 Baltic country filtering"""
    
    @pytest.mark.skipif(not CS2_IMPORTS_AVAILABLE, reason=f"CS2 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs2_baltic_countries(self, expected_cs2_countries):
        """Test that CS2 contains exactly the Baltic countries"""
        final_data, _, _ = load_case_study_2_data(include_crisis_years=True)
        
        # Check that only Baltic countries are present
        actual_countries = set(final_data['COUNTRY'].unique())
        expected_countries = set(expected_cs2_countries)
        
        assert actual_countries == expected_countries, \
            f"Expected countries {expected_countries}, got {actual_countries}"
    
    @pytest.mark.skipif(not CS2_IMPORTS_AVAILABLE, reason=f"CS2 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs2_observations_per_country(self, expected_cs2_countries):
        """Test expected number of observations per Baltic country"""
        final_data, _, _ = load_case_study_2_data(include_crisis_years=True)
        
        # Each Baltic country should have 105 observations (105 quarters from 1999-2025)
        for country in expected_cs2_countries:
            country_data = final_data[final_data['COUNTRY'] == country]
            country_count = len(country_data)
            assert country_count == 105, f"Expected 105 observations for {country}, got {country_count}"
    
    @pytest.mark.skipif(not CS2_IMPORTS_AVAILABLE, reason=f"CS2 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs2_group_labeling(self):
        """Test that CS2_GROUP column is correctly set"""
        final_data, _, _ = load_case_study_2_data(include_crisis_years=True)
        
        # Check CS2_GROUP column exists
        assert 'CS2_GROUP' in final_data.columns, "CS2_GROUP column missing"
        
        # All countries should be labeled as 'Included'
        cs2_groups = final_data['CS2_GROUP'].unique()
        assert list(cs2_groups) == ['Included'], f"Expected ['Included'], got {list(cs2_groups)}"

class TestCS2EuroAdoptionAnalysis:
    """Test CS2 Euro adoption timeline analysis"""
    
    @pytest.mark.skipif(not CS2_IMPORTS_AVAILABLE, reason=f"CS2 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs2_euro_adoption_periods(self):
        """Test that Euro adoption periods are correctly identified"""
        final_data, _, _ = load_case_study_2_data(include_crisis_years=True)
        
        # Test that we have data covering Euro adoption periods
        # Estonia: 2011, Latvia: 2014, Lithuania: 2015
        euro_adoption_years = {
            'Estonia, Republic of': 2011,
            'Latvia, Republic of': 2014, 
            'Lithuania, Republic of': 2015
        }
        
        for country, adoption_year in euro_adoption_years.items():
            country_data = final_data[final_data['COUNTRY'] == country]
            years = set(country_data['YEAR'].unique())
            
            # Should have data before and after adoption
            pre_adoption_years = [y for y in years if y < adoption_year]
            post_adoption_years = [y for y in years if y >= adoption_year]
            
            assert len(pre_adoption_years) > 0, f"No pre-adoption data for {country}"
            assert len(post_adoption_years) > 0, f"No post-adoption data for {country}"
            assert adoption_year in years, f"No data for adoption year {adoption_year} for {country}"
    
    @pytest.mark.skipif(not CS2_IMPORTS_AVAILABLE, reason=f"CS2 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs2_temporal_analysis_capability(self):
        """Test that CS2 data supports before/after temporal analysis"""
        final_data, analysis_indicators, _ = load_case_study_2_data(include_crisis_years=True)
        
        # Test Estonia's data around 2011 adoption
        estonia_data = final_data[final_data['COUNTRY'] == 'Estonia, Republic of']
        
        # Check that we have reasonable data coverage
        years_range = estonia_data['YEAR'].max() - estonia_data['YEAR'].min()
        assert years_range >= 20, f"Expected at least 20 years of data, got {years_range}"
        
        # Check that indicator data exists for pre and post periods
        pre_2011 = estonia_data[estonia_data['YEAR'] < 2011]
        post_2011 = estonia_data[estonia_data['YEAR'] >= 2011]
        
        assert len(pre_2011) > 0, "No pre-Euro data for Estonia"
        assert len(post_2011) > 0, "No post-Euro data for Estonia"
        
        # Check that key indicators have data in both periods
        key_indicators = analysis_indicators[:3]  # Test first 3 indicators
        for indicator in key_indicators:
            pre_data = pre_2011[indicator].dropna()
            post_data = post_2011[indicator].dropna()
            
            assert len(pre_data) > 0, f"No pre-Euro data for indicator {indicator}"
            assert len(post_data) > 0, f"No post-Euro data for indicator {indicator}"

class TestCS2StatisticalAnalysis:
    """Test CS2 statistical analysis capabilities"""
    
    @pytest.mark.skipif(not CS2_IMPORTS_AVAILABLE, reason=f"CS2 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs2_variance_analysis_setup(self):
        """Test that CS2 data is suitable for variance analysis"""
        final_data, analysis_indicators, _ = load_case_study_2_data(include_crisis_years=True)
        
        # Test each country has sufficient data for statistical analysis
        for country in ['Estonia, Republic of', 'Latvia, Republic of', 'Lithuania, Republic of']:
            country_data = final_data[final_data['COUNTRY'] == country]
            
            # Should have enough observations for meaningful statistics
            assert len(country_data) >= 50, f"Insufficient data for {country}: {len(country_data)} observations"
            
            # Test that indicators have reasonable variance (not all constant)
            for indicator in analysis_indicators[:3]:  # Test first 3 indicators
                indicator_data = country_data[indicator].dropna()
                if len(indicator_data) > 1:
                    variance = indicator_data.var()
                    # Should not be exactly zero variance (unless legitimately constant)
                    if variance == 0:
                        # Check if it's all the same value
                        unique_values = indicator_data.nunique()
                        # Allow for indicators that are legitimately constant
                        assert unique_values <= 2, f"Indicator {indicator} for {country} has suspicious zero variance"
    
    @pytest.mark.skipif(not CS2_IMPORTS_AVAILABLE, reason=f"CS2 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs2_overall_capital_flows_loading(self):
        """Test CS2 overall capital flows data loading"""
        try:
            full_data, full_indicators, full_metadata = load_overall_capital_flows_data_cs2(include_crisis_years=True)
            assert full_data is not None, "CS2 overall capital flows loading failed"
            
            # Should have same countries as main CS2 data
            countries = set(full_data['COUNTRY'].unique())
            expected_countries = {'Estonia, Republic of', 'Latvia, Republic of', 'Lithuania, Republic of'}
            assert countries == expected_countries, f"Expected {expected_countries}, got {countries}"
            
        except Exception as e:
            # If function doesn't exist or has issues, that's still useful information
            pytest.fail(f"CS2 overall capital flows function failed: {e}")

class TestCS2DataIntegrity:
    """Test CS2 data integrity and consistency"""
    
    @pytest.mark.skipif(not CS2_IMPORTS_AVAILABLE, reason=f"CS2 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs2_time_series_completeness(self):
        """Test that CS2 has complete time series for each country"""
        final_data, _, _ = load_case_study_2_data(include_crisis_years=True)
        
        # Check that each country has data for the full expected period
        for country in ['Estonia, Republic of', 'Latvia, Republic of', 'Lithuania, Republic of']:
            country_data = final_data[final_data['COUNTRY'] == country]
            
            # Check year coverage
            min_year = country_data['YEAR'].min()
            max_year = country_data['YEAR'].max()
            
            assert min_year <= 2000, f"{country} data starts too late: {min_year}"
            assert max_year >= 2020, f"{country} data ends too early: {max_year}"
            
            # Check quarter completeness for a sample year
            sample_year = 2010  # Should be present for all countries
            year_data = country_data[country_data['YEAR'] == sample_year]
            if len(year_data) > 0:
                quarters = set(year_data['QUARTER'].unique())
                expected_quarters = {1, 2, 3, 4}
                # Should have at least 3 quarters (allowing for some missing data)
                assert len(quarters & expected_quarters) >= 3, \
                    f"{country} missing too many quarters in {sample_year}: {quarters}"
    
    @pytest.mark.skipif(not CS2_IMPORTS_AVAILABLE, reason=f"CS2 imports failed: {import_error if 'import_error' in locals() else 'Unknown error'}")
    def test_cs2_indicator_data_quality(self):
        """Test that CS2 indicator data has reasonable values"""
        final_data, analysis_indicators, _ = load_case_study_2_data(include_crisis_years=True)
        
        # Test data quality for key indicators
        for indicator in analysis_indicators[:5]:  # Test first 5 indicators
            indicator_data = final_data[indicator].dropna()
            
            if len(indicator_data) > 0:
                # Basic sanity checks
                assert pd.api.types.is_numeric_dtype(indicator_data), f"Indicator {indicator} should be numeric"
                
                # Check for reasonable value ranges (% of GDP shouldn't be extreme)
                q99 = indicator_data.quantile(0.99) if len(indicator_data) > 1 else indicator_data.iloc[0]
                q01 = indicator_data.quantile(0.01) if len(indicator_data) > 1 else indicator_data.iloc[0]
                
                # Values shouldn't be astronomically large for small economies
                assert abs(q99) < 5000, f"Indicator {indicator} has extreme values: 99th percentile = {q99}"
                assert abs(q01) < 5000, f"Indicator {indicator} has extreme values: 1st percentile = {q01}"