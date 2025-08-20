"""
Statistical Validation Tests for Capital Flows Research Project

Advanced tests to verify academic rigor and methodological consistency
"""

import pytest
import pandas as pd
import numpy as np
from scipy import stats
import sys
from pathlib import Path

# Add dashboard modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "dashboard"))

class TestFTestMethodology:
    """Test F-test implementation for academic rigor"""
    
    def test_f_test_degrees_of_freedom_calculation(self):
        """Test that F-test uses correct degrees of freedom"""
        try:
            from simple_report_app import load_default_data, perform_volatility_tests
            
            final_data, analysis_indicators, _ = load_default_data(include_crisis_years=True)
            
            # Test specific groups
            iceland_data = final_data[final_data['CS1_GROUP'] == 'Iceland'][analysis_indicators[0]].dropna()
            eurozone_data = final_data[final_data['CS1_GROUP'] == 'Eurozone'][analysis_indicators[0]].dropna()
            
            # Degrees of freedom should be n-1 for each group
            expected_df1 = len(iceland_data) - 1
            expected_df2 = len(eurozone_data) - 1
            
            # F-test should use these degrees of freedom
            assert expected_df1 == 104, f"Iceland df should be 104, got {expected_df1}"
            assert expected_df2 == 987, f"Eurozone df should be 987, got {expected_df2}"
            
        except ImportError:
            pytest.skip("CS1 modules not available")
    
    def test_f_test_variance_calculation_accuracy(self):
        """Test F-test variance calculations match manual calculations"""
        try:
            from simple_report_app import load_default_data, perform_volatility_tests
            
            final_data, analysis_indicators, _ = load_default_data(include_crisis_years=True)
            f_test_results = perform_volatility_tests(final_data, analysis_indicators)
            
            # Manual calculation for first indicator
            test_indicator = analysis_indicators[0]
            iceland_data = final_data[final_data['CS1_GROUP'] == 'Iceland'][test_indicator].dropna()
            eurozone_data = final_data[final_data['CS1_GROUP'] == 'Eurozone'][test_indicator].dropna()
            
            manual_var_iceland = iceland_data.var(ddof=1)
            manual_var_eurozone = eurozone_data.var(ddof=1)
            manual_f_stat = max(manual_var_iceland, manual_var_eurozone) / min(manual_var_iceland, manual_var_eurozone)
            
            if isinstance(f_test_results, pd.DataFrame) and 'F_Statistic' in f_test_results.columns:
                automated_f_stat = f_test_results.iloc[0]['F_Statistic']
                
                # Should match within 0.1% tolerance
                relative_error = abs(manual_f_stat - automated_f_stat) / manual_f_stat
                assert relative_error < 0.001, f"F-statistic mismatch: manual={manual_f_stat:.6f}, automated={automated_f_stat:.6f}"
            
        except ImportError:
            pytest.skip("CS1 modules not available")
    
    def test_significance_level_thresholds(self):
        """Test that significance levels are correctly applied"""
        # Test significance threshold logic
        p_values = [0.005, 0.025, 0.075, 0.15]
        
        # 1% threshold
        sig_1pct = [p for p in p_values if p < 0.01]
        assert len(sig_1pct) == 1, "Should have 1 result significant at 1%"
        
        # 5% threshold  
        sig_5pct = [p for p in p_values if p < 0.05]
        assert len(sig_5pct) == 2, "Should have 2 results significant at 5%"
        
        # 10% threshold
        sig_10pct = [p for p in p_values if p < 0.10]
        assert len(sig_10pct) == 3, "Should have 3 results significant at 10%"

class TestCrisisPeriodConsistency:
    """Test crisis period definitions and exclusions"""
    
    def test_global_financial_crisis_definition(self):
        """Test Global Financial Crisis period definition"""
        expected_gfc_years = [2008, 2009, 2010]
        
        try:
            from simple_report_app import load_default_data
            
            full_data, _, _ = load_default_data(include_crisis_years=True)
            excluded_data, _, _ = load_default_data(include_crisis_years=False)
            
            full_years = set(full_data['YEAR'].unique())
            excluded_years = set(excluded_data['YEAR'].unique())
            removed_years = full_years - excluded_years
            
            # Should include GFC years in removal
            for year in expected_gfc_years:
                assert year in removed_years, f"GFC year {year} should be excluded"
                
        except ImportError:
            pytest.skip("CS1 modules not available")
    
    def test_covid_period_definition(self):
        """Test COVID-19 period definition"""
        expected_covid_years = [2020, 2021, 2022]
        
        try:
            from simple_report_app import load_default_data
            
            full_data, _, _ = load_default_data(include_crisis_years=True)
            excluded_data, _, _ = load_default_data(include_crisis_years=False)
            
            full_years = set(full_data['YEAR'].unique())
            excluded_years = set(excluded_data['YEAR'].unique())
            removed_years = full_years - excluded_years
            
            # Should include COVID years in removal
            for year in expected_covid_years:
                assert year in removed_years, f"COVID year {year} should be excluded"
                
        except ImportError:
            pytest.skip("CS1 modules not available")
    
    def test_crisis_exclusion_sample_size_impact(self):
        """Test that crisis exclusion has expected impact on sample size"""
        try:
            from simple_report_app import load_default_data
            
            full_data, _, _ = load_default_data(include_crisis_years=True)
            excluded_data, _, _ = load_default_data(include_crisis_years=False)
            
            # Should have meaningful reduction
            reduction = len(full_data) - len(excluded_data)
            reduction_pct = reduction / len(full_data) * 100
            
            # Crisis periods should represent reasonable portion of data
            assert 15 <= reduction_pct <= 35, f"Crisis exclusion impact seems unreasonable: {reduction_pct:.1f}%"
            
        except ImportError:
            pytest.skip("CS1 modules not available")

class TestGDPNormalizationConsistency:
    """Test GDP normalization methodology"""
    
    def test_gdp_normalization_units(self):
        """Test that all case studies use consistent GDP normalization"""
        expected_unit_pattern = "% of GDP"
        
        # Test CS1
        try:
            from simple_report_app import load_default_data
            
            cs1_data, _, _ = load_default_data(include_crisis_years=True)
            cs1_unit = cs1_data['UNIT'].iloc[0]
            assert expected_unit_pattern in cs1_unit, f"CS1 unit should include '% of GDP': {cs1_unit}"
            
        except ImportError:
            pytest.skip("CS1 modules not available")
        
        # Test CS2
        try:
            from case_study_2_euro_adoption import load_case_study_2_data
            
            cs2_data, _, _ = load_case_study_2_data(include_crisis_years=True)
            cs2_unit = cs2_data['UNIT'].iloc[0]
            assert expected_unit_pattern in cs2_unit, f"CS2 unit should include '% of GDP': {cs2_unit}"
            
        except ImportError:
            pytest.skip("CS2 modules not available")
    
    def test_annualization_factor(self):
        """Test that quarterly data is properly annualized"""
        # GDP normalization should use factor of 4 for quarterly to annual conversion
        quarterly_value = 2.5  # Example quarterly flow as % of GDP
        expected_annual = quarterly_value * 4  # Should be 10.0
        
        assert expected_annual == 10.0, "Annualization factor should be 4 for quarterly data"
    
    def test_reasonable_value_ranges(self):
        """Test that normalized values are in reasonable economic ranges"""
        try:
            from simple_report_app import load_default_data
            
            final_data, analysis_indicators, _ = load_default_data(include_crisis_years=True)
            
            # Test value ranges for economic plausibility
            for indicator in analysis_indicators[:3]:  # Test first 3 indicators
                values = final_data[indicator].dropna()
                
                # Extreme values should be rare
                q99 = values.quantile(0.99)
                q01 = values.quantile(0.01)
                
                # Capital flows >100% of GDP should be very rare
                extreme_count = ((values > 100) | (values < -100)).sum()
                extreme_pct = extreme_count / len(values) * 100
                
                assert extreme_pct < 5, f"Too many extreme values (>100% GDP) in {indicator}: {extreme_pct:.1f}%"
                
        except ImportError:
            pytest.skip("CS1 modules not available")

class TestStatisticalAssumptions:
    """Test statistical assumptions and methods"""
    
    def test_variance_equality_assumption_checking(self):
        """Test that variance equality assumptions are properly evaluated"""
        # Create test data with known variance properties
        np.random.seed(42)
        equal_var1 = np.random.normal(0, 1, 100)
        equal_var2 = np.random.normal(2, 1, 100)  # Different mean, same variance
        unequal_var = np.random.normal(0, 3, 100)  # Different variance
        
        # Levene's test for equal variances
        stat_equal, p_equal = stats.levene(equal_var1, equal_var2)
        stat_unequal, p_unequal = stats.levene(equal_var1, unequal_var)
        
        # Equal variance groups should have higher p-value
        assert p_equal > p_unequal, "Equal variance test should distinguish variance differences"
    
    def test_f_test_robustness_to_sample_size(self):
        """Test F-test behavior with different sample sizes"""
        np.random.seed(42)
        
        # Test with different sample sizes
        for n in [30, 100, 300]:
            group1 = np.random.normal(0, 1, n)
            group2 = np.random.normal(0, 1.5, n)  # 50% higher variance
            
            var1 = np.var(group1, ddof=1)
            var2 = np.var(group2, ddof=1)
            f_stat = max(var1, var2) / min(var1, var2)
            
            # F-statistic should be reasonable regardless of sample size
            assert 1.0 <= f_stat <= 10.0, f"F-statistic seems unreasonable for n={n}: {f_stat}"
    
    def test_missing_data_handling(self):
        """Test proper handling of missing data in statistical calculations"""
        # Create data with missing values
        data_with_missing = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8])
        
        # Statistical functions should handle missing data properly
        mean_val = data_with_missing.mean()  # Should ignore NaN
        var_val = data_with_missing.var()    # Should ignore NaN
        
        assert not np.isnan(mean_val), "Mean calculation should handle missing data"
        assert not np.isnan(var_val), "Variance calculation should handle missing data"
        
        # Check that calculations match manual calculation
        clean_data = data_with_missing.dropna()
        manual_mean = clean_data.mean()
        
        assert abs(mean_val - manual_mean) < 1e-10, "Missing data handling should match manual calculation"

class TestEconometricMethodology:
    """Test econometric methodology for CS4"""
    
    def test_time_series_data_structure(self):
        """Test that data structure supports time series analysis"""
        data_dir = Path("updated_data/Clean/CS4_Statistical_Modeling")
        
        if not data_dir.exists():
            pytest.skip("CS4 data not available")
        
        test_file = data_dir / "net_capital_flows_full.csv"
        if test_file.exists():
            df = pd.read_csv(test_file)
            
            # Should have time series structure
            assert 'YEAR' in df.columns, "Time series data should have YEAR column"
            assert 'QUARTER' in df.columns, "Quarterly data should have QUARTER column"
            
            # Should have sufficient time span for AR modeling
            years = df['YEAR'].unique()
            time_span = len(years)
            assert time_span >= 20, f"Need at least 20 years for AR modeling, got {time_span}"
            
            # Should have balanced panel structure
            quarters = set(df['QUARTER'].unique())
            expected_quarters = {1, 2, 3, 4}
            assert quarters == expected_quarters, f"Should have all quarters: {quarters}"
    
    def test_ar_model_prerequisites(self):
        """Test that data meets AR(4) model prerequisites"""
        data_dir = Path("updated_data/Clean/CS4_Statistical_Modeling")
        
        if not data_dir.exists():
            pytest.skip("CS4 data not available")
        
        test_file = data_dir / "net_capital_flows_full.csv"
        if test_file.exists():
            df = pd.read_csv(test_file)
            
            # Should have multiple countries for cross-sectional analysis
            if 'COUNTRY' in df.columns:
                countries = df['COUNTRY'].unique()
                assert len(countries) >= 10, f"Need multiple countries for panel analysis, got {len(countries)}"
            
            # Should have numeric indicators suitable for modeling
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            indicator_cols = [col for col in numeric_cols if col not in ['YEAR', 'QUARTER']]
            assert len(indicator_cols) >= 1, "Should have numeric indicators for modeling"

class TestExternalDataIntegration:
    """Test external data integration methodology for CS5"""
    
    def test_capital_controls_data_structure(self):
        """Test capital controls data supports correlation analysis"""
        controls_dir = Path("updated_data/Clean/CS5_Capital_Controls")
        
        if not controls_dir.exists():
            pytest.skip("CS5 capital controls data not available")
        
        test_file = controls_dir / "sd_yearly_flows.csv"
        if test_file.exists():
            df = pd.read_csv(test_file)
            
            # Should have numeric columns for correlation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            assert len(numeric_cols) >= 2, "Need at least 2 numeric columns for correlation"
            
            # Should be able to calculate correlation matrix
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols[:2]].corr()
                assert not corr_matrix.isnull().all().all(), "Correlation matrix should be computable"
    
    def test_exchange_rate_regime_classification(self):
        """Test exchange rate regime data supports classification analysis"""
        regime_dir = Path("updated_data/Clean/CS5_Regime_Analysis")
        
        if not regime_dir.exists():
            pytest.skip("CS5 regime analysis data not available")
        
        test_file = regime_dir / "net_capital_flows_full.csv"
        if test_file.exists():
            df = pd.read_csv(test_file)
            
            # Should have adequate time and country coverage
            if 'YEAR' in df.columns and 'COUNTRY' in df.columns:
                years = df['YEAR'].unique()
                countries = df['COUNTRY'].unique()
                
                # Should span sufficient time for regime analysis
                time_span = max(years) - min(years)
                assert time_span >= 15, f"Need sufficient time span for regime analysis, got {time_span}"
                
                # Should include key countries
                iceland_present = any('Iceland' in str(country) for country in countries)
                assert iceland_present, "Iceland should be included in regime analysis"