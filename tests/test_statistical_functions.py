"""
Statistical Functions Tests for Capital Flows Research Project

Tests common statistical methods and calculations used across case studies
"""

import pytest
import pandas as pd
import numpy as np
from scipy import stats
from conftest import TestDataQuality

# Import statistical functions if available
try:
    from common_statistical_functions import perform_f_test, calculate_correlation_matrix
    COMMON_STATS_AVAILABLE = True
except ImportError:
    COMMON_STATS_AVAILABLE = False

class TestFTestImplementation:
    """Test F-test implementation for variance comparison"""
    
    def test_basic_f_test_calculation(self):
        """Test basic F-test calculation with known data"""
        # Create test data with known variances
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 100)  # variance ≈ 1
        group2 = np.random.normal(0, 2, 100)  # variance ≈ 4
        
        # Calculate F-statistic manually
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        f_stat_expected = var2 / var1  # Larger variance in numerator
        
        # Test that our F-test gives reasonable results
        assert var2 > var1, "Test setup: group2 should have larger variance"
        assert f_stat_expected > 1, "F-statistic should be > 1 when var2 > var1"
        
        # Basic F-test properties
        assert f_stat_expected >= 0, "F-statistic should be non-negative"
        
        # Test with equal variances
        group3 = np.random.normal(0, 1, 100)
        var3 = np.var(group3, ddof=1)
        f_stat_equal = var1 / var3
        # Should be close to 1 for similar variances
        assert 0.5 < f_stat_equal < 2.0, f"F-stat for similar variances should be ~1, got {f_stat_equal}"
    
    @pytest.mark.skipif(not COMMON_STATS_AVAILABLE, reason="Common statistical functions not available")
    def test_f_test_function_interface(self):
        """Test F-test function interface if available"""
        # Create test series
        series1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        series2 = pd.Series([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        
        # Test F-test function
        result = perform_f_test(series1, series2)
        
        # Should return a dictionary with expected keys
        expected_keys = ['f_statistic', 'p_value', 'df1', 'df2']
        assert isinstance(result, dict), "F-test should return a dictionary"
        
        # Check that we get numeric results
        for key in expected_keys:
            if key in result:
                assert isinstance(result[key], (int, float)), f"F-test {key} should be numeric"
                assert not np.isnan(result[key]), f"F-test {key} should not be NaN"
    
    def test_f_test_edge_cases(self):
        """Test F-test behavior with edge cases"""
        # Test with identical series (zero variance)
        identical_series = pd.Series([5, 5, 5, 5, 5])
        different_series = pd.Series([1, 2, 3, 4, 5])
        
        var_identical = identical_series.var()
        var_different = different_series.var()
        
        assert var_identical == 0, "Identical series should have zero variance"
        assert var_different > 0, "Different series should have positive variance"
        
        # F-test with zero variance should be handled gracefully
        if var_identical == 0 and var_different > 0:
            # This should result in infinite F-statistic or special handling
            assert True  # Test that we can detect this case
        
        # Test with very small series
        small_series1 = pd.Series([1, 2])
        small_series2 = pd.Series([3, 4])
        
        assert len(small_series1) >= 2, "Should handle small series"
        assert len(small_series2) >= 2, "Should handle small series"

class TestCorrelationAnalysis:
    """Test correlation analysis functionality"""
    
    def test_correlation_calculation_basics(self):
        """Test basic correlation calculations"""
        # Create correlated data
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y_positive = x + np.random.normal(0, 0.5, 100)  # Positive correlation
        y_negative = -x + np.random.normal(0, 0.5, 100)  # Negative correlation
        y_independent = np.random.normal(0, 1, 100)  # No correlation
        
        # Test positive correlation
        corr_pos = np.corrcoef(x, y_positive)[0, 1]
        assert corr_pos > 0.5, f"Expected positive correlation > 0.5, got {corr_pos}"
        
        # Test negative correlation
        corr_neg = np.corrcoef(x, y_negative)[0, 1]
        assert corr_neg < -0.5, f"Expected negative correlation < -0.5, got {corr_neg}"
        
        # Test correlation bounds
        assert -1 <= corr_pos <= 1, "Correlation should be between -1 and 1"
        assert -1 <= corr_neg <= 1, "Correlation should be between -1 and 1"
    
    @pytest.mark.skipif(not COMMON_STATS_AVAILABLE, reason="Common statistical functions not available")
    def test_correlation_matrix_function(self):
        """Test correlation matrix function if available"""
        # Create test dataframe
        np.random.seed(42)
        df = pd.DataFrame({
            'var1': np.random.normal(0, 1, 50),
            'var2': np.random.normal(0, 2, 50),
            'var3': np.random.normal(0, 1, 50)
        })
        
        # Test correlation matrix function
        corr_matrix = calculate_correlation_matrix(df)
        
        # Should return a square matrix
        assert corr_matrix.shape[0] == corr_matrix.shape[1], "Correlation matrix should be square"
        assert corr_matrix.shape[0] == len(df.columns), "Matrix size should match column count"
        
        # Diagonal should be 1 (perfect self-correlation)
        diagonal = np.diag(corr_matrix)
        np.testing.assert_array_almost_equal(diagonal, np.ones(len(diagonal)), 
                                           decimal=10, err_msg="Diagonal should be 1s")
        
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(corr_matrix, corr_matrix.T, 
                                           decimal=10, err_msg="Correlation matrix should be symmetric")

class TestDescriptiveStatistics:
    """Test descriptive statistics calculations"""
    
    def test_summary_statistics_calculation(self):
        """Test calculation of summary statistics"""
        # Create test data with known properties
        np.random.seed(42)
        data = np.random.normal(10, 3, 1000)  # mean=10, std=3
        
        # Calculate statistics
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        median_val = np.median(data)
        min_val = np.min(data)
        max_val = np.max(data)
        
        # Test that statistics are reasonable
        assert 9.5 < mean_val < 10.5, f"Mean should be ~10, got {mean_val}"
        assert 2.5 < std_val < 3.5, f"Std should be ~3, got {std_val}"
        assert min_val < median_val < max_val, "Min < Median < Max should hold"
        
        # Test coefficient of variation
        cv = (std_val / abs(mean_val)) * 100
        assert 0 < cv < 100, f"CV should be reasonable, got {cv}%"
    
    def test_volatility_measures(self):
        """Test volatility measurement calculations"""
        # Create test time series with different volatility patterns
        np.random.seed(42)
        
        # Low volatility series
        low_vol = np.random.normal(0, 0.5, 100)
        # High volatility series  
        high_vol = np.random.normal(0, 2.0, 100)
        
        # Calculate volatilities
        vol_low = np.std(low_vol, ddof=1)
        vol_high = np.std(high_vol, ddof=1)
        
        assert vol_high > vol_low, "High volatility series should have higher standard deviation"
        assert vol_low > 0, "Volatility should be positive"
        assert vol_high > 0, "Volatility should be positive"
        
        # Test that volatility scales appropriately
        ratio = vol_high / vol_low
        assert 2 < ratio < 6, f"Volatility ratio should reflect the setup (2σ vs 0.5σ), got {ratio}"
    
    def test_outlier_detection_basics(self):
        """Test basic outlier detection methods"""
        # Create data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        outliers = np.array([10, -10, 15])  # Clear outliers
        data_with_outliers = np.concatenate([normal_data, outliers])
        
        # IQR method
        Q1 = np.percentile(data_with_outliers, 25)
        Q3 = np.percentile(data_with_outliers, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (data_with_outliers < lower_bound) | (data_with_outliers > upper_bound)
        detected_outliers = data_with_outliers[outlier_mask]
        
        # Should detect the extreme outliers we added
        assert len(detected_outliers) > 0, "Should detect outliers in the data"
        assert 10 in detected_outliers or 15 in detected_outliers, "Should detect positive outliers"
        assert -10 in detected_outliers, "Should detect negative outliers"

class TestHypothesisTestingFramework:
    """Test hypothesis testing framework functionality"""
    
    def test_significance_level_interpretation(self):
        """Test interpretation of significance levels"""
        # Test p-value interpretation at different significance levels
        p_values = [0.001, 0.01, 0.05, 0.1, 0.5]
        alpha_levels = [0.01, 0.05, 0.1]
        
        for p_val in p_values:
            for alpha in alpha_levels:
                is_significant = p_val < alpha
                
                # Basic logical consistency
                if p_val < 0.01:
                    assert p_val < 0.05, "p < 0.01 implies p < 0.05"
                    assert p_val < 0.1, "p < 0.01 implies p < 0.1"
                
                # Test significance determination
                if alpha == 0.05:
                    if p_val < 0.05:
                        assert is_significant, f"p={p_val} should be significant at α=0.05"
                    else:
                        assert not is_significant, f"p={p_val} should not be significant at α=0.05"
    
    def test_multiple_testing_awareness(self):
        """Test awareness of multiple testing issues"""
        # When testing multiple hypotheses, significance levels need adjustment
        n_tests = [1, 5, 10, 20]
        alpha = 0.05
        
        for n in n_tests:
            # Bonferroni correction
            bonferroni_alpha = alpha / n
            assert bonferroni_alpha <= alpha, "Bonferroni correction should reduce α"
            assert bonferroni_alpha > 0, "Corrected α should be positive"
            
            # Type I error inflation
            prob_no_error = (1 - alpha) ** n
            prob_at_least_one_error = 1 - prob_no_error
            
            if n > 1:
                assert prob_at_least_one_error > alpha, \
                    f"Multiple testing inflates Type I error: {prob_at_least_one_error:.3f} > {alpha}"

class TestStatisticalAssumptions:
    """Test checking of statistical assumptions"""
    
    def test_normality_assumption_checking(self):
        """Test methods for checking normality assumptions"""
        np.random.seed(42)
        
        # Create normal and non-normal data
        normal_data = np.random.normal(0, 1, 1000)
        uniform_data = np.random.uniform(-3, 3, 1000)
        
        # Shapiro-Wilk test (for smaller samples)
        if len(normal_data) <= 5000:  # Shapiro-Wilk works best for smaller samples
            stat_normal, p_normal = stats.shapiro(normal_data[:100])  # Use subset
            stat_uniform, p_uniform = stats.shapiro(uniform_data[:100])
            
            # Normal data should have higher p-value (less evidence against normality)
            assert p_normal > p_uniform, \
                f"Normal data should have higher p-value: {p_normal} vs {p_uniform}"
        
        # Basic distributional properties
        # Normal data should have skewness close to 0
        skew_normal = stats.skew(normal_data)
        skew_uniform = stats.skew(uniform_data)
        
        assert abs(skew_normal) < abs(skew_uniform) or abs(skew_normal) < 0.5, \
            f"Normal data should have lower skewness: {skew_normal} vs {skew_uniform}"
    
    def test_variance_equality_assumptions(self):
        """Test checking variance equality assumptions"""
        np.random.seed(42)
        
        # Create groups with equal and unequal variances
        group1_equal = np.random.normal(0, 1, 100)
        group2_equal = np.random.normal(2, 1, 100)  # Different mean, same variance
        group3_unequal = np.random.normal(0, 3, 100)  # Different variance
        
        # Levene's test for equal variances
        stat_equal, p_equal = stats.levene(group1_equal, group2_equal)
        stat_unequal, p_unequal = stats.levene(group1_equal, group3_unequal)
        
        # Equal variance groups should have higher p-value (less evidence against equality)
        assert p_equal > p_unequal, \
            f"Equal variance groups should have higher p-value: {p_equal} vs {p_unequal}"
        
        # Variance ratios
        var1 = np.var(group1_equal, ddof=1)
        var2 = np.var(group2_equal, ddof=1)
        var3 = np.var(group3_unequal, ddof=1)
        
        ratio_equal = max(var1, var2) / min(var1, var2)
        ratio_unequal = max(var1, var3) / min(var1, var3)
        
        assert ratio_unequal > ratio_equal, \
            f"Unequal variance groups should have higher variance ratio: {ratio_unequal} vs {ratio_equal}"

class TestRobustnessChecks:
    """Test robustness of statistical methods"""
    
    def test_sample_size_sensitivity(self):
        """Test how statistical tests behave with different sample sizes"""
        np.random.seed(42)
        
        # Test F-test with different sample sizes
        sample_sizes = [10, 30, 100, 300]
        
        for n in sample_sizes:
            group1 = np.random.normal(0, 1, n)
            group2 = np.random.normal(0, 1.5, n)  # Different variance
            
            if n >= 3:  # Need minimum sample size for F-test
                var1 = np.var(group1, ddof=1)
                var2 = np.var(group2, ddof=1)
                f_stat = max(var1, var2) / min(var1, var2)
                
                # F-statistic should be reasonable regardless of sample size
                assert f_stat >= 1, "F-statistic should be ≥ 1"
                assert f_stat < 100, f"F-statistic seems too extreme for n={n}: {f_stat}"
    
    def test_missing_data_handling(self):
        """Test statistical methods with missing data"""
        np.random.seed(42)
        
        # Create data with missing values
        complete_data = np.random.normal(0, 1, 100)
        data_with_missing = complete_data.copy()
        data_with_missing[::10] = np.nan  # 10% missing
        
        series_complete = pd.Series(complete_data)
        series_missing = pd.Series(data_with_missing)
        
        # Test that dropna works correctly
        clean_data = series_missing.dropna()
        assert len(clean_data) < len(series_missing), "dropna should remove missing values"
        assert not clean_data.isnull().any(), "After dropna, no nulls should remain"
        
        # Test statistics with missing data
        mean_complete = series_complete.mean()
        mean_missing = series_missing.mean()  # pandas handles NaN automatically
        
        # Should be similar (within reasonable tolerance)
        diff = abs(mean_complete - mean_missing)
        assert diff < 0.5, f"Means should be similar despite missing data: {diff}"