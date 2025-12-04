"""
CS4 Robustness Testing and Validation Suite

Comprehensive testing framework to validate CS4 statistical analysis:
1. Statistical validation and diagnostic checks
2. Cross-validation with existing case studies
3. Sensitivity analysis and model robustness
4. Edge case and error handling tests
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from cs4_statistical_analysis import CS4AnalysisFramework, CS4DataLoader, CS4TimeSeriesAnalysis

class CS4RobustnessTests:
    """Comprehensive robustness testing for CS4 statistical framework"""
    
    def __init__(self):
        self.framework = CS4AnalysisFramework()
        self.loader = CS4DataLoader()
        self.ts_analyzer = CS4TimeSeriesAnalysis()
        self.test_results = {}
        
    def run_all_tests(self, verbose=True):
        """Run complete robustness testing suite"""
        print("=" * 80)
        print("CS4 ROBUSTNESS TESTING SUITE")
        print("=" * 80)
        
        # 1. Statistical Validation
        print("\n📊 1. STATISTICAL VALIDATION")
        print("-" * 40)
        self.test_results['statistical'] = self.statistical_validation()
        
        # 2. Edge Case Testing
        print("\n🔧 2. EDGE CASE TESTING")
        print("-" * 40)
        self.test_results['edge_cases'] = self.edge_case_testing()
        
        # 3. Cross-Validation with Other Case Studies
        print("\n🔄 3. CROSS-VALIDATION WITH CS1/CS3")
        print("-" * 40)
        self.test_results['cross_validation'] = self.cross_validation()
        
        # 4. Sensitivity Analysis
        print("\n📈 4. SENSITIVITY ANALYSIS")
        print("-" * 40)
        self.test_results['sensitivity'] = self.sensitivity_analysis()
        
        # 5. Generate Summary Report
        print("\n📋 5. SUMMARY REPORT")
        print("-" * 40)
        self.generate_summary_report()
        
        return self.test_results
    
    def statistical_validation(self):
        """Validate statistical results and check model diagnostics"""
        results = {
            'f_tests': {},
            'ar4_diagnostics': {},
            'rmse_validation': {}
        }
        
        print("Testing F-test results consistency...")
        
        # Load test data
        test_data = self.loader.load_indicator_data("Net Direct Investment", include_crisis_years=True)
        if test_data is None:
            print("❌ Failed to load test data")
            return results
        
        # Test F-test symmetry and consistency
        iceland = test_data['Iceland'].dropna()
        eurozone = test_data['eurozone_avg'].dropna()
        
        # Calculate F-test manually
        var_iceland = np.var(iceland, ddof=1)
        var_eurozone = np.var(eurozone, ddof=1)
        f_stat = var_iceland / var_eurozone if var_iceland > var_eurozone else var_eurozone / var_iceland
        
        results['f_tests']['manual_f_stat'] = f_stat
        results['f_tests']['variance_ratio'] = var_iceland / var_eurozone
        
        # Test AR(4) model diagnostics
        print("Testing AR(4) model diagnostics...")
        ar4_result = self.ts_analyzer.fit_ar4_model(iceland)
        
        if ar4_result:
            residuals = ar4_result['residuals']
            
            # Ljung-Box test for residual autocorrelation
            lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
            results['ar4_diagnostics']['ljung_box_pvalue'] = lb_test['lb_pvalue'].min()
            results['ar4_diagnostics']['residual_autocorr'] = "No autocorrelation" if lb_test['lb_pvalue'].min() > 0.05 else "Autocorrelation detected"
            
            # Check stationarity
            adf_test = adfuller(iceland.dropna())
            results['ar4_diagnostics']['adf_statistic'] = adf_test[0]
            results['ar4_diagnostics']['adf_pvalue'] = adf_test[1]
            results['ar4_diagnostics']['stationary'] = "Yes" if adf_test[1] < 0.05 else "No"
            
            # Model fit statistics
            results['ar4_diagnostics']['aic'] = ar4_result['aic']
            results['ar4_diagnostics']['n_obs'] = ar4_result['n_observations']
            
            print(f"✅ AR(4) Diagnostics: {results['ar4_diagnostics']['residual_autocorr']}")
            print(f"   Stationarity: {results['ar4_diagnostics']['stationary']} (p={adf_test[1]:.4f})")
        
        # RMSE validation
        print("Testing RMSE prediction accuracy...")
        rmse_value = self.ts_analyzer.calculate_rmse_prediction(iceland)
        if rmse_value:
            # Calculate as percentage of mean
            mean_val = abs(iceland.mean())
            rmse_pct = (rmse_value / mean_val * 100) if mean_val > 0 else np.nan
            results['rmse_validation']['rmse'] = rmse_value
            results['rmse_validation']['rmse_pct_of_mean'] = rmse_pct
            results['rmse_validation']['prediction_quality'] = "Good" if rmse_pct < 100 else "Poor"
            print(f"✅ RMSE: {rmse_value:.2f} ({rmse_pct:.1f}% of mean)")
        
        return results
    
    def edge_case_testing(self):
        """Test edge cases and error handling"""
        results = {
            'missing_data': {},
            'insufficient_data': {},
            'extreme_values': {}
        }
        
        print("Testing missing data handling...")
        
        # Create test series with missing data
        test_series = pd.Series(np.random.randn(100))
        test_series[20:30] = np.nan  # 10% missing
        test_series[50:60] = np.nan  # Another 10% missing
        
        # Test AR(4) with missing data
        ar4_result = self.ts_analyzer.fit_ar4_model(test_series)
        results['missing_data']['ar4_handles_missing'] = ar4_result is not None
        
        # Test RMSE with missing data
        rmse_result = self.ts_analyzer.calculate_rmse_prediction(test_series)
        results['missing_data']['rmse_handles_missing'] = rmse_result is not None
        
        print(f"✅ Missing data handling: AR(4)={results['missing_data']['ar4_handles_missing']}, RMSE={results['missing_data']['rmse_handles_missing']}")
        
        # Test with insufficient data
        print("Testing insufficient data handling...")
        short_series = pd.Series([1, 2, 3, 4, 5])  # Only 5 observations
        
        ar4_short = self.ts_analyzer.fit_ar4_model(short_series)
        results['insufficient_data']['ar4_handles_short'] = ar4_short is None  # Should return None
        
        rmse_short = self.ts_analyzer.calculate_rmse_prediction(short_series)
        results['insufficient_data']['rmse_handles_short'] = rmse_short is None  # Should return None
        
        print(f"✅ Insufficient data protection: AR(4)={not ar4_short}, RMSE={not rmse_short}")
        
        # Test with extreme values
        print("Testing extreme value handling...")
        extreme_series = pd.Series(np.random.randn(100))
        extreme_series[50] = 1000  # Add extreme outlier
        
        ar4_extreme = self.ts_analyzer.fit_ar4_model(extreme_series)
        results['extreme_values']['ar4_handles_extreme'] = ar4_extreme is not None
        
        if ar4_extreme:
            half_life = self.ts_analyzer.calculate_half_life_impulse_response(ar4_extreme['coefficients'])
            results['extreme_values']['half_life_with_extreme'] = half_life
        
        print(f"✅ Extreme value handling: Model fits={results['extreme_values']['ar4_handles_extreme']}")
        
        return results
    
    def cross_validation(self):
        """Cross-validate results with CS1 and CS3 findings"""
        results = {
            'cs1_consistency': {},
            'cs3_consistency': {},
            'economic_sense': {}
        }
        
        print("Cross-validating with CS1 (Iceland vs Eurozone)...")
        
        # Run analysis for comparison
        full_results = self.framework.run_comprehensive_analysis(include_crisis_years=True)
        
        if 'summary_tables' in full_results:
            std_table = full_results['summary_tables']['standard_deviations_ftest']
            
            # Check Iceland vs Eurozone from CS4
            iceland_std = None
            eurozone_std = None
            
            for _, row in std_table.iterrows():
                if row['Indicator'] == 'Net Direct Investment':
                    iceland_std = float(row['Iceland'])
                    eurozone_val = row.get('Eurozone Avg', '')
                    if eurozone_val and isinstance(eurozone_val, str):
                        eurozone_std = float(eurozone_val.replace('*', ''))
                    break
            
            if iceland_std and eurozone_std:
                variance_ratio = (iceland_std / eurozone_std) ** 2
                results['cs1_consistency']['iceland_more_volatile'] = iceland_std > eurozone_std
                results['cs1_consistency']['variance_ratio'] = variance_ratio
                
                # Expected from CS1: Iceland should show higher volatility
                results['cs1_consistency']['consistent_with_cs1'] = iceland_std > eurozone_std
                print(f"✅ CS1 Consistency: Iceland volatility {'higher' if iceland_std > eurozone_std else 'lower'} than Eurozone")
                print(f"   Variance ratio: {variance_ratio:.2f}")
        
        print("Cross-validating with CS3 (Iceland vs SOE)...")
        
        # Check Iceland vs SOE from CS4
        if 'summary_tables' in full_results:
            for _, row in std_table.iterrows():
                if row['Indicator'] == 'Net Direct Investment':
                    soe_val = row.get('SOE Avg', '')
                    if soe_val and isinstance(soe_val, str):
                        soe_std = float(soe_val.replace('*', ''))
                        results['cs3_consistency']['iceland_vs_soe'] = iceland_std / soe_std if soe_std else None
                        results['cs3_consistency']['pattern_similar'] = abs(iceland_std - soe_std) < 5  # Within 5% threshold
                        print(f"✅ CS3 Consistency: Iceland vs SOE ratio = {iceland_std/soe_std:.2f}")
                    break
        
        # Economic sense check
        print("Checking economic sense of results...")
        
        # Half-lives should be low (1-3 quarters) for financial flows
        if 'half_life_ar4' in full_results.get('summary_tables', {}):
            hl_table = full_results['summary_tables']['half_life_ar4']
            half_lives = []
            for _, row in hl_table.iterrows():
                for col in hl_table.columns[1:]:  # Skip indicator column
                    val = row[col]
                    if val != 'N/A' and isinstance(val, (int, float)):
                        half_lives.append(val)
            
            avg_half_life = np.mean(half_lives) if half_lives else None
            results['economic_sense']['avg_half_life'] = avg_half_life
            results['economic_sense']['half_life_reasonable'] = avg_half_life <= 3 if avg_half_life else False
            print(f"✅ Economic Sense: Average half-life = {avg_half_life:.1f} quarters (expected: 1-3)")
        
        return results
    
    def sensitivity_analysis(self):
        """Test sensitivity to model specifications and parameters"""
        results = {
            'ar_order_sensitivity': {},
            'aggregation_sensitivity': {},
            'time_period_sensitivity': {}
        }
        
        print("Testing AR model order sensitivity...")
        
        # Load test data
        test_data = self.loader.load_indicator_data("Net Capital Flows", include_crisis_years=True)
        if test_data is not None:
            iceland = test_data['Iceland'].dropna()
            
            # Test AR(2) vs AR(4)
            from statsmodels.tsa.ar_model import AutoReg
            
            # AR(2) model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ar2_model = AutoReg(iceland, lags=2, trend='c').fit()
                ar4_model = AutoReg(iceland, lags=4, trend='c').fit()
            
            results['ar_order_sensitivity']['ar2_aic'] = ar2_model.aic
            results['ar_order_sensitivity']['ar4_aic'] = ar4_model.aic
            results['ar_order_sensitivity']['preferred_model'] = 'AR(2)' if ar2_model.aic < ar4_model.aic else 'AR(4)'
            
            print(f"✅ AR Order: AR(2) AIC={ar2_model.aic:.1f}, AR(4) AIC={ar4_model.aic:.1f}")
            print(f"   Preferred: {results['ar_order_sensitivity']['preferred_model']}")
        
        print("Testing aggregation method sensitivity...")
        
        # Compare sum vs average for Eurozone
        if test_data is not None:
            eurozone_sum = test_data['eurozone_sum'].dropna()
            eurozone_avg = test_data['eurozone_avg'].dropna()
            
            # Calculate correlation
            if len(eurozone_sum) > 0 and len(eurozone_avg) > 0:
                # Standardize both series for comparison
                sum_std = (eurozone_sum - eurozone_sum.mean()) / eurozone_sum.std()
                avg_std = (eurozone_avg - eurozone_avg.mean()) / eurozone_avg.std()
                
                correlation = np.corrcoef(sum_std, avg_std)[0, 1]
                results['aggregation_sensitivity']['sum_avg_correlation'] = correlation
                results['aggregation_sensitivity']['aggregation_matters'] = correlation < 0.95
                
                print(f"✅ Aggregation: Sum vs Avg correlation = {correlation:.3f}")
                print(f"   Aggregation method {'matters' if correlation < 0.95 else 'does not matter'}")
        
        print("Testing time period sensitivity...")
        
        # Compare full vs crisis-excluded
        full_results = self.framework.run_comprehensive_analysis(include_crisis_years=True)
        crisis_results = self.framework.run_comprehensive_analysis(include_crisis_years=False)
        
        if 'summary_tables' in full_results and 'summary_tables' in crisis_results:
            full_std = full_results['summary_tables']['standard_deviations_ftest']
            crisis_std = crisis_results['summary_tables']['standard_deviations_ftest']
            
            # Compare Iceland standard deviations
            iceland_full = []
            iceland_crisis = []
            
            for _, row in full_std.iterrows():
                iceland_full.append(float(row['Iceland']))
            for _, row in crisis_std.iterrows():
                iceland_crisis.append(float(row['Iceland']))
            
            if iceland_full and iceland_crisis:
                avg_change = np.mean([(c/f - 1) * 100 for f, c in zip(iceland_full, iceland_crisis)])
                results['time_period_sensitivity']['avg_std_change_pct'] = avg_change
                results['time_period_sensitivity']['crisis_impact'] = 'Significant' if abs(avg_change) > 20 else 'Moderate'
                
                print(f"✅ Time Period: Crisis exclusion changes std by {avg_change:.1f}%")
                print(f"   Impact: {results['time_period_sensitivity']['crisis_impact']}")
        
        return results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report of all tests"""
        print("\n" + "=" * 80)
        print("ROBUSTNESS TESTING SUMMARY")
        print("=" * 80)
        
        # Overall assessment
        all_passed = True
        critical_issues = []
        warnings = []
        
        # Check statistical validation
        if 'statistical' in self.test_results:
            stat_results = self.test_results['statistical']
            
            # Check AR(4) diagnostics
            if 'ar4_diagnostics' in stat_results:
                if stat_results['ar4_diagnostics'].get('residual_autocorr') == "Autocorrelation detected":
                    warnings.append("AR(4) residuals show autocorrelation")
                if stat_results['ar4_diagnostics'].get('stationary') == "No":
                    warnings.append("Time series may be non-stationary")
            
            # Check RMSE
            if 'rmse_validation' in stat_results:
                if stat_results['rmse_validation'].get('prediction_quality') == "Poor":
                    warnings.append("RMSE prediction accuracy is poor")
        
        # Check edge cases
        if 'edge_cases' in self.test_results:
            edge_results = self.test_results['edge_cases']
            if not edge_results.get('missing_data', {}).get('ar4_handles_missing'):
                critical_issues.append("AR(4) fails with missing data")
            if edge_results.get('insufficient_data', {}).get('ar4_handles_short'):
                critical_issues.append("Insufficient data protection failed")
        
        # Check cross-validation
        if 'cross_validation' in self.test_results:
            cv_results = self.test_results['cross_validation']
            if not cv_results.get('cs1_consistency', {}).get('consistent_with_cs1'):
                warnings.append("Results inconsistent with CS1 findings")
            if not cv_results.get('economic_sense', {}).get('half_life_reasonable'):
                warnings.append("Half-lives outside expected range")
        
        # Print summary
        print("\n📊 TEST RESULTS OVERVIEW:")
        print("-" * 40)
        
        if critical_issues:
            print("❌ CRITICAL ISSUES:")
            for issue in critical_issues:
                print(f"   - {issue}")
            all_passed = False
        
        if warnings:
            print("⚠️  WARNINGS:")
            for warning in warnings:
                print(f"   - {warning}")
        
        if all_passed and not warnings:
            print("✅ ALL TESTS PASSED - Framework is robust!")
        elif all_passed:
            print("✅ TESTS PASSED with warnings - Framework is functional")
        else:
            print("❌ CRITICAL ISSUES FOUND - Framework needs attention")
        
        # Statistical summary
        print("\n📈 STATISTICAL SUMMARY:")
        print("-" * 40)
        
        if 'statistical' in self.test_results:
            stat = self.test_results['statistical']
            if 'ar4_diagnostics' in stat:
                print(f"AIC: {stat['ar4_diagnostics'].get('aic', 'N/A')}")
                print(f"Observations: {stat['ar4_diagnostics'].get('n_obs', 'N/A')}")
            if 'rmse_validation' in stat:
                print(f"RMSE: {stat['rmse_validation'].get('rmse', 'N/A'):.2f}")
                print(f"RMSE % of mean: {stat['rmse_validation'].get('rmse_pct_of_mean', 'N/A'):.1f}%")
        
        # Sensitivity summary
        print("\n🔄 SENSITIVITY ANALYSIS:")
        print("-" * 40)
        
        if 'sensitivity' in self.test_results:
            sens = self.test_results['sensitivity']
            if 'ar_order_sensitivity' in sens:
                print(f"Preferred model: {sens['ar_order_sensitivity'].get('preferred_model', 'N/A')}")
            if 'aggregation_sensitivity' in sens:
                print(f"Sum vs Avg correlation: {sens['aggregation_sensitivity'].get('sum_avg_correlation', 'N/A'):.3f}")
            if 'time_period_sensitivity' in sens:
                print(f"Crisis impact: {sens['time_period_sensitivity'].get('crisis_impact', 'N/A')}")
        
        print("\n" + "=" * 80)
        print("END OF ROBUSTNESS TESTING REPORT")
        print("=" * 80)
        
        return all_passed


def run_robustness_tests():
    """Main function to run all robustness tests"""
    tester = CS4RobustnessTests()
    results = tester.run_all_tests(verbose=True)
    return results


if __name__ == "__main__":
    print("🚀 Starting CS4 Robustness Testing Suite")
    print("This will run comprehensive validation tests...")
    print()
    
    results = run_robustness_tests()
    
    print("\n✅ Robustness testing complete!")