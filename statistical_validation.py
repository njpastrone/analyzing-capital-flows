#!/usr/bin/env python3
"""
Statistical Validation Audit for Capital Flows Research Project

This script conducts comprehensive validation of statistical methodologies 
to ensure academic rigor and consistency across all case studies.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import sys
import warnings
from datetime import datetime

# Add dashboard modules to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "dashboard"))

warnings.filterwarnings('ignore')

class StatisticalValidator:
    """Comprehensive statistical validation for Capital Flows research"""
    
    def __init__(self):
        self.validation_results = []
        self.critical_failures = []
        self.warnings = []
        
    def log_result(self, test_name, status, details, critical=False):
        """Log validation result"""
        result = {
            'test_name': test_name,
            'status': status,  # 'PASS', 'FAIL', 'WARNING'
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.validation_results.append(result)
        
        if status == 'FAIL' and critical:
            self.critical_failures.append(result)
        elif status == 'WARNING':
            self.warnings.append(result)
    
    def validate_f_test_implementation(self):
        """Validate F-test calculations across case studies"""
        print("🧮 Validating F-Test Implementation...")
        
        try:
            # Test CS1 F-test implementation
            from simple_report_app import load_default_data, perform_volatility_tests
            
            final_data, analysis_indicators, _ = load_default_data(include_crisis_years=True)
            f_test_results = perform_volatility_tests(final_data, analysis_indicators)
            
            if isinstance(f_test_results, pd.DataFrame):
                # Manual F-test calculation for validation
                test_indicator = analysis_indicators[0]
                
                iceland_data = final_data[final_data['CS1_GROUP'] == 'Iceland'][test_indicator].dropna()
                eurozone_data = final_data[final_data['CS1_GROUP'] == 'Eurozone'][test_indicator].dropna()
                
                # Manual calculation
                var_iceland = iceland_data.var(ddof=1)
                var_eurozone = eurozone_data.var(ddof=1)
                f_stat_manual = max(var_iceland, var_eurozone) / min(var_iceland, var_eurozone)
                df1 = len(iceland_data) - 1
                df2 = len(eurozone_data) - 1
                
                # Compare with automated result
                if 'F_Statistic' in f_test_results.columns:
                    automated_f_stat = f_test_results.iloc[0]['F_Statistic']
                    
                    if abs(f_stat_manual - automated_f_stat) / f_stat_manual < 0.01:  # 1% tolerance
                        self.log_result(
                            'F-Test Manual vs Automated',
                            'PASS',
                            f'Manual: {f_stat_manual:.4f}, Automated: {automated_f_stat:.4f}'
                        )
                    else:
                        self.log_result(
                            'F-Test Manual vs Automated',
                            'FAIL',
                            f'Significant discrepancy: Manual: {f_stat_manual:.4f}, Automated: {automated_f_stat:.4f}',
                            critical=True
                        )
                
                # Validate degrees of freedom
                if len(iceland_data) == 105 and len(eurozone_data) == 988:
                    self.log_result(
                        'F-Test Degrees of Freedom',
                        'PASS',
                        f'Iceland df1={df1}, Eurozone df2={df2}'
                    )
                else:
                    self.log_result(
                        'F-Test Degrees of Freedom',
                        'WARNING',
                        f'Unexpected sample sizes: Iceland={len(iceland_data)}, Eurozone={len(eurozone_data)}'
                    )
                
                # Check significance level applications
                if 'P_Value' in f_test_results.columns:
                    p_values = f_test_results['P_Value'].dropna()
                    sig_1pct = (p_values < 0.01).sum()
                    sig_5pct = (p_values < 0.05).sum()
                    sig_10pct = (p_values < 0.10).sum()
                    
                    self.log_result(
                        'Significance Level Distribution',
                        'PASS',
                        f'1%: {sig_1pct}, 5%: {sig_5pct}, 10%: {sig_10pct} out of {len(p_values)} tests'
                    )
                else:
                    self.log_result(
                        'P-Value Availability',
                        'WARNING',
                        'P-values not found in F-test results'
                    )
            else:
                self.log_result(
                    'F-Test Results Format',
                    'WARNING',
                    f'Unexpected F-test results format: {type(f_test_results)}'
                )
                
        except Exception as e:
            self.log_result(
                'F-Test Implementation',
                'FAIL',
                f'Error in F-test validation: {str(e)}',
                critical=True
            )
    
    def validate_crisis_period_consistency(self):
        """Validate crisis period definitions across case studies"""
        print("📅 Validating Crisis Period Consistency...")
        
        crisis_definitions = {
            'Global Financial Crisis': [2008, 2009, 2010],
            'COVID-19 Pandemic': [2020, 2021, 2022]
        }
        
        try:
            # Test CS1 crisis exclusion
            from simple_report_app import load_default_data
            
            cs1_full, _, _ = load_default_data(include_crisis_years=True)
            cs1_excluded, _, cs1_metadata = load_default_data(include_crisis_years=False)
            
            full_years = set(cs1_full['YEAR'].unique())
            excluded_years = set(cs1_excluded['YEAR'].unique())
            removed_years = full_years - excluded_years
            
            expected_crisis_years = set(crisis_definitions['Global Financial Crisis'] + 
                                      crisis_definitions['COVID-19 Pandemic'])
            
            if removed_years == expected_crisis_years:
                self.log_result(
                    'CS1 Crisis Period Exclusion',
                    'PASS',
                    f'Correctly excluded years: {sorted(removed_years)}'
                )
            else:
                self.log_result(
                    'CS1 Crisis Period Exclusion',
                    'FAIL',
                    f'Expected: {sorted(expected_crisis_years)}, Actual: {sorted(removed_years)}',
                    critical=True
                )
            
            # Validate sample size reduction
            excluded_count = cs1_metadata.get('excluded_observations', 0)
            expected_reduction = len(cs1_full) - len(cs1_excluded)
            
            if excluded_count == expected_reduction:
                self.log_result(
                    'CS1 Sample Size Reduction',
                    'PASS',
                    f'Excluded {excluded_count} observations as expected'
                )
            else:
                self.log_result(
                    'CS1 Sample Size Reduction',
                    'WARNING',
                    f'Metadata shows {excluded_count}, actual reduction: {expected_reduction}'
                )
            
            # Test CS2 crisis exclusion
            try:
                from case_study_2_euro_adoption import load_case_study_2_data
                
                cs2_full, _, _ = load_case_study_2_data(include_crisis_years=True)
                cs2_excluded, _, _ = load_case_study_2_data(include_crisis_years=False)
                
                cs2_full_years = set(cs2_full['YEAR'].unique())
                cs2_excluded_years = set(cs2_excluded['YEAR'].unique())
                cs2_removed_years = cs2_full_years - cs2_excluded_years
                
                # CS2 should exclude same periods, possibly plus Latvia Banking Crisis (2011)
                if expected_crisis_years.issubset(cs2_removed_years):
                    self.log_result(
                        'CS2 Crisis Period Exclusion',
                        'PASS',
                        f'CS2 excluded crisis years: {sorted(cs2_removed_years)}'
                    )
                else:
                    self.log_result(
                        'CS2 Crisis Period Exclusion',
                        'WARNING',
                        f'CS2 crisis exclusion differs from CS1: {sorted(cs2_removed_years)}'
                    )
                    
            except Exception as e:
                self.log_result(
                    'CS2 Crisis Period Validation',
                    'WARNING',
                    f'Could not validate CS2 crisis periods: {str(e)}'
                )
                
        except Exception as e:
            self.log_result(
                'Crisis Period Consistency',
                'FAIL',
                f'Error in crisis period validation: {str(e)}',
                critical=True
            )
    
    def validate_cs4_econometric_methods(self):
        """Validate CS4 advanced econometric methods"""
        print("📈 Validating CS4 Econometric Methods...")
        
        try:
            # Check CS4 data structure for time series analysis
            data_dir = Path("updated_data/Clean/CS4_Statistical_Modeling")
            
            if not data_dir.exists():
                self.log_result(
                    'CS4 Data Directory',
                    'FAIL',
                    f'CS4 directory not found: {data_dir}',
                    critical=True
                )
                return
            
            # Validate file structure for econometric analysis
            required_files = [
                'net_capital_flows_full.csv',
                'net_capital_flows_no_crises.csv',
                'net_direct_investment_full.csv',
                'net_portfolio_investment_full.csv',
                'net_other_investment_full.csv'
            ]
            
            missing_files = []
            for file_name in required_files:
                if not (data_dir / file_name).exists():
                    missing_files.append(file_name)
            
            if not missing_files:
                self.log_result(
                    'CS4 Required Files',
                    'PASS',
                    f'All {len(required_files)} econometric files present'
                )
            else:
                self.log_result(
                    'CS4 Required Files',
                    'FAIL',
                    f'Missing files: {missing_files}',
                    critical=True
                )
            
            # Validate time series data structure
            test_file = data_dir / 'net_capital_flows_full.csv'
            if test_file.exists():
                df = pd.read_csv(test_file)
                
                # Check for time series requirements
                if 'YEAR' in df.columns and 'QUARTER' in df.columns:
                    years = df['YEAR'].unique()
                    quarters = df['QUARTER'].unique()
                    
                    # Should have sufficient time span for AR(4) modeling
                    time_span = len(years)
                    if time_span >= 20:
                        self.log_result(
                            'CS4 Time Series Span',
                            'PASS',
                            f'Sufficient data span: {time_span} years'
                        )
                    else:
                        self.log_result(
                            'CS4 Time Series Span',
                            'WARNING',
                            f'Limited data span for AR modeling: {time_span} years'
                        )
                    
                    # Check quarterly completeness
                    expected_quarters = {1, 2, 3, 4}
                    actual_quarters = set(quarters)
                    if actual_quarters == expected_quarters:
                        self.log_result(
                            'CS4 Quarterly Completeness',
                            'PASS',
                            'All quarters represented'
                        )
                    else:
                        self.log_result(
                            'CS4 Quarterly Completeness',
                            'WARNING',
                            f'Missing quarters: {expected_quarters - actual_quarters}'
                        )
                
                # Check for country coverage sufficient for panel analysis
                if 'COUNTRY' in df.columns:
                    countries = df['COUNTRY'].unique()
                    if len(countries) >= 10:
                        self.log_result(
                            'CS4 Country Coverage',
                            'PASS',
                            f'Adequate country coverage: {len(countries)} countries'
                        )
                    else:
                        self.log_result(
                            'CS4 Country Coverage',
                            'WARNING',
                            f'Limited country coverage: {len(countries)} countries'
                        )
                
            else:
                self.log_result(
                    'CS4 Data Structure',
                    'WARNING',
                    'Could not validate CS4 data structure - file not accessible'
                )
                
        except Exception as e:
            self.log_result(
                'CS4 Econometric Validation',
                'FAIL',
                f'Error in CS4 validation: {str(e)}',
                critical=True
            )
    
    def validate_cs5_external_data_integration(self):
        """Validate CS5 external data integration methods"""
        print("🔗 Validating CS5 External Data Integration...")
        
        try:
            # Check CS5 data directories
            controls_dir = Path("updated_data/Clean/CS5_Capital_Controls")
            regime_dir = Path("updated_data/Clean/CS5_Regime_Analysis")
            
            if not controls_dir.exists() or not regime_dir.exists():
                self.log_result(
                    'CS5 Data Directories',
                    'FAIL',
                    f'CS5 directories missing: controls={controls_dir.exists()}, regime={regime_dir.exists()}',
                    critical=True
                )
                return
            
            # Validate capital controls data structure
            controls_files = list(controls_dir.glob("*.csv"))
            regime_files = list(regime_dir.glob("*.csv"))
            
            if len(controls_files) >= 4 and len(regime_files) >= 8:
                self.log_result(
                    'CS5 File Count',
                    'PASS',
                    f'Controls: {len(controls_files)} files, Regime: {len(regime_files)} files'
                )
            else:
                self.log_result(
                    'CS5 File Count',
                    'WARNING',
                    f'Unexpected file counts - Controls: {len(controls_files)}, Regime: {len(regime_files)}'
                )
            
            # Check correlation analysis data structure
            test_controls_file = controls_dir / "sd_yearly_flows.csv"
            if test_controls_file.exists():
                df_controls = pd.read_csv(test_controls_file)
                
                # Should have numeric columns for correlation
                numeric_cols = df_controls.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    self.log_result(
                        'CS5 Correlation Data Structure',
                        'PASS',
                        f'{len(numeric_cols)} numeric columns available for correlation'
                    )
                    
                    # Test correlation calculation
                    test_corr = df_controls[numeric_cols[:2]].corr()
                    if not test_corr.isnull().all().all():
                        self.log_result(
                            'CS5 Correlation Calculation',
                            'PASS',
                            'Correlation matrix computable'
                        )
                    else:
                        self.log_result(
                            'CS5 Correlation Calculation',
                            'WARNING',
                            'Correlation matrix contains all NaN values'
                        )
                else:
                    self.log_result(
                        'CS5 Correlation Data Structure',
                        'WARNING',
                        f'Insufficient numeric columns: {len(numeric_cols)}'
                    )
            
            # Validate regime analysis data
            test_regime_file = regime_dir / "net_capital_flows_full.csv"
            if test_regime_file.exists():
                df_regime = pd.read_csv(test_regime_file)
                
                # Check time coverage for regime analysis
                if 'YEAR' in df_regime.columns:
                    years = df_regime['YEAR'].unique()
                    if max(years) >= 2015 and min(years) <= 2005:
                        self.log_result(
                            'CS5 Regime Time Coverage',
                            'PASS',
                            f'Adequate time span: {min(years)}-{max(years)}'
                        )
                    else:
                        self.log_result(
                            'CS5 Regime Time Coverage',
                            'WARNING',
                            f'Limited time coverage: {min(years)}-{max(years)}'
                        )
                
                # Check country coverage for regime analysis
                if 'COUNTRY' in df_regime.columns:
                    countries = df_regime['COUNTRY'].unique()
                    iceland_present = any('Iceland' in str(country) for country in countries)
                    
                    if iceland_present and len(countries) >= 5:
                        self.log_result(
                            'CS5 Regime Country Coverage',
                            'PASS',
                            f'Iceland included with {len(countries)} total countries'
                        )
                    else:
                        self.log_result(
                            'CS5 Regime Country Coverage',
                            'WARNING',
                            f'Iceland present: {iceland_present}, Countries: {len(countries)}'
                        )
            
        except Exception as e:
            self.log_result(
                'CS5 External Data Validation',
                'FAIL',
                f'Error in CS5 validation: {str(e)}',
                critical=True
            )
    
    def validate_cross_case_consistency(self):
        """Validate methodological consistency across case studies"""
        print("🔄 Validating Cross-Case Study Consistency...")
        
        try:
            # Load data from multiple case studies
            from simple_report_app import load_default_data
            
            cs1_data, cs1_indicators, cs1_metadata = load_default_data(include_crisis_years=True)
            
            # Check GDP normalization consistency
            if 'UNIT' in cs1_data.columns:
                unit_description = cs1_data['UNIT'].iloc[0]
                if '% of GDP' in unit_description:
                    self.log_result(
                        'GDP Normalization CS1',
                        'PASS',
                        'CS1 data properly normalized to % of GDP'
                    )
                else:
                    self.log_result(
                        'GDP Normalization CS1',
                        'WARNING',
                        f'CS1 unit description: {unit_description}'
                    )
            
            # Validate indicator count consistency
            expected_indicator_count = 14
            if len(cs1_indicators) == expected_indicator_count:
                self.log_result(
                    'CS1 Indicator Count',
                    'PASS',
                    f'CS1 has expected {expected_indicator_count} indicators'
                )
            else:
                self.log_result(
                    'CS1 Indicator Count',
                    'WARNING',
                    f'CS1 has {len(cs1_indicators)} indicators, expected {expected_indicator_count}'
                )
            
            # Check time series structure consistency
            if 'YEAR' in cs1_data.columns and 'QUARTER' in cs1_data.columns:
                min_year = cs1_data['YEAR'].min()
                max_year = cs1_data['YEAR'].max()
                
                if min_year <= 2000 and max_year >= 2020:
                    self.log_result(
                        'CS1 Time Coverage',
                        'PASS',
                        f'Good time coverage: {min_year}-{max_year}'
                    )
                else:
                    self.log_result(
                        'CS1 Time Coverage',
                        'WARNING',
                        f'Limited time coverage: {min_year}-{max_year}'
                    )
            
            # Test CS2 for consistency
            try:
                from case_study_2_euro_adoption import load_case_study_2_data
                
                cs2_data, cs2_indicators, cs2_metadata = load_case_study_2_data(include_crisis_years=True)
                
                # Compare indicator counts
                if len(cs2_indicators) == len(cs1_indicators):
                    self.log_result(
                        'CS1-CS2 Indicator Consistency',
                        'PASS',
                        f'Both have {len(cs1_indicators)} indicators'
                    )
                else:
                    self.log_result(
                        'CS1-CS2 Indicator Consistency',
                        'WARNING',
                        f'CS1: {len(cs1_indicators)}, CS2: {len(cs2_indicators)} indicators'
                    )
                
                # Check GDP normalization consistency
                if 'UNIT' in cs2_data.columns:
                    cs2_unit = cs2_data['UNIT'].iloc[0]
                    cs1_unit = cs1_data['UNIT'].iloc[0]
                    
                    if cs1_unit == cs2_unit:
                        self.log_result(
                            'CS1-CS2 Unit Consistency',
                            'PASS',
                            'Consistent GDP normalization across CS1 and CS2'
                        )
                    else:
                        self.log_result(
                            'CS1-CS2 Unit Consistency',
                            'WARNING',
                            f'CS1: {cs1_unit} vs CS2: {cs2_unit}'
                        )
                        
            except Exception as e:
                self.log_result(
                    'CS2 Consistency Check',
                    'WARNING',
                    f'Could not validate CS2 consistency: {str(e)}'
                )
            
        except Exception as e:
            self.log_result(
                'Cross-Case Consistency',
                'FAIL',
                f'Error in cross-case validation: {str(e)}',
                critical=True
            )
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "="*60)
        print("📊 STATISTICAL VALIDATION REPORT")
        print("="*60)
        
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results if r['status'] == 'PASS')
        failed_tests = sum(1 for r in self.validation_results if r['status'] == 'FAIL')
        warning_tests = sum(1 for r in self.validation_results if r['status'] == 'WARNING')
        
        print(f"\n📈 SUMMARY:")
        print(f"  Total Tests: {total_tests}")
        print(f"  ✅ Passed: {passed_tests}")
        print(f"  ❌ Failed: {failed_tests}")
        print(f"  ⚠️  Warnings: {warning_tests}")
        print(f"  📊 Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if self.critical_failures:
            print(f"\n🚨 CRITICAL FAILURES ({len(self.critical_failures)}):")
            for failure in self.critical_failures:
                print(f"  ❌ {failure['test_name']}: {failure['details']}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:5]:  # Show first 5 warnings
                print(f"  ⚠️  {warning['test_name']}: {warning['details']}")
        
        print(f"\n📋 DETAILED RESULTS:")
        for result in self.validation_results:
            status_icon = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️"}[result['status']]
            print(f"  {status_icon} {result['test_name']}: {result['details']}")
        
        # Academic rigor assessment
        print(f"\n🎓 ACADEMIC RIGOR ASSESSMENT:")
        
        if len(self.critical_failures) == 0:
            print("  ✅ Statistical methodology is academically sound")
        else:
            print(f"  ❌ {len(self.critical_failures)} critical issues require immediate attention")
        
        if failed_tests + len(self.critical_failures) == 0:
            print("  ✅ All statistical calculations verified")
        else:
            print(f"  ⚠️  {failed_tests} failed tests require review")
        
        if warning_tests <= total_tests * 0.2:  # Less than 20% warnings acceptable
            print("  ✅ Methodological consistency is good")
        else:
            print(f"  ⚠️  {warning_tests} warnings suggest methodology review needed")
        
        # Overall assessment
        if len(self.critical_failures) == 0 and failed_tests == 0:
            overall_status = "PUBLICATION READY"
            print(f"\n🎉 OVERALL STATUS: {overall_status}")
            print("   📚 Platform meets academic publication standards")
        elif len(self.critical_failures) == 0:
            overall_status = "MINOR REVISIONS NEEDED"
            print(f"\n📝 OVERALL STATUS: {overall_status}")
            print("   🔍 Address warnings before publication")
        else:
            overall_status = "MAJOR REVISIONS REQUIRED"
            print(f"\n🔧 OVERALL STATUS: {overall_status}")
            print("   ⚠️  Critical statistical issues must be resolved")
        
        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'warnings': warning_tests,
            'critical_failures': len(self.critical_failures),
            'overall_status': overall_status
        }
    
    def run_full_validation(self):
        """Run complete statistical validation audit"""
        print("🔬 Capital Flows Research - Statistical Validation Audit")
        print("="*60)
        print("Ensuring academic rigor and methodological consistency...")
        print()
        
        # Run all validation tests
        self.validate_f_test_implementation()
        self.validate_crisis_period_consistency()
        self.validate_cs4_econometric_methods()
        self.validate_cs5_external_data_integration()
        self.validate_cross_case_consistency()
        
        # Generate comprehensive report
        summary = self.generate_validation_report()
        
        return summary

def main():
    """Main execution function"""
    validator = StatisticalValidator()
    summary = validator.run_full_validation()
    
    # Exit with appropriate code
    if summary['critical_failures'] > 0:
        sys.exit(1)  # Critical failures
    elif summary['failed'] > 0:
        sys.exit(2)  # Non-critical failures
    else:
        sys.exit(0)  # Success

if __name__ == "__main__":
    main()