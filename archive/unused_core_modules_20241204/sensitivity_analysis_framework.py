"""
Sensitivity Analysis Framework for Capital Flows Research

Comprehensive framework for assessing the robustness of statistical findings
to different methodological choices including outlier treatment, crisis period
definitions, and threshold selections.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=RuntimeWarning)

class SensitivityAnalysisFramework:
    """Comprehensive sensitivity analysis for robust statistical conclusions"""
    
    def __init__(self):
        self.results_cache = {}
        self.sensitivity_tests = {}
        
    def run_comprehensive_sensitivity_analysis(self, 
                                             data: pd.DataFrame,
                                             indicators: List[str],
                                             group_column: str = 'CS1_GROUP',
                                             group_values: List[str] = ['Iceland', 'Eurozone']) -> Dict[str, Any]:
        """
        Run comprehensive sensitivity analysis across multiple dimensions
        
        Args:
            data: Dataset to analyze
            indicators: List of indicator columns
            group_column: Column defining groups for comparison
            group_values: Values of groups to compare
            
        Returns:
            Comprehensive sensitivity analysis results
        """
        logger.info("Running comprehensive sensitivity analysis...")
        
        try:
            sensitivity_results = {
                'winsorization_sensitivity': self._test_winsorization_sensitivity(
                    data, indicators, group_column, group_values
                ),
                'crisis_period_sensitivity': self._test_crisis_period_sensitivity(
                    data, indicators, group_column, group_values
                ),
                'threshold_sensitivity': self._test_threshold_sensitivity(
                    data, indicators, group_column, group_values
                ),
                'sample_period_sensitivity': self._test_sample_period_sensitivity(
                    data, indicators, group_column, group_values
                ),
                'statistical_method_sensitivity': self._test_statistical_method_sensitivity(
                    data, indicators, group_column, group_values
                ),
                'summary': {}
            }
            
            # Generate comprehensive summary
            sensitivity_results['summary'] = self._generate_sensitivity_summary(sensitivity_results)
            
            logger.info("Comprehensive sensitivity analysis completed")
            return sensitivity_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive sensitivity analysis: {str(e)}")
            raise
    
    def _test_winsorization_sensitivity(self, 
                                      data: pd.DataFrame, 
                                      indicators: List[str],
                                      group_column: str,
                                      group_values: List[str]) -> Dict[str, Any]:
        """Test sensitivity to different winsorization levels"""
        
        logger.info("Testing winsorization sensitivity...")
        
        winsorization_levels = [0.00, 0.01, 0.025, 0.05, 0.10]  # 0%, 1%, 2.5%, 5%, 10%
        winsorization_results = {}
        
        for level in winsorization_levels:
            level_name = f"{level*100:.1f}%"
            logger.info(f"  Testing {level_name} winsorization...")
            
            try:
                # Apply winsorization
                data_winsorized = self._apply_winsorization(data.copy(), indicators, level)
                
                # Perform F-tests
                f_test_results = self._perform_f_tests_by_groups(
                    data_winsorized, indicators, group_column, group_values
                )
                
                # Calculate summary statistics
                winsorization_results[level_name] = {
                    'f_test_results': f_test_results,
                    'significant_count_5pct': sum(1 for r in f_test_results if r.get('p_value', 1) < 0.05),
                    'significant_count_1pct': sum(1 for r in f_test_results if r.get('p_value', 1) < 0.01),
                    'avg_f_statistic': np.mean([r.get('f_statistic', 0) for r in f_test_results if pd.notna(r.get('f_statistic'))]),
                    'winsorization_level': level
                }
                
            except Exception as e:
                logger.warning(f"Error testing {level_name} winsorization: {str(e)}")
                winsorization_results[level_name] = {'error': str(e)}
        
        # Analyze consistency across winsorization levels
        consistency_analysis = self._analyze_winsorization_consistency(winsorization_results, indicators)
        winsorization_results['consistency_analysis'] = consistency_analysis
        
        return winsorization_results
    
    def _test_crisis_period_sensitivity(self, 
                                       data: pd.DataFrame,
                                       indicators: List[str],
                                       group_column: str,
                                       group_values: List[str]) -> Dict[str, Any]:
        """Test sensitivity to different crisis period definitions"""
        
        logger.info("Testing crisis period sensitivity...")
        
        crisis_definitions = {
            'no_exclusion': [],
            'gfc_only': [2008, 2009, 2010],
            'covid_only': [2020, 2021, 2022], 
            'both_crises': [2008, 2009, 2010, 2020, 2021, 2022],
            'extended_gfc': [2007, 2008, 2009, 2010, 2011],
            'extended_covid': [2020, 2021, 2022, 2023]
        }
        
        crisis_results = {}
        
        for crisis_name, excluded_years in crisis_definitions.items():
            logger.info(f"  Testing '{crisis_name}' crisis definition...")
            
            try:
                # Filter data
                if excluded_years:
                    data_filtered = data[~data['YEAR'].isin(excluded_years)].copy()
                else:
                    data_filtered = data.copy()
                
                if len(data_filtered) == 0:
                    continue
                
                # Perform F-tests
                f_test_results = self._perform_f_tests_by_groups(
                    data_filtered, indicators, group_column, group_values
                )
                
                crisis_results[crisis_name] = {
                    'f_test_results': f_test_results,
                    'sample_size': len(data_filtered),
                    'excluded_years': excluded_years,
                    'significant_count_5pct': sum(1 for r in f_test_results if r.get('p_value', 1) < 0.05),
                    'significant_count_1pct': sum(1 for r in f_test_results if r.get('p_value', 1) < 0.01)
                }
                
            except Exception as e:
                logger.warning(f"Error testing '{crisis_name}' crisis definition: {str(e)}")
                crisis_results[crisis_name] = {'error': str(e)}
        
        # Analyze consistency across crisis definitions
        consistency_analysis = self._analyze_crisis_consistency(crisis_results, indicators)
        crisis_results['consistency_analysis'] = consistency_analysis
        
        return crisis_results
    
    def _test_threshold_sensitivity(self, 
                                  data: pd.DataFrame,
                                  indicators: List[str],
                                  group_column: str,
                                  group_values: List[str]) -> Dict[str, Any]:
        """Test sensitivity to different significance thresholds"""
        
        logger.info("Testing threshold sensitivity...")
        
        # Perform F-tests once
        f_test_results = self._perform_f_tests_by_groups(
            data, indicators, group_column, group_values
        )
        
        # Test different significance thresholds
        thresholds = [0.001, 0.01, 0.05, 0.10, 0.15, 0.20]
        threshold_results = {}
        
        for threshold in thresholds:
            threshold_name = f"{threshold*100:.1f}%"
            
            significant_indicators = [
                r['indicator'] for r in f_test_results 
                if r.get('p_value', 1) < threshold
            ]
            
            threshold_results[threshold_name] = {
                'threshold': threshold,
                'significant_count': len(significant_indicators),
                'significant_indicators': significant_indicators,
                'significance_rate': len(significant_indicators) / len(indicators) * 100
            }
        
        # Analyze threshold stability
        stability_analysis = self._analyze_threshold_stability(threshold_results)
        threshold_results['stability_analysis'] = stability_analysis
        
        return threshold_results
    
    def _test_sample_period_sensitivity(self, 
                                       data: pd.DataFrame,
                                       indicators: List[str],
                                       group_column: str,
                                       group_values: List[str]) -> Dict[str, Any]:
        """Test sensitivity to different sample periods"""
        
        logger.info("Testing sample period sensitivity...")
        
        if 'YEAR' not in data.columns:
            return {'error': 'YEAR column not found in data'}
        
        years = sorted(data['YEAR'].unique())
        min_year, max_year = min(years), max(years)
        
        # Define different sample periods
        sample_periods = {
            'full_sample': (min_year, max_year),
            'pre_2010': (min_year, 2009),
            'post_2010': (2010, max_year),
            'pre_2015': (min_year, 2014),
            'post_2015': (2015, max_year),
            'middle_period': (min_year + 5, max_year - 5) if max_year - min_year > 10 else (min_year, max_year)
        }
        
        period_results = {}
        
        for period_name, (start_year, end_year) in sample_periods.items():
            if start_year >= end_year:
                continue
                
            logger.info(f"  Testing {period_name}: {start_year}-{end_year}")
            
            try:
                # Filter data by period
                data_period = data[
                    (data['YEAR'] >= start_year) & (data['YEAR'] <= end_year)
                ].copy()
                
                if len(data_period) < 20:  # Minimum sample size
                    continue
                
                # Perform F-tests
                f_test_results = self._perform_f_tests_by_groups(
                    data_period, indicators, group_column, group_values
                )
                
                period_results[period_name] = {
                    'period': (start_year, end_year),
                    'sample_size': len(data_period),
                    'f_test_results': f_test_results,
                    'significant_count_5pct': sum(1 for r in f_test_results if r.get('p_value', 1) < 0.05),
                    'years_covered': end_year - start_year + 1
                }
                
            except Exception as e:
                logger.warning(f"Error testing period {period_name}: {str(e)}")
                period_results[period_name] = {'error': str(e)}
        
        # Analyze period consistency
        consistency_analysis = self._analyze_period_consistency(period_results, indicators)
        period_results['consistency_analysis'] = consistency_analysis
        
        return period_results
    
    def _test_statistical_method_sensitivity(self, 
                                           data: pd.DataFrame,
                                           indicators: List[str],
                                           group_column: str,
                                           group_values: List[str]) -> Dict[str, Any]:
        """Test sensitivity to different statistical methods"""
        
        logger.info("Testing statistical method sensitivity...")
        
        method_results = {}
        
        # F-test (current method)
        try:
            f_test_results = self._perform_f_tests_by_groups(
                data, indicators, group_column, group_values
            )
            method_results['f_test'] = {
                'results': f_test_results,
                'significant_count_5pct': sum(1 for r in f_test_results if r.get('p_value', 1) < 0.05),
                'method_description': 'F-test for equality of variances'
            }
        except Exception as e:
            method_results['f_test'] = {'error': str(e)}
        
        # Levene's test
        try:
            levene_results = self._perform_levene_tests(
                data, indicators, group_column, group_values
            )
            method_results['levene_test'] = {
                'results': levene_results,
                'significant_count_5pct': sum(1 for r in levene_results if r.get('p_value', 1) < 0.05),
                'method_description': "Levene's test for equality of variances (more robust to non-normality)"
            }
        except Exception as e:
            method_results['levene_test'] = {'error': str(e)}
        
        # Bartlett's test
        try:
            bartlett_results = self._perform_bartlett_tests(
                data, indicators, group_column, group_values
            )
            method_results['bartlett_test'] = {
                'results': bartlett_results,
                'significant_count_5pct': sum(1 for r in bartlett_results if r.get('p_value', 1) < 0.05),
                'method_description': "Bartlett's test for equality of variances (assumes normality)"
            }
        except Exception as e:
            method_results['bartlett_test'] = {'error': str(e)}
        
        # Brown-Forsythe test
        try:
            bf_results = self._perform_brown_forsythe_tests(
                data, indicators, group_column, group_values
            )
            method_results['brown_forsythe_test'] = {
                'results': bf_results,
                'significant_count_5pct': sum(1 for r in bf_results if r.get('p_value', 1) < 0.05),
                'method_description': 'Brown-Forsythe test for equality of variances (robust to non-normality)'
            }
        except Exception as e:
            method_results['brown_forsythe_test'] = {'error': str(e)}
        
        # Analyze method consistency
        consistency_analysis = self._analyze_method_consistency(method_results, indicators)
        method_results['consistency_analysis'] = consistency_analysis
        
        return method_results
    
    def _apply_winsorization(self, 
                           data: pd.DataFrame, 
                           indicators: List[str], 
                           level: float) -> pd.DataFrame:
        """Apply winsorization at specified level"""
        
        if level == 0.0:
            return data
        
        for indicator in indicators:
            if indicator in data.columns and data[indicator].notna().sum() > 0:
                values = data[indicator].dropna()
                if len(values) > 2:
                    lower_bound = values.quantile(level)
                    upper_bound = values.quantile(1 - level)
                    
                    data.loc[data[indicator] < lower_bound, indicator] = lower_bound
                    data.loc[data[indicator] > upper_bound, indicator] = upper_bound
        
        return data
    
    def _perform_f_tests_by_groups(self, 
                                  data: pd.DataFrame,
                                  indicators: List[str],
                                  group_column: str,
                                  group_values: List[str]) -> List[Dict]:
        """Perform F-tests for each indicator"""
        
        results = []
        
        if len(group_values) != 2:
            logger.warning("F-test requires exactly 2 groups")
            return results
        
        for indicator in indicators:
            if indicator not in data.columns:
                continue
                
            try:
                group1_data = data[data[group_column] == group_values[0]][indicator].dropna()
                group2_data = data[data[group_column] == group_values[1]][indicator].dropna()
                
                if len(group1_data) < 2 or len(group2_data) < 2:
                    continue
                
                # Perform F-test
                var1 = group1_data.var(ddof=1)
                var2 = group2_data.var(ddof=1)
                
                if var1 <= 0 or var2 <= 0:
                    continue
                
                f_statistic = max(var1, var2) / min(var1, var2)
                df1 = len(group1_data) - 1 if var1 >= var2 else len(group2_data) - 1
                df2 = len(group2_data) - 1 if var1 >= var2 else len(group1_data) - 1
                
                p_value = 2 * (1 - stats.f.cdf(f_statistic, df1, df2))
                
                results.append({
                    'indicator': indicator,
                    'f_statistic': f_statistic,
                    'p_value': p_value,
                    'df1': df1,
                    'df2': df2,
                    'group1_var': var1,
                    'group2_var': var2,
                    'group1_n': len(group1_data),
                    'group2_n': len(group2_data)
                })
                
            except Exception as e:
                logger.warning(f"Error in F-test for {indicator}: {str(e)}")
                continue
        
        return results
    
    def _perform_levene_tests(self, 
                            data: pd.DataFrame,
                            indicators: List[str],
                            group_column: str,
                            group_values: List[str]) -> List[Dict]:
        """Perform Levene's tests for each indicator"""
        
        results = []
        
        for indicator in indicators:
            if indicator not in data.columns:
                continue
                
            try:
                groups_data = []
                for group_value in group_values:
                    group_data = data[data[group_column] == group_value][indicator].dropna()
                    if len(group_data) >= 2:
                        groups_data.append(group_data)
                
                if len(groups_data) >= 2:
                    statistic, p_value = stats.levene(*groups_data)
                    
                    results.append({
                        'indicator': indicator,
                        'statistic': statistic,
                        'p_value': p_value,
                        'method': 'levene'
                    })
                    
            except Exception as e:
                logger.warning(f"Error in Levene's test for {indicator}: {str(e)}")
                continue
        
        return results
    
    def _perform_bartlett_tests(self, 
                              data: pd.DataFrame,
                              indicators: List[str],
                              group_column: str,
                              group_values: List[str]) -> List[Dict]:
        """Perform Bartlett's tests for each indicator"""
        
        results = []
        
        for indicator in indicators:
            if indicator not in data.columns:
                continue
                
            try:
                groups_data = []
                for group_value in group_values:
                    group_data = data[data[group_column] == group_value][indicator].dropna()
                    if len(group_data) >= 2:
                        groups_data.append(group_data)
                
                if len(groups_data) >= 2:
                    statistic, p_value = stats.bartlett(*groups_data)
                    
                    results.append({
                        'indicator': indicator,
                        'statistic': statistic,
                        'p_value': p_value,
                        'method': 'bartlett'
                    })
                    
            except Exception as e:
                logger.warning(f"Error in Bartlett's test for {indicator}: {str(e)}")
                continue
        
        return results
    
    def _perform_brown_forsythe_tests(self, 
                                    data: pd.DataFrame,
                                    indicators: List[str],
                                    group_column: str,
                                    group_values: List[str]) -> List[Dict]:
        """Perform Brown-Forsythe tests for each indicator"""
        
        results = []
        
        for indicator in indicators:
            if indicator not in data.columns:
                continue
                
            try:
                groups_data = []
                for group_value in group_values:
                    group_data = data[data[group_column] == group_value][indicator].dropna()
                    if len(group_data) >= 2:
                        groups_data.append(group_data)
                
                if len(groups_data) >= 2:
                    # Brown-Forsythe is Levene's test with median
                    statistic, p_value = stats.levene(*groups_data, center='median')
                    
                    results.append({
                        'indicator': indicator,
                        'statistic': statistic,
                        'p_value': p_value,
                        'method': 'brown_forsythe'
                    })
                    
            except Exception as e:
                logger.warning(f"Error in Brown-Forsythe test for {indicator}: {str(e)}")
                continue
        
        return results
    
    def _analyze_winsorization_consistency(self, 
                                         winsorization_results: Dict,
                                         indicators: List[str]) -> Dict[str, Any]:
        """Analyze consistency across winsorization levels"""
        
        # Remove consistency_analysis and error entries for analysis
        levels_data = {k: v for k, v in winsorization_results.items() 
                      if k not in ['consistency_analysis'] and 'error' not in v}
        
        if len(levels_data) < 2:
            return {'error': 'Insufficient data for consistency analysis'}
        
        # Track which indicators are significant across different levels
        indicator_significance = {}
        
        for indicator in indicators:
            significance_pattern = {}
            for level_name, level_data in levels_data.items():
                f_test_results = level_data.get('f_test_results', [])
                indicator_result = next((r for r in f_test_results if r['indicator'] == indicator), None)
                
                if indicator_result:
                    significance_pattern[level_name] = indicator_result.get('p_value', 1) < 0.05
                else:
                    significance_pattern[level_name] = None
            
            indicator_significance[indicator] = significance_pattern
        
        # Calculate consistency metrics
        consistent_indicators = []
        inconsistent_indicators = []
        
        for indicator, pattern in indicator_significance.items():
            valid_results = [v for v in pattern.values() if v is not None]
            if len(valid_results) > 1:
                if len(set(valid_results)) == 1:  # All same
                    consistent_indicators.append(indicator)
                else:
                    inconsistent_indicators.append(indicator)
        
        consistency_rate = len(consistent_indicators) / len(indicators) * 100 if indicators else 0
        
        return {
            'consistent_indicators': consistent_indicators,
            'inconsistent_indicators': inconsistent_indicators,
            'consistency_rate': consistency_rate,
            'total_indicators': len(indicators),
            'levels_tested': list(levels_data.keys()),
            'indicator_patterns': indicator_significance
        }
    
    def _analyze_crisis_consistency(self, 
                                  crisis_results: Dict,
                                  indicators: List[str]) -> Dict[str, Any]:
        """Analyze consistency across crisis definitions"""
        
        # Remove consistency_analysis and error entries
        definitions_data = {k: v for k, v in crisis_results.items() 
                           if k not in ['consistency_analysis'] and 'error' not in v}
        
        if len(definitions_data) < 2:
            return {'error': 'Insufficient data for consistency analysis'}
        
        # Analyze significance patterns
        significance_counts = {}
        for def_name, def_data in definitions_data.items():
            significance_counts[def_name] = def_data.get('significant_count_5pct', 0)
        
        # Calculate consistency metrics
        max_significant = max(significance_counts.values())
        min_significant = min(significance_counts.values())
        range_significant = max_significant - min_significant
        
        avg_significant = np.mean(list(significance_counts.values()))
        std_significant = np.std(list(significance_counts.values()))
        
        return {
            'significance_counts': significance_counts,
            'range_significant': range_significant,
            'avg_significant': avg_significant,
            'std_significant': std_significant,
            'coefficient_of_variation': std_significant / avg_significant * 100 if avg_significant > 0 else 0,
            'most_significant_definition': max(significance_counts, key=significance_counts.get),
            'least_significant_definition': min(significance_counts, key=significance_counts.get)
        }
    
    def _analyze_threshold_stability(self, threshold_results: Dict) -> Dict[str, Any]:
        """Analyze stability across significance thresholds"""
        
        # Remove analysis entries
        thresholds_data = {k: v for k, v in threshold_results.items() 
                          if k not in ['stability_analysis']}
        
        # Extract significance counts
        significance_counts = [data['significant_count'] for data in thresholds_data.values()]
        thresholds = [data['threshold'] for data in thresholds_data.values()]
        
        # Calculate stability metrics
        correlation = np.corrcoef(thresholds, significance_counts)[0, 1] if len(thresholds) > 1 else 0
        
        return {
            'threshold_correlation': correlation,
            'significance_range': max(significance_counts) - min(significance_counts),
            'threshold_sensitivity': 'High' if correlation > 0.8 else 'Medium' if correlation > 0.5 else 'Low'
        }
    
    def _analyze_period_consistency(self, 
                                  period_results: Dict,
                                  indicators: List[str]) -> Dict[str, Any]:
        """Analyze consistency across sample periods"""
        
        # Remove analysis and error entries
        periods_data = {k: v for k, v in period_results.items() 
                       if k not in ['consistency_analysis'] and 'error' not in v}
        
        if len(periods_data) < 2:
            return {'error': 'Insufficient data for consistency analysis'}
        
        # Analyze significance counts
        significance_counts = {}
        for period_name, period_data in periods_data.items():
            significance_counts[period_name] = period_data.get('significant_count_5pct', 0)
        
        # Calculate consistency
        counts = list(significance_counts.values())
        range_significant = max(counts) - min(counts) if counts else 0
        avg_significant = np.mean(counts) if counts else 0
        std_significant = np.std(counts) if counts else 0
        
        return {
            'significance_counts': significance_counts,
            'range_significant': range_significant,
            'avg_significant': avg_significant,
            'std_significant': std_significant,
            'consistency_rating': 'High' if range_significant <= 2 else 'Medium' if range_significant <= 5 else 'Low'
        }
    
    def _analyze_method_consistency(self, 
                                  method_results: Dict,
                                  indicators: List[str]) -> Dict[str, Any]:
        """Analyze consistency across statistical methods"""
        
        # Extract significance counts
        method_counts = {}
        for method_name, method_data in method_results.items():
            if method_name != 'consistency_analysis' and 'error' not in method_data:
                method_counts[method_name] = method_data.get('significant_count_5pct', 0)
        
        if len(method_counts) < 2:
            return {'error': 'Insufficient methods for consistency analysis'}
        
        # Calculate consistency metrics
        counts = list(method_counts.values())
        range_significant = max(counts) - min(counts)
        avg_significant = np.mean(counts)
        
        return {
            'method_counts': method_counts,
            'range_significant': range_significant,
            'avg_significant': avg_significant,
            'consistency_rating': 'High' if range_significant <= 1 else 'Medium' if range_significant <= 3 else 'Low',
            'most_conservative_method': min(method_counts, key=method_counts.get),
            'most_liberal_method': max(method_counts, key=method_counts.get)
        }
    
    def _generate_sensitivity_summary(self, sensitivity_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary of sensitivity analysis"""
        
        summary = {
            'overall_robustness': 'High',  # Default
            'key_findings': [],
            'recommendations': [],
            'robustness_score': 0,
            'total_tests': 0,
            'robust_tests': 0
        }
        
        # Analyze winsorization sensitivity
        if 'winsorization_sensitivity' in sensitivity_results:
            wins_data = sensitivity_results['winsorization_sensitivity']
            if 'consistency_analysis' in wins_data:
                consistency = wins_data['consistency_analysis']
                consistency_rate = consistency.get('consistency_rate', 0)
                
                summary['key_findings'].append(f"Winsorization consistency: {consistency_rate:.1f}%")
                
                if consistency_rate >= 80:
                    summary['recommendations'].append("Results are robust to outlier treatment")
                else:
                    summary['recommendations'].append("Consider reporting both original and winsorized results")
        
        # Analyze crisis period sensitivity
        if 'crisis_period_sensitivity' in sensitivity_results:
            crisis_data = sensitivity_results['crisis_period_sensitivity']
            if 'consistency_analysis' in crisis_data:
                consistency = crisis_data['consistency_analysis']
                cv = consistency.get('coefficient_of_variation', 0)
                
                summary['key_findings'].append(f"Crisis period sensitivity CV: {cv:.1f}%")
                
                if cv < 20:
                    summary['recommendations'].append("Results are stable across crisis definitions")
                else:
                    summary['recommendations'].append("Crisis period definition affects results - report sensitivity")
        
        # Analyze method consistency
        if 'statistical_method_sensitivity' in sensitivity_results:
            method_data = sensitivity_results['statistical_method_sensitivity']
            if 'consistency_analysis' in method_data:
                consistency = method_data['consistency_analysis']
                rating = consistency.get('consistency_rating', 'Unknown')
                
                summary['key_findings'].append(f"Method consistency: {rating}")
                
                if rating == 'High':
                    summary['recommendations'].append("Statistical methods are consistent")
                else:
                    summary['recommendations'].append("Consider multiple statistical approaches")
        
        # Overall assessment
        robustness_indicators = [
            sensitivity_results.get('winsorization_sensitivity', {}).get('consistency_analysis', {}).get('consistency_rate', 50),
            100 - sensitivity_results.get('crisis_period_sensitivity', {}).get('consistency_analysis', {}).get('coefficient_of_variation', 50),
            85 if sensitivity_results.get('statistical_method_sensitivity', {}).get('consistency_analysis', {}).get('consistency_rating') == 'High' else 60
        ]
        
        overall_score = np.mean([score for score in robustness_indicators if score is not None])
        summary['robustness_score'] = overall_score
        
        if overall_score >= 80:
            summary['overall_robustness'] = 'High'
        elif overall_score >= 60:
            summary['overall_robustness'] = 'Medium'
        else:
            summary['overall_robustness'] = 'Low'
        
        return summary


if __name__ == "__main__":
    # Test the sensitivity analysis framework
    print("Testing Sensitivity Analysis Framework...")
    
    try:
        # Create test data
        np.random.seed(42)
        test_data = pd.DataFrame({
            'COUNTRY': ['Iceland'] * 100 + ['Germany'] * 100,
            'CS1_GROUP': ['Iceland'] * 100 + ['Eurozone'] * 100,
            'YEAR': list(range(2000, 2025)) * 8,
            'Indicator1': np.random.normal(5, 2, 200),
            'Indicator2': np.random.normal(3, 1.5, 200)
        })
        
        # Run sensitivity analysis
        framework = SensitivityAnalysisFramework()
        results = framework.run_comprehensive_sensitivity_analysis(
            test_data, ['Indicator1', 'Indicator2']
        )
        
        print(f"✓ Sensitivity analysis completed")
        print(f"  Overall robustness: {results['summary'].get('overall_robustness', 'Unknown')}")
        print(f"  Robustness score: {results['summary'].get('robustness_score', 0):.1f}/100")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("Sensitivity analysis framework test completed!")