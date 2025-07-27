"""
Statistical testing classes for Capital Flows Research
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging

class StatisticalAnalyzer:
    """Base class for statistical analysis operations"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.logger = logging.getLogger(__name__)
        
    def calculate_descriptive_stats(self, data: pd.Series) -> Dict:
        """Calculate comprehensive descriptive statistics"""
        if len(data.dropna()) < 2:
            return {'error': 'Insufficient data'}
        
        clean_data = data.dropna()
        
        return {
            'n': len(clean_data),
            'mean': clean_data.mean(),
            'median': clean_data.median(),
            'std': clean_data.std(),
            'variance': clean_data.var(),
            'cv': (clean_data.std() / abs(clean_data.mean())) * 100 if clean_data.mean() != 0 else np.inf,
            'skewness': stats.skew(clean_data),
            'kurtosis': stats.kurtosis(clean_data),
            'min': clean_data.min(),
            'max': clean_data.max(),
            'q25': clean_data.quantile(0.25),
            'q75': clean_data.quantile(0.75),
            'iqr': clean_data.quantile(0.75) - clean_data.quantile(0.25)
        }
    
    def calculate_group_statistics(self, data: pd.DataFrame, group_col: str, 
                                 indicators: List[str]) -> pd.DataFrame:
        """Calculate statistics for all groups and indicators"""
        results = []
        
        for group in data[group_col].unique():
            group_data = data[data[group_col] == group]
            
            for indicator in indicators:
                values = group_data[indicator].dropna()
                
                if len(values) > 1:
                    stats_dict = self.calculate_descriptive_stats(values)
                    stats_dict.update({
                        'Group': group,
                        'Indicator': indicator.replace('_PGDP', ''),
                        'Indicator_Full': indicator
                    })
                    results.append(stats_dict)
        
        return pd.DataFrame(results)


class VolatilityTester(StatisticalAnalyzer):
    """Specialized class for volatility testing between groups"""
    
    def __init__(self, significance_level: float = 0.05):
        super().__init__(significance_level)
        
    def f_test_equal_variances(self, group1_data: pd.Series, group2_data: pd.Series,
                              group1_name: str = "Group1", group2_name: str = "Group2") -> Dict:
        """Perform F-test for equal variances between two groups"""
        
        # Clean data
        data1 = group1_data.dropna()
        data2 = group2_data.dropna()
        
        if len(data1) < 2 or len(data2) < 2:
            return {'error': 'Insufficient data for F-test'}
        
        # Calculate variances
        var1 = data1.var()
        var2 = data2.var()
        
        # F-statistic (larger variance in numerator)
        if var1 >= var2:
            f_stat = var1 / var2
            df1, df2 = len(data1) - 1, len(data2) - 1
            larger_group = group1_name
        else:
            f_stat = var2 / var1
            df1, df2 = len(data2) - 1, len(data1) - 1
            larger_group = group2_name
        
        # Two-tailed p-value
        p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 
                         1 - stats.f.cdf(f_stat, df1, df2))
        
        # Critical value
        critical_value = stats.f.ppf(1 - self.significance_level/2, df1, df2)
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'critical_value': critical_value,
            'df1': df1,
            'df2': df2,
            'significant': p_value < self.significance_level,
            'reject_null': f_stat > critical_value,
            'larger_variance_group': larger_group,
            f'{group1_name}_variance': var1,
            f'{group2_name}_variance': var2,
            f'{group1_name}_n': len(data1),
            f'{group2_name}_n': len(data2),
            'variance_ratio': var1 / var2 if var2 != 0 else np.inf
        }
    
    def perform_volatility_analysis(self, data: pd.DataFrame, indicators: List[str],
                                   group_col: str = 'GROUP', 
                                   group1: str = 'Iceland', group2: str = 'Eurozone') -> pd.DataFrame:
        """Perform comprehensive volatility analysis for all indicators"""
        
        results = []
        
        for indicator in indicators:
            # Get data for each group
            group1_data = data[data[group_col] == group1][indicator]
            group2_data = data[data[group_col] == group2][indicator]
            
            # Perform F-test
            f_test_result = self.f_test_equal_variances(
                group1_data, group2_data, group1, group2
            )
            
            if 'error' not in f_test_result:
                result_dict = {
                    'Indicator': indicator.replace('_PGDP', ''),
                    'Indicator_Full': indicator,
                    'F_Statistic': f_test_result['f_statistic'],
                    'P_Value': f_test_result['p_value'],
                    'Significant_5pct': f_test_result['p_value'] < 0.05,
                    'Significant_1pct': f_test_result['p_value'] < 0.01,
                    'Significant_0_1pct': f_test_result['p_value'] < 0.001,
                    f'{group1}_Higher_Volatility': f_test_result[f'{group1}_variance'] > f_test_result[f'{group2}_variance'],
                    f'{group1}_Variance': f_test_result[f'{group1}_variance'],
                    f'{group2}_Variance': f_test_result[f'{group2}_variance'],
                    'Variance_Ratio': f_test_result['variance_ratio'],
                    f'{group1}_N': f_test_result[f'{group1}_n'],
                    f'{group2}_N': f_test_result[f'{group2}_n']
                }
                results.append(result_dict)
        
        results_df = pd.DataFrame(results)
        
        # Log summary
        if len(results_df) > 0:
            total = len(results_df)
            higher_vol = results_df[f'{group1}_Higher_Volatility'].sum()
            significant = results_df['Significant_5pct'].sum()
            
            self.logger.info(f"Volatility analysis complete:")
            self.logger.info(f"  {group1} higher volatility: {higher_vol}/{total} ({higher_vol/total*100:.1f}%)")
            self.logger.info(f"  Significant differences: {significant}/{total} ({significant/total*100:.1f}%)")
        
        return results_df
    
    def t_test_equal_means(self, group1_data: pd.Series, group2_data: pd.Series,
                          equal_var: bool = False) -> Dict:
        """Perform t-test for equal means between two groups"""
        
        data1 = group1_data.dropna()
        data2 = group2_data.dropna()
        
        if len(data1) < 2 or len(data2) < 2:
            return {'error': 'Insufficient data for t-test'}
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
        
        # Degrees of freedom
        if equal_var:
            df = len(data1) + len(data2) - 2
        else:
            # Welch's t-test degrees of freedom
            s1, s2 = data1.var(), data2.var()
            n1, n2 = len(data1), len(data2)
            df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
        
        # Critical value
        critical_value = stats.t.ppf(1 - self.significance_level/2, df)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'critical_value': critical_value,
            'degrees_of_freedom': df,
            'significant': p_value < self.significance_level,
            'reject_null': abs(t_stat) > critical_value,
            'mean_difference': data1.mean() - data2.mean(),
            'equal_var_assumed': equal_var
        }


class EffectSizeCalculator:
    """Calculate effect sizes for group comparisons"""
    
    @staticmethod
    def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
        """Calculate Cohen's d effect size"""
        data1 = group1.dropna()
        data2 = group2.dropna()
        
        if len(data1) < 2 or len(data2) < 2:
            return np.nan
        
        # Pooled standard deviation
        n1, n2 = len(data1), len(data2)
        pooled_std = np.sqrt(((n1-1)*data1.var() + (n2-1)*data2.var()) / (n1+n2-2))
        
        if pooled_std == 0:
            return np.nan
        
        return (data1.mean() - data2.mean()) / pooled_std
    
    @staticmethod
    def hedges_g(group1: pd.Series, group2: pd.Series) -> float:
        """Calculate Hedges' g effect size (bias-corrected Cohen's d)"""
        d = EffectSizeCalculator.cohens_d(group1, group2)
        
        if np.isnan(d):
            return np.nan
        
        data1 = group1.dropna()
        data2 = group2.dropna()
        df = len(data1) + len(data2) - 2
        
        # Bias correction factor
        correction = 1 - (3 / (4*df - 1))
        
        return d * correction