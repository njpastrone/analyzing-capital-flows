"""
Case Study 4 Backend Statistical Analysis Framework

This module provides comprehensive statistical analysis functions for comparing
Iceland vs multiple comparator groups to evaluate currency regime effects on capital flows.

Statistical Methodologies:
1. F-tests for variance equality (homoscedasticity)
2. AR(4) models with impulse response half-life calculation
3. RMSE calculation using rolling prediction methodology

Data Structure:
- Location: updated_data/Clean/CS4_Statistical_Modeling/
- Indicators: Net Direct Investment, Net Portfolio Investment, Net Other Investment, Net Capital Flows
- Time Periods: Full time series and crisis-excluded versions
- Comparator Groups: Eurozone (sum/avg), SOEs (sum/avg), Baltics (sum/avg)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CS4DataLoader:
    """Data loading and validation for CS4 statistical analysis"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize CS4 data loader with data directory path"""
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / "updated_data" / "Clean" / "CS4_Statistical_Modeling"
        else:
            self.data_dir = Path(data_dir)
        
        # Define indicators and their file mappings
        self.indicators = {
            'Net Direct Investment': {'full': 'net_direct_investment_full.csv', 'no_crises': 'net_direct_investment_no_crises.csv'},
            'Net Portfolio Investment': {'full': 'net_portfolio_investment_full.csv', 'no_crises': 'net_portfolio_investment_no_crises.csv'}, 
            'Net Other Investment': {'full': 'net_other_investment_full.csv', 'no_crises': 'net_other_investment_no_crises.csv'},
            'Net Capital Flows': {'full': 'net_capital_flows_full.csv', 'no_crises': 'net_capital_flows_no_crises.csv'}
        }
        
        # Define comparator groups with new naming convention
        self.comparator_groups = ['eurozone_pgdp_weighted', 'eurozone_pgdp_simple', 'soe_pgdp_weighted', 'soe_pgdp_simple', 'baltics_pgdp_weighted', 'baltics_pgdp_simple']
        self.group_labels = {
            'eurozone_pgdp_weighted': 'Eurozone Weighted Avg',
            'eurozone_pgdp_simple': 'Eurozone Simple Avg', 
            'soe_pgdp_weighted': 'SOE Weighted Avg',
            'soe_pgdp_simple': 'SOE Simple Avg',
            'baltics_pgdp_weighted': 'Baltics Weighted Avg',
            'baltics_pgdp_simple': 'Baltics Simple Avg'
        }
    
    def load_indicator_data(self, indicator: str, include_crisis_years: bool = True) -> Optional[pd.DataFrame]:
        """
        Load data for a specific indicator
        
        Args:
            indicator: One of the indicator names from self.indicators
            include_crisis_years: If True, load full data; if False, load crisis-excluded data
            
        Returns:
            DataFrame with columns: YEAR, QUARTER, UNIT, Iceland, and comparator groups
        """
        try:
            if indicator not in self.indicators:
                raise ValueError(f"Unknown indicator: {indicator}. Available: {list(self.indicators.keys())}")
            
            # Select appropriate file
            file_type = 'full' if include_crisis_years else 'no_crises'
            filename = self.indicators[indicator][file_type]
            file_path = self.data_dir / filename
            
            if not file_path.exists():
                logger.error(f"Data file not found: {file_path}")
                return None
            
            # Load and validate data
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['YEAR', 'QUARTER', 'UNIT', 'iceland_pgdp'] + self.comparator_groups
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return None
            
            # Create date column for easier handling
            df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + 'Q' + df['QUARTER'].astype(str))
            
            logger.info(f"✅ Loaded {indicator} ({'full' if include_crisis_years else 'crisis-excluded'}): {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {indicator} data: {str(e)}")
            return None
    
    def load_all_indicators(self, include_crisis_years: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load data for all indicators
        
        Args:
            include_crisis_years: If True, load full data; if False, load crisis-excluded data
            
        Returns:
            Dictionary mapping indicator names to DataFrames
        """
        data = {}
        for indicator in self.indicators:
            df = self.load_indicator_data(indicator, include_crisis_years)
            if df is not None:
                data[indicator] = df
        
        logger.info(f"Loaded {len(data)}/{len(self.indicators)} indicators successfully")
        return data
    
    def validate_data_completeness(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate data completeness and quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'total_observations': len(df),
            'time_range': (df['YEAR'].min(), df['YEAR'].max()),
            'missing_data': {},
            'data_quality': 'Good'
        }
        
        # Check for missing data in each series
        data_cols = ['Iceland'] + self.comparator_groups
        for col in data_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct = (missing_count / len(df)) * 100
                results['missing_data'][col] = {
                    'count': missing_count,
                    'percentage': missing_pct
                }
                
                if missing_pct > 10:  # More than 10% missing
                    results['data_quality'] = 'Warning'
                    logger.warning(f"High missing data in {col}: {missing_pct:.1f}%")
        
        return results


class CS4StatisticalTests:
    """Statistical test implementations for CS4 analysis"""
    
    def __init__(self):
        self.significance_levels = {0.01: '***', 0.05: '**', 0.10: '*'}
    
    def f_test_variance_equality(self, series1: pd.Series, series2: pd.Series, 
                               group1_name: str = "Iceland", group2_name: str = "Comparator") -> Dict[str, any]:
        """
        Perform F-test for variance equality (homoscedasticity)
        
        Null Hypothesis: σ²(group1) = σ²(group2)
        Alternative: σ²(group1) ≠ σ²(group2)
        
        Args:
            series1: First time series (typically Iceland)
            series2: Second time series (comparator group)
            group1_name: Name of first group
            group2_name: Name of second group
            
        Returns:
            Dictionary with test results
        """
        try:
            # Remove NaN values
            s1 = series1.dropna()
            s2 = series2.dropna()
            
            if len(s1) < 2 or len(s2) < 2:
                return {
                    'f_statistic': np.nan,
                    'p_value': np.nan,
                    'significance': '',
                    'variance_1': np.nan,
                    'variance_2': np.nan,
                    'std_1': np.nan,
                    'std_2': np.nan,
                    'error': 'Insufficient data points'
                }
            
            # Calculate variances and standard deviations
            var1 = np.var(s1, ddof=1)  # Sample variance
            var2 = np.var(s2, ddof=1)
            std1 = np.std(s1, ddof=1)
            std2 = np.std(s2, ddof=1)
            
            # F-statistic (larger variance in numerator)
            if var1 >= var2:
                f_stat = var1 / var2
                df1, df2 = len(s1) - 1, len(s2) - 1
            else:
                f_stat = var2 / var1
                df1, df2 = len(s2) - 1, len(s1) - 1
            
            # Two-tailed p-value
            p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
            
            # Determine significance
            significance = ''
            for alpha, symbol in self.significance_levels.items():
                if p_value < alpha:
                    significance = symbol
                    break
            
            return {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significance': significance,
                'variance_1': var1,
                'variance_2': var2,
                'std_1': std1,
                'std_2': std2,
                'group_1': group1_name,
                'group_2': group2_name,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error in F-test: {str(e)}")
            return {
                'f_statistic': np.nan,
                'p_value': np.nan,
                'significance': '',
                'error': str(e)
            }


class CS4TimeSeriesAnalysis:
    """Time series analysis functions for CS4"""
    
    def __init__(self):
        pass
    
    def fit_ar4_model(self, series: pd.Series) -> Optional[Dict[str, any]]:
        """
        Fit AR(4) model to time series
        
        Model: y_t = φ₁y_{t-1} + φ₂y_{t-2} + φ₃y_{t-3} + φ₄y_{t-4} + ε_t
        
        Args:
            series: Time series data
            
        Returns:
            Dictionary with model results including coefficients
        """
        try:
            # Remove NaN values and ensure sufficient data
            clean_series = series.dropna()
            
            if len(clean_series) < 8:  # Need at least 8 observations for AR(4)
                logger.warning("Insufficient data for AR(4) model")
                return None
            
            # Fit AR(4) model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = AutoReg(clean_series, lags=4, trend='c')
                fitted_model = model.fit()
            
            # Extract coefficients (excluding constant term)
            ar_coeffs = fitted_model.params[1:5].values  # φ₁, φ₂, φ₃, φ₄
            
            return {
                'coefficients': ar_coeffs,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'fitted_model': fitted_model,
                'residuals': fitted_model.resid,
                'n_observations': len(clean_series)
            }
            
        except Exception as e:
            logger.error(f"Error fitting AR(4) model: {str(e)}")
            return None
    
    def calculate_half_life_impulse_response(self, ar_coefficients: np.ndarray) -> Union[int, str]:
        """
        Calculate half-life from AR(4) impulse response function
        
        Generate impulse response: track how unit shock propagates through time
        Half-life = number of quarters for shock to lose ≥50% of initial value
        
        Args:
            ar_coefficients: Array of AR coefficients [φ₁, φ₂, φ₃, φ₄]
            
        Returns:
            Half-life in quarters, or "N/A" if not found within 20 quarters
        """
        try:
            if len(ar_coefficients) != 4:
                logger.error("AR coefficients must be length 4")
                return "N/A"
            
            impulse_response = [1.0]  # Initial shock = 1
            
            # Generate impulse response up to 20 quarters
            for t in range(1, 21):
                response = 0
                for lag in range(min(4, t)):
                    if t-1-lag >= 0:
                        response += ar_coefficients[lag] * impulse_response[t-1-lag]
                impulse_response.append(response)
            
            # Find when response ≤ 0.5 (50% of initial shock)
            for quarter, response in enumerate(impulse_response):
                if abs(response) <= 0.5:
                    return quarter
            
            return "N/A"  # No half-life found within 20 quarters
            
        except Exception as e:
            logger.error(f"Error calculating half-life: {str(e)}")
            return "N/A"
    
    def calculate_rmse_prediction(self, series: pd.Series, forecast_periods: int = 4) -> Optional[float]:
        """
        Calculate RMSE using rolling prediction methodology
        
        Methodology:
        1. Training Data: Use all data except last 4 quarters
        2. Fit AR(4) on training data (one time only)
        3. Predict: Generate 4-step-ahead forecast for last 4 quarters
        4. RMSE: √(Σ(actual - predicted)² / 4)
        
        Args:
            series: Time series data
            forecast_periods: Number of periods to forecast (default: 4)
            
        Returns:
            RMSE value or None if calculation fails
        """
        try:
            clean_series = series.dropna()
            
            if len(clean_series) < 8:  # Need at least 8 observations
                logger.warning("Insufficient data for RMSE calculation")
                return None
            
            # Split data: training (all except last 4) and test (last 4)
            n_train = len(clean_series) - forecast_periods
            train_data = clean_series.iloc[:n_train]
            test_data = clean_series.iloc[n_train:]
            
            if len(train_data) < 8:
                logger.warning("Insufficient training data for AR(4) model")
                return None
            
            # Fit AR(4) model on training data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = AutoReg(train_data, lags=4, trend='c')
                fitted_model = model.fit()
            
            # Generate forecast for test period
            forecast = fitted_model.forecast(steps=len(test_data))
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((test_data.values - forecast.values)**2))
            
            return rmse
            
        except Exception as e:
            logger.error(f"Error calculating RMSE: {str(e)}")
            return None


class CS4AnalysisFramework:
    """Main analysis framework combining all CS4 statistical methodologies"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_loader = CS4DataLoader(data_dir)
        self.statistical_tests = CS4StatisticalTests()
        self.time_series_analysis = CS4TimeSeriesAnalysis()
    
    def run_comprehensive_analysis(self, include_crisis_years: bool = True) -> Dict[str, any]:
        """
        Run comprehensive CS4 statistical analysis
        
        Args:
            include_crisis_years: If True, analyze full data; if False, analyze crisis-excluded data
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info(f"🚀 Starting CS4 comprehensive analysis ({'full' if include_crisis_years else 'crisis-excluded'})")
        
        # Load all indicator data
        data = self.data_loader.load_all_indicators(include_crisis_years)
        
        if not data:
            logger.error("No data loaded - cannot proceed with analysis")
            return {}
        
        results = {
            'metadata': {
                'include_crisis_years': include_crisis_years,
                'analysis_type': 'full' if include_crisis_years else 'crisis-excluded',
                'indicators_analyzed': list(data.keys()),
                'comparator_groups': self.data_loader.comparator_groups
            },
            'f_tests': {},
            'ar4_analysis': {},
            'rmse_analysis': {},
            'summary_tables': {}
        }
        
        # Run analysis for each indicator
        for indicator_name, df in data.items():
            logger.info(f"📊 Analyzing {indicator_name}")
            
            # Initialize results for this indicator
            results['f_tests'][indicator_name] = {}
            results['ar4_analysis'][indicator_name] = {}
            results['rmse_analysis'][indicator_name] = {}
            
            # Get Iceland data
            iceland_series = df['iceland_pgdp']
            
            # Analyze each comparator group
            for group_col in self.data_loader.comparator_groups:
                group_name = self.data_loader.group_labels[group_col]
                comparator_series = df[group_col]
                
                # F-test for variance equality
                f_test_result = self.statistical_tests.f_test_variance_equality(
                    iceland_series, comparator_series, "Iceland", group_name
                )
                results['f_tests'][indicator_name][group_name] = f_test_result
                
                # AR(4) analysis for Iceland (done once per indicator)
                if group_col == self.data_loader.comparator_groups[0]:  # Only do once
                    ar4_iceland = self.time_series_analysis.fit_ar4_model(iceland_series)
                    if ar4_iceland:
                        half_life = self.time_series_analysis.calculate_half_life_impulse_response(
                            ar4_iceland['coefficients']
                        )
                        results['ar4_analysis'][indicator_name]['Iceland'] = {
                            'coefficients': ar4_iceland['coefficients'],
                            'half_life': half_life,
                            'aic': ar4_iceland['aic'],
                            'n_observations': ar4_iceland['n_observations']
                        }
                
                # AR(4) analysis for comparator group
                ar4_comp = self.time_series_analysis.fit_ar4_model(comparator_series)
                if ar4_comp:
                    half_life = self.time_series_analysis.calculate_half_life_impulse_response(
                        ar4_comp['coefficients']
                    )
                    results['ar4_analysis'][indicator_name][group_name] = {
                        'coefficients': ar4_comp['coefficients'],
                        'half_life': half_life,
                        'aic': ar4_comp['aic'],
                        'n_observations': ar4_comp['n_observations']
                    }
                
                # RMSE analysis for Iceland (done once per indicator)
                if group_col == self.data_loader.comparator_groups[0]:  # Only do once
                    rmse_iceland = self.time_series_analysis.calculate_rmse_prediction(iceland_series)
                    results['rmse_analysis'][indicator_name]['Iceland'] = rmse_iceland
                
                # RMSE analysis for comparator group
                rmse_comp = self.time_series_analysis.calculate_rmse_prediction(comparator_series)
                results['rmse_analysis'][indicator_name][group_name] = rmse_comp
        
        # Generate summary tables
        results['summary_tables'] = self._generate_summary_tables(results)
        
        logger.info("✅ CS4 comprehensive analysis completed")
        return results
    
    def _generate_summary_tables(self, results: Dict[str, any]) -> Dict[str, pd.DataFrame]:
        """Generate the three summary tables as specified"""
        
        indicators = results['metadata']['indicators_analyzed']
        groups = ['Iceland'] + list(self.data_loader.group_labels.values())
        
        # Table 1: Standard deviations + F-test significance stars
        std_data = []
        for indicator in indicators:
            row = {'Indicator': indicator}
            
            # Iceland standard deviation (from F-test results)
            for group_name in self.data_loader.group_labels.values():
                f_result = results['f_tests'][indicator].get(group_name, {})
                if 'std_1' in f_result:  # std_1 is Iceland
                    row['Iceland'] = f"{f_result['std_1']:.3f}"
                    break
            
            # Comparator group standard deviations with significance stars
            for group_name in self.data_loader.group_labels.values():
                f_result = results['f_tests'][indicator].get(group_name, {})
                if 'std_2' in f_result and 'significance' in f_result:
                    std_val = f_result['std_2']
                    significance = f_result['significance']
                    row[group_name] = f"{std_val:.3f}{significance}"
            
            std_data.append(row)
        
        table1 = pd.DataFrame(std_data)
        
        # Table 2: Half-life values from AR(4) impulse response analysis
        halflife_data = []
        for indicator in indicators:
            row = {'Indicator': indicator}
            
            for group in groups:
                ar4_result = results['ar4_analysis'][indicator].get(group, {})
                half_life = ar4_result.get('half_life', 'N/A')
                row[group] = half_life
            
            halflife_data.append(row)
        
        table2 = pd.DataFrame(halflife_data)
        
        # Table 3: RMSE values from prediction analysis
        rmse_data = []
        for indicator in indicators:
            row = {'Indicator': indicator}
            
            for group in groups:
                rmse_value = results['rmse_analysis'][indicator].get(group, None)
                if rmse_value is not None:
                    row[group] = f"{rmse_value:.4f}"
                else:
                    row[group] = 'N/A'
            
            rmse_data.append(row)
        
        table3 = pd.DataFrame(rmse_data)
        
        return {
            'standard_deviations_ftest': table1,
            'half_life_ar4': table2, 
            'rmse_prediction': table3
        }


# Convenience function for quick testing
def run_cs4_analysis(include_crisis_years: bool = True) -> Dict[str, any]:
    """
    Convenience function to run complete CS4 analysis
    
    Args:
        include_crisis_years: If True, analyze full data; if False, analyze crisis-excluded data
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    framework = CS4AnalysisFramework()
    return framework.run_comprehensive_analysis(include_crisis_years)


if __name__ == "__main__":
    # Test the framework
    print("🧪 Testing CS4 Statistical Analysis Framework")
    
    # Test data loading
    loader = CS4DataLoader()
    test_data = loader.load_indicator_data("Net Direct Investment", include_crisis_years=True)
    if test_data is not None:
        print(f"✅ Data loading successful: {test_data.shape}")
        validation = loader.validate_data_completeness(test_data)
        print(f"✅ Data validation: {validation['data_quality']}")
    
    # Test full analysis (commented out for now to prevent long execution)
    # results = run_cs4_analysis(include_crisis_years=True)
    # print(f"✅ Analysis completed with {len(results.get('f_tests', {}))} indicators")