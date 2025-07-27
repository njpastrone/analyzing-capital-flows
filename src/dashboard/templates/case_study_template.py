"""
Base template class for case studies
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

# Import core classes
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.data_processor import BOPDataProcessor
from core.statistical_tests import VolatilityTester, StatisticalAnalyzer
from core.visualizer import StatisticalVisualizer, InteractiveVisualizer


class CaseStudyTemplate(ABC):
    """Abstract base class for case study templates"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.data_processor = None
        self.statistical_analyzer = None
        self.visualizer = None
        self.results = {}
        self.metadata = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def load_data(self, **kwargs) -> bool:
        """Load and validate required data"""
        pass
    
    @abstractmethod
    def process_data(self, **kwargs) -> bool:
        """Process raw data into analysis-ready format"""
        pass
    
    @abstractmethod
    def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """Run the main analysis"""
        pass
    
    @abstractmethod
    def generate_visualizations(self, **kwargs) -> Dict[str, Any]:
        """Generate all required visualizations"""
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of analysis results"""
        return {
            'name': self.name,
            'description': self.description,
            'metadata': self.metadata,
            'results_keys': list(self.results.keys()) if self.results else []
        }
    
    def export_results(self, output_dir: str, formats: List[str] = ['csv']) -> List[str]:
        """Export analysis results in specified formats"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        exported_files = []
        
        for result_key, result_data in self.results.items():
            if isinstance(result_data, pd.DataFrame):
                for fmt in formats:
                    filename = f"{self.name}_{result_key}.{fmt}"
                    filepath = output_path / filename
                    
                    if fmt == 'csv':
                        result_data.to_csv(filepath, index=False)
                    elif fmt == 'excel':
                        result_data.to_excel(filepath, index=False)
                    
                    exported_files.append(str(filepath))
        
        return exported_files


class VolatilityAnalysisTemplate(CaseStudyTemplate):
    """Template for volatility analysis between groups"""
    
    def __init__(self, name: str = "Volatility Analysis", 
                 description: str = "Capital flow volatility comparison between groups"):
        super().__init__(name, description)
        
        self.data_processor = BOPDataProcessor()
        self.volatility_tester = VolatilityTester()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = StatisticalVisualizer()
        self.interactive_visualizer = InteractiveVisualizer()
        
        # Analysis parameters
        self.group_definitions = {}
        self.analysis_indicators = []
        self.processed_data = None
    
    def load_data(self, bop_file: str, gdp_file: str, **kwargs) -> bool:
        """Load BOP and GDP data files"""
        try:
            self.logger.info(f"Loading data for {self.name}")
            
            # Load BOP data
            bop_data = self.data_processor.load_data(bop_file, 'bop_raw')
            
            # Load GDP data
            gdp_data = self.data_processor.load_data(gdp_file, 'gdp_raw')
            
            self.metadata['data_loaded'] = True
            self.metadata['bop_shape'] = bop_data.shape
            self.metadata['gdp_shape'] = gdp_data.shape
            
            self.logger.info("Data loading completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            self.metadata['data_loaded'] = False
            return False
    
    def process_data(self, group_definitions: Dict[str, List[str]], 
                    remove_luxembourg: bool = True, **kwargs) -> bool:
        """Process data for volatility analysis"""
        try:
            self.logger.info("Processing data for analysis")
            
            # Store group definitions
            self.group_definitions = group_definitions
            
            # Process BOP data
            bop_pivot = self.data_processor.process_bop_data(
                self.data_processor.raw_data['bop_raw']
            )
            
            # Process GDP data
            gdp_pivot = self.data_processor.process_gdp_data(
                self.data_processor.raw_data['gdp_raw']
            )
            
            # Join and normalize data
            normalized_data = self.data_processor.join_bop_gdp(
                remove_luxembourg=remove_luxembourg
            )
            
            # Create groups
            final_data = self.data_processor.create_groups(group_definitions)
            
            # Get analysis-ready data
            self.processed_data, self.analysis_indicators = \
                self.data_processor.get_analysis_ready_data()
            
            self.metadata['processing_completed'] = True
            self.metadata['final_shape'] = self.processed_data.shape
            self.metadata['n_indicators'] = len(self.analysis_indicators)
            self.metadata['groups'] = list(group_definitions.keys())
            
            self.logger.info("Data processing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            self.metadata['processing_completed'] = False
            return False
    
    def run_analysis(self, group1: str = "Iceland", group2: str = "Eurozone", 
                    significance_level: float = 0.05, **kwargs) -> Dict[str, Any]:
        """Run volatility analysis"""
        try:
            self.logger.info("Running volatility analysis")
            
            if self.processed_data is None:
                raise ValueError("Data must be processed before running analysis")
            
            # Update significance level
            self.volatility_tester.significance_level = significance_level
            
            # Calculate descriptive statistics for all groups
            group_stats = self.statistical_analyzer.calculate_group_statistics(
                self.processed_data, 'GROUP', self.analysis_indicators
            )
            
            # Perform volatility tests
            volatility_results = self.volatility_tester.perform_volatility_analysis(
                self.processed_data, self.analysis_indicators, 'GROUP', group1, group2
            )
            
            # Store results
            self.results['group_statistics'] = group_stats
            self.results['volatility_tests'] = volatility_results
            
            # Calculate summary statistics
            if len(volatility_results) > 0:
                total_indicators = len(volatility_results)
                higher_volatility = volatility_results[f'{group1}_Higher_Volatility'].sum()
                significant_5pct = volatility_results['Significant_5pct'].sum()
                significant_1pct = volatility_results['Significant_1pct'].sum()
                
                summary_stats = {
                    'total_indicators': total_indicators,
                    'higher_volatility_count': higher_volatility,
                    'higher_volatility_pct': (higher_volatility / total_indicators) * 100,
                    'significant_5pct_count': significant_5pct,
                    'significant_5pct_pct': (significant_5pct / total_indicators) * 100,
                    'significant_1pct_count': significant_1pct,
                    'significant_1pct_pct': (significant_1pct / total_indicators) * 100,
                    'group1': group1,
                    'group2': group2
                }
                
                self.results['summary_statistics'] = summary_stats
                self.metadata['analysis_completed'] = True
                
                self.logger.info(f"Analysis completed: {higher_volatility}/{total_indicators} "
                               f"indicators show {group1} higher volatility")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error running analysis: {str(e)}")
            self.metadata['analysis_completed'] = False
            return {}
    
    def generate_visualizations(self, output_dir: str = "output", 
                              save_plots: bool = True, **kwargs) -> Dict[str, Any]:
        """Generate all visualizations"""
        try:
            self.logger.info("Generating visualizations")
            
            if self.processed_data is None:
                raise ValueError("Data must be processed before generating visualizations")
            
            visualizations = {}
            
            # 1. Statistical comparison boxplots
            boxplot_fig = self.visualizer.create_boxplots_comparison(
                self.processed_data, self.analysis_indicators,
                title=f"{self.name} - Statistical Comparison"
            )
            visualizations['boxplots'] = boxplot_fig
            
            # 2. Time series plots
            time_series_fig = self.visualizer.create_time_series_plots(
                self.processed_data, self.analysis_indicators,
                title=f"{self.name} - Time Series Analysis"
            )
            visualizations['time_series'] = time_series_fig
            
            # 3. Interactive boxplots
            interactive_boxplots = self.interactive_visualizer.create_interactive_boxplots(
                self.processed_data, self.analysis_indicators
            )
            visualizations['interactive_boxplots'] = interactive_boxplots
            
            # 4. Interactive time series
            interactive_time_series = self.interactive_visualizer.create_interactive_time_series(
                self.processed_data, self.analysis_indicators
            )
            visualizations['interactive_time_series'] = interactive_time_series
            
            # Save plots if requested
            if save_plots:
                saved_files = []
                for viz_name, fig in visualizations.items():
                    if hasattr(fig, 'savefig'):  # Matplotlib figure
                        files = self.visualizer.save_plot(fig, f"{self.name}_{viz_name}", output_dir)
                        saved_files.extend(files)
                
                self.metadata['saved_visualizations'] = saved_files
            
            self.results['visualizations'] = visualizations
            self.metadata['visualizations_completed'] = True
            
            self.logger.info("Visualizations generated successfully")
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            self.metadata['visualizations_completed'] = False
            return {}
    
    def get_analysis_summary(self) -> str:
        """Get formatted analysis summary"""
        if 'summary_statistics' not in self.results:
            return "Analysis not completed yet."
        
        stats = self.results['summary_statistics']
        
        summary = f"""
        ## {self.name} - Analysis Summary
        
        **Hypothesis**: {stats['group1']} shows higher capital flow volatility than {stats['group2']}
        
        ### Key Findings:
        - **Total Indicators Analyzed**: {stats['total_indicators']}
        - **{stats['group1']} Higher Volatility**: {stats['higher_volatility_count']}/{stats['total_indicators']} ({stats['higher_volatility_pct']:.1f}%)
        - **Statistically Significant (5%)**: {stats['significant_5pct_count']}/{stats['total_indicators']} ({stats['significant_5pct_pct']:.1f}%)
        - **Statistically Significant (1%)**: {stats['significant_1pct_count']}/{stats['total_indicators']} ({stats['significant_1pct_pct']:.1f}%)
        
        ### Conclusion:
        {'Strong evidence supports' if stats['higher_volatility_pct'] > 60 else 'Mixed evidence for'} the hypothesis that {stats['group1']} has higher capital flow volatility.
        """
        
        return summary