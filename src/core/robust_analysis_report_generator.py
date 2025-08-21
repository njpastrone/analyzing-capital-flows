"""
Robust Analysis Report Generator for Capital Flows Research

Generates lightweight PDF reports comparing original and winsorized analysis results
with focus on key statistical findings and sensitivity assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import io
import base64

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for professional reports
plt.style.use('default')
sns.set_palette("colorblind")

class RobustAnalysisReportGenerator:
    """Generates comprehensive robust analysis reports"""
    
    def __init__(self):
        self.output_dir = Path("output/robust_analysis_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_comparison_report(self, 
                                 original_results: pd.DataFrame,
                                 winsorized_results: pd.DataFrame,
                                 impact_analysis: pd.DataFrame,
                                 case_study: str = "CS1",
                                 include_crisis_years: bool = True) -> str:
        """
        Generate comprehensive comparison report
        
        Args:
            original_results: F-test results from original data
            winsorized_results: F-test results from winsorized data
            impact_analysis: Winsorization impact analysis
            case_study: Case study identifier
            include_crisis_years: Whether crisis years are included
            
        Returns:
            Path to generated report file
        """
        try:
            # Create report filename
            crisis_suffix = "_crisis_excluded" if not include_crisis_years else "_full"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"robust_analysis_{case_study.lower()}{crisis_suffix}_{timestamp}.html"
            report_path = self.output_dir / report_filename
            
            # Generate HTML report
            html_content = self._generate_html_report(
                original_results, winsorized_results, impact_analysis, 
                case_study, include_crisis_years
            )
            
            # Save report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Generated robust analysis report: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating comparison report: {str(e)}")
            raise
    
    def _generate_html_report(self,
                            original_results: pd.DataFrame,
                            winsorized_results: pd.DataFrame,
                            impact_analysis: pd.DataFrame,
                            case_study: str,
                            include_crisis_years: bool) -> str:
        """Generate HTML content for the robust analysis report"""
        
        # Calculate key metrics
        changed_conclusions = self._identify_changed_conclusions(original_results, winsorized_results)
        sensitivity_summary = self._generate_sensitivity_summary(changed_conclusions, impact_analysis)
        
        # Generate visualizations
        comparison_chart = self._create_comparison_chart(original_results, winsorized_results)
        impact_chart = self._create_impact_visualization(impact_analysis)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Robust Analysis Report - {case_study}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 300;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    font-size: 1.2em;
                    opacity: 0.9;
                }}
                .section {{
                    background: white;
                    padding: 25px;
                    margin-bottom: 25px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .section h2 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                    margin-top: 0;
                }}
                .section h3 {{
                    color: #34495e;
                    margin-top: 25px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    border-left: 4px solid #3498db;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #2c3e50;
                    margin: 0;
                }}
                .metric-label {{
                    color: #7f8c8d;
                    font-size: 0.9em;
                    margin: 5px 0 0 0;
                }}
                .table-container {{
                    overflow-x: auto;
                    margin: 20px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: white;
                    border-radius: 8px;
                    overflow: hidden;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background: #34495e;
                    color: white;
                    font-weight: 600;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .chart-container {{
                    text-align: center;
                    margin: 25px 0;
                }}
                .chart-container img {{
                    max-width: 100%;
                    border-radius: 8px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                }}
                .alert {{
                    padding: 15px;
                    border-radius: 5px;
                    margin: 15px 0;
                }}
                .alert-info {{
                    background-color: #d1ecf1;
                    border: 1px solid #bee5eb;
                    color: #0c5460;
                }}
                .alert-warning {{
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                    color: #856404;
                }}
                .alert-success {{
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    color: #155724;
                }}
                .methodology {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-left: 4px solid #17a2b8;
                    margin: 20px 0;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding: 20px;
                    color: #7f8c8d;
                    border-top: 1px solid #ecf0f1;
                }}
                .conclusion-box {{
                    background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
                    color: white;
                    padding: 25px;
                    border-radius: 10px;
                    margin: 25px 0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🛡️ Robust Analysis Report</h1>
                <p>{case_study} - Outlier Sensitivity Assessment</p>
                <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Executive Summary</h2>
                {sensitivity_summary}
            </div>
            
            <div class="section">
                <h2>📈 Statistical Comparison</h2>
                <h3>Key Performance Metrics</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <p class="metric-value">{len(original_results)}</p>
                        <p class="metric-label">Indicators Analyzed</p>
                    </div>
                    <div class="metric-card">
                        <p class="metric-value">{(original_results.get('P_Value', pd.Series([1])) < 0.05).sum()}</p>
                        <p class="metric-label">Original Significant Results</p>
                    </div>
                    <div class="metric-card">
                        <p class="metric-value">{(winsorized_results.get('P_Value', pd.Series([1])) < 0.05).sum()}</p>
                        <p class="metric-label">Winsorized Significant Results</p>
                    </div>
                    <div class="metric-card">
                        <p class="metric-value">{len(changed_conclusions)}</p>
                        <p class="metric-label">Conclusions Changed</p>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h3>F-Statistic Comparison</h3>
                    <img src="data:image/png;base64,{comparison_chart}" alt="F-Statistic Comparison Chart">
                </div>
                
                <h3>Detailed Results Comparison</h3>
                <div class="table-container">
                    {self._create_comparison_table(original_results, winsorized_results)}
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 Winsorization Impact Analysis</h2>
                <p>This section examines how 5% symmetric winsorization affected the statistical properties of each indicator.</p>
                
                <div class="chart-container">
                    <h3>Data Modification Impact</h3>
                    <img src="data:image/png;base64,{impact_chart}" alt="Winsorization Impact Chart">
                </div>
                
                <h3>Impact Summary Statistics</h3>
                <div class="table-container">
                    {self._create_impact_table(impact_analysis)}
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 Sensitivity Assessment</h2>
                {self._generate_sensitivity_assessment(changed_conclusions, impact_analysis)}
            </div>
            
            <div class="section">
                <h2>📚 Methodology</h2>
                <div class="methodology">
                    <h3>Winsorization Procedure</h3>
                    <p><strong>Method:</strong> Symmetric 5% winsorization</p>
                    <p><strong>Implementation:</strong> Values below the 5th percentile are replaced with the 5th percentile value; 
                    values above the 95th percentile are replaced with the 95th percentile value.</p>
                    <p><strong>Grouping:</strong> Applied indicator-by-indicator within country groups to preserve cross-sectional relationships.</p>
                    <p><strong>Temporal Structure:</strong> All time periods and crisis period definitions maintained.</p>
                    
                    <h3>Statistical Testing</h3>
                    <p><strong>F-Test:</strong> Tests for equality of variances between Iceland and Eurozone groups.</p>
                    <p><strong>Significance Levels:</strong> 1%, 5%, and 10% thresholds applied.</p>
                    <p><strong>Crisis Periods:</strong> {'Included' if include_crisis_years else 'Excluded'} - Global Financial Crisis (2008-2010) and COVID-19 (2020-2022).</p>
                </div>
            </div>
            
            <div class="conclusion-box">
                <h2>🎯 Overall Assessment</h2>
                {self._generate_overall_conclusion(changed_conclusions, len(original_results))}
            </div>
            
            <div class="footer">
                <p>Capital Flows Research Project - Robust Analysis Report</p>
                <p>Generated using 5% symmetric winsorization methodology</p>
                <p>Report ID: {timestamp}</p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _identify_changed_conclusions(self, 
                                    original_results: pd.DataFrame, 
                                    winsorized_results: pd.DataFrame) -> List[Dict]:
        """Identify indicators where statistical conclusions changed"""
        
        changed_conclusions = []
        
        if 'P_Value' not in original_results.columns or 'P_Value' not in winsorized_results.columns:
            return changed_conclusions
        
        for idx, orig_row in original_results.iterrows():
            indicator = orig_row.get('Indicator', f'Indicator_{idx}')
            wins_row = winsorized_results[winsorized_results['Indicator'] == indicator]
            
            if not wins_row.empty:
                orig_p = orig_row.get('P_Value', 1.0)
                wins_p = wins_row.iloc[0].get('P_Value', 1.0)
                
                orig_sig = orig_p < 0.05
                wins_sig = wins_p < 0.05
                
                if orig_sig != wins_sig:
                    changed_conclusions.append({
                        'indicator': indicator,
                        'original_p_value': orig_p,
                        'winsorized_p_value': wins_p,
                        'original_significant': orig_sig,
                        'winsorized_significant': wins_sig,
                        'change_direction': 'Lost Significance' if orig_sig else 'Gained Significance'
                    })
        
        return changed_conclusions
    
    def _generate_sensitivity_summary(self, 
                                    changed_conclusions: List[Dict], 
                                    impact_analysis: pd.DataFrame) -> str:
        """Generate executive summary of sensitivity analysis"""
        
        total_indicators = len(impact_analysis) if not impact_analysis.empty else 0
        changed_count = len(changed_conclusions)
        
        if changed_count == 0:
            return """
            <div class="alert alert-success">
                <h3>✅ Robust Statistical Results</h3>
                <p>All statistical conclusions remain unchanged after outlier adjustment through winsorization. 
                This indicates that the findings are <strong>not sensitive to extreme values</strong> and can be 
                considered statistically robust.</p>
            </div>
            """
        elif changed_count <= 0.1 * total_indicators:
            return f"""
            <div class="alert alert-info">
                <h3>📊 Mostly Robust Results</h3>
                <p><strong>{changed_count} out of {total_indicators}</strong> indicators ({changed_count/total_indicators*100:.1f}%) 
                changed statistical significance after winsorization. This suggests that the majority of findings are robust to outliers, 
                with only minor sensitivity detected in a small subset of indicators.</p>
            </div>
            """
        else:
            return f"""
            <div class="alert alert-warning">
                <h3>⚠️ Significant Outlier Sensitivity</h3>
                <p><strong>{changed_count} out of {total_indicators}</strong> indicators ({changed_count/total_indicators*100:.1f}%) 
                changed statistical significance after winsorization. This indicates substantial sensitivity to extreme values 
                and suggests that outlier treatment should be carefully considered in the final analysis.</p>
            </div>
            """
    
    def _create_comparison_chart(self, 
                               original_results: pd.DataFrame, 
                               winsorized_results: pd.DataFrame) -> str:
        """Create F-statistic comparison chart"""
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Extract F-statistics
            orig_f_stats = []
            wins_f_stats = []
            indicators = []
            
            for idx, orig_row in original_results.iterrows():
                indicator = orig_row.get('Indicator', f'Indicator_{idx}')
                wins_row = winsorized_results[winsorized_results['Indicator'] == indicator]
                
                if not wins_row.empty:
                    orig_f = orig_row.get('F_Statistic', 0)
                    wins_f = wins_row.iloc[0].get('F_Statistic', 0)
                    
                    if pd.notna(orig_f) and pd.notna(wins_f):
                        orig_f_stats.append(orig_f)
                        wins_f_stats.append(wins_f)
                        # Shorten indicator names for display
                        short_name = indicator.replace('_PGDP', '').replace('_', ' ')[:20]
                        indicators.append(short_name)
            
            if len(orig_f_stats) > 0:
                x = np.arange(len(indicators))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, orig_f_stats, width, label='Original Data', alpha=0.8)
                bars2 = ax.bar(x + width/2, wins_f_stats, width, label='Winsorized Data', alpha=0.8)
                
                ax.set_xlabel('Indicators', fontsize=12)
                ax.set_ylabel('F-Statistic', fontsize=12)
                ax.set_title('F-Statistic Comparison: Original vs Winsorized Data', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(indicators, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Convert to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                return image_base64
            else:
                plt.close()
                return ""
        
        except Exception as e:
            logger.warning(f"Could not create comparison chart: {str(e)}")
            plt.close()
            return ""
    
    def _create_impact_visualization(self, impact_analysis: pd.DataFrame) -> str:
        """Create winsorization impact visualization"""
        
        try:
            if impact_analysis.empty or 'Pct_Values_Changed' not in impact_analysis.columns:
                return ""
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Chart 1: Percentage of data affected
            indicators = [ind.replace('_PGDP', '').replace('_', ' ')[:20] for ind in impact_analysis['Indicator']]
            pct_affected = impact_analysis['Pct_Values_Changed']
            
            colors = ['#ff7f7f' if x > 15 else '#ffff7f' if x > 5 else '#7fff7f' for x in pct_affected]
            
            bars = ax1.barh(indicators, pct_affected, color=colors, alpha=0.8)
            ax1.set_xlabel('Percentage of Data Modified (%)')
            ax1.set_title('Winsorization Impact by Indicator', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, pct_affected):
                ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{value:.1f}%', va='center', fontsize=9)
            
            # Chart 2: Standard deviation changes
            if 'Std_Change_Pct' in impact_analysis.columns:
                std_changes = impact_analysis['Std_Change_Pct']
                
                scatter = ax2.scatter(pct_affected, std_changes, alpha=0.7, s=100, 
                                    c=['red' if x > 0 else 'blue' for x in std_changes])
                ax2.set_xlabel('Percentage of Data Modified (%)')
                ax2.set_ylabel('Standard Deviation Change (%)')
                ax2.set_title('Impact on Volatility Measures', fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                
                # Add trend line
                if len(pct_affected) > 1:
                    z = np.polyfit(pct_affected, std_changes, 1)
                    p = np.poly1d(z)
                    ax2.plot(pct_affected, p(pct_affected), "r--", alpha=0.8)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.warning(f"Could not create impact visualization: {str(e)}")
            plt.close()
            return ""
    
    def _create_comparison_table(self, 
                               original_results: pd.DataFrame, 
                               winsorized_results: pd.DataFrame) -> str:
        """Create HTML table comparing results"""
        
        try:
            comparison_data = []
            
            for idx, orig_row in original_results.iterrows():
                indicator = orig_row.get('Indicator', f'Indicator_{idx}')
                wins_row = winsorized_results[winsorized_results['Indicator'] == indicator]
                
                if not wins_row.empty:
                    wins_stats = wins_row.iloc[0]
                    
                    orig_p = orig_row.get('P_Value', np.nan)
                    wins_p = wins_stats.get('P_Value', np.nan)
                    
                    comparison_data.append({
                        'Indicator': indicator.replace('_PGDP', '').replace('_', ' '),
                        'Original F-Stat': f"{orig_row.get('F_Statistic', np.nan):.3f}",
                        'Winsorized F-Stat': f"{wins_stats.get('F_Statistic', np.nan):.3f}",
                        'Original P-Value': f"{orig_p:.4f}" if pd.notna(orig_p) else "N/A",
                        'Winsorized P-Value': f"{wins_p:.4f}" if pd.notna(wins_p) else "N/A",
                        'Original Sig': "Yes" if pd.notna(orig_p) and orig_p < 0.05 else "No",
                        'Winsorized Sig': "Yes" if pd.notna(wins_p) and wins_p < 0.05 else "No",
                        'Changed': "⚠️ Yes" if (pd.notna(orig_p) and pd.notna(wins_p) and 
                                               (orig_p < 0.05) != (wins_p < 0.05)) else "✅ No"
                    })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                return df_comparison.to_html(index=False, escape=False, classes='table')
            else:
                return "<p>No comparison data available</p>"
                
        except Exception as e:
            logger.warning(f"Could not create comparison table: {str(e)}")
            return "<p>Error generating comparison table</p>"
    
    def _create_impact_table(self, impact_analysis: pd.DataFrame) -> str:
        """Create HTML table for impact analysis"""
        
        try:
            if impact_analysis.empty:
                return "<p>No impact analysis data available</p>"
            
            # Select key columns and format
            display_cols = ['Indicator', 'Pct_Values_Changed', 'Mean_Change_Pct', 'Std_Change_Pct']
            available_cols = [col for col in display_cols if col in impact_analysis.columns]
            
            if available_cols:
                display_df = impact_analysis[available_cols].copy()
                
                # Format display names
                if 'Indicator' in display_df.columns:
                    display_df['Indicator'] = display_df['Indicator'].str.replace('_PGDP', '').str.replace('_', ' ')
                
                # Format numeric columns
                numeric_cols = display_df.select_dtypes(include=[np.number]).columns
                display_df[numeric_cols] = display_df[numeric_cols].round(3)
                
                # Rename columns for display
                display_df.columns = ['Indicator', 'Data Modified (%)', 'Mean Change (%)', 'Std Dev Change (%)'][:len(display_df.columns)]
                
                return display_df.to_html(index=False, classes='table')
            else:
                return "<p>No impact analysis columns available</p>"
                
        except Exception as e:
            logger.warning(f"Could not create impact table: {str(e)}")
            return "<p>Error generating impact table</p>"
    
    def _generate_sensitivity_assessment(self, 
                                       changed_conclusions: List[Dict], 
                                       impact_analysis: pd.DataFrame) -> str:
        """Generate detailed sensitivity assessment"""
        
        if len(changed_conclusions) == 0:
            return """
            <div class="alert alert-success">
                <h3>✅ No Sensitivity Detected</h3>
                <p>All indicators maintained their statistical significance status after winsorization. 
                This provides strong evidence for the robustness of the statistical conclusions.</p>
                
                <h4>Implications:</h4>
                <ul>
                    <li>Results are not driven by extreme values or outliers</li>
                    <li>Statistical conclusions can be considered reliable</li>
                    <li>Policy implications based on these findings are well-supported</li>
                </ul>
            </div>
            """
        
        # Create detailed assessment for changed indicators
        assessment_html = """
        <div class="alert alert-warning">
            <h3>⚠️ Sensitivity Detected</h3>
            <p>The following indicators changed statistical significance after outlier adjustment:</p>
        </div>
        
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Indicator</th>
                        <th>Change Direction</th>
                        <th>Original P-Value</th>
                        <th>Winsorized P-Value</th>
                        <th>Assessment</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for change in changed_conclusions:
            assessment = "Outliers were driving significance" if change['original_significant'] else "Outliers were masking significance"
            
            assessment_html += f"""
                    <tr>
                        <td>{change['indicator'].replace('_PGDP', '').replace('_', ' ')}</td>
                        <td>{change['change_direction']}</td>
                        <td>{change['original_p_value']:.4f}</td>
                        <td>{change['winsorized_p_value']:.4f}</td>
                        <td>{assessment}</td>
                    </tr>
            """
        
        assessment_html += """
                </tbody>
            </table>
        </div>
        
        <div class="alert alert-info">
            <h4>📋 Recommendations:</h4>
            <ul>
                <li>Carefully examine the indicators that changed significance</li>
                <li>Consider reporting both original and winsorized results</li>
                <li>Investigate potential sources of extreme values</li>
                <li>Use winsorized results for more conservative statistical conclusions</li>
            </ul>
        </div>
        """
        
        return assessment_html
    
    def _generate_overall_conclusion(self, changed_conclusions: List[Dict], total_indicators: int) -> str:
        """Generate overall conclusion for the report"""
        
        changed_count = len(changed_conclusions)
        
        if changed_count == 0:
            return """
            <h3>🎯 Robust Statistical Findings</h3>
            <p>This analysis demonstrates that all statistical conclusions are <strong>robust to outlier effects</strong>. 
            The consistency between original and winsorized results provides strong confidence in the reliability 
            of the research findings.</p>
            <p><strong>Recommendation:</strong> Proceed with confidence using the original analysis results, 
            as they have been validated against outlier sensitivity.</p>
            """
        elif changed_count <= 0.1 * total_indicators:
            return f"""
            <h3>📊 Generally Robust with Minor Sensitivity</h3>
            <p>The analysis shows that <strong>{100-changed_count/total_indicators*100:.1f}% of results are robust</strong> 
            to outlier effects, with only {changed_count} out of {total_indicators} indicators showing sensitivity.</p>
            <p><strong>Recommendation:</strong> Use original results as primary findings, but acknowledge the outlier 
            sensitivity in the affected indicators and consider presenting both sets of results for transparency.</p>
            """
        else:
            return f"""
            <h3>⚠️ Significant Outlier Sensitivity</h3>
            <p>The analysis reveals substantial sensitivity to outliers, with <strong>{changed_count} out of {total_indicators} 
            indicators</strong> ({changed_count/total_indicators*100:.1f}%) changing significance after winsorization.</p>
            <p><strong>Recommendation:</strong> Carefully review the data for extreme values and consider using 
            winsorized results as the primary findings to ensure robust statistical conclusions.</p>
            """


def generate_lightweight_pdf_report(case_study: str = "CS1", 
                                  include_crisis_years: bool = True) -> Optional[str]:
    """
    Generate a lightweight PDF report for a specific case study
    
    Args:
        case_study: Case study identifier 
        include_crisis_years: Whether to include crisis periods
        
    Returns:
        Path to generated report or None if error
    """
    try:
        # Import required modules
        sys.path.append(str(Path(__file__).parent.parent / "dashboard"))
        from simple_report_app import perform_volatility_tests
        from winsorized_data_loader import (
            load_original_vs_winsorized_comparison,
            calculate_winsorization_impact
        )
        
        # Load data and perform analysis
        df_original, df_winsorized = load_original_vs_winsorized_comparison(case_study, include_crisis_years)
        
        # Get indicators
        metadata_cols = ['COUNTRY', 'INDICATOR', 'UNIT', 'YEAR', 'QUARTER', 'TIME_PERIOD',
                        'CS1_GROUP', 'CS2_GROUP', 'CS3_GROUP', 'CS4_GROUP', 'CS5_GROUP']
        indicators = [col for col in df_original.columns if col not in metadata_cols]
        
        # Perform statistical tests
        original_results = perform_volatility_tests(df_original, indicators)
        winsorized_results = perform_volatility_tests(df_winsorized, indicators)
        
        # Calculate impact analysis
        impact_analysis = calculate_winsorization_impact(df_original, df_winsorized, indicators)
        
        # Generate report
        generator = RobustAnalysisReportGenerator()
        report_path = generator.generate_comparison_report(
            original_results, winsorized_results, impact_analysis, 
            case_study, include_crisis_years
        )
        
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating lightweight PDF report: {str(e)}")
        return None


if __name__ == "__main__":
    # Test the report generator
    print("Testing Robust Analysis Report Generator...")
    
    try:
        report_path = generate_lightweight_pdf_report("CS1", True)
        if report_path:
            print(f"✓ Generated test report: {report_path}")
        else:
            print("✗ Report generation failed")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("Report generator test completed!")