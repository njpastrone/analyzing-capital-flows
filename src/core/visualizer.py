"""
Visualization classes for Capital Flows Research
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import logging

class BaseVisualizer:
    """Base class for creating visualizations"""
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid', 
                 color_palette: List[str] = None):
        self.style = style
        self.color_palette = color_palette or ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Set matplotlib style
        plt.style.use(self.style)
        
        # Set seaborn palette
        sns.set_palette(self.color_palette)
        
        self.logger = logging.getLogger(__name__)
    
    def save_plot(self, fig, filename: str, output_dir: str = "output", 
                  formats: List[str] = ['png']) -> List[str]:
        """Save plot in multiple formats"""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_files = []
        for fmt in formats:
            filepath = os.path.join(output_dir, f"{filename}.{fmt}")
            
            if hasattr(fig, 'write_image'):  # Plotly figure
                fig.write_image(filepath)
            else:  # Matplotlib figure
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
            
            saved_files.append(filepath)
        
        return saved_files


class StatisticalVisualizer(BaseVisualizer):
    """Specialized visualizer for statistical plots"""
    
    def create_boxplots_comparison(self, data: pd.DataFrame, indicators: List[str],
                                 group_col: str = 'GROUP', 
                                 title: str = "Statistical Comparison") -> plt.Figure:
        """Create side-by-side boxplots for means and standard deviations"""
        
        # Prepare data for boxplots
        stats_data = []
        
        for group in data[group_col].unique():
            group_data = data[data[group_col] == group]
            
            for indicator in indicators:
                values = group_data[indicator].dropna()
                if len(values) > 1:
                    mean_val = values.mean()
                    std_val = values.std()
                    
                    stats_data.extend([
                        {'GROUP': group, 'Statistic': 'Mean', 'Value': mean_val},
                        {'GROUP': group, 'Statistic': 'Standard Deviation', 'Value': std_val}
                    ])
        
        stats_df = pd.DataFrame(stats_data)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Boxplot for Means
        mean_data = stats_df[stats_df['Statistic'] == 'Mean']
        groups = mean_data['GROUP'].unique()
        mean_values = [mean_data[mean_data['GROUP'] == group]['Value'] for group in groups]
        
        bp1 = ax1.boxplot(mean_values, labels=groups, patch_artist=True)
        for i, box in enumerate(bp1['boxes']):
            box.set_facecolor(self.color_palette[i % len(self.color_palette)])
        
        ax1.set_title('Distribution of Means Across All Indicators', fontweight='bold')
        ax1.set_ylabel('Mean (% of GDP, annualized)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Boxplot for Standard Deviations
        std_data = stats_df[stats_df['Statistic'] == 'Standard Deviation']
        std_values = [std_data[std_data['GROUP'] == group]['Value'] for group in groups]
        
        bp2 = ax2.boxplot(std_values, labels=groups, patch_artist=True)
        for i, box in enumerate(bp2['boxes']):
            box.set_facecolor(self.color_palette[i % len(self.color_palette)])
        
        ax2.set_title('Distribution of Standard Deviations Across All Indicators', fontweight='bold')
        ax2.set_ylabel('Standard Deviation (% of GDP, annualized)')
        ax2.grid(True, alpha=0.3)
        
        # Add summary statistics
        mean_summary = {group: mean_data[mean_data['GROUP'] == group]['Value'].mean() 
                       for group in groups}
        std_summary = {group: std_data[std_data['GROUP'] == group]['Value'].mean() 
                      for group in groups}
        
        # Add text annotations
        ax1.text(0.02, 0.98, '\\n'.join([f'{k}: {v:.2f}%' for k, v in mean_summary.items()]),
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        volatility_ratio = list(std_summary.values())[1] / list(std_summary.values())[0] if len(std_summary) == 2 else 1
        ax2.text(0.02, 0.98, '\\n'.join([f'{k}: {v:.2f}%' for k, v in std_summary.items()]) + 
                f'\\nRatio: {volatility_ratio:.2f}x',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_time_series_plots(self, data: pd.DataFrame, indicators: List[str],
                               group_col: str = 'GROUP', date_col: str = 'Date',
                               top_n: int = 6, title: str = "Time Series Analysis") -> plt.Figure:
        """Create time series plots for top volatile indicators"""
        
        # Create date column if it doesn't exist
        if date_col not in data.columns:
            data = data.copy()
            data[date_col] = pd.to_datetime(
                data['YEAR'].astype(str) + '-' + 
                ((data['QUARTER'] - 1) * 3 + 1).astype(str) + '-01'
            )
        
        # Select top indicators (by F-statistic if available, otherwise by variance)
        selected_indicators = indicators[:top_n]
        
        # Create subplots
        n_rows = (top_n + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(20, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        groups = data[group_col].unique()
        
        for i, indicator in enumerate(selected_indicators):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            for j, group in enumerate(groups):
                group_data = data[data[group_col] == group].sort_values(date_col)
                
                if group == groups[0]:  # First group (e.g., Eurozone) - show average
                    if len(groups) > 1:
                        # Calculate group average for multi-country groups
                        avg_data = data[data[group_col] == group].groupby(date_col)[indicator].mean()
                        ax.plot(avg_data.index, avg_data.values, 
                               color=self.color_palette[j], linewidth=2.5, 
                               label=f'{group} Average', linestyle='--')
                    else:
                        ax.plot(group_data[date_col], group_data[indicator],
                               color=self.color_palette[j], linewidth=2.5, 
                               label=group, marker='o', markersize=2)
                else:  # Second group (e.g., Iceland) - show individual
                    ax.plot(group_data[date_col], group_data[indicator],
                           color=self.color_palette[j], linewidth=2.5, 
                           label=group, marker='o', markersize=2)
            
            # Formatting
            clean_name = indicator.replace('_PGDP', '')
            display_title = (clean_name[:45] + '...' if len(clean_name) > 45 else clean_name)
            ax.set_title(display_title, fontweight='bold', fontsize=10)
            ax.set_ylabel('% of GDP (annualized)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        
        # Hide unused subplots
        for i in range(len(selected_indicators), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        return fig


class InteractiveVisualizer(BaseVisualizer):
    """Plotly-based interactive visualizations"""
    
    def create_interactive_boxplots(self, data: pd.DataFrame, indicators: List[str],
                                  group_col: str = 'GROUP') -> go.Figure:
        """Create interactive boxplots using Plotly"""
        
        # Prepare data
        stats_data = []
        for group in data[group_col].unique():
            group_data = data[data[group_col] == group]
            
            for indicator in indicators:
                values = group_data[indicator].dropna()
                if len(values) > 1:
                    stats_data.extend([
                        {'GROUP': group, 'Statistic': 'Mean', 'Value': values.mean(),
                         'Indicator': indicator.replace('_PGDP', '')},
                        {'GROUP': group, 'Statistic': 'Standard Deviation', 'Value': values.std(),
                         'Indicator': indicator.replace('_PGDP', '')}
                    ])
        
        stats_df = pd.DataFrame(stats_data)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Distribution of Means', 'Distribution of Standard Deviations'],
            horizontal_spacing=0.1
        )
        
        groups = stats_df['GROUP'].unique()
        colors = self.color_palette[:len(groups)]
        
        # Mean boxplots
        mean_data = stats_df[stats_df['Statistic'] == 'Mean']
        for i, group in enumerate(groups):
            group_means = mean_data[mean_data['GROUP'] == group]['Value']
            fig.add_trace(
                go.Box(y=group_means, name=group, marker_color=colors[i],
                      legendgroup=group, showlegend=True),
                row=1, col=1
            )
        
        # Standard deviation boxplots
        std_data = stats_df[stats_df['Statistic'] == 'Standard Deviation']
        for i, group in enumerate(groups):
            group_stds = std_data[std_data['GROUP'] == group]['Value']
            fig.add_trace(
                go.Box(y=group_stds, name=group, marker_color=colors[i],
                      legendgroup=group, showlegend=False),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Interactive Statistical Comparison",
            height=500,
            template="plotly_white"
        )
        
        fig.update_yaxes(title_text="Mean (% of GDP)", row=1, col=1)
        fig.update_yaxes(title_text="Standard Deviation (% of GDP)", row=1, col=2)
        
        return fig
    
    def create_interactive_time_series(self, data: pd.DataFrame, indicators: List[str],
                                     group_col: str = 'GROUP', date_col: str = 'Date') -> go.Figure:
        """Create interactive time series plot"""
        
        # Create date column if needed
        if date_col not in data.columns:
            data = data.copy()
            data[date_col] = pd.to_datetime(
                data['YEAR'].astype(str) + '-' + 
                ((data['QUARTER'] - 1) * 3 + 1).astype(str) + '-01'
            )
        
        # Create figure
        fig = go.Figure()
        
        groups = data[group_col].unique()
        colors = self.color_palette[:len(groups)]
        
        # Add traces for each group and indicator
        for indicator in indicators[:3]:  # Limit to first 3 for readability
            for i, group in enumerate(groups):
                group_data = data[data[group_col] == group].sort_values(date_col)
                
                clean_name = indicator.replace('_PGDP', '')
                trace_name = f"{group} - {clean_name[:30]}..."
                
                fig.add_trace(
                    go.Scatter(
                        x=group_data[date_col],
                        y=group_data[indicator],
                        mode='lines+markers',
                        name=trace_name,
                        line=dict(color=colors[i], width=2),
                        marker=dict(size=4),
                        visible=True if indicator == indicators[0] else 'legendonly'
                    )
                )
        
        # Update layout
        fig.update_layout(
            title="Interactive Time Series Analysis",
            xaxis_title="Date",
            yaxis_title="% of GDP (annualized)",
            height=600,
            template="plotly_white",
            hovermode='x unified'
        )
        
        return fig