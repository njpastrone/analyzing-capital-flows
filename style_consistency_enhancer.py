#!/usr/bin/env python3
"""
Style Consistency Enhancement for Capital Flows Research Platform

Analyzes and improves visual consistency across dashboard applications
to achieve professional presentation standards.
"""

import re
from pathlib import Path
import ast

class StyleConsistencyEnhancer:
    """Enhances style consistency across dashboard applications"""
    
    def __init__(self):
        self.style_issues = []
        self.improvements = []
        
    def analyze_chart_styling(self):
        """Analyze chart styling patterns across dashboard files"""
        print("🎨 Analyzing Chart Styling Consistency...")
        
        dashboard_files = [
            "src/dashboard/main_app.py",
            "src/dashboard/simple_report_app.py", 
            "src/dashboard/case_study_2_euro_adoption.py"
        ]
        
        color_schemes = {}
        layout_patterns = {}
        
        for file_path in dashboard_files:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Extract color schemes
                color_matches = re.findall(r'color[s]?\s*=\s*[\'"]([^\'\"]+)[\'"]', content)
                color_list_matches = re.findall(r'color[s]?\s*=\s*\[(.*?)\]', content, re.DOTALL)
                
                # Extract plotly layout patterns
                layout_matches = re.findall(r'fig\.update_layout\((.*?)\)', content, re.DOTALL)
                
                file_name = Path(file_path).stem
                color_schemes[file_name] = color_matches + color_list_matches
                layout_patterns[file_name] = layout_matches
        
        # Analyze consistency
        print("\n📊 Chart Styling Analysis:")
        
        # Color scheme analysis
        all_colors = []
        for file_name, colors in color_schemes.items():
            print(f"  📁 {file_name}: {len(colors)} color definitions")
            all_colors.extend(colors)
        
        unique_colors = set(all_colors)
        if len(unique_colors) <= 8:
            print(f"  ✅ Good color consistency: {len(unique_colors)} unique color patterns")
        else:
            print(f"  ⚠️  High color variation: {len(unique_colors)} unique patterns")
            self.style_issues.append("Inconsistent color schemes across files")
        
        # Layout consistency analysis
        layout_keywords = ['title', 'xaxis', 'yaxis', 'font', 'margin']
        layout_consistency = {}
        
        for keyword in layout_keywords:
            keyword_count = 0
            for file_layouts in layout_patterns.values():
                for layout in file_layouts:
                    if keyword in layout.lower():
                        keyword_count += 1
            layout_consistency[keyword] = keyword_count
        
        print(f"\n📏 Layout Element Usage:")
        for element, count in layout_consistency.items():
            print(f"  📐 {element}: used {count} times")
        
        if layout_consistency.get('title', 0) >= 5 and layout_consistency.get('font', 0) >= 3:
            print("  ✅ Consistent layout patterns detected")
        else:
            print("  ⚠️  Inconsistent layout patterns")
            self.style_issues.append("Inconsistent chart layout patterns")
        
        return color_schemes, layout_patterns
    
    def analyze_table_formatting(self):
        """Analyze table formatting consistency"""
        print("\n📋 Analyzing Table Formatting Consistency...")
        
        dashboard_files = [
            "src/dashboard/main_app.py",
            "src/dashboard/simple_report_app.py", 
            "src/dashboard/case_study_2_euro_adoption.py"
        ]
        
        table_methods = {}
        
        for file_path in dashboard_files:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Find table display methods
                dataframe_calls = re.findall(r'st\.dataframe\([^)]*\)', content)
                table_calls = re.findall(r'st\.table\([^)]*\)', content)
                metric_calls = re.findall(r'st\.metric\([^)]*\)', content)
                
                file_name = Path(file_path).stem
                table_methods[file_name] = {
                    'dataframe': len(dataframe_calls),
                    'table': len(table_calls),
                    'metric': len(metric_calls)
                }
        
        print("\n📊 Table Method Usage:")
        total_dataframe = sum(methods['dataframe'] for methods in table_methods.values())
        total_table = sum(methods['table'] for methods in table_methods.values())
        total_metric = sum(methods['metric'] for methods in table_methods.values())
        
        print(f"  📊 st.dataframe: {total_dataframe} uses")
        print(f"  📋 st.table: {total_table} uses") 
        print(f"  📈 st.metric: {total_metric} uses")
        
        # Check for consistency preference
        if total_dataframe > total_table * 2:
            print("  ✅ Consistent preference for st.dataframe")
        elif total_table > total_dataframe * 2:
            print("  ✅ Consistent preference for st.table")
        else:
            print("  ⚠️  Mixed table display methods")
            self.style_issues.append("Inconsistent table display methods")
        
        return table_methods
    
    def analyze_statistical_notation(self):
        """Analyze statistical notation consistency"""
        print("\n🔢 Analyzing Statistical Notation Consistency...")
        
        dashboard_files = list(Path("src/dashboard").glob("*.py"))
        
        notation_patterns = {
            'significance_stars': 0,
            'p_value_format': 0,
            'f_statistic_format': 0,
            'percentage_format': 0
        }
        
        for file_path in dashboard_files:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for significance notation
            if re.search(r'\*{1,3}', content):
                notation_patterns['significance_stars'] += 1
            
            # Check for p-value formatting
            if re.search(r'p[-_\s]*value|p\s*=|p\s*<', content, re.IGNORECASE):
                notation_patterns['p_value_format'] += 1
            
            # Check for F-statistic formatting
            if re.search(r'f[-_\s]*stat|f\s*=', content, re.IGNORECASE):
                notation_patterns['f_statistic_format'] += 1
            
            # Check for percentage formatting
            if re.search(r'\.2f\%|\.1f\%|{.*?:.+?%}', content):
                notation_patterns['percentage_format'] += 1
        
        print("\n📊 Statistical Notation Usage:")
        total_files = len(dashboard_files)
        for pattern, count in notation_patterns.items():
            coverage = (count / total_files) * 100
            print(f"  📐 {pattern}: {count}/{total_files} files ({coverage:.1f}%)")
        
        avg_coverage = sum(notation_patterns.values()) / len(notation_patterns) / total_files * 100
        
        if avg_coverage >= 70:
            print(f"  ✅ Good statistical notation consistency: {avg_coverage:.1f}%")
        else:
            print(f"  ⚠️  Inconsistent statistical notation: {avg_coverage:.1f}%")
            self.style_issues.append("Inconsistent statistical notation")
        
        return notation_patterns
    
    def generate_style_guide(self):
        """Generate style guide recommendations"""
        print("\n📐 Generating Style Guide Recommendations...")
        
        style_guide = """
# Capital Flows Research - Style Guide

## Chart Styling Standards

### Color Palette
```python
# Primary color scheme for all charts
CAPITAL_FLOWS_COLORS = {
    'primary': '#1f77b4',      # Blue - main data
    'secondary': '#ff7f0e',    # Orange - comparison data  
    'accent': '#2ca02c',       # Green - positive values
    'warning': '#d62728',      # Red - negative/crisis values
    'neutral': '#9467bd',      # Purple - neutral data
    'background': '#f8f9fa'    # Light gray background
}
```

### Chart Layout Template
```python
def apply_standard_layout(fig, title="", subtitle=""):
    '''Apply consistent layout across all charts'''
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:12px'>{subtitle}</span>",
            x=0.5,
            font=dict(size=16, family="Arial, sans-serif")
        ),
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=80, b=60),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right", 
            x=1
        )
    )
    return fig
```

## Table Formatting Standards

### Preferred Method
```python
# Use st.dataframe for interactive tables
st.dataframe(
    data,
    use_container_width=True,
    hide_index=True
)

# Use st.table only for small summary tables
st.table(summary_data)
```

### Statistical Table Formatting
```python
def format_statistical_table(df):
    '''Standard formatting for statistical results'''
    # Format p-values
    df['P-Value'] = df['P-Value'].apply(lambda x: f"{x:.3f}" if x >= 0.001 else "<0.001")
    
    # Format F-statistics  
    df['F-Statistic'] = df['F-Statistic'].apply(lambda x: f"{x:.2f}")
    
    # Add significance stars
    df['Significance'] = df['P-Value'].apply(get_significance_stars)
    
    return df

def get_significance_stars(p_value):
    '''Convert p-values to significance stars'''
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**" 
    elif p_value < 0.05:
        return "*"
    else:
        return ""
```

## Statistical Notation Standards

### P-Values
- Format: `p = 0.023` or `p < 0.001` for very small values
- Significance: * p<0.05, ** p<0.01, *** p<0.001

### F-Statistics  
- Format: `F(df1, df2) = 3.45`
- Always include degrees of freedom

### Percentages
- Format: `23.4%` for data displays
- Format: `{value:.1f}%` for dynamic content

### Crisis Period Notation
- GFC: "Global Financial Crisis (2008-2010)"
- COVID: "COVID-19 Pandemic (2020-2022)"
"""
        
        style_guide_path = Path("STYLE_GUIDE.md")
        with open(style_guide_path, 'w') as f:
            f.write(style_guide)
        
        print(f"  📄 Style guide created: {style_guide_path}")
        self.improvements.append("Generated comprehensive style guide")
        
        return style_guide_path
    
    def suggest_improvements(self):
        """Generate specific improvement suggestions"""
        print("\n💡 Style Improvement Suggestions:")
        
        if not self.style_issues:
            print("  ✅ No major style consistency issues detected")
            return
        
        for i, issue in enumerate(self.style_issues, 1):
            print(f"  {i}. {issue}")
        
        print("\n🔧 Recommended Actions:")
        
        if "Inconsistent color schemes" in self.style_issues:
            print("  🎨 Standardize color palette across all charts")
            print("     - Define CAPITAL_FLOWS_COLORS constant")
            print("     - Use consistent colors for similar data types")
        
        if "Inconsistent chart layout" in self.style_issues:
            print("  📏 Create standard chart layout template")
            print("     - Uniform title styling and positioning")
            print("     - Consistent margin and legend placement")
        
        if "Inconsistent table display" in self.style_issues:
            print("  📋 Standardize table display methods")
            print("     - Prefer st.dataframe for large datasets")
            print("     - Use st.table only for small summaries")
        
        if "Inconsistent statistical notation" in self.style_issues:
            print("  🔢 Standardize statistical notation")
            print("     - Uniform p-value formatting")
            print("     - Consistent significance star notation")
    
    def run_style_analysis(self):
        """Run complete style consistency analysis"""
        print("🎨 Capital Flows Research - Style Consistency Analysis")
        print("="*60)
        
        # Run all analyses
        color_schemes, layout_patterns = self.analyze_chart_styling()
        table_methods = self.analyze_table_formatting()
        notation_patterns = self.analyze_statistical_notation()
        
        # Generate improvements
        style_guide_path = self.generate_style_guide()
        self.suggest_improvements()
        
        # Summary
        print(f"\n📊 STYLE ANALYSIS SUMMARY:")
        print(f"  Issues Identified: {len(self.style_issues)}")
        print(f"  Improvements Generated: {len(self.improvements)}")
        
        if len(self.style_issues) <= 2:
            print("  ✅ Good overall style consistency")
        else:
            print("  ⚠️  Multiple style improvements recommended")
        
        return {
            'issues': self.style_issues,
            'improvements': self.improvements,
            'style_guide': style_guide_path
        }

def main():
    """Main execution function"""
    enhancer = StyleConsistencyEnhancer()
    results = enhancer.run_style_analysis()
    
    if len(results['issues']) == 0:
        print("\n🎉 Style consistency is excellent!")
    else:
        print(f"\n📝 {len(results['issues'])} style improvements identified")

if __name__ == "__main__":
    main()