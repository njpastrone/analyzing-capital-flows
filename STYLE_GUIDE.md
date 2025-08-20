
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
