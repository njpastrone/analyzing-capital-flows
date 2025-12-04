# Notebook Template Structure

Each notebook should follow this consistent structure for clarity and reproducibility:

## Standard Notebook Sections

### 1. Title and Metadata
```python
# Case Study X: [Title]
# Research Pipeline - Capital Flows Analysis
# Date: [Creation Date]
# Author: [Author Name]
```

### 2. Executive Summary (Markdown cell)
- Research question
- Key findings (filled after analysis)
- Methodology overview

### 3. Setup and Imports
```python
# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Configuration
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.6f}'.format)
plt.style.use('seaborn-v0_8-whitegrid')

# Paths
DATA_DIR = Path('../data')
OUTPUT_DIR = Path('../outputs')
```

### 4. Data Loading
```python
# Load data with clear documentation
print("Loading data from: comprehensive_df_PGDP_labeled.csv")
print("Source: IMF Balance of Payments Statistics")
print("Period: 1999-2024")
print("Frequency: Quarterly")

df = pd.read_csv(DATA_DIR / 'comprehensive_df_PGDP_labeled.csv')
print(f"Loaded {len(df):,} observations")
```

### 5. Data Exploration
- Show data structure
- Summary statistics
- Missing data analysis
- Group definitions

### 6. Statistical Analysis (WITH TRANSPARENCY)
```python
# EXAMPLE: F-test with full transparency
print("=" * 60)
print("F-TEST FOR VARIANCE EQUALITY")
print("=" * 60)

# State hypothesis
print("\nHypothesis:")
print("H0: σ²_Iceland = σ²_Eurozone (variances are equal)")
print("H1: σ²_Iceland ≠ σ²_Eurozone (variances are different)")
print("Significance level: α = 0.05")

# Show the calculation
iceland_data = df[df['GROUP'] == 'Iceland']['INDICATOR_VALUE'].dropna()
eurozone_data = df[df['GROUP'] == 'Eurozone']['INDICATOR_VALUE'].dropna()

print(f"\nSample sizes:")
print(f"Iceland: n = {len(iceland_data)}")
print(f"Eurozone: n = {len(eurozone_data)}")

# Calculate variances
var_iceland = iceland_data.var()
var_eurozone = eurozone_data.var()

print(f"\nVariances:")
print(f"Iceland: σ² = {var_iceland:.6f}")
print(f"Eurozone: σ² = {var_eurozone:.6f}")

# F-statistic
f_stat = var_iceland / var_eurozone
print(f"\nF-statistic = σ²_Iceland / σ²_Eurozone = {f_stat:.4f}")

# Critical values and p-value
df1 = len(iceland_data) - 1
df2 = len(eurozone_data) - 1
p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))

print(f"\nDegrees of freedom: df1={df1}, df2={df2}")
print(f"P-value: {p_value:.6f}")
print(f"\nConclusion: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'}")
```

### 7. Visualization
```python
# Create publication-ready figures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Time series
ax1.plot(iceland_ts, label='Iceland')
ax1.plot(eurozone_ts, label='Eurozone')
ax1.set_xlabel('Year')
ax1.set_ylabel('Capital Flows (% of GDP)')
ax1.legend()
ax1.set_title('Capital Flow Volatility Comparison')

# Box plots
ax2.boxplot([iceland_data, eurozone_data], labels=['Iceland', 'Eurozone'])
ax2.set_ylabel('Capital Flows (% of GDP)')
ax2.set_title('Distribution Comparison')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figures' / 'cs1_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 8. Results Summary
```python
# Create summary table
results = pd.DataFrame({
    'Group': ['Iceland', 'Eurozone'],
    'Mean': [iceland_data.mean(), eurozone_data.mean()],
    'Std Dev': [iceland_data.std(), eurozone_data.std()],
    'Variance': [var_iceland, var_eurozone],
    'N': [len(iceland_data), len(eurozone_data)]
})

print("\nSUMMARY STATISTICS TABLE")
print(results.to_string(index=False))

# Save for verification
results.to_csv(OUTPUT_DIR / 'tables' / 'cs1_summary_stats.csv', index=False)
```

### 9. Verification Against Dashboard
```python
# Load dashboard results for comparison
dashboard_results = pd.read_csv('../verification/dashboard_results/cs1_results.csv')

print("\nVERIFICATION: Comparing with Dashboard Results")
print("-" * 50)

# Compare key statistics
for metric in ['mean', 'variance', 'f_statistic']:
    notebook_val = results[metric]
    dashboard_val = dashboard_results[metric]
    difference = abs(notebook_val - dashboard_val)
    match = "✓" if difference < 0.0001 else "✗"

    print(f"{metric:15} | Notebook: {notebook_val:10.6f} | "
          f"Dashboard: {dashboard_val:10.6f} | {match}")
```

### 10. Conclusions
- Summary of findings
- Interpretation
- Limitations
- Next steps

### 11. Export Results
```python
# Save all outputs for reproducibility
print("\nExporting results...")

# Statistical outputs
with open(OUTPUT_DIR / 'statistics' / 'cs1_full_output.txt', 'w') as f:
    f.write(f"Case Study 1: Iceland vs Eurozone\n")
    f.write(f"Date: {pd.Timestamp.now()}\n")
    f.write(f"=" * 60 + "\n")
    f.write(f"F-statistic: {f_stat:.6f}\n")
    f.write(f"P-value: {p_value:.6f}\n")
    # ... additional statistics

print("✓ Results exported to outputs/")
```

## Best Practices

1. **Show Your Work**: Display intermediate calculations, not just final results
2. **State Assumptions**: Make all assumptions explicit
3. **Document Sources**: Reference data sources and methodological papers
4. **Version Control**: Include date and version in outputs
5. **Error Handling**: Check for data quality issues explicitly
6. **Reproducibility**: Set random seeds where applicable
7. **Clear Narrative**: Use markdown cells to explain what you're doing and why

## Verification Checklist

- [ ] Results match dashboard within reasonable tolerance
- [ ] All calculations are shown transparently
- [ ] Assumptions are clearly stated
- [ ] Output files are created successfully
- [ ] Notebook runs from top to bottom without errors
- [ ] Figures are saved in high resolution
- [ ] Tables are exported to CSV