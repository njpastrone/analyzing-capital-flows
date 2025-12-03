# Research Pipeline Implementation Plan

**Created**: December 2024
**Purpose**: Create traceable, verifiable notebooks for academic review
**Status**: Planning Phase

---

## Executive Summary

This document outlines the plan to create a transparent, traceable research pipeline using Jupyter notebooks. Each case study will have its own comprehensive notebook showing every calculation step, suitable for academic peer review and publication.

## Problem Statement

The current codebase contains:
- 47,000+ lines of code with 43% duplication
- Complex Streamlit UI obscuring core calculations
- 4x duplication across report versions (regular, outlier-adjusted, PDF versions)
- Difficult to manually verify calculations for research publication

## Solution: Traceable Research Notebooks

### Structure: 5 Comprehensive Notebooks

```
research_pipeline/
├── notebooks/
│   ├── CS1_Iceland_vs_Eurozone.ipynb
│   ├── CS2_Baltic_Euro_Adoption.ipynb
│   ├── CS3_Small_Open_Economies.ipynb
│   ├── CS4_Statistical_Framework.ipynb
│   ├── CS5_Capital_Controls_Regimes.ipynb
│   └── 00_Data_Pipeline.ipynb          # Optional: Show data cleaning
├── data/
│   └── (cleaned CSV files from updated_data/Clean/)
└── verification/
    ├── dashboard_screenshots/          # Original results for comparison
    └── notebook_outputs/               # CSV exports from notebooks
```

## Detailed Notebook Contents

### CS1_Iceland_vs_Eurozone.ipynb

**Research Question**: Does Iceland exhibit significantly higher capital flow volatility compared to Eurozone countries?

**Structure**:
1. **Introduction & Methodology**
   - Research context and motivation
   - Statistical approach (F-tests for variance equality)
   - Data sources (IMF Balance of Payments, 1999-2024)

2. **Data Loading**
   ```python
   # Load pre-cleaned data
   df = pd.read_csv('../data/comprehensive_df_PGDP_labeled.csv')
   # Filter for Case Study 1 groups
   cs1_data = df[df['CS1_GROUP'].notna()]
   ```

3. **Exploratory Analysis**
   - Summary statistics by group (Iceland vs Eurozone)
   - Time series visualization
   - Distribution analysis

4. **Statistical Tests** (with full transparency)
   ```python
   # Example of traceable calculation:
   # Show formula
   print("F-test: F = Var(Iceland) / Var(Eurozone)")
   # Show intermediate values
   print(f"Iceland variance: {iceland_var:.6f}")
   print(f"Eurozone variance: {eurozone_var:.6f}")
   # Calculate and show result
   f_stat = iceland_var / eurozone_var
   print(f"F-statistic: {f_stat:.4f}")
   ```

5. **Results Tables**
   - Formatted summary table
   - Export to CSV for verification

6. **Verification**
   - Compare with dashboard results
   - Document any differences

### CS2_Baltic_Euro_Adoption.ipynb

**Research Question**: How did Euro adoption affect capital flow volatility in Baltic countries?

**Structure**:
1. **Introduction**
   - Estonia (adopted 2011), Latvia (2014), Lithuania (2015)
   - Before/after methodology
   - Crisis period handling options

2. **Data Preparation**
   ```python
   # Define pre/post Euro periods for each country
   estonia_pre = data[(data['COUNTRY'] == 'Estonia') & (data['YEAR'] < 2011)]
   estonia_post = data[(data['COUNTRY'] == 'Estonia') & (data['YEAR'] >= 2011)]
   ```

3. **Country-by-Country Analysis**
   - Estonia: pre-2011 vs post-2011 volatility
   - Latvia: pre-2014 vs post-2014 volatility
   - Lithuania: pre-2015 vs post-2015 volatility

4. **Statistical Tests**
   - F-tests for variance changes
   - Temporal stability tests
   - Full calculation transparency

5. **Comparative Results**
   - Cross-country comparison table
   - Visualization of volatility changes

6. **Export & Verification**

### CS3_Small_Open_Economies.ipynb

**Research Question**: How does Iceland's volatility compare to other small open economies?

**Structure**:
1. **Introduction**
   - Iceland vs 6 comparable economies
   - Selection criteria for comparators
   - Size-adjusted volatility analysis

2. **Data Loading**
   ```python
   # Load CS3 group data
   cs3_data = df[df['CS3_GROUP'].notna()]
   countries = cs3_data['COUNTRY'].unique()
   ```

3. **Comparative Analysis**
   - Volatility measures for each country
   - Relative volatility rankings
   - Size adjustments

4. **Statistical Testing**
   - Pairwise F-tests
   - Group comparisons
   - Full calculation transparency

5. **Results & Verification**

### CS4_Statistical_Framework.ipynb

**Research Question**: Comprehensive statistical analysis using advanced methodologies

**Structure**:
1. **Advanced Methodologies**
   - F-tests for variance equality
   - AR(4) models with impulse response
   - RMSE prediction analysis

2. **Data Loading**
   ```python
   # Import from CS4-specific cleaned data
   from pathlib import Path
   cs4_dir = Path('../data/CS4_Statistical_Modeling/')
   ```

3. **F-Test Analysis**
   - Iceland vs Eurozone (weighted/simple avg)
   - Iceland vs Small Open Economies
   - Iceland vs Baltics
   - Show all variance calculations

4. **AR(4) Time Series Models**
   ```python
   # Model fitting with full transparency
   from statsmodels.tsa.ar_model import AutoReg
   model = AutoReg(series, lags=4)
   results = model.fit()
   # Show coefficients and diagnostics
   ```

5. **RMSE Prediction Analysis**
   - Rolling window predictions
   - Forecast accuracy metrics

6. **Summary Tables & Verification**

### CS5_Capital_Controls_Regimes.ipynb

**Research Question**: How do capital controls and exchange rate regimes affect volatility?

**Structure**:
1. **External Data Integration**
   - Fernández et al. (2016) capital controls database
   - Ilzetzki-Reinhart-Rogoff (2019) regime classification

2. **Data Merging**
   ```python
   # Merge external data with capital flows
   controls_data = pd.read_csv('../data/capital_controls.csv')
   merged = pd.merge(flows_data, controls_data, on=['COUNTRY', 'YEAR'])
   ```

3. **Capital Controls Analysis** (1999-2017)
   - Correlation with flow volatility
   - Iceland-specific patterns
   - Scatter plots with trend lines

4. **Exchange Rate Regime Analysis** (1999-2019)
   - 6-regime classification system
   - F-tests by regime type
   - Regime transition effects

5. **Results & Policy Implications**

## Key Features Across All Notebooks

### 1. Full Transparency
Every calculation shows:
- Mathematical formula
- Intermediate values
- Final result
- Statistical interpretation

### 2. Example Traceable Calculation Cell

```python
# ================================================
# F-TEST: DIRECT INVESTMENT VOLATILITY
# ================================================

# Mathematical Formula
"""
F-test for Equality of Variances
H₀: σ²(Iceland) = σ²(Eurozone)
H₁: σ²(Iceland) ≠ σ²(Eurozone)
Test Statistic: F = S₁²/S₂² ~ F(n₁-1, n₂-1)
"""

# Step 1: Extract data
iceland_data = df[df['CS1_GROUP'] == 'Iceland']['Direct_Investment_PGDP'].dropna()
eurozone_data = df[df['CS1_GROUP'] == 'Eurozone']['Direct_Investment_PGDP'].dropna()
print(f"Sample sizes: Iceland={len(iceland_data)}, Eurozone={len(eurozone_data)}")

# Step 2: Calculate variances
var_iceland = iceland_data.var(ddof=1)
var_eurozone = eurozone_data.var(ddof=1)
print(f"Variances: Iceland={var_iceland:.6f}, Eurozone={var_eurozone:.6f}")

# Step 3: Calculate F-statistic
f_stat = var_iceland / var_eurozone
print(f"F = {var_iceland:.6f} / {var_eurozone:.6f} = {f_stat:.4f}")

# Step 4: Calculate p-value
from scipy.stats import f
df1, df2 = len(iceland_data) - 1, len(eurozone_data) - 1
p_value = 2 * min(f.cdf(f_stat, df1, df2), 1 - f.cdf(f_stat, df1, df2))
print(f"P-value (two-tailed): {p_value:.6f}")
print(f"Significant at 5%: {'Yes' if p_value < 0.05 else 'No'}")

# Step 5: Save for verification
results = pd.DataFrame({
    'indicator': ['Direct_Investment'],
    'f_statistic': [f_stat],
    'p_value': [p_value],
    'var_iceland': [var_iceland],
    'var_eurozone': [var_eurozone]
})
results.to_csv('../verification/notebook_outputs/cs1_f_test.csv', index=False)
```

### 3. Verification Process

Each notebook includes:
1. Load original dashboard results
2. Compare calculated values
3. Flag any discrepancies
4. Document differences if any

## Implementation Timeline

### Day 1: Foundation
- Set up notebook structure
- CS1 notebook (cleanest case study)
- CS4 notebook (uses existing clean module)

### Day 2: Temporal Analysis
- CS2 Baltic countries notebook
- CS3 Small Open Economies notebook

### Day 3: External Data
- CS5 Capital Controls & Regimes notebook
- Data pipeline documentation (optional)

### Day 4: Verification & Polish
- Run all notebooks end-to-end
- Compare all results with dashboard
- Document any discrepancies
- Final adjustments

## Verification Strategy

### Step 1: Baseline Capture
1. Run existing dashboard for each case study
2. Screenshot all results tables
3. Export data to CSV where possible
4. Document exact values for key statistics

### Step 2: Notebook Execution
1. Run each notebook completely
2. Export all results to CSV
3. Create summary comparison tables

### Step 3: Comparison
1. Automated comparison where possible
2. Manual verification of key statistics
3. Document any differences with explanations

### Step 4: Sign-off
1. Confirm all critical values match
2. Document verification completion
3. Prepare for academic review

## Benefits of This Approach

✅ **Complete Traceability**: Every calculation visible and documented
✅ **Academic Standards**: Suitable for peer review and publication
✅ **Modular Structure**: One notebook per research question
✅ **Reproducible**: Anyone can re-run and verify results
✅ **No Code Duplication**: Uses existing cleaned data
✅ **Manageable Scope**: ~500 lines per notebook vs 47,000 total
✅ **Version Controlled**: Easy to track changes in git

## Risk Mitigation

### What We're NOT Doing
❌ Not refactoring existing code
❌ Not creating parallel implementations
❌ Not modifying working dashboard
❌ Not duplicating data cleaning

### What We ARE Doing
✅ Creating transparent documentation
✅ Using existing cleaned data
✅ Showing every calculation step
✅ Verifying against known results

## Success Criteria

1. **All notebooks run without errors**
2. **Results match dashboard within tolerance** (< 0.0001 difference)
3. **Every calculation is traceable**
4. **Academic reviewer can understand and reproduce**
5. **No hidden calculations or black boxes**

## Next Steps

1. Create `research_pipeline/` directory structure
2. Set up first notebook (CS1) as proof of concept
3. Verify CS1 results match dashboard exactly
4. Proceed with remaining notebooks
5. Complete verification documentation

---

**Note**: This plan prioritizes transparency and verifiability over code elegance. The goal is to create notebooks that can withstand academic scrutiny while leveraging existing cleaned data and verified calculations.