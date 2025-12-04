# Research Pipeline Implementation Plan - FINAL VERSION

**Created**: December 2024
**Last Updated**: December 4, 2024
**Purpose**: Create traceable, verifiable notebooks for academic review
**Status**: Ready for Implementation

---

## Executive Summary

Transform 19,506 lines of dashboard code into ~2,500 lines of transparent Jupyter notebooks that show every calculation step. Extract ONLY the mathematical/statistical logic, ignoring all UI code.

## Current State (After Consolidation)

The codebase has been reduced from 47,000+ to 19,506 lines, but still contains:
- **Total Python Code**: 19,506 lines (after 59% reduction)
- **Core Statistics**: ~500 lines buried in UI/dashboard code
- **Target**: 2,500 lines of transparent notebooks
- **Expected Reduction**: 87% while maintaining full verifiability

## Implementation Structure

```
research_pipeline/
├── lib/
│   └── stats_core.py                      # ~150 lines of extracted functions
├── notebooks/
│   ├── CS1_Iceland_vs_Eurozone.ipynb     # ~400 lines
│   ├── CS2_Baltic_Euro_Adoption.ipynb    # ~500 lines
│   ├── CS3_Small_Open_Economies.ipynb    # ~400 lines
│   ├── CS4_Statistical_Framework.ipynb   # ~600 lines
│   └── CS5_Capital_Controls_Regimes.ipynb # ~500 lines
├── verification/
│   ├── extract_baseline.py               # One-time dashboard export
│   └── baseline_results/                 # Saved dashboard outputs
└── data/                                 # Symlink to updated_data/Clean/
```

## Precise Extraction Points

### From `src/dashboard/reports/cs1_report.py` (3,333 lines → extract ~50 lines)

**Extract Lines 232-258**: F-test calculation
```python
def perform_volatility_tests(data, indicators):
    # Lines 240-248: Core F-test logic
    iceland_var = iceland_data.var()
    eurozone_var = eurozone_data.var()
    f_stat = iceland_var / eurozone_var
    p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
```

**Extract Lines 139-168**: Summary statistics
```python
def calculate_group_statistics(data, group_col, indicators):
    # Lines 152-164: Mean, std, CV calculations
    mean_val = values.mean()
    std_val = values.std()
    cv = (std_val / abs(mean_val)) * 100
```

### From `src/core/cs4_statistical_analysis.py` (587 lines → extract ~200 lines)

**Extract Lines 200-245**: F-test with significance levels
```python
# Lines 202-216: Variance calculation and F-statistic
var1 = np.var(s1, ddof=1)
var2 = np.var(s2, ddof=1)
f_stat = var1 / var2 if var1 >= var2 else var2 / var1
```

**Extract Lines 254-294**: AR(4) model fitting
```python
# Lines 277-281: AR(4) fitting
model = AutoReg(clean_series, lags=4, trend='c')
fitted_model = model.fit()
ar_coeffs = fitted_model.params[1:5].values
```

**Extract Lines 296-340**: Half-life calculation from impulse response

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