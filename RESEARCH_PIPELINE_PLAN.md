# Research Pipeline Implementation Plan v2
**Last Updated**: December 4, 2024
**Goal**: Extract ~500 lines of statistics from 19,506 lines of code

## Quick Summary

Extract ONLY the mathematical calculations from the dashboard into transparent Jupyter notebooks. Ignore ALL UI code.

## File Structure
```
research_pipeline/
├── lib/
│   └── stats_core.py              # 150 lines of pure statistics
├── notebooks/
│   ├── CS1_Iceland_vs_Eurozone.ipynb
│   ├── CS2_Baltic_Euro_Adoption.ipynb
│   ├── CS3_Small_Open_Economies.ipynb
│   ├── CS4_Statistical_Framework.ipynb
│   └── CS5_Capital_Controls_Regimes.ipynb
└── verification/
    ├── extract_baseline.py        # Save dashboard results
    └── baseline_results/          # For comparison
```

## Core Library: stats_core.py

```python
"""Minimal statistics extracted from dashboard"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg

def calculate_f_statistic(group1_data, group2_data):
    """F-test for variance equality (from cs1_report.py:240-248)"""
    var1 = group1_data.var()
    var2 = group2_data.var()
    f_stat = var1 / var2
    df1, df2 = len(group1_data) - 1, len(group2_data) - 1
    p_value = 2 * min(stats.f.cdf(f_stat, df1, df2),
                      1 - stats.f.cdf(f_stat, df1, df2))
    return {'f_statistic': f_stat, 'p_value': p_value,
            'var1': var1, 'var2': var2}

def fit_ar4_model(series):
    """AR(4) model (from cs4_statistical_analysis.py:254-294)"""
    clean = series.dropna()
    if len(clean) < 8: return None
    model = AutoReg(clean, lags=4, trend='c')
    fitted = model.fit()
    return {'coefficients': fitted.params[1:5].values,
            'aic': fitted.aic, 'bic': fitted.bic}

def calculate_temporal_change(pre_data, post_data):
    """Before/after for CS2 Baltic analysis"""
    return {
        'pre_var': pre_data.var(),
        'post_var': post_data.var(),
        'var_ratio': post_data.var() / pre_data.var(),
        'var_change_pct': ((post_data.var() - pre_data.var()) /
                           pre_data.var() * 100)
    }

# Euro adoption dates for CS2
EURO_DATES = {
    'Estonia, Republic of': 2011,
    'Latvia, Republic of': 2014,
    'Lithuania, Republic of': 2015
}

# Crisis years for exclusion
CRISIS_YEARS = [2008, 2009, 2010, 2020, 2021, 2022]
```

## Extraction Map

### FROM: cs1_report.py (3,333 lines)
**EXTRACT**: Lines 240-248 (F-test), Lines 152-164 (summary stats)

### FROM: cs4_statistical_analysis.py (587 lines)
**EXTRACT**: Lines 202-216 (F-test), Lines 277-281 (AR4), Lines 296-340 (half-life)

### FROM: cs2_shared_functions.py (312 lines)
**EXTRACT**: Lines 33-61 (Euro dates), Lines 63-95 (period labels)

## CS1 Notebook Example

```python
# Cell 1: Setup
import pandas as pd
import sys
sys.path.append('../lib')
from stats_core import calculate_f_statistic

# Cell 2: Load Data
df = pd.read_csv('../data/comprehensive_df_PGDP_labeled.csv')
iceland = df[df['CS1_GROUP'] == 'Iceland']
eurozone = df[df['CS1_GROUP'] == 'Eurozone']

# Cell 3: F-test with Full Transparency
indicators = ['Direct_Investment_PGDP', 'Portfolio_Investment_PGDP']
results = []

for ind in indicators:
    ice_vals = iceland[ind].dropna()
    euro_vals = eurozone[ind].dropna()

    print(f"\n{'='*50}")
    print(f"F-TEST: {ind}")
    print(f"{'='*50}")
    print(f"Iceland: n={len(ice_vals)}, var={ice_vals.var():.6f}")
    print(f"Eurozone: n={len(euro_vals)}, var={euro_vals.var():.6f}")

    result = calculate_f_statistic(ice_vals, euro_vals)
    print(f"F-statistic: {result['f_statistic']:.4f}")
    print(f"P-value: {result['p_value']:.6f}")
    print(f"Significant at 5%: {result['p_value'] < 0.05}")

    results.append({
        'indicator': ind,
        'f_stat': result['f_statistic'],
        'p_value': result['p_value'],
        'significant': result['p_value'] < 0.05
    })

# Cell 4: Save Results
pd.DataFrame(results).to_csv('../outputs/CS1_results.csv', index=False)

# Cell 5: Verify Against Dashboard
baseline = pd.read_csv('../verification/baseline_results/CS1_baseline.csv')
# Compare and document any differences
```

## Baseline Extraction Script

```python
# verification/extract_baseline.py
# Run ONCE to save dashboard results

import sys
sys.path.append('../../src/dashboard/reports')
from cs1_report import perform_volatility_tests, load_default_data

# CS1 Baseline
data, indicators, _ = load_default_data()
cs1_results = perform_volatility_tests(data, indicators)
cs1_results.to_csv('baseline_results/CS1_baseline.csv', index=False)

# CS2 Baseline
from cs2_shared_functions import calculate_pre_post_statistics
# ... extract CS2 results

print("Baselines saved for verification")
```

## Implementation Order

### Day 1: Foundation
1. Create stats_core.py (150 lines)
2. Create extract_baseline.py (100 lines)
3. Run baseline extraction
4. Create CS1 notebook (400 lines)
5. Verify CS1 results match

### Day 2: Similar Patterns
1. CS3 notebook (reuses CS1 logic) (400 lines)
2. CS4 notebook (uses clean source) (600 lines)

### Day 3: Complex Cases
1. CS2 notebook (temporal analysis) (500 lines)
2. CS5 notebook (external data) (500 lines)

### Day 4: Final Verification
1. Run all notebooks
2. Compare with baselines
3. Document discrepancies

## Total Line Count
- stats_core.py: 150 lines
- 5 notebooks: ~2,400 lines
- extract_baseline.py: 100 lines
- **TOTAL**: ~2,650 lines (vs 19,506 current)
- **REDUCTION**: 86%

## Success Criteria
✅ Every calculation shown step-by-step
✅ Results match dashboard (tolerance 0.0001)
✅ No imports from src/dashboard
✅ Anyone can verify the math

## NOT Doing
❌ Refactoring dashboard code
❌ Creating abstractions
❌ Handling UI/visualization
❌ Optimizing performance