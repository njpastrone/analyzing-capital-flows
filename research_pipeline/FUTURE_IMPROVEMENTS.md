# Future Improvements for Research Pipeline

**Created**: December 9, 2024
**Purpose**: Track needed improvements to make notebooks fully independent for academic publication

## 🎯 Priority 1: Notebook Independence

### Remove External Dependencies
Currently, all notebooks depend on `../lib/stats_core.py` for statistical functions. This needs to be eliminated for true portability.

**Required Changes**:
1. **Embed Statistical Functions**
   - Copy `calculate_f_statistic()` function into each notebook
   - Copy `get_significance_stars()` function into each notebook
   - Copy `calculate_temporal_change()` into CS2
   - Copy `fit_ar4_model()` and related functions into CS4
   - Copy constants (EURO_ADOPTION_DATES, CRISIS_YEARS) where needed

2. **Remove Import Statements**
   - Delete all `sys.path.append('../lib')` statements
   - Delete all `from stats_core import ...` statements
   - Replace with local function definitions

3. **Test in Isolation**
   - Create fresh virtual environment
   - Install only: pandas, numpy, scipy, matplotlib, statsmodels
   - Run each notebook independently
   - Verify all outputs match current results

### Example Implementation
```python
# Instead of:
sys.path.append('../lib')
from stats_core import calculate_f_statistic

# Use:
def calculate_f_statistic(group1_data, group2_data, group1_name="Group1", group2_name="Group2"):
    """F-test for equality of variances between two groups."""
    # [Full function implementation here]
    ...
```

## 🎯 Priority 2: Export Preparation

### Requirements Documentation
Create `requirements.txt` in research_pipeline/ with minimal dependencies:
```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.6.0
statsmodels>=0.14.0
jupyter>=1.0.0
```

### PDF Generation
1. Run all notebooks with outputs
2. Export as PDF with outputs included
3. Verify all visualizations render correctly
4. Ensure page breaks are appropriate

### Data Dictionary
Add to each notebook a data dictionary cell explaining:
- Column names and meanings
- Units (% of GDP, millions USD, etc.)
- Time periods covered
- Missing data handling

## 🎯 Priority 3: Academic Publication Preparation

### Methodology Documentation
Each notebook should include:
1. **Introduction Section**
   - Research question
   - Hypothesis
   - Data sources

2. **Methodology Section**
   - Statistical tests used
   - Assumptions
   - Limitations

3. **Results Interpretation**
   - Key findings
   - Policy implications
   - Comparison with literature

### Citation Guidelines
Add cell with proper citations:
```python
"""
Data Sources:
- IMF Balance of Payments Statistics (BOP), 1999-2024
- IMF World Economic Outlook (WEO) for GDP data
- Fernández et al. (2016) Capital Control Measures Database
- Ilzetzki, Reinhart, and Rogoff (2019) Exchange Rate Classification

Statistical Methods:
- F-test for variance equality: Snedecor & Cochran (1989)
- AR(4) models: Box & Jenkins (1976)
- Winsorization: Tukey (1962)
"""
```

## 🎯 Priority 4: Reproducibility Enhancement

### Environment Specification
Create `environment.yml` for conda:
```yaml
name: capital-flows-research
channels:
  - conda-forge
dependencies:
  - python=3.9
  - pandas=2.0.3
  - numpy=1.24.3
  - scipy=1.10.1
  - matplotlib=3.7.1
  - statsmodels=0.14.0
  - jupyter=1.0.0
```

### Execution Instructions
Add README.md in research_pipeline/:
```markdown
# Research Pipeline Execution

## Setup
1. Create environment: `conda env create -f environment.yml`
2. Activate: `conda activate capital-flows-research`
3. Start Jupyter: `jupyter notebook`

## Running Notebooks
Execute notebooks in order:
1. CS1_Iceland_vs_Eurozone.ipynb
2. CS2_Baltic_Euro_Adoption.ipynb
3. CS3_Small_Open_Economies.ipynb
4. CS4_Statistical_Framework.ipynb
5. CS5_Capital_Controls_Regimes.ipynb

## Verification
Results in `outputs/` should match `verification/baseline_results/`
```

## 🎯 Priority 5: Quality Assurance

### Code Review Checklist
- [ ] All functions have docstrings
- [ ] Variable names are descriptive
- [ ] Magic numbers are explained
- [ ] Assumptions are documented
- [ ] Edge cases are handled

### Statistical Validation
- [ ] Verify F-test calculations
- [ ] Check p-value computations
- [ ] Validate significance levels
- [ ] Confirm degrees of freedom

### Output Validation
- [ ] Compare with dashboard results
- [ ] Check for numerical precision
- [ ] Verify chart accuracy
- [ ] Confirm statistical conclusions

## 📅 Implementation Timeline

### Week 1: Independence
- Embed functions in notebooks
- Remove external dependencies
- Test in isolation

### Week 2: Documentation
- Add methodology sections
- Create data dictionaries
- Include citations

### Week 3: Export & Package
- Generate PDFs
- Create environment files
- Write execution instructions

### Week 4: Final Review
- Quality assurance checks
- Peer review
- Prepare for submission

## 🚀 Success Criteria

The notebooks will be ready for academic publication when:
1. ✅ Zero external dependencies (except standard libraries)
2. ✅ Complete reproducibility in fresh environment
3. ✅ Full methodology documentation
4. ✅ Results match dashboard within tolerance
5. ✅ PDF exports with all outputs included
6. ✅ Peer review completed

---

**Note**: Prioritize notebook independence first, as this is essential for academic submission. Other improvements can be iterative.