# Capital Flows Research - Testing Framework

## Overview

This comprehensive testing framework ensures the reliability and integrity of the Capital Flows Research project's data pipeline and analytical components. The test suite prevents issues like the recent trailing space filename problems that broke data loading across multiple case studies.

## Test Structure

```
tests/
├── conftest.py                     # Shared fixtures and utilities
├── test_data_loading.py            # Core data access and file path tests
├── test_case_study_1.py           # CS1: Iceland vs Eurozone tests
├── test_case_study_2.py           # CS2: Baltic Euro adoption tests
├── test_case_study_3.py           # CS3: Iceland vs SOE tests
├── test_case_study_4.py           # CS4: Statistical modeling tests
├── test_case_study_5.py           # CS5: Capital controls & regimes tests
├── test_statistical_functions.py  # Common statistical methods tests
└── README.md                      # This file
```

## Test Categories

### 1. Data Loading Verification (`test_data_loading.py`)
**Purpose**: Prevent data pipeline breaks and ensure file access reliability

**Key Tests**:
- ✅ Comprehensive dataset exists and is accessible
- ✅ No trailing spaces in filenames (prevents recent pipeline issues)
- ✅ CS4 specialized files exist
- ✅ CS5 data directories and files are accessible
- ✅ Cross-platform path handling
- ✅ File encoding and format consistency
- ✅ Error handling for missing/corrupted files

**Value**: Catches infrastructure issues before they break user experience.

### 2. Case Study Specific Tests (`test_case_study_*.py`)

#### CS1: Iceland vs Eurozone (`test_case_study_1.py`)
**Data Integrity Checks**:
- ✅ Full period loading (1,093 rows, 14 indicators)
- ✅ Crisis-excluded loading with proper filtering
- ✅ Iceland vs Eurozone group filtering (105 vs 988 observations)
- ✅ Luxembourg exclusion verification
- ✅ Statistical function integration

#### CS2: Baltic Euro Adoption (`test_case_study_2.py`)
**Temporal Analysis Validation**:
- ✅ Baltic country filtering (Estonia, Latvia, Lithuania)
- ✅ Euro adoption period coverage (2011, 2014, 2015)
- ✅ Before/after temporal analysis capability
- ✅ 105 observations per country verification

#### CS3: Iceland vs Small Open Economies (`test_case_study_3.py`)
**Comparative Analysis Setup**:
- ✅ Small open economy country selection
- ✅ Iceland vs Comparator group integrity
- ✅ Multi-country aggregation potential
- ✅ Cross-country comparison data alignment

#### CS4: Statistical Analysis (`test_case_study_4.py`)
**Advanced Modeling Capability**:
- ✅ Specialized data file access (12 files)
- ✅ Portfolio investment disaggregation
- ✅ Full vs crisis-excluded version consistency
- ✅ Variance modeling data sufficiency
- ✅ Crisis period identification accuracy

#### CS5: Capital Controls & Exchange Rate Regimes (`test_case_study_5.py`)
**External Data Integration**:
- ✅ Capital controls data structure (4 files)
- ✅ Exchange rate regime analysis (8 files)
- ✅ Correlation analysis capability
- ✅ Time series alignment potential
- ✅ External data merge compatibility

### 3. Statistical Functions Tests (`test_statistical_functions.py`)
**Core Statistical Methods Validation**:
- ✅ F-test implementation for variance comparison
- ✅ Correlation analysis functionality
- ✅ Descriptive statistics calculation
- ✅ Hypothesis testing framework
- ✅ Statistical assumptions checking
- ✅ Robustness to sample size and missing data

## Running Tests

### Quick Test Run
```bash
# Run all tests
python run_tests.py

# Or use pytest directly
python -m pytest tests/ -v
```

### Individual Test Categories
```bash
# Test specific components
python -m pytest tests/test_data_loading.py -v
python -m pytest tests/test_case_study_1.py -v
python -m pytest tests/test_statistical_functions.py -v
```

### Advanced Testing Options
```bash
# Run with coverage reporting
python -m pytest tests/ --cov=src --cov-report=html

# Run only fast tests (skip slow data loading)
python -m pytest tests/ -m "not slow"

# Show detailed failure information
python -m pytest tests/ --tb=long
```

## Expected Results

**✅ Healthy Pipeline Status**:
```
=================== 90 passed, 2 skipped, 3 warnings ===================
🎉 All tests passed!
✅ Capital Flows data pipeline is healthy and reliable
```

**Test Coverage**:
- **Data Access**: 14 tests covering file existence, path handling, encoding
- **CS1-CS5 Functionality**: 65 tests covering each case study's specific requirements
- **Statistical Methods**: 13 tests covering core analytical functions
- **Total**: 92 comprehensive tests with 98% pass rate

## Error Prevention

### Issues This Framework Prevents

**1. Trailing Space Filename Issues** (Recently Fixed):
```python
# ❌ This would break:
file_path = "comprehensive_df_PGDP_labeled.csv "  # trailing space
# ✅ Test catches this:
assert TestDataQuality.check_no_trailing_spaces(filename)
```

**2. Missing Data File Issues**:
```python
# ✅ Test verifies before runtime:
assert comprehensive_file.exists(), "Data file missing"
assert comprehensive_file.stat().st_size > 0, "Data file empty"
```

**3. Country Filtering Errors**:
```python
# ✅ Test verifies expected countries:
actual_countries = set(data['COUNTRY'].unique())
assert actual_countries == expected_countries
```

**4. Statistical Calculation Errors**:
```python
# ✅ Test verifies F-test implementation:
f_test_results = perform_volatility_tests(data, indicators)
assert len(f_test_results) == len(indicators)
```

### Data Integrity Validation

**Expected Data Shapes**:
- CS1: 1,093 rows (Iceland: 105, Eurozone: 988)
- CS2: 315 rows (3 countries × 105 observations)
- CS3: 763 rows (Iceland + 7 SOE countries)
- CS4: 12 specialized files with consistent structure
- CS5: 12 files (4 controls + 8 regimes)

**Quality Checks**:
- ✅ No infinite values in statistical calculations
- ✅ Reasonable value ranges for economic indicators
- ✅ Consistent time series structure across datasets
- ✅ Proper handling of crisis period exclusions

## Maintenance

### Adding New Tests

**For New Case Studies**:
1. Create `test_case_study_N.py` following existing patterns
2. Add data loading, filtering, and integrity tests
3. Update `conftest.py` with new fixtures if needed

**For New Statistical Methods**:
1. Add tests to `test_statistical_functions.py`
2. Test edge cases, assumptions, and robustness
3. Verify integration with existing case studies

### Updating Test Expectations

**When Data Changes**:
1. Update expected shapes in `conftest.py` fixtures
2. Modify country lists if case study composition changes  
3. Adjust time range expectations if data coverage changes

**When Methods Change**:
1. Update function signature tests
2. Modify expected output format tests
3. Add backward compatibility checks if needed

## Integration with CI/CD

### GitHub Actions Integration
```yaml
# .github/workflows/tests.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python run_tests.py
```

### Pre-commit Hooks
```bash
# Run tests before each commit
git add .pre-commit-config.yaml
pre-commit install
```

## Performance Considerations

**Test Execution Time**: ~1.2 seconds for full suite
**Slowest Tests**: 
- CS3 data loading: 0.07s
- Statistical calculations: 0.01s each

**Memory Usage**: Minimal - tests load data efficiently and clean up automatically

## Troubleshooting

### Common Issues

**ImportError for Case Study Functions**:
- Check that `src/dashboard` is in Python path
- Verify case study modules are not broken
- Run individual case study tests to isolate issues

**File Not Found Errors**:
- Verify data consolidation is complete
- Check for trailing spaces in filenames
- Ensure updated_data/Clean/ structure is intact

**Statistical Test Failures**:
- Check data quality and ranges
- Verify statistical assumptions are met
- Review calculation methods for edge cases

### Debug Mode
```bash
# Run with maximum verbosity
python -m pytest tests/ -v --tb=long --show-capture=all

# Run single test for debugging
python -m pytest tests/test_data_loading.py::TestDataFileAccess::test_comprehensive_dataset_exists -v -s
```

This testing framework provides a robust foundation for maintaining the reliability and quality of the Capital Flows Research platform, preventing issues before they impact users and ensuring all analytical components function correctly.