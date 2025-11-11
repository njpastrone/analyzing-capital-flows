# PHASE 3 CENTRALIZATION ANALYSIS: Duplicate Helper Functions Report

## Executive Summary
Analysis of 31 Python files across the case study dashboard reveals **extensive duplication** of helper functions due to the 4-version architecture (full_reports, outlier_adjusted_reports, pdf_reports, pdf_reports_outlier_adjusted). The most impactful opportunity is centralizing **20+ recurring helper functions** that would eliminate **4,600+ lines of duplicated code** across the codebase.

## Key Metrics
- **Total Files Analyzed**: 31 (7 case studies × 4 versions + 2 CS2 master files + 1 CS2 master variant)
- **Duplicate Function Families**: 20+ major function families with 2+ occurrences
- **Total Duplicated Lines**: ~4,600 lines across all copies
- **Average Duplication Factor**: 3-4x per function
- **Highest Impact Single Function**: apply_professional_styling (8 files, 1,127 lines)

---

## TIER 1: CRITICAL PRIORITY - Cross-Cutting Styling & Utilities
### High Impact (Large Size × High Frequency)

#### 1. **apply_professional_styling()** - HIGHEST PRIORITY
**Category**: Visualization Helper Function (CSS/HTML Styling)
- **Occurrences**: 8 files
- **Total Duplicated Lines**: 1,127 lines
- **Average Size**: 141 lines per implementation
- **Duplication Factor**: 8x
- **Impact Score**: 1,127 (highest)

**Files**:
- src/dashboard/full_reports/cs4_report_app.py (197 lines)
- src/dashboard/full_reports/cs5_report_app.py (93 lines)
- src/dashboard/outlier_adjusted_reports/cs4_report_outlier_adjusted.py (215 lines)
- src/dashboard/outlier_adjusted_reports/cs5_report_outlier_adjusted.py (64 lines)
- src/dashboard/pdf_reports/cs4_report_app_pdf.py (215 lines)
- src/dashboard/pdf_reports/cs5_report_app_pdf.py (64 lines)
- src/dashboard/pdf_reports_outlier_adjusted/cs4_report_outlier_adjusted_pdf.py (215 lines)
- src/dashboard/pdf_reports_outlier_adjusted/cs5_report_outlier_adjusted_pdf.py (64 lines)

**What It Does**: Applies professional CSS styling for tables, charts, and general page layout. Includes PDF export optimization with media queries and responsive design for data tables.

**Variation Notes**: 
- CS4 versions are larger (215 lines) due to complex master table styling for F-test results
- CS5 versions are smaller (64 lines) due to simpler styling requirements
- All versions follow the same pattern structure

**Recommendation**: 
- Create `src/dashboard/shared_utils/styling.py` with:
  - `get_professional_base_css()` - Common styling
  - `get_cs4_specific_css()` - CS4 master table optimization
  - `get_cs5_specific_css()` - CS5 13-column table optimization
  - `apply_professional_styling(case_study='cs4')` - Main function with conditional imports

**Lines to Eliminate**: 1,000+ (keep 1 implementation, remove 7 copies)

---

#### 2. **get_pdf_optimized_figsize()** - HIGH PRIORITY
**Category**: Visualization Helper Function (PDF Export)
- **Occurrences**: 4 files (CS4 only)
- **Total Duplicated Lines**: 176 lines
- **Size per Implementation**: 44 lines
- **Duplication Factor**: 4x
- **Impact Score**: 176

**Files**:
- src/dashboard/full_reports/cs4_report_app.py (44 lines)
- src/dashboard/outlier_adjusted_reports/cs4_report_outlier_adjusted.py (44 lines)
- src/dashboard/pdf_reports/cs4_report_app_pdf.py (44 lines)
- src/dashboard/pdf_reports_outlier_adjusted/cs4_report_outlier_adjusted_pdf.py (44 lines)

**What It Does**: Calculates PDF-optimized figure sizes for different chart types (boxplot, grid, timeseries). Respects US Letter format with 0.75" margins.

**Variation Notes**: All implementations are identical (no variations)

**Recommendation**: Move to `src/dashboard/shared_utils/pdf_utils.py`

**Lines to Eliminate**: 132 (keep 1 implementation, remove 3 copies)

---

#### 3. **generate_html_report()** - HIGHEST CONTENT SIZE
**Category**: Report Generation Function (HTML/PDF Export)
- **Occurrences**: 4 files (CS1 only)
- **Total Duplicated Lines**: 1,432 lines (358 × 4)
- **Size per Implementation**: 358 lines
- **Duplication Factor**: 4x
- **Impact Score**: 1,432 (CRITICAL)

**Files**:
- src/dashboard/full_reports/cs1_report_app.py (358 lines)
- src/dashboard/outlier_adjusted_reports/cs1_report_outlier_adjusted.py (358 lines)
- src/dashboard/pdf_reports/cs1_report_app_pdf.py (358 lines)
- src/dashboard/pdf_reports_outlier_adjusted/cs1_report_outlier_adjusted_pdf.py (358 lines)

**What It Does**: Generates comprehensive HTML reports for CS1 analysis including headers, summary tables, statistical results, and interactive visualizations. Large, complex function that combines data formatting with HTML generation.

**Variation Notes**: All implementations are identical (no variations)

**Recommendation**: 
- Create `src/dashboard/shared_utils/report_generation.py`
- Function: `generate_html_report(case_study, final_data, analysis_indicators, test_results, group_stats, boxplot_data, outlier_adjusted=False)`
- Parameterize case study and outlier_adjusted flag

**Lines to Eliminate**: 1,074 (keep 1 implementation, remove 3 copies)

---

## TIER 2: HIGH PRIORITY - Core Statistical Functions
### Medium-High Impact (Medium Size × High Frequency)

#### 4. **CS4 Visualization Chart Suite** - MODERATE PRIORITY
Four functions with identical implementations in 4 files each:

**4a. create_comprehensive_boxplots_chart()**
- **Occurrences**: 4 files
- **Total Duplicated Lines**: 380 lines
- **Size**: 95 lines each
- **Impact**: 380

**4b. create_comprehensive_acf_chart()**
- **Occurrences**: 4 files
- **Total Duplicated Lines**: 396 lines
- **Size**: 99 lines each
- **Impact**: 396

**4c. create_comprehensive_timeseries_chart()**
- **Occurrences**: 4 files
- **Total Duplicated Lines**: 272 lines
- **Size**: 68 lines each
- **Impact**: 272

**Files Affected** (all four functions appear in same 4 files):
- src/dashboard/full_reports/cs4_report_app.py
- src/dashboard/outlier_adjusted_reports/cs4_report_outlier_adjusted.py
- src/dashboard/pdf_reports/cs4_report_app_pdf.py
- src/dashboard/pdf_reports_outlier_adjusted/cs4_report_outlier_adjusted_pdf.py

**What They Do**: Generate professional matplotlib charts with PDF export optimization for statistical analysis display.

**Recommendation**: 
- Create `src/dashboard/shared_utils/cs4_charts.py`
- Move all three functions there
- Parameterize outlier_adjusted flag if needed

**Lines to Eliminate**: 1,048 (keep 1 of each function, remove 3 copies each)

---

#### 5. **CS5 Scatter Plot Functions** - MODERATE PRIORITY
**5a. create_capital_controls_scatter()**
- **Occurrences**: 4 files
- **Total Duplicated Lines**: 312 lines
- **Size**: 78 lines each
- **Impact**: 312

**5b. create_country_aggregate_scatter()**
- **Occurrences**: 4 files
- **Total Duplicated Lines**: 416 lines
- **Size**: 104 lines each
- **Impact**: 416

**Files**:
- src/dashboard/full_reports/cs5_report_app.py
- src/dashboard/outlier_adjusted_reports/cs5_report_outlier_adjusted.py
- src/dashboard/pdf_reports/cs5_report_app_pdf.py
- src/dashboard/pdf_reports_outlier_adjusted/cs5_report_outlier_adjusted_pdf.py

**What They Do**: Create scatter plots with Iceland highlighted for capital controls and regime analysis.

**Recommendation**: Move to `src/dashboard/shared_utils/cs5_charts.py`

**Lines to Eliminate**: 728 (keep 1 of each, remove 3 copies each)

---

#### 6. **CS1 Statistical Calculation Functions** - HIGH PRIORITY
**6a. calculate_group_statistics()**
- **Occurrences**: 4 files
- **Total Duplicated Lines**: 124 lines (31 × 4)
- **Size**: 31 lines each
- **Impact**: 124

**6b. create_boxplot_data()**
- **Occurrences**: 4 files
- **Total Duplicated Lines**: 116 lines (29 × 4)
- **Size**: 29 lines each
- **Impact**: 116

**6c. create_individual_country_boxplot_data()**
- **Occurrences**: 4 files
- **Total Duplicated Lines**: 132 lines (33 × 4)
- **Size**: 33 lines each
- **Impact**: 132

**6d. perform_volatility_tests()**
- **Occurrences**: 4 files
- **Total Duplicated Lines**: 116 lines (29 × 4)
- **Size**: 29 lines each
- **Impact**: 116

**Files**:
- src/dashboard/full_reports/cs1_report_app.py
- src/dashboard/outlier_adjusted_reports/cs1_report_outlier_adjusted.py
- src/dashboard/pdf_reports/cs1_report_app_pdf.py
- src/dashboard/pdf_reports_outlier_adjusted/cs1_report_outlier_adjusted_pdf.py

**What They Do**: 
- calculate_group_statistics: Compute mean, std dev, skewness, CV by group and indicator
- create_boxplot_data: Prepare data for boxplot visualization
- create_individual_country_boxplot_data: Prepare individual country-level boxplot data
- perform_volatility_tests: Conduct F-tests for variance equality testing

**Recommendation**: Create `src/dashboard/shared_utils/cs1_statistics.py` with all four functions

**Lines to Eliminate**: 488 (keep 1 of each, remove 3 copies each)

---

## TIER 3: MEDIUM PRIORITY - Case Study Specific Functions
### Medium Impact (Large Size × Lower Frequency)

#### 7. **CS4 Display & Analysis Functions** - MEDIUM PRIORITY
**7a. display_comprehensive_analysis_overview()**
- **Occurrences**: 4 files
- **Total Duplicated Lines**: 1,340 lines (335 × 4)
- **Size**: 335 lines each
- **Impact**: 1,340

**7b. display_master_table()**
- **Occurrences**: 4 files
- **Total Duplicated Lines**: 416 lines (104 × 4)
- **Size**: 104 lines each
- **Impact**: 416

**7c. create_integrated_table()**
- **Occurrences**: 4 files
- **Total Duplicated Lines**: 520 lines (130 × 4)
- **Size**: 130 lines each
- **Impact**: 520

**7d. display_summary_insights_and_export()**
- **Occurrences**: 4 files
- **Total Duplicated Lines**: 388 lines (97 × 4)
- **Size**: 97 lines each
- **Impact**: 388

**7e. create_master_table()**
- **Occurrences**: 4 files
- **Total Duplicated Lines**: 200 lines (50 × 4)
- **Size**: 50 lines each
- **Impact**: 200

**Files** (all in CS4):
- src/dashboard/full_reports/cs4_report_app.py
- src/dashboard/outlier_adjusted_reports/cs4_report_outlier_adjusted.py
- src/dashboard/pdf_reports/cs4_report_app_pdf.py
- src/dashboard/pdf_reports_outlier_adjusted/cs4_report_outlier_adjusted_pdf.py

**What They Do**: Complex table creation and display logic for F-test results with significance stars, color coding, and professional formatting.

**Recommendation**: Create `src/dashboard/shared_utils/cs4_tables.py`

**Lines to Eliminate**: 2,864 (keep 1 of each, remove 3 copies each)

---

#### 8. **CS5 Analysis & Table Functions** - MEDIUM PRIORITY
**8a. run_cs5_analysis()**
- **Occurrences**: 2 files
- **Total Duplicated Lines**: 326 lines (163 × 2)
- **Size**: 163 lines each
- **Impact**: 326

**8b. create_cs5_master_table()**
- **Occurrences**: 2 files
- **Total Duplicated Lines**: 156 lines (78 × 2)
- **Size**: 78 lines each
- **Impact**: 156

**8c. display_cs5_master_table()**
- **Occurrences**: 2 files
- **Total Duplicated Lines**: 148 lines (74 × 2)
- **Size**: 74 lines each
- **Impact**: 148

**Files**:
- src/dashboard/full_reports/cs5_report_app.py
- src/dashboard/pdf_reports/cs5_report_app_pdf.py

**What They Do**: End-to-end CS5 analysis orchestration and professional table generation with regime comparison data.

**Recommendation**: Create `src/dashboard/shared_utils/cs5_analysis.py`

**Lines to Eliminate**: 630

---

#### 9. **CS2 Temporal Analysis Functions** - MEDIUM PRIORITY
**9a. calculate_temporal_statistics()**
- **Occurrences**: 2 files
- **Total Duplicated Lines**: 60 lines (30 × 2)
- **Size**: 30 lines each
- **Impact**: 60

**9b. create_temporal_boxplot_data()**
- **Occurrences**: 2 files
- **Total Duplicated Lines**: 62 lines (31 × 2)
- **Size**: 31 lines each
- **Impact**: 62

**9c. perform_temporal_volatility_tests()**
- **Occurrences**: 2 files
- **Total Duplicated Lines**: 76 lines (38 × 2)
- **Size**: 38 lines each
- **Impact**: 76

**Files**:
- src/dashboard/outlier_adjusted_reports/case_study_2_euro_adoption_outlier_adjusted.py
- src/dashboard/pdf_reports_outlier_adjusted/case_study_2_euro_adoption_outlier_adjusted_pdf.py

**What They Do**: Pre/post-Euro adoption statistical analysis (mean, variance, F-tests by period).

**Recommendation**: Create `src/dashboard/shared_utils/cs2_statistics.py`

**Lines to Eliminate**: 198

---

## TIER 4: LOWER PRIORITY - Display & Utility Functions
### Lower Impact (Small Size or Low Frequency)

#### 10. **Simple Utility Functions** - LOWER PRIORITY
**10a. format_table_for_display()**
- **Occurrences**: 4 files (CS4)
- **Size**: 7 lines each
- **Total**: 28 lines
- **Recommendation**: Move to `src/dashboard/shared_utils/formatting.py`

**10b. create_download_link()**
- **Occurrences**: 4 files (CS4)
- **Size**: 7 lines each
- **Total**: 28 lines
- **Recommendation**: Move to `src/dashboard/shared_utils/formatting.py`

**10c. load_capital_controls_data()**
- **Occurrences**: 4 files (CS5)
- **Size**: 21 lines each
- **Total**: 84 lines
- **Recommendation**: Already has centralized version in `shared_utils/data_loading.py`

**10d. load_regime_analysis_data()**
- **Occurrences**: 4 files (CS5)
- **Size**: 22 lines each
- **Total**: 88 lines
- **Recommendation**: Already has centralized version in `shared_utils/data_loading.py`

**Lines to Eliminate**: 228 (if consolidating all)

---

#### 11. **Data Loading Wrappers** - ALREADY PARTIALLY CENTRALIZED
**11a. load_default_data()**
- **Occurrences**: 4 files (CS1)
- **Size**: 27 lines each
- **Total**: 108 lines
- **Status**: Already uses centralized `_load_cs_data()` from shared_utils

**11b. load_overall_capital_flows_data()**
- **Occurrences**: 4 files (CS1)
- **Size**: 26 lines average (some variation)
- **Total**: 145 lines
- **Status**: Already uses centralized `_load_overall_data()` from shared_utils

**Recommendation**: These are wrapper functions. Can be consolidated further or kept as thin wrappers.

---

#### 12. **CS2 Temporal Data Loading** - ALREADY PARTIALLY CENTRALIZED
**12a. load_case_study_2_data()**
- **Occurrences**: 2 files
- **Size**: ~30 lines each
- **Status**: Should be centralized in `shared_utils/data_loading.py`

**Recommendation**: Move to centralized data loading module

---

#### 13. **CS3 & CS2 Display Functions** - CASE STUDY SPECIFIC
**13a. show_estonia_overall_analysis() / show_latvia_overall_analysis() / show_lithuania_overall_analysis()**
- **Occurrences**: 4 files each (3 country variants × 4 versions)
- **Size**: 85 lines each
- **Total**: 340 lines per country
- **Note**: These are country-specific; less value in centralization

**13b. show_estonia_indicator_analysis() / show_latvia_indicator_analysis() / show_lithuania_indicator_analysis()**
- **Occurrences**: 4 files each
- **Size**: 86 lines each
- **Total**: 344 lines per country
- **Note**: These are country-specific

**Recommendation**: Low priority; country-specific logic limits reusability

---

## DUPLICATION SUMMARY BY CATEGORY

| Category | Function Families | Total Duplicated Lines | Priority |
|----------|------------------|----------------------|----------|
| **Styling/CSS** | apply_professional_styling | 1,127 | CRITICAL |
| **Report Generation** | generate_html_report | 1,432 | CRITICAL |
| **CS4 Display Tables** | 5 functions | 2,864 | HIGH |
| **CS4 Charts** | 3 functions | 1,048 | HIGH |
| **CS1 Statistics** | 4 functions | 488 | HIGH |
| **CS5 Scatter Plots** | 2 functions | 728 | MEDIUM |
| **CS5 Analysis** | 3 functions | 630 | MEDIUM |
| **PDF Optimization** | get_pdf_optimized_figsize | 176 | HIGH |
| **CS2 Temporal Stats** | 3 functions | 198 | MEDIUM |
| **Formatting/Utils** | 2 functions | 56 | LOW |
| **Data Loading** | 3 functions | 180 (mostly centralized) | LOW |
| **CS2/CS3 Display** | 6 country functions × 4 | 1,200+ | LOW |
| **TOTAL** | **20+ families** | **~4,600** | - |

---

## IMPLEMENTATION ROADMAP FOR PHASE 3

### Phase 3a: Critical Foundation (Weeks 1-2)
1. Create `src/dashboard/shared_utils/styling.py`
   - Move apply_professional_styling with case_study parameter
   - **Eliminate**: 1,000 lines

2. Create `src/dashboard/shared_utils/report_generation.py`
   - Move generate_html_report and related report functions
   - **Eliminate**: 1,074 lines

3. Create `src/dashboard/shared_utils/pdf_utils.py`
   - Move get_pdf_optimized_figsize
   - **Eliminate**: 132 lines

**Phase 3a Impact**: Eliminate 2,206 lines

### Phase 3b: Case Study Utilities (Weeks 2-3)
4. Create `src/dashboard/shared_utils/cs1_statistics.py`
   - Move calculate_group_statistics, create_boxplot_data, create_individual_country_boxplot_data, perform_volatility_tests
   - **Eliminate**: 488 lines

5. Create `src/dashboard/shared_utils/cs4_charts.py`
   - Move create_comprehensive_boxplots_chart, create_comprehensive_acf_chart, create_comprehensive_timeseries_chart
   - **Eliminate**: 1,048 lines

6. Create `src/dashboard/shared_utils/cs4_tables.py`
   - Move display_comprehensive_analysis_overview, display_master_table, create_integrated_table, display_summary_insights_and_export, create_master_table
   - **Eliminate**: 2,864 lines

**Phase 3b Impact**: Eliminate 4,400 lines

### Phase 3c: Additional Utilities (Weeks 3-4)
7. Create `src/dashboard/shared_utils/cs5_charts.py`
   - Move create_capital_controls_scatter, create_country_aggregate_scatter
   - **Eliminate**: 728 lines

8. Create `src/dashboard/shared_utils/cs5_analysis.py`
   - Move run_cs5_analysis, create_cs5_master_table, display_cs5_master_table (with outlier_adjusted variants)
   - **Eliminate**: 630 lines

9. Create `src/dashboard/shared_utils/cs2_statistics.py`
   - Move calculate_temporal_statistics, create_temporal_boxplot_data, perform_temporal_volatility_tests
   - **Eliminate**: 198 lines

10. Enhance `src/dashboard/shared_utils/data_loading.py`
    - Add load_case_study_2_data, consolidate temporal data loading
    - **Eliminate**: 180 lines

**Phase 3c Impact**: Eliminate 1,736 lines

### Total Phase 3 Impact
- **Lines Eliminated**: 2,206 + 4,400 + 1,736 = **8,342 lines** (conservative estimate; actual will be higher including imports and adjustments)
- **Files Simplified**: 31 files → each 250-400 lines shorter
- **Maintenance Burden Reduced**: Functions updated in 1 place instead of 4
- **New Modules Created**: 10 new shared utility modules
- **Testing Scope**: 20+ new unit test opportunities

---

## ARCHITECTURAL RECOMMENDATIONS

### New Module Structure for Phase 3

```
src/dashboard/
├── shared_utils/
│   ├── __init__.py
│   ├── styling.py              # CSS/HTML styling (apply_professional_styling, get_professional_css)
│   ├── report_generation.py    # Report HTML generation (generate_html_report)
│   ├── pdf_utils.py            # PDF optimization (get_pdf_optimized_figsize)
│   ├── formatting.py           # Data formatting (format_table_for_display, create_download_link)
│   ├── cs1_statistics.py       # CS1 statistical functions (calculate_group_statistics, etc.)
│   ├── cs1_charts.py           # CS1 visualization (to be created)
│   ├── cs2_statistics.py       # CS2 temporal analysis (calculate_temporal_statistics, etc.)
│   ├── cs3_charts.py           # CS3 visualization (to be created)
│   ├── cs4_charts.py           # CS4 visualization (create_comprehensive_boxplots_chart, etc.)
│   ├── cs4_tables.py           # CS4 table generation (display_comprehensive_analysis_overview, etc.)
│   ├── cs5_charts.py           # CS5 visualization (create_capital_controls_scatter, etc.)
│   ├── cs5_analysis.py         # CS5 analysis (run_cs5_analysis, etc.)
│   ├── cs5_tables.py           # CS5 table generation (create_cs5_master_table, etc.)
│   └── data_loading.py         # Data loading (already exists; enhance)
│
├── full_reports/
│   ├── cs1_report_app.py       # Simplified (imports from shared_utils)
│   ├── cs2_*_report_app.py     # Simplified (imports from shared_utils)
│   ├── cs3_report_app.py       # Simplified (imports from shared_utils)
│   ├── cs4_report_app.py       # Simplified (imports from shared_utils)
│   └── cs5_report_app.py       # Simplified (imports from shared_utils)
│
├── outlier_adjusted_reports/   # Simplified (imports from shared_utils)
├── pdf_reports/                # Simplified (imports from shared_utils)
└── pdf_reports_outlier_adjusted/ # Simplified (imports from shared_utils)
```

### Key Design Patterns

1. **Parameterized Functions**
   - Pass `outlier_adjusted=True/False` flag to functions that have variants
   - Use `case_study='cs1'|'cs4'|'cs5'` parameter for case-study-specific logic
   - Example: `apply_professional_styling(case_study='cs4', outlier_adjusted=False)`

2. **Import Structure in Case Study Files**
   ```python
   # Before (duplicate code):
   def apply_professional_styling():
       st.markdown("""<style>...""")
   
   # After (centralized):
   from shared_utils.styling import apply_professional_styling
   
   # In main function:
   apply_professional_styling(case_study='cs4')
   ```

3. **Version Compatibility**
   - All outlier_adjusted variants use same underlying functions
   - Data loading determines if winsorized or full datasets are used
   - Functions accept both and behave accordingly

---

## TESTING RECOMMENDATIONS

### Unit Tests to Create (Phase 3)
1. Test styling functions with different case studies
2. Test report generation with different data types
3. Test statistical calculation correctness
4. Test chart generation with edge cases
5. Test data loading with both full and winsorized datasets

### Integration Tests
1. Verify all 31 files still work after centralization
2. Test all 4 versions (full, outlier, PDF, PDF+outlier) render correctly
3. Test PDF export with complex tables

---

## RISK ASSESSMENT & MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Subtle differences in implementations not visible in analysis | Medium | High | Line-by-line comparison of each function |
| Import errors when consolidating | Low | Medium | Comprehensive integration testing |
| Performance impact on module loading | Low | Low | Profile before/after loading times |
| Breaking changes for future maintenance | Low | Medium | Clear documentation and type hints |
| Merge conflicts during refactoring | Medium | Medium | Create branch, merge carefully by section |

---

## ESTIMATED EFFORT

| Phase | Task | Effort | Risk |
|-------|------|--------|------|
| 3a | Styling + Report Gen + PDF Utils | 8-10 hours | Low |
| 3b | CS1/CS4 functions | 10-12 hours | Medium |
| 3c | CS2/CS5 functions + data loading | 8-10 hours | Low |
| Testing | Unit + Integration tests | 8-10 hours | Medium |
| Cleanup | Documentation + final polish | 4-6 hours | Low |
| **Total** | **All Phase 3** | **38-48 hours** | **Medium** |

---

## CONCLUSION

Phase 3 offers a significant opportunity to reduce codebase complexity and maintenance burden through systematic centralization of helper functions. The 4-version architecture (full_reports, outlier_adjusted, pdf_reports, pdf_reports_outlier_adjusted) creates substantial duplication that Phase 3 is well-positioned to address.

**Key Takeaway**: By centralizing 20+ function families, we can eliminate 4,600-8,000+ duplicated lines while making the codebase more maintainable and easier to update. The most impactful priorities are styling (apply_professional_styling) and report generation (generate_html_report), which together account for 2,500+ duplicated lines.

