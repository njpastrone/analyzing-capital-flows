# Phase 3: Duplicate Function Quick Reference Guide

## One-Liner Summary
**31 files contain 20+ duplicated helper functions totaling 4,600+ lines. Top priorities: apply_professional_styling (1,127 dup lines), generate_html_report (1,432 dup lines), and CS4 functions (2,864+ dup lines).**

---

## Files Analyzed (31 Total)

### Full Reports (7 files)
1. `/home/user/analyzing-capital-flows/src/dashboard/full_reports/cs1_report_app.py`
2. `/home/user/analyzing-capital-flows/src/dashboard/full_reports/cs2_estonia_report_app.py`
3. `/home/user/analyzing-capital-flows/src/dashboard/full_reports/cs2_latvia_report_app.py`
4. `/home/user/analyzing-capital-flows/src/dashboard/full_reports/cs2_lithuania_report_app.py`
5. `/home/user/analyzing-capital-flows/src/dashboard/full_reports/cs3_report_app.py`
6. `/home/user/analyzing-capital-flows/src/dashboard/full_reports/cs4_report_app.py`
7. `/home/user/analyzing-capital-flows/src/dashboard/full_reports/cs5_report_app.py`

### Outlier-Adjusted Reports (8 files)
8. `/home/user/analyzing-capital-flows/src/dashboard/outlier_adjusted_reports/case_study_2_euro_adoption_outlier_adjusted.py`
9. `/home/user/analyzing-capital-flows/src/dashboard/outlier_adjusted_reports/cs1_report_outlier_adjusted.py`
10. `/home/user/analyzing-capital-flows/src/dashboard/outlier_adjusted_reports/cs2_estonia_report_outlier_adjusted.py`
11. `/home/user/analyzing-capital-flows/src/dashboard/outlier_adjusted_reports/cs2_latvia_report_outlier_adjusted.py`
12. `/home/user/analyzing-capital-flows/src/dashboard/outlier_adjusted_reports/cs2_lithuania_report_outlier_adjusted.py`
13. `/home/user/analyzing-capital-flows/src/dashboard/outlier_adjusted_reports/cs3_report_outlier_adjusted.py`
14. `/home/user/analyzing-capital-flows/src/dashboard/outlier_adjusted_reports/cs4_report_outlier_adjusted.py`
15. `/home/user/analyzing-capital-flows/src/dashboard/outlier_adjusted_reports/cs5_report_outlier_adjusted.py`

### PDF Reports (7 files)
16. `/home/user/analyzing-capital-flows/src/dashboard/pdf_reports/cs1_report_app_pdf.py`
17. `/home/user/analyzing-capital-flows/src/dashboard/pdf_reports/cs2_estonia_report_app_pdf.py`
18. `/home/user/analyzing-capital-flows/src/dashboard/pdf_reports/cs2_latvia_report_app_pdf.py`
19. `/home/user/analyzing-capital-flows/src/dashboard/pdf_reports/cs2_lithuania_report_app_pdf.py`
20. `/home/user/analyzing-capital-flows/src/dashboard/pdf_reports/cs3_report_app_pdf.py`
21. `/home/user/analyzing-capital-flows/src/dashboard/pdf_reports/cs4_report_app_pdf.py`
22. `/home/user/analyzing-capital-flows/src/dashboard/pdf_reports/cs5_report_app_pdf.py`

### PDF Reports Outlier-Adjusted (8 files)
23. `/home/user/analyzing-capital-flows/src/dashboard/pdf_reports_outlier_adjusted/case_study_2_euro_adoption_outlier_adjusted_pdf.py`
24. `/home/user/analyzing-capital-flows/src/dashboard/pdf_reports_outlier_adjusted/cs1_report_outlier_adjusted_pdf.py`
25. `/home/user/analyzing-capital-flows/src/dashboard/pdf_reports_outlier_adjusted/cs2_estonia_report_outlier_adjusted_pdf.py`
26. `/home/user/analyzing-capital-flows/src/dashboard/pdf_reports_outlier_adjusted/cs2_latvia_report_outlier_adjusted_pdf.py`
27. `/home/user/analyzing-capital-flows/src/dashboard/pdf_reports_outlier_adjusted/cs2_lithuania_report_outlier_adjusted_pdf.py`
28. `/home/user/analyzing-capital-flows/src/dashboard/pdf_reports_outlier_adjusted/cs3_report_outlier_adjusted_pdf.py`
29. `/home/user/analyzing-capital-flows/src/dashboard/pdf_reports_outlier_adjusted/cs4_report_outlier_adjusted_pdf.py`
30. `/home/user/analyzing-capital-flows/src/dashboard/pdf_reports_outlier_adjusted/cs5_report_outlier_adjusted_pdf.py`

---

## Top 10 Duplicate Function Families

| Rank | Function Name | Occurrences | Duplicated Lines | Category | Priority |
|------|---------------|------------|------------------|----------|----------|
| 1 | **generate_html_report** | 4 | 1,432 | Report Gen | CRITICAL |
| 2 | **apply_professional_styling** | 8 | 1,127 | Styling | CRITICAL |
| 3 | **display_comprehensive_analysis_overview** | 4 | 1,340 | CS4 Display | HIGH |
| 4 | **create_comprehensive_acf_chart** | 4 | 396 | CS4 Charts | HIGH |
| 5 | **create_comprehensive_boxplots_chart** | 4 | 380 | CS4 Charts | HIGH |
| 6 | **create_integrated_table** | 4 | 520 | CS4 Tables | HIGH |
| 7 | **create_country_aggregate_scatter** | 4 | 416 | CS5 Charts | MEDIUM |
| 8 | **display_summary_insights_and_export** | 4 | 388 | CS4 Analysis | MEDIUM |
| 9 | **create_capital_controls_scatter** | 4 | 312 | CS5 Charts | MEDIUM |
| 10 | **create_comprehensive_timeseries_chart** | 4 | 272 | CS4 Charts | HIGH |

---

## Function Locations by Family

### TIER 1: CRITICAL - apply_professional_styling()
Located in 8 files:
```
src/dashboard/full_reports/cs4_report_app.py (197 lines)
src/dashboard/full_reports/cs5_report_app.py (93 lines)
src/dashboard/outlier_adjusted_reports/cs4_report_outlier_adjusted.py (215 lines)
src/dashboard/outlier_adjusted_reports/cs5_report_outlier_adjusted.py (64 lines)
src/dashboard/pdf_reports/cs4_report_app_pdf.py (215 lines)
src/dashboard/pdf_reports/cs5_report_app_pdf.py (64 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs4_report_outlier_adjusted_pdf.py (215 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs5_report_outlier_adjusted_pdf.py (64 lines)
```
**Action**: Create `src/dashboard/shared_utils/styling.py` with case_study parameter

---

### TIER 1: CRITICAL - generate_html_report()
Located in 4 files (CS1 only):
```
src/dashboard/full_reports/cs1_report_app.py (358 lines)
src/dashboard/outlier_adjusted_reports/cs1_report_outlier_adjusted.py (358 lines)
src/dashboard/pdf_reports/cs1_report_app_pdf.py (358 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs1_report_outlier_adjusted_pdf.py (358 lines)
```
**Action**: Create `src/dashboard/shared_utils/report_generation.py`

---

### TIER 2: HIGH - CS4 Statistical Functions

#### generate_overall_html_content() - 4 files (CS1)
```
src/dashboard/full_reports/cs1_report_app.py (100 lines)
src/dashboard/outlier_adjusted_reports/cs1_report_outlier_adjusted.py (100 lines)
src/dashboard/pdf_reports/cs1_report_app_pdf.py (100 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs1_report_outlier_adjusted_pdf.py (100 lines)
```

#### create_all_flows_time_series_charts() - 4 files (CS1)
```
src/dashboard/full_reports/cs1_report_app.py (117 lines)
src/dashboard/outlier_adjusted_reports/cs1_report_outlier_adjusted.py (117 lines)
src/dashboard/pdf_reports/cs1_report_app_pdf.py (117 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs1_report_outlier_adjusted_pdf.py (117 lines)
```

#### get_pdf_optimized_figsize() - 4 files (CS4)
```
src/dashboard/full_reports/cs4_report_app.py (44 lines)
src/dashboard/outlier_adjusted_reports/cs4_report_outlier_adjusted.py (44 lines)
src/dashboard/pdf_reports/cs4_report_app_pdf.py (44 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs4_report_outlier_adjusted_pdf.py (44 lines)
```
**Action**: Create `src/dashboard/shared_utils/pdf_utils.py`

---

### TIER 2: HIGH - CS1 Statistical Functions

#### calculate_group_statistics() - 4 files
```
src/dashboard/full_reports/cs1_report_app.py (31 lines)
src/dashboard/outlier_adjusted_reports/cs1_report_outlier_adjusted.py (31 lines)
src/dashboard/pdf_reports/cs1_report_app_pdf.py (31 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs1_report_outlier_adjusted_pdf.py (31 lines)
```

#### create_boxplot_data() - 4 files
```
src/dashboard/full_reports/cs1_report_app.py (29 lines)
src/dashboard/outlier_adjusted_reports/cs1_report_outlier_adjusted.py (29 lines)
src/dashboard/pdf_reports/cs1_report_app_pdf.py (29 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs1_report_outlier_adjusted_pdf.py (29 lines)
```

#### create_individual_country_boxplot_data() - 4 files
```
src/dashboard/full_reports/cs1_report_app.py (33 lines)
src/dashboard/outlier_adjusted_reports/cs1_report_outlier_adjusted.py (33 lines)
src/dashboard/pdf_reports/cs1_report_app_pdf.py (33 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs1_report_outlier_adjusted_pdf.py (33 lines)
```

#### perform_volatility_tests() - 4 files
```
src/dashboard/full_reports/cs1_report_app.py (29 lines)
src/dashboard/outlier_adjusted_reports/cs1_report_outlier_adjusted.py (29 lines)
src/dashboard/pdf_reports/cs1_report_app_pdf.py (29 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs1_report_outlier_adjusted_pdf.py (29 lines)
```
**Action**: Create `src/dashboard/shared_utils/cs1_statistics.py`

---

### TIER 2: HIGH - CS4 Chart Functions

#### create_comprehensive_boxplots_chart() - 4 files
```
src/dashboard/full_reports/cs4_report_app.py (95 lines)
src/dashboard/outlier_adjusted_reports/cs4_report_outlier_adjusted.py (95 lines)
src/dashboard/pdf_reports/cs4_report_app_pdf.py (95 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs4_report_outlier_adjusted_pdf.py (95 lines)
```

#### create_comprehensive_acf_chart() - 4 files
```
src/dashboard/full_reports/cs4_report_app.py (99 lines)
src/dashboard/outlier_adjusted_reports/cs4_report_outlier_adjusted.py (99 lines)
src/dashboard/pdf_reports/cs4_report_app_pdf.py (99 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs4_report_outlier_adjusted_pdf.py (99 lines)
```

#### create_comprehensive_timeseries_chart() - 4 files
```
src/dashboard/full_reports/cs4_report_app.py (68 lines)
src/dashboard/outlier_adjusted_reports/cs4_report_outlier_adjusted.py (68 lines)
src/dashboard/pdf_reports/cs4_report_app_pdf.py (68 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs4_report_outlier_adjusted_pdf.py (68 lines)
```
**Action**: Create `src/dashboard/shared_utils/cs4_charts.py`

---

### TIER 3: MEDIUM - CS4 Table/Display Functions

#### display_comprehensive_analysis_overview() - 4 files
```
src/dashboard/full_reports/cs4_report_app.py (335 lines)
src/dashboard/outlier_adjusted_reports/cs4_report_outlier_adjusted.py (335 lines)
src/dashboard/pdf_reports/cs4_report_app_pdf.py (335 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs4_report_outlier_adjusted_pdf.py (335 lines)
```

#### display_master_table() - 4 files
```
src/dashboard/full_reports/cs4_report_app.py (104 lines)
src/dashboard/outlier_adjusted_reports/cs4_report_outlier_adjusted.py (104 lines)
src/dashboard/pdf_reports/cs4_report_app_pdf.py (104 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs4_report_outlier_adjusted_pdf.py (104 lines)
```

#### create_integrated_table() - 4 files
```
src/dashboard/full_reports/cs4_report_app.py (130 lines)
src/dashboard/outlier_adjusted_reports/cs4_report_outlier_adjusted.py (130 lines)
src/dashboard/pdf_reports/cs4_report_app_pdf.py (130 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs4_report_outlier_adjusted_pdf.py (130 lines)
```

#### display_summary_insights_and_export() - 4 files
```
src/dashboard/full_reports/cs4_report_app.py (97 lines)
src/dashboard/outlier_adjusted_reports/cs4_report_outlier_adjusted.py (97 lines)
src/dashboard/pdf_reports/cs4_report_app_pdf.py (97 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs4_report_outlier_adjusted_pdf.py (97 lines)
```

#### create_master_table() - 4 files
```
src/dashboard/full_reports/cs4_report_app.py (50 lines)
src/dashboard/outlier_adjusted_reports/cs4_report_outlier_adjusted.py (50 lines)
src/dashboard/pdf_reports/cs4_report_app_pdf.py (50 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs4_report_outlier_adjusted_pdf.py (50 lines)
```
**Action**: Create `src/dashboard/shared_utils/cs4_tables.py`

---

### TIER 3: MEDIUM - CS5 Chart Functions

#### create_capital_controls_scatter() - 4 files
```
src/dashboard/full_reports/cs5_report_app.py (78 lines)
src/dashboard/outlier_adjusted_reports/cs5_report_outlier_adjusted.py (78 lines)
src/dashboard/pdf_reports/cs5_report_app_pdf.py (78 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs5_report_outlier_adjusted_pdf.py (78 lines)
```

#### create_country_aggregate_scatter() - 4 files
```
src/dashboard/full_reports/cs5_report_app.py (104 lines)
src/dashboard/outlier_adjusted_reports/cs5_report_outlier_adjusted.py (104 lines)
src/dashboard/pdf_reports/cs5_report_app_pdf.py (104 lines)
src/dashboard/pdf_reports_outlier_adjusted/cs5_report_outlier_adjusted_pdf.py (104 lines)
```
**Action**: Create `src/dashboard/shared_utils/cs5_charts.py`

---

### TIER 3: MEDIUM - CS5 Analysis Functions

#### run_cs5_analysis() - 2 files
```
src/dashboard/full_reports/cs5_report_app.py (163 lines)
src/dashboard/pdf_reports/cs5_report_app_pdf.py (163 lines)
```

#### create_cs5_master_table() - 2 files
```
src/dashboard/full_reports/cs5_report_app.py (78 lines)
src/dashboard/pdf_reports/cs5_report_app_pdf.py (78 lines)
```

#### display_cs5_master_table() - 2 files
```
src/dashboard/full_reports/cs5_report_app.py (74 lines)
src/dashboard/pdf_reports/cs5_report_app_pdf.py (74 lines)
```
**Action**: Create `src/dashboard/shared_utils/cs5_analysis.py`

---

### TIER 3: MEDIUM - CS2 Temporal Functions

#### calculate_temporal_statistics() - 2 files
```
src/dashboard/outlier_adjusted_reports/case_study_2_euro_adoption_outlier_adjusted.py (30 lines)
src/dashboard/pdf_reports_outlier_adjusted/case_study_2_euro_adoption_outlier_adjusted_pdf.py (30 lines)
```

#### create_temporal_boxplot_data() - 2 files
```
src/dashboard/outlier_adjusted_reports/case_study_2_euro_adoption_outlier_adjusted.py (31 lines)
src/dashboard/pdf_reports_outlier_adjusted/case_study_2_euro_adoption_outlier_adjusted_pdf.py (31 lines)
```

#### perform_temporal_volatility_tests() - 2 files
```
src/dashboard/outlier_adjusted_reports/case_study_2_euro_adoption_outlier_adjusted.py (38 lines)
src/dashboard/pdf_reports_outlier_adjusted/case_study_2_euro_adoption_outlier_adjusted_pdf.py (38 lines)
```
**Action**: Create `src/dashboard/shared_utils/cs2_statistics.py`

---

## Phase 3 Implementation Order

### Week 1: Critical Foundation
1. **src/dashboard/shared_utils/styling.py**
   - apply_professional_styling(case_study='cs4', outlier_adjusted=False)
   - Lines saved: 1,000+

2. **src/dashboard/shared_utils/report_generation.py**
   - generate_html_report()
   - Lines saved: 1,074

3. **src/dashboard/shared_utils/pdf_utils.py**
   - get_pdf_optimized_figsize()
   - Lines saved: 132

**Total Week 1**: 2,206 lines

### Week 2: CS1 & CS4 Core
4. **src/dashboard/shared_utils/cs1_statistics.py**
   - calculate_group_statistics()
   - create_boxplot_data()
   - create_individual_country_boxplot_data()
   - perform_volatility_tests()
   - Lines saved: 488

5. **src/dashboard/shared_utils/cs4_charts.py**
   - create_comprehensive_boxplots_chart()
   - create_comprehensive_acf_chart()
   - create_comprehensive_timeseries_chart()
   - Lines saved: 1,048

6. **src/dashboard/shared_utils/cs4_tables.py**
   - display_comprehensive_analysis_overview()
   - display_master_table()
   - create_integrated_table()
   - display_summary_insights_and_export()
   - create_master_table()
   - Lines saved: 2,864

**Total Week 2**: 4,400 lines

### Week 3: CS2/CS5 & Utilities
7. **src/dashboard/shared_utils/cs5_charts.py**
   - create_capital_controls_scatter()
   - create_country_aggregate_scatter()
   - Lines saved: 728

8. **src/dashboard/shared_utils/cs5_analysis.py**
   - run_cs5_analysis()
   - create_cs5_master_table()
   - display_cs5_master_table()
   - Lines saved: 630

9. **src/dashboard/shared_utils/cs2_statistics.py**
   - calculate_temporal_statistics()
   - create_temporal_boxplot_data()
   - perform_temporal_volatility_tests()
   - Lines saved: 198

10. **Enhance src/dashboard/shared_utils/data_loading.py**
    - Add load_case_study_2_data()
    - Lines saved: 180

**Total Week 3**: 1,736 lines

**Grand Total**: 8,342 lines eliminated

