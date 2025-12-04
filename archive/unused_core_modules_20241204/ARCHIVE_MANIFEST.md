# Archive Manifest - Unused Core Modules
**Date Archived**: December 4, 2024
**Reason**: Code cleanup - Phase 4 optimization

## Summary
Archived 4 unused core modules totaling 2,393 lines of code that were not being imported or used anywhere in the active codebase.

## Archived Files

### 1. sensitivity_analysis_framework.py (804 lines)
- **Purpose**: Comprehensive sensitivity analysis for robust statistical conclusions
- **Status**: Never imported by any active code
- **Reason for archiving**: Unused legacy module

### 2. robust_analysis_report_generator.py (773 lines)
- **Purpose**: Automated report generation for statistical robustness
- **Status**: Never imported by any active code
- **Dependencies**: Only file that imported winsorized_data_loader.py
- **Reason for archiving**: Unused legacy module

### 3. cs4_robustness_tests.py (451 lines)
- **Purpose**: Robustness testing framework for CS4 analysis
- **Status**: Never imported by any active code
- **Reason for archiving**: Unused legacy module

### 4. winsorized_data_loader.py (365 lines)
- **Purpose**: Outlier-adjusted data handling and processing
- **Status**: Only imported by robust_analysis_report_generator.py (also unused)
- **Reason for archiving**: Unused legacy module

## Important Notes

### Winsorized/Outlier-Adjusted Analysis Still Works
The winsorized (outlier-adjusted) analysis functionality continues to work perfectly without these modules because it uses:
- Pre-processed CSV files from the R data pipeline (`comprehensive_df_PGDP_labeled_winsorized.csv`)
- Parameter switching in dashboard reports (`data_type='winsorized'`)
- Configuration in `dashboard_config.py` that routes to appropriate CSV files
- NOT these archived modules

### Verification Performed
Before archiving, the following tests were conducted:
1. ✅ No imports found in any Python files (except self-references)
2. ✅ No references in test files
3. ✅ No dynamic imports or string references
4. ✅ Dashboard works perfectly with files renamed
5. ✅ Winsorized data loading confirmed working without these files

### Recovery Instructions
If these modules are ever needed:
```bash
# To restore all files:
cp archive/unused_core_modules_20241204/*.py src/core/

# To restore specific file:
cp archive/unused_core_modules_20241204/sensitivity_analysis_framework.py src/core/
```

## Line Count Summary
- **Total Lines Archived**: 2,393
- **Percentage of Codebase**: ~11% of total Python code
- **Impact on Functionality**: None - all features continue working

## Related Documentation
- See `CODEBASE_AUDIT_REPORT.md` for detailed analysis
- See `CONSOLIDATION_COMPLETION_PLAN.md` for Phase 4 optimization context