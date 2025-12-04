# Dashboard Consolidation Completion Plan

## Current Status Overview
- **Completion**: 85% (14,553 active lines vs 12,000 target)
- **Main Gap**: CS2 not fully consolidated (2,162 lines in case_study_2_euro_adoption.py)
- **Functional Status**: ✅ All case studies working with runtime errors fixed

## Phase 1: Complete CS2 Consolidation (Priority: CRITICAL)
**Estimated Time**: 3-4 hours
**Risk Level**: Medium
**Line Reduction**: ~1,800 lines

### Step 1.1: Merge case_study_2_euro_adoption.py into cs2_report.py
```bash
# Current structure:
# - cs2_report.py (341 lines) - just a wrapper
# - case_study_2_euro_adoption.py (2,162 lines) - actual implementation

# Target structure:
# - cs2_report.py (~700-800 lines) - fully consolidated with country parameter
```

**Actions**:
1. Copy all 14 functions from case_study_2_euro_adoption.py into cs2_report.py
2. Update functions to use the country parameter system already in place
3. Ensure data_type and ui_config parameters flow through all functions
4. Remove import statements calling the old file
5. Test each country configuration (Estonia, Latvia, Lithuania)

### Step 1.2: Archive case_study_2_euro_adoption.py
```bash
mv src/dashboard/case_study_2_euro_adoption.py \
   src/dashboard/archive_20241203/case_study_2_euro_adoption.py
```

### Step 1.3: Update main_app.py imports
- Remove any remaining references to case_study_2_euro_adoption
- Ensure all CS2 calls go through cs2_report.py

## Phase 2: Directory Cleanup (Priority: HIGH)
**Estimated Time**: 30 minutes
**Risk Level**: Low
**Line Reduction**: 0 (but improves organization)

### Step 2.1: Remove empty directories
```bash
# These directories are empty and no longer needed
rm -rf src/dashboard/pdf_reports
rm -rf src/dashboard/pdf_reports_outlier_adjusted
```

### Step 2.2: Archive remaining PDF generation files
```bash
# Move PDF generation files to archive (found 6 files)
mkdir -p src/dashboard/archive_20241203/pdf_generation
mv src/dashboard/pdf_reports/*.py \
   src/dashboard/archive_20241203/pdf_generation/
mv src/dashboard/pdf_reports_outlier_adjusted/*.py \
   src/dashboard/archive_20241203/pdf_generation/
```

### Step 2.3: Keep pdfs/ directory
```bash
# This contains generated PDFs - keep for reference
# src/dashboard/pdfs/ - KEEP (contains output files)
```

## Phase 3: Archive Reorganization (Priority: MEDIUM)
**Estimated Time**: 15 minutes
**Risk Level**: Very Low

### Step 3.1: Move archive to project root
```bash
# Create organized archive structure at project root
mkdir -p archive/dashboard_consolidation_20241203

# Move archive from dashboard subdirectory to project root
mv src/dashboard/archive_20241203/* \
   archive/dashboard_consolidation_20241203/

# Remove old archive location
rmdir src/dashboard/archive_20241203
```

### Step 3.2: Create archive manifest
```bash
# Generate manifest file listing all archived files
cd archive/dashboard_consolidation_20241203
find . -type f -name "*.py" | sort > ARCHIVE_MANIFEST.txt
echo "Total files: $(wc -l < ARCHIVE_MANIFEST.txt)" >> ARCHIVE_MANIFEST.txt
echo "Total lines: $(find . -name "*.py" | xargs wc -l | tail -1)" >> ARCHIVE_MANIFEST.txt
```

## Phase 4: Final Optimization Review (Priority: LOW)
**Estimated Time**: 2 hours
**Risk Level**: Low
**Line Reduction**: ~500-1000 lines

### Step 4.1: Review helper files for duplication
```bash
# Check for overlapping functionality
# - common_statistical_functions.py (300 lines)
# - cs3_complete_functions.py (288 lines)
```

### Step 4.2: Consider main_app.py reduction
- Review the 3,483 lines in main_app.py
- Look for repeated patterns that could be parameterized
- Consider extracting more functions to shared_utils

## Phase 5: Testing & Validation (Priority: CRITICAL)
**Estimated Time**: 1 hour
**Risk Level**: N/A (Testing phase)

### Step 5.1: Run test suites
```bash
# Run all test suites after each phase
python test_consolidated_dashboard.py
python test_runtime_errors.py
```

### Step 5.2: Manual testing checklist
- [ ] Launch dashboard: `streamlit run src/dashboard/main_app.py`
- [ ] Test CS1: Full and winsorized modes
- [ ] Test CS2: All three countries (Estonia, Latvia, Lithuania)
- [ ] Test CS3: Full and crisis-excluded modes
- [ ] Test CS4: Verify statistical tables load
- [ ] Test CS5: Check capital controls and regime analysis
- [ ] Test PDF export for at least one case study

### Step 5.3: Statistical verification
- [ ] Run one analysis and compare output with pre-consolidation results
- [ ] Verify F-test values match (tolerance < 0.0001)
- [ ] Check that visualizations render correctly

## Implementation Order & Timeline

### Day 1 (4-5 hours)
1. **Phase 1**: Complete CS2 Consolidation
2. **Phase 5.1**: Run automated tests
3. **Phase 5.2**: Quick manual testing of CS2

### Day 2 (1-2 hours)
1. **Phase 2**: Directory Cleanup
2. **Phase 3**: Archive Reorganization
3. **Phase 5**: Complete testing suite

### Optional Future Work (2-3 hours)
1. **Phase 4**: Final optimization review
2. Additional documentation updates

## Success Metrics

### Target Achievements
- [ ] Total active lines: < 13,000 (from current 14,553)
- [ ] CS2 fully consolidated into single file
- [ ] All empty directories removed
- [ ] Archive properly organized at project root
- [ ] All tests passing (10/10)
- [ ] Manual testing checklist complete

### Final Structure Goal
```
src/dashboard/
├── reports/
│   ├── cs1_report.py (~3,300 lines)
│   ├── cs2_report.py (~800 lines)  ← TARGET
│   ├── cs3_report.py (~2,000 lines)
│   ├── cs4_report.py (~1,400 lines)
│   └── cs5_report.py (~700 lines)
├── main_app.py (~3,500 lines)
├── config/
├── shared_utils/
└── pdfs/ (keep for outputs)

archive/dashboard_consolidation_20241203/
└── [all original 27,587 lines properly organized]
```

## Risk Mitigation

### Before Starting
1. Create a backup branch: `git checkout -b consolidation-completion`
2. Document current working state
3. Ensure test suites are passing in current state

### If Issues Arise
1. CS2 consolidation breaks functionality:
   - Revert cs2_report.py to wrapper approach
   - Keep case_study_2_euro_adoption.py active
   - Document why full consolidation wasn't possible

2. Test failures after changes:
   - Use git diff to review exact changes
   - Run tests after each function migration
   - Consider partial consolidation if full merge fails

## Notes

- The primary goal is functionality over line count
- The 65% reduction already achieved is substantial
- CS2 consolidation is the main remaining task
- All other optimizations are optional

## Commands Summary

```bash
# Quick implementation commands for copy-paste

# 1. Backup current state
git add -A && git commit -m "Pre-CS2 consolidation checkpoint"

# 2. After CS2 consolidation
python test_consolidated_dashboard.py
python test_runtime_errors.py

# 3. Directory cleanup
rm -rf src/dashboard/pdf_reports
rm -rf src/dashboard/pdf_reports_outlier_adjusted

# 4. Archive reorganization
mkdir -p archive/dashboard_consolidation_20241203
mv src/dashboard/archive_20241203/* archive/dashboard_consolidation_20241203/
rmdir src/dashboard/archive_20241203

# 5. Final testing
cd src/dashboard && streamlit run main_app.py
```

## Completion Checklist

- [ ] CS2 fully consolidated
- [ ] case_study_2_euro_adoption.py archived
- [ ] Empty directories removed
- [ ] Archive moved to project root
- [ ] All tests passing
- [ ] Manual testing complete
- [ ] Line count < 13,000
- [ ] Documentation updated
- [ ] Git commit created