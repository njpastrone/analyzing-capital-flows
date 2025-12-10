# Comprehensive Codebase Audit Report
**Date**: December 4, 2024
**Current Size**: 21,899 lines (active Python code)
**Dashboard Size**: 18,765 lines in src/

## Executive Summary

The codebase faces three primary challenges inflating its size:

1. **Monolithic Dashboard Architecture** (3,473 lines in main_app.py alone)
2. **Duplicated Statistical Implementations** (3,111 lines in core modules with overlapping functionality)
3. **Repetitive Report Generation Code** (~8,800 lines across 7 case study reports)

## 1. Size Distribution Analysis

### Top 10 Largest Files
| File | Lines | % of Total | Primary Issue |
|------|-------|------------|---------------|
| main_app.py | 3,473 | 15.9% | Monolithic controller with 44 functions |
| cs1_report.py | 3,333 | 15.2% | Complete case study implementation |
| cs3_report.py | 2,020 | 9.2% | Full analysis + visualization |
| cs4_report.py | 1,381 | 6.3% | Statistical framework duplication |
| sensitivity_analysis_framework.py | 804 | 3.7% | Unused robust analysis code |
| robust_analysis_report_generator.py | 773 | 3.5% | Unused report generation |
| cs5_report.py | 709 | 3.2% | Policy regime analysis |
| cs4_statistical_analysis.py | 587 | 2.7% | Core statistical functions |
| data_loading.py | 522 | 2.4% | Data loading utilities |
| CS2 reports (3 files) | 1,452 | 6.6% | Country-specific implementations |

**Key Finding**: Top 10 files account for 68.7% of the codebase

## 2. Feature Inflation Analysis

### 2.1 Unused/Dead Code (~1,942 lines)
- **sensitivity_analysis_framework.py** (804 lines) - Not referenced in active dashboard
- **robust_analysis_report_generator.py** (773 lines) - Legacy report generation
- **winsorized_data_loader.py** (365 lines) - Only imported by robust_analysis_report_generator (also unused)
- All three files in src/core/ form a chain of unused dependencies

**IMPORTANT CLARIFICATION**: The winsorized/outlier-adjusted analysis IS working in the reports, but through:
  - Data type parameters ('full' vs 'winsorized') in each report file
  - Pre-processed winsorized CSV files from the R pipeline
  - NOT through these unused "robust analysis" modules
  - The outlier-adjusted functionality is embedded in the report files themselves

### 2.2 Duplication Patterns

#### Statistical Function Duplication (~600 lines)
Multiple implementations of same statistical operations:
- **common_statistical_functions.py** (300 lines)
- **cs3_complete_functions.py** (288 lines)
- Overlapping functionality with cs4_statistical_analysis.py

#### Data Loading Redundancy
- 10 different files implement `load.*data` functions
- Each case study has its own data loading logic
- No centralized data access layer

#### Visualization Code Pattern (~2,000 lines)
- 40 chart display instances across reports
- Each report reimplements similar chart creation
- No shared visualization components

### 2.3 Monolithic Architecture Issues

#### main_app.py Problems (3,473 lines)
- 44 function definitions
- Manages 11 tabs with repetitive code
- Each tab setup ~200-300 lines of similar structure
- Inline CSS and styling repeated throughout

#### Report File Pattern
Each case study report (CS1-CS5) contains:
- Complete data loading (~200 lines)
- Full analysis implementation (~500-800 lines)
- Visualization code (~300-500 lines)
- Export functionality (~100 lines)
- UI layout code (~200-300 lines)

## 3. Biggest Challenges for Cleanup

### Challenge 1: Tightly Coupled Architecture
**Issue**: Each case study is a self-contained monolith
**Impact**: Cannot extract shared components without breaking functionality
**Solution Required**: Complete architectural refactor to separate concerns

### Challenge 2: Streamlit-Specific Code Interweaving
**Issue**: Business logic mixed with Streamlit UI code throughout
**Impact**: ~40% of code is UI-specific, making extraction difficult
**Example**: Statistical calculations directly output to `st.write()` instead of returning values

### Challenge 3: Data Pipeline Redundancy
**Issue**: Multiple data loading strategies across modules
**Impact**: ~1,000 lines of redundant data access code
**Files Affected**:
- data_loader.py (131 lines)
- winsorized_data_loader.py (365 lines)
- data_loading.py (522 lines)
- Plus inline loading in each report

### Challenge 4: Lack of Component Abstraction
**Issue**: No reusable components for common patterns
**Missing Abstractions**:
- Chart generation templates
- Statistical test runners
- Report section generators
- Data transformation pipelines

## 4. Feature-Specific Inflation

### 4.1 Outlier Adjustment Features (~2,000 lines)
- Parallel implementation for winsorized data
- Duplicates entire analysis pipeline
- Could be handled with parameter flags

### 4.2 PDF Export Functionality (~1,500 lines)
- Each report has custom PDF generation
- No shared PDF template system
- Repeated margin/styling configurations

### 4.3 Spinner/Loading Feedback (~500 lines)
- spinner_utils.py (286 lines)
- Plus inline spinner code throughout
- Could be 50 lines with proper abstraction

### 4.4 Configuration Sprawl (~600 lines)
- dashboard_config.py (250 lines)
- constants.py (332 lines)
- Plus inline configs in each module

## 5. Quantified Cleanup Potential

### Immediate Wins (Low Risk)
| Action | Lines Saved | Effort |
|--------|------------|--------|
| Remove dead core modules | 1,942 | 30 min |
| Consolidate statistical functions | 300 | 1 hour |
| Extract shared visualizations | 500 | 2 hours |
| **Total** | **2,742** | **3.5 hours** |

### Medium-Term Refactoring (Medium Risk)
| Action | Lines Saved | Effort |
|--------|------------|--------|
| Create data access layer | 800 | 4 hours |
| Abstract report components | 1,500 | 6 hours |
| Consolidate configuration | 400 | 2 hours |
| **Total** | **2,700** | **12 hours** |

### Long-Term Architecture (High Risk)
| Action | Lines Saved | Effort |
|--------|------------|--------|
| Separate UI from business logic | 3,000 | 20 hours |
| Implement component system | 2,000 | 15 hours |
| Create analysis pipeline framework | 1,500 | 10 hours |
| **Total** | **6,500** | **45 hours** |

## 6. Critical Observations

### 6.1 The 80/20 Problem
- 80% of functionality could be achieved with 20% of the code
- Current implementation prioritizes feature completeness over efficiency
- Each case study reimplements the entire stack

### 6.2 Academic vs Production Code
- Code written for research exploration, not production
- Prioritizes correctness and completeness over DRY principles
- Inline documentation and verbose implementations

### 6.3 Streamlit Lock-in
- Deep integration with Streamlit throughout
- Business logic not separable from UI
- Makes testing and refactoring extremely difficult

## 7. Recommendations

### Immediate Actions (This Week)
1. **Delete dead code**: Remove sensitivity_analysis_framework.py and robust_analysis_report_generator.py
2. **Consolidate helpers**: Merge statistical and data loading functions
3. **Document dependencies**: Create module dependency graph

### Short-Term Plan (This Month)
1. **Extract business logic**: Separate calculations from UI code
2. **Create shared components**: Build reusable visualization library
3. **Implement data layer**: Centralized data access with caching

### Long-Term Strategy (Next Quarter)
1. **Migrate to notebook pipeline**: As per RESEARCH_PIPELINE_PLAN.md
2. **Modular architecture**: Separate concerns properly
3. **Test coverage**: Add tests before major refactoring

## 8. Conclusion

The codebase's size is primarily inflated by:
1. **Architectural debt**: Monolithic, tightly-coupled design
2. **Duplication**: Same patterns implemented 5-7 times
3. **Dead code**: ~1,942 lines of unused functionality
4. **UI coupling**: Business logic intertwined with Streamlit

**Realistic cleanup potential**:
- **Quick wins**: 2,742 lines (12.5% reduction)
- **With refactoring**: 5,077 lines (23% reduction)
- **Complete overhaul**: 11,577 lines (53% reduction)

**Current blockers**:
- Tight Streamlit coupling makes extraction difficult
- Each case study is self-contained by design
- No test coverage to ensure refactoring safety

**Recommended approach**: Focus on the Research Pipeline Plan (notebooks) rather than refactoring existing code, as the technical debt is too deeply embedded for cost-effective cleanup.