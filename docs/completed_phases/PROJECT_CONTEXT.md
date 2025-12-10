# Project Context: Critical Information for Research Pipeline

**Created**: December 2024
**Last Updated**: December 4, 2024
**Purpose**: Preserve critical context about technical debt and verification requirements
**Status**: Dashboard Consolidation Complete → Research Pipeline Phase

---

## The Current Situation

### Codebase Statistics (Updated December 4, 2024)
- **Total Lines**: 19,506 lines of Python code (reduced from 47,000+)
- **Dashboard Code**: ~16,000 lines in src/dashboard/ (reduced from 43,443)
- **Archived Code**: 45,998 lines (dashboard consolidation + unused core modules)
- **Duplication Rate**: Reduced through consolidation
- **Duplication Pattern**: 4x for many components:
  - Original version
  - Outlier-adjusted version
  - PDF-specific version
  - Outlier-adjusted PDF version

### The Core Problem

The user needs to submit this code for academic research publication, which requires:
1. **Complete verifiability** - every calculation must be traceable
2. **Manual checking** - ability to verify all numbers are correct
3. **Transparency** - no black boxes or hidden calculations
4. **Standalone execution** - reviewers need to run and verify

However, the current codebase makes this nearly impossible:
- 47,000 lines is too much to manually review
- High duplication makes it unclear which version is canonical
- UI code obscures the actual calculations
- Dependencies between files are complex and undocumented

## Critical Insights from Analysis

### 1. The Real Data Flow
```
Raw IMF Data (5 CSVs)
    ↓
R/Quarto Cleaning (Cleaning_All_Datasets.qmd)
    ↓
Cleaned CSVs (updated_data/Clean/)
    ↓
Python Analysis (src/core/ and src/dashboard/)
    ↓
Results (displayed in Streamlit)
```

### 2. Code Distribution Breakdown
- **UI/Dashboard Code**: ~57% (Streamlit, HTML, CSS)
- **Visualization**: ~10% (Matplotlib, Plotly)
- **Core Statistics**: ~9% (actual calculations)
- **Data Loading**: ~5%
- **Duplicate Code**: ~24% (counted within above categories)

### 3. What's Actually Essential
The core statistical calculations are relatively simple:
- **F-tests**: Comparing variances between groups
- **Temporal analysis**: Before/after comparisons
- **AR(4) models**: Time series analysis
- **Correlations**: Relationships between variables

Most complexity comes from:
- UI/UX implementation
- PDF generation
- Interactive features
- Duplicate implementations

## Why Certain Approaches Are Dangerous

### ❌ DON'T: Try to "Refactor" or "Simplify"

**Why it's dangerous:**
1. **Hidden Dependencies**: With 4x duplication, versions may have diverged with important bug fixes
2. **Breaking Changes**: One small change could cascade into broken functionality
3. **Verification Nightmare**: How would you prove the "simplified" version is correct?
4. **Time Sink**: Could take weeks and introduce new bugs
5. **Research Integrity Risk**: Any error could invalidate published research

**User Quote**: "My concern is that this is really dangerous."

### ❌ DON'T: Create a Parallel "Minimal" Implementation

**Why it's dangerous:**
1. **Divergence Risk**: Two implementations might produce different results
2. **Which is Correct?**: If results differ, hard to know which is right
3. **Maintenance Burden**: Now you have two codebases to maintain
4. **Context Window Issues**: Adding 2,000+ lines fills up AI assistant context

### ❌ DON'T: Extract and Reorganize Code

**Initial Tempting Idea**: Extract ~7,000 lines of "core" code from 47,000

**Why it won't work:**
1. **Interdependencies**: Functions may depend on UI state or hidden globals
2. **Subtle Differences**: "Duplicate" code might have important variations
3. **Testing Gap**: No way to ensure extraction preserves functionality
4. **Academic Timeline**: User needs this for publication, not months from now

## The Approved Solution

### ✅ DO: Create Traceable Jupyter Notebooks

**Why this works:**
1. **No Code Changes**: Uses existing, working data and calculations
2. **Full Transparency**: Every step visible and documented
3. **Academic Standard**: Jupyter notebooks are expected in research
4. **Verifiable**: Can compare directly with dashboard results
5. **Manageable Scope**: ~500 lines per notebook, not 47,000 total

### Key Requirements (User-Specified)

1. **"We really have to trace all steps"**
   - Every calculation must be shown
   - Intermediate values displayed
   - Formulas documented

2. **"I think my code will eventually be reviewed"**
   - Must meet academic standards
   - Suitable for peer review
   - Reproducible by others

3. **"We really need standalone code I can verify and run"**
   - Independent execution
   - No Streamlit dependency
   - Clear inputs and outputs

4. **"It would be good if we had one notebook for each case study"**
   - CS1: Iceland vs Eurozone
   - CS2: Baltic Euro Adoption
   - CS3: Small Open Economies
   - CS4: Statistical Framework
   - CS5: Capital Controls & Regimes

## Critical Context for Future Sessions

### Remember These Points

1. **The code works** - Don't break it trying to "improve" it
2. **Duplication is complex** - Not simple copy-paste, versions have diverged
3. **User needs verification** - Not refactoring or optimization
4. **Academic stakes are high** - Errors could invalidate published research
5. **Time is limited** - Solution must be implementable in days, not weeks

### The User's Journey

1. Started with working analysis but massive technical debt
2. Discovered 43% code duplication across multiple versions
3. Considered dangerous refactoring approaches
4. Settled on creating traceable notebooks for verification
5. Needs to preserve all context for future work sessions

### Success Metrics

✅ Every calculation is traceable
✅ Results match existing dashboard
✅ Academic reviewers can understand and verify
✅ No modifications to working code
✅ Completed in reasonable timeframe (4 days)

## Guidelines for Future Development

### Before Making ANY Changes

1. **Read this document completely**
2. **Review TECHNICAL_DEBT.md** for duplication details
3. **Check RESEARCH_PIPELINE_PLAN.md** for approved approach
4. **Don't refactor existing code**
5. **Focus on verification, not optimization**

### If Asked About Simplification

**User**: "Can we simplify the codebase?"
**Response**: "The approved approach is to create verification notebooks as documented in RESEARCH_PIPELINE_PLAN.md. Refactoring the existing code is too risky given the academic publication requirements."

### If Context is Lost

Key files to review:
1. **PROJECT_CONTEXT.md** (this file) - Critical context
2. **RESEARCH_PIPELINE_PLAN.md** - Implementation plan
3. **TECHNICAL_DEBT.md** - Detailed duplication analysis
4. **CLAUDE.md** - Should reference these documents

## The Bottom Line

**Mission**: Create traceable, verifiable notebooks for academic review
**Method**: Extract calculations into transparent Jupyter notebooks
**Mandate**: Don't break working code by trying to "fix" it
**Timeline**: 4 days to implementation
**Success**: When every number can be traced and verified

---

**Critical Reminder**: The existing code, despite its flaws, produces correct results that have been validated. The goal is to make these results transparent and verifiable, not to rewrite or "improve" the system. Academic integrity depends on maintaining consistency with existing, validated results.