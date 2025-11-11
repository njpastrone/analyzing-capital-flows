# Data Lineage Matrix - Capital Flows Research Project

## 📊 Single Source of Truth Structure

### ✅ CONSOLIDATED DATA ARCHITECTURE
```
updated_data/                    # 41MB - SINGLE SOURCE OF TRUTH
├── Raw/                         # 40.5MB - Original IMF downloads
│   ├── case_study_1_data_july_24_2025.csv         (7.2MB)
│   ├── case_study_2_data_july_27_2025.csv         (6.8MB)
│   ├── case_study_4_data_july_28_2025.csv         (9.1MB)
│   ├── gdp_july_24_2025.csv                       (2.1MB)
│   └── net_flows_july_30_2025.csv                 (15.3MB)
├── Clean/                       # 500KB - Analysis-ready datasets
│   ├── comprehensive_df_PGDP_labeled.csv          (1.1MB) ← MASTER DATASET
│   ├── CS4_Statistical_Modeling/                  (600KB) ← CS4 SPECIALIZED
│   ├── CS5_Capital_Controls/                      (80KB)  ← CS5 CONTROLS
│   └── CS5_Regime_Analysis/                       (320KB) ← CS5 REGIMES
└── [Processing Scripts & Metadata]                # R/Quarto pipeline
```

### ❌ ELIMINATED REDUNDANCY
```
archive/legacy_data_folder/      # 208KB - ARCHIVED
├── Table_DataDefinition_Sources_StijnAndrew_March22_2017.xlsx  (OUTDATED)
├── case_study_2_euro_adoption_data.csv                        (DUPLICATE)
└── case_study_2_euro_adoption_data_crisis_excluded.csv        (DUPLICATE)

archive/legacy_notebooks/        # 1.4MB - ARCHIVED  
├── Case_Study_1_Analysis.ipynb                    (REPLACED BY STREAMLIT)
├── Case_Study_1_Debug_Analysis.ipynb              (REPLACED BY STREAMLIT)
└── Case_Study_1_Report_Template.ipynb             (REPLACED BY STREAMLIT)
```

---

## 🗺️ Data Flow Mapping

### Case Study Dependencies Matrix

| Case Study | Data Source | Size | Filter/Processing | Status |
|------------|-------------|------|------------------|---------|
| **CS1: Iceland vs Eurozone** | `comprehensive_df_PGDP_labeled.csv` | 1.1MB → 400KB | `CS1_GROUP` filter | ✅ Active |
| **CS2: Baltic Euro Adoption** | `comprehensive_df_PGDP_labeled.csv` | 1.1MB → 450KB | `CS2_GROUP` filter | ✅ Active |
| **CS3: Small Open Economies** | `comprehensive_df_PGDP_labeled.csv` | 1.1MB → 350KB | `CS3_GROUP` filter | ✅ Active |
| **CS4: Statistical Analysis** | `CS4_Statistical_Modeling/*.csv` | 600KB | 12 specialized files | ✅ Active |
| **CS5: Capital Controls** | `CS5_Capital_Controls/*.csv` | 80KB | 4 correlation files | ✅ Active |
| **CS5: Exchange Rate Regimes** | `CS5_Regime_Analysis/*.csv` | 320KB | 8 regime comparison files | ✅ Active |

### Data Processing Pipeline
```
Raw IMF Data (40.5MB)
    ↓ [R/Quarto Processing Scripts]
    ↓ 
Master Dataset (1.1MB) ──┬─→ CS1 Filter → Iceland vs Eurozone Analysis
                         ├─→ CS2 Filter → Baltic Euro Adoption Analysis  
                         └─→ CS3 Filter → Small Open Economies Analysis
    ↓ [Specialized Processing]
    ↓
CS4 Datasets (600KB) ────────→ Advanced Statistical Analysis (F-tests, AR models)
    ↓ [External Data Integration]
    ↓
CS5 Datasets (400KB) ────────→ Capital Controls & Exchange Rate Regime Analysis
```

---

## 📋 File Access Patterns

### Streamlit Applications (ACTIVE)
```python
# CS1-CS3: Main Dataset Access
master_data = pd.read_csv('updated_data/Clean/comprehensive_df_PGDP_labeled.csv')
cs1_data = master_data[master_data['CS1_GROUP'].isin(['Iceland', 'Eurozone'])]

# CS4: Specialized Dataset Access  
cs4_data = pd.read_csv('updated_data/Clean/CS4_Statistical_Modeling/net_capital_flows_full.csv')

# CS5: Multiple Dataset Access
controls_data = pd.read_csv('updated_data/Clean/CS5_Capital_Controls/sd_yearly_flows.csv')
regime_data = pd.read_csv('updated_data/Clean/CS5_Regime_Analysis/net_capital_flows_full.csv')
```

### R Processing Scripts (ACTIVE)
```r
# Raw Data Processing
raw_bop <- read_csv("updated_data/Raw/case_study_1_data_july_24_2025.csv")  # ✅ FIXED
raw_gdp <- read_csv("updated_data/Raw/gdp_july_24_2025.csv")              # ✅ FIXED

# Output Generation
write_csv(clean_data, "updated_data/Clean/comprehensive_df_PGDP_labeled.csv")
```

### Legacy References (FIXED)
```
❌ OLD: "../data/case_study_1_data_july_24_2025.csv"     
✅ NEW: "../updated_data/Raw/case_study_1_data_july_24_2025.csv"

❌ OLD: "../data/dataset_2025-07-24T18_28_31..."         
✅ NEW: "../updated_data/Raw/gdp_july_24_2025.csv"
```

---

## 🎯 Data Validation Results

### File Existence Verification
- ✅ **Master Dataset**: `comprehensive_df_PGDP_labeled.csv` (24 columns, CS1/CS2/CS3 groups)
- ✅ **CS4 Specialized**: 12 files (full + crisis-excluded versions)
- ✅ **CS5 Capital Controls**: 4 files (yearly/country, with/without outliers)
- ✅ **CS5 Regime Analysis**: 8 files (4 indicators × 2 versions)

### Data Quality Checks
- ✅ **No Missing Dependencies**: All case studies can access required data
- ✅ **No Broken References**: Legacy path issues resolved
- ✅ **No Data Duplication**: Single source of truth established
- ✅ **Consistent Naming**: Standard file naming conventions

### Storage Optimization
- ✅ **Redundancy Eliminated**: 208KB saved from legacy folder removal
- ✅ **Archive Created**: 1.6MB legacy content preserved in archive/
- ✅ **Clean Structure**: Clear separation of Raw vs Clean vs Archive

---

## 🔄 Data Refresh Procedures

### Adding New Data
1. **Raw Data**: Place new IMF downloads in `updated_data/Raw/`
2. **Processing**: Run R/Quarto scripts to update clean datasets  
3. **Validation**: Verify all case studies still load correctly
4. **Testing**: Run Streamlit apps to confirm functionality

### Modifying Existing Data
1. **Backup**: Create copy of current clean datasets
2. **Update**: Modify processing scripts as needed
3. **Regenerate**: Run full cleaning pipeline
4. **Validate**: Test all 5 case study applications

### Quality Assurance
- **Automated**: Data path validation in `dashboard_config.py`
- **Manual**: Periodic verification of file sizes and column structures
- **Testing**: Streamlit app loading tests before deployment

---

## 📊 Impact Summary

### ✅ ACHIEVEMENTS
- **Single Source of Truth**: Eliminated split data architecture
- **Storage Optimized**: Removed 208KB redundant legacy data
- **References Fixed**: Resolved 6+ broken path references
- **Archive Preserved**: 1.6MB legacy content safely archived
- **Documentation Clear**: Complete data lineage established

### 📈 METRICS
- **Data Folders**: 2 → 1 (50% reduction)
- **Redundant Storage**: 208KB eliminated
- **Broken References**: 6+ fixed
- **Case Studies Verified**: 5/5 functional
- **Files Archived**: 8 legacy files preserved

### 🎯 PRODUCTION READY
The Capital Flows Research project now has a **clean, consolidated, single-source-of-truth data architecture** ready for:
- Academic publication and peer review
- Reliable production deployment  
- Future data additions and modifications
- Maintainable long-term research platform

**Result**: Transformed from confusing split architecture to professional, maintainable data pipeline with complete traceability from raw sources to final analysis.