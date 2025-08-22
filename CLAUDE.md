# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **capital flows research analysis project** that examines capital flow volatility across different economies, time periods, and policy regimes. The project investigates how monetary policy frameworks, currency unions, exchange rate regimes, and capital controls affect financial stability in small open economies, with Iceland as the primary focus.

### Core Research Framework
1. **Cross-Sectional Analysis**: Iceland vs Eurozone volatility comparisons
2. **Temporal Analysis**: Before/after Euro adoption effects in Baltic countries  
3. **Small Open Economy Analysis**: Iceland vs comparable small economies
4. **Statistical Modeling**: Advanced time series and variance analysis
5. **Policy Regime Analysis**: Capital controls and exchange rate regime effects

## Current Project Structure

**Directory Organization:**
```
analyzing-capital-flows/
├── src/                           # Source code and analysis modules
│   ├── core/                      # Python statistical analysis frameworks
│   │   ├── cs4_statistical_analysis.py  # Advanced statistical testing (F-tests, AR(4), RMSE)
│   │   ├── data_loader.py         # Data loading utilities
│   │   ├── config.py             # Configuration management
│   │   └── [other analysis modules]
│   ├── dashboard/                 # Streamlit web applications (MAIN INTERFACE)
│   │   ├── main_app.py          # Multi-tab master dashboard (11 tabs)
│   │   ├── simple_report_app.py # Case Study 1: Iceland vs Eurozone
│   │   ├── case_study_2_euro_adoption.py  # CS2 Master: Baltic Euro Adoption
│   │   ├── cs2_[country]_report_app.py    # Individual Baltic country reports
│   │   ├── cs3_report_app.py    # CS3: Small Open Economies
│   │   ├── cs4_report_app.py    # CS4: Statistical Analysis Framework
│   │   └── cs5_report_app.py    # CS5: Capital Controls & Exchange Rate Regimes
│   └── case_study_one/           # Legacy notebooks and early analysis
├── updated_data/                  # ACTIVE DATA PIPELINE (R-based cleaning)
│   ├── Clean/                     # Python-ready processed datasets
│   │   ├── comprehensive_df_PGDP_labeled.csv  # Master dataset
│   │   ├── CS4_Statistical_Modeling/          # CS4 advanced analysis data
│   │   ├── CS5_Capital_Controls/               # Capital controls correlation data
│   │   └── CS5_Regime_Analysis/                # Exchange rate regime data
│   ├── Raw/                       # Raw IMF API downloads
│   ├── Metadata/                  # Data definitions and sources
│   ├── Other Data (Not IMF)/      # External data sources
│   └── [R/Quarto cleaning scripts]  # Data processing pipeline
├── data/                         # Legacy/deprecated data folder
└── output/                       # Generated visualizations and reports
```

## Case Study Implementation Status

### ✅ **Case Study 1: Iceland vs Eurozone (1999-2024)**
- **Status**: Complete production implementation
- **Methodology**: Cross-sectional volatility comparison using F-tests
- **Key Findings**: Iceland shows significantly higher volatility (10/13 indicators at 5% level)
- **Features**: 
  - Professional boxplots and time series visualizations
  - Comprehensive statistical testing framework
  - Export functionality for charts and data
- **File**: `simple_report_app.py`

### ✅ **Case Study 2: Baltic Euro Adoption**
- **Status**: Complete with dual analysis versions
- **Countries**: Estonia (2011), Latvia (2014), Lithuania (2015)
- **Methodology**: Before/after temporal volatility analysis
- **Versions**:
  - **Full Series**: Complete time windows with asymmetric periods
  - **Crisis-Excluded**: Removes 2008-2010 GFC and 2020-2022 COVID periods
- **Features**: Individual country reports + master comparative analysis
- **Files**: `case_study_2_euro_adoption.py`, `cs2_[country]_report_app.py`

### ✅ **Case Study 3: Small Open Economies**
- **Status**: Complete implementation
- **Methodology**: Iceland compared to 6 comparable small open economies
- **Countries Analyzed**: Aruba, Bahamas, Brunei Darussalam, Malta, Mauritius, Seychelles
- **Data Limitation**: Bermuda excluded due to missing GDP data (required for % GDP normalization)
- **Focus**: Size-adjusted volatility analysis beyond currency union effects
- **File**: `cs3_report_app.py`

### ✅ **Case Study 4: Statistical Analysis Framework**
- **Status**: Complete with advanced methodologies
- **Methodology**: 
  - **F-tests**: Variance equality testing with significance stars
  - **AR(4) Models**: Impulse response half-life calculations
  - **RMSE Analysis**: Rolling prediction methodology
- **Data**: 6 indicators (including portfolio investment disaggregation)
- **Comparators**: Eurozone, Small Open Economies, Baltics (weighted & simple averages)
- **Features**: Professional table generation with color coding, ACF analysis
- **Files**: `cs4_report_app.py`, `src/core/cs4_statistical_analysis.py`

### ✅ **Case Study 5: Capital Controls & Exchange Rate Regimes**
- **Status**: Complete implementation
- **Methodology**: 
  - **Capital Controls Analysis (1999-2017)**: Correlation between restrictions and volatility
  - **Exchange Rate Regime Analysis (1999-2019)**: 6-regime classification system
- **External Data Sources**:
  - Fernández et al. (2016) Capital Control Measures Database
  - Ilzetzki, Reinhart, and Rogoff (2019) Exchange Rate Classification
- **Features**: Iceland-highlighted scatter plots, F-test regime comparison table
- **File**: `cs5_report_app.py`

## Data Pipeline Architecture

### **R/Quarto Data Cleaning (Primary Pipeline)**
- **Location**: `updated_data/` directory
- **Master Script**: `Cleaning_All_Datasets.qmd`
- **Process**:
  1. **Raw Data Import**: IMF BOP and GDP data from quarterly API downloads
  2. **Format Detection**: Automatic detection of timeseries-per-row vs long format
  3. **Data Reshaping**: Pivot operations for consistent structure
  4. **Scale Corrections**: Convert to millions USD, apply proper scaling
  5. **Normalization**: Convert to % of GDP (quarterly × 4 / annual GDP × 100)
  6. **Group Labeling**: Create case study group identifiers
  7. **Output Generation**: Multiple formats (USD, % GDP, labeled versions)

### **Python Analysis Layer**
- **Data Loading**: Uses pre-cleaned CSV files from `updated_data/Clean/`
- **Statistical Processing**: Advanced analysis, hypothesis testing, visualization
- **No Raw Data Processing**: Python focuses purely on analysis, not cleaning

### **External Data Integration (CS5)**
- **Capital Controls**: Processed in dedicated R scripts
- **Exchange Rate Regimes**: Cleaned from original classification datasets
- **Integration**: Merged with main BOP data for correlation analysis

## Technical Stack

### **Core Dependencies**
- **Python**: 3.8+ required
- **Streamlit**: 1.28.0+ (Interactive dashboard framework)
- **Statistical**: pandas, numpy, scipy, statsmodels (advanced time series)
- **Visualization**: matplotlib, seaborn, plotly (static + interactive)
- **Data I/O**: openpyxl (Excel), reportlab (PDF generation)

### **R Environment**
- **Quarto**: Document generation and data processing
- **tidyverse**: Core data manipulation (dplyr, ggplot2, readr, stringr)
- **Data Processing**: Advanced reshaping, cleaning, and transformation

### **Professional Features**
- **PDF Export Optimization**: US Letter format with 0.75" margins
- **Colorblind-Safe Palette**: Consistent across all visualizations
- **Professional Tables**: HTML generation with significance color coding
- **Academic Standards**: Rigorous statistical methodology and presentation

## Running the Analysis

### **Main Dashboard (Recommended)**
```bash
cd src/dashboard/
streamlit run main_app.py
```
**Features**: 11-tab comprehensive interface with all case studies integrated

### **Individual Case Studies**
```bash
# Case Study 1: Iceland vs Eurozone
streamlit run simple_report_app.py

# Case Study 4: Statistical Analysis
streamlit run cs4_report_app.py

# Case Study 5: Capital Controls & Regimes
streamlit run cs5_report_app.py
```

### **Data Processing (R)**
```bash
# Open in RStudio and render
quarto render "updated_data/Cleaning_All_Datasets.qmd"
```

## Key Development Patterns

### **Data Access Pattern**
- **R scripts clean and process** raw IMF data
- **Python applications consume** pre-cleaned CSV files
- **No Python data cleaning** - analysis layer only
- **Consistent file paths** using `updated_data/Clean/` structure

### **Statistical Analysis Architecture**
1. **Data Loading**: Utility functions in `src/core/`
2. **Statistical Testing**: Dedicated analysis modules (e.g., CS4 framework)
3. **Visualization**: Professional matplotlib/plotly integration
4. **Export**: PDF-optimized charts and downloadable data

### **Streamlit App Structure**
- **Page Config**: Handled in main functions to avoid import conflicts
- **Styling**: Applied via functions, not module-level calls
- **Professional CSS**: Optimized for PDF export and academic presentation
- **Tab Organization**: Logical flow from overview to detailed analysis

## Data Sources and Quality

### **Primary Data**
- **IMF Balance of Payments**: Quarterly capital flow data (1999-2025)
- **IMF World Economic Outlook**: Annual GDP data for normalization
- **Coverage**: 20+ countries including Iceland, Eurozone, Baltic states, Small Open Economies

### **External Data (CS5)**
- **Capital Controls**: Fernández et al. database (1999-2017)
- **Exchange Rate Regimes**: Ilzetzki-Reinhart-Rogoff classification (1999-2019)

### **Data Quality Features**
- **Crisis Period Handling**: 2008-2010 GFC and 2020-2022 COVID exclusion options
- **Missing Data Management**: Graceful degradation with statistical adjustments
- **Outlier Detection**: Robust statistical methods with removal options
- **Validation**: Cross-verification with original sources

### **Known Data Limitations**
- **CS3 Bermuda Exclusion**: Bermuda has complete Balance of Payments data but missing GDP data in IMF World Economic Outlook database, preventing % GDP normalization. Automatically excluded from CS3 analysis (6 countries analyzed instead of 7 originally planned)
- **Non-IMF Member Territories**: Some offshore financial centers may have limited standardized reporting despite data availability
- **GDP Normalization Dependency**: All capital flows indicators require corresponding GDP data for cross-country comparability

## Working with the Codebase

### **Adding New Analysis**
1. **Data**: Add to appropriate `updated_data/Clean/` subfolder
2. **Analysis**: Create statistical functions in `src/core/`
3. **Visualization**: Build Streamlit app in `src/dashboard/`
4. **Integration**: Add to main dashboard tabs

### **Modifying Existing Case Studies**
- **Statistical Tests**: Modify analysis frameworks in `src/core/`
- **Visualizations**: Update chart generation functions
- **Data**: Re-run R cleaning scripts if source data changes

### **Common Tasks**
- **New Countries**: Add to country groupings in data cleaning scripts
- **Additional Indicators**: Extend BOP indicator processing in R scripts
- **Statistical Methods**: Add to core analysis modules
- **Export Formats**: Enhance visualization export capabilities

### **Critical Notes**
- **Always verify data compatibility** when adding new countries or periods
- **Maintain statistical assumptions** (variance equality, normality) for parametric tests
- **Keep methodological consistency** across case studies
- **Test PDF export compatibility** for new visualizations
- **Update data period documentation** when adding new time series

This project bridges rigorous academic research with practical policy insights, requiring attention to both methodological soundness and professional presentation standards.