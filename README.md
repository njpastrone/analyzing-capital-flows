# Capital Flows Research Analysis

A comprehensive research platform for analyzing capital flow volatility across different monetary regimes, with a focus on small open economies and currency union decisions.

App hosted online [here](https://analyzing-capital-flows.streamlit.app/).

## Overview

This project examines how monetary policy frameworks, currency unions, and external shocks affect financial stability through rigorous statistical analysis of IMF Balance of Payments data. The research provides evidence-based insights for monetary policy and currency union decisions.

### Key Research Questions

1. **How does capital flow volatility vary across different monetary regimes?**
2. **What are the effects of Euro adoption on capital flow stability?**
3. **How do global financial crises affect capital flow patterns differently across countries?**
4. **What do volatility patterns suggest for currency union and monetary policy decisions?**

### Research Findings

- **Iceland vs Eurozone**: Iceland shows significantly higher capital flow volatility (10/13 indicators at 5% significance level), suggesting Euro adoption could reduce financial volatility
- **Baltic Euro Adoption**: Analysis of Estonia (2011), Latvia (2014), and Lithuania (2015) provides evidence on pre/post adoption volatility changes
- **Crisis Impact**: Separate analysis tracks effects of Global Financial Crisis (2008-2010) and COVID-19 pandemic (2020-2022)

## Quick Start

### Prerequisites

- **R Environment**: RStudio with Quarto support
- **Python 3.8+**: For dashboard and advanced processing
- **Git**: For version control

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd analyzing-capital-flows
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install R dependencies**:
   ```r
   # Open R/RStudio and run:
   install.packages(c("tidyverse", "readr", "stringr", "ggplot2", "knitr", "gridExtra"))
   ```

### Running the Analysis

**Interactive Dashboard** (Recommended):
```bash
cd src/dashboard/
streamlit run main_app.py
```
*Features: 11-tab comprehensive interface with all case studies integrated*

**Individual Case Studies (Full Reports)**:
```bash
cd src/dashboard/

# Case Study 1: Iceland vs Eurozone
streamlit run full_reports/cs1_report_app.py

# Case Study 2: Estonia, Latvia, Lithuania
streamlit run full_reports/cs2_estonia_report_app.py
streamlit run full_reports/cs2_latvia_report_app.py
streamlit run full_reports/cs2_lithuania_report_app.py

# Case Study 3: Small Open Economies
streamlit run full_reports/cs3_report_app.py

# Case Study 4: Statistical Analysis Framework
streamlit run full_reports/cs4_report_app.py

# Case Study 5: Capital Controls & Exchange Rate Regimes
streamlit run full_reports/cs5_report_app.py
```

**Outlier-Adjusted Reports** (Robust Analysis):
```bash
cd src/dashboard/

# Run outlier-adjusted versions using winsorized data
streamlit run outlier_adjusted_reports/cs1_report_outlier_adjusted.py
streamlit run outlier_adjusted_reports/cs2_estonia_report_outlier_adjusted.py
streamlit run outlier_adjusted_reports/cs3_report_outlier_adjusted.py
streamlit run outlier_adjusted_reports/cs4_report_outlier_adjusted.py
streamlit run outlier_adjusted_reports/cs5_report_outlier_adjusted.py
```

**R/Quarto Data Processing**:
```bash
# Open in RStudio and render
quarto render "updated_data/Cleaning_All_Datasets.qmd"
```

## Project Structure

```
analyzing-capital-flows/
├── src/                           # Source code and analysis modules
│   ├── core/                      # Python statistical analysis frameworks
│   │   ├── cs4_statistical_analysis.py  # Advanced statistical testing (F-tests, AR(4), RMSE)
│   │   ├── data_loader.py         # Data loading utilities
│   │   ├── robust_analysis_report_generator.py  # Robust analysis framework
│   │   ├── winsorized_data_loader.py  # Outlier-adjusted data handling
│   │   └── sensitivity_analysis_framework.py   # Sensitivity testing
│   ├── dashboard/                 # Streamlit web applications (MAIN INTERFACE)
│   │   ├── main_app.py          # Multi-tab master dashboard (11 tabs)
│   │   ├── case_study_2_euro_adoption.py  # CS2 Master: Baltic Euro Adoption
│   │   ├── full_reports/        # Standalone reports for PDF export
│   │   │   ├── cs1_report_app.py        # CS1: Iceland vs Eurozone
│   │   │   ├── cs2_[country]_report_app.py  # Individual Baltic country reports
│   │   │   ├── cs3_report_app.py        # CS3: Small Open Economies
│   │   │   ├── cs4_report_app.py        # CS4: Statistical Analysis Framework
│   │   │   └── cs5_report_app.py        # CS5: Capital Controls & Exchange Rate Regimes
│   │   ├── outlier_adjusted_reports/    # Winsorized data analysis versions
│   │   │   ├── cs1_report_outlier_adjusted.py  # CS1: Outlier-adjusted analysis
│   │   │   ├── case_study_2_euro_adoption_outlier_adjusted.py  # CS2: Winsorized master
│   │   │   ├── cs2_[country]_report_outlier_adjusted.py  # CS2: Individual outlier-adjusted
│   │   │   ├── cs3_report_outlier_adjusted.py  # CS3: Outlier-adjusted analysis
│   │   │   ├── cs4_report_outlier_adjusted.py  # CS4: Outlier-adjusted analysis
│   │   │   └── cs5_report_outlier_adjusted.py  # CS5: Outlier-adjusted analysis
│   │   └── pdfs/               # Generated PDF reports (full and outlier-adjusted)
│   └── case_study_one/           # Legacy notebooks and early analysis
├── updated_data/                  # ACTIVE DATA PIPELINE (R-based cleaning)
│   ├── Clean/                     # Python-ready processed datasets
│   │   ├── comprehensive_df_PGDP_labeled.csv  # Master dataset
│   │   ├── comprehensive_df_PGDP_labeled_winsorized.csv  # Outlier-adjusted master dataset
│   │   ├── CS4_Statistical_Modeling/          # CS4 advanced analysis data
│   │   ├── CS4_Statistical_Modeling_winsorized/  # CS4 outlier-adjusted data
│   │   ├── CS5_Capital_Controls/               # Capital controls correlation data
│   │   ├── CS5_Capital_Controls_winsorized/    # CS5 outlier-adjusted data
│   │   ├── CS5_Regime_Analysis/                # Exchange rate regime data
│   │   └── CS5_Regime_Analysis_winsorized/     # CS5 outlier-adjusted regime data
│   ├── Raw/                       # Raw IMF API downloads
│   ├── Metadata/                  # Data definitions and sources
│   ├── Other Data (Not IMF)/      # External data sources
│   └── [R/Quarto cleaning scripts]  # Data processing pipeline
├── tests/                        # Comprehensive test suite (108 tests)
├── data/                         # Legacy/deprecated data folder  
├── output/                       # Generated visualizations and reports
├── analyzing-capital-flows.Rproj # RStudio project configuration
├── requirements.txt              # Python dependencies
├── CLAUDE.md                     # Claude Code interaction guide (detailed)
└── README.md                     # This file
```

## Key Modules and Scripts

### Core Analysis Modules (`/src/core/`)

- **`cs4_statistical_analysis.py`**: Advanced statistical testing framework (F-tests, AR(4), RMSE analysis)
- **`data_loader.py`**: Comprehensive data loading utilities with robust path handling
- **`robust_analysis_report_generator.py`**: Automated report generation for statistical robustness
- **`winsorized_data_loader.py`**: Outlier-adjusted data handling and processing
- **`sensitivity_analysis_framework.py`**: Sensitivity testing and validation framework

### Dashboard Applications (`/src/dashboard/`)

- **`main_app.py`**: 11-tab master dashboard integrating all case studies
- **`case_study_2_euro_adoption.py`**: CS2 master analysis for Baltic Euro adoption
- **`full_reports/`**: Standalone report applications optimized for PDF export
  - Individual case study reports (CS1-CS5) with professional formatting
- **`outlier_adjusted_reports/`**: Complete outlier-adjusted analysis suite
  - Parallel structure providing robust analysis using winsorized data

### Data Processing Pipeline (`/updated_data/`)

- **`Cleaning_All_Datasets.qmd`**: Master R/Quarto data cleaning and processing script
- **`winsorize_datasets.R`**: Systematic outlier treatment and winsorization
- **Specialized subdirectories**: CS4 statistical modeling data, CS5 external data integration

## Case Studies

### ✅ Case Study 1: Iceland vs Eurozone (1999-2024)
**Question**: How would Euro adoption affect Iceland's capital flow volatility?

- **Methodology**: Cross-sectional comparison of capital flow volatility using F-tests
- **Key Finding**: Iceland shows significantly higher volatility (10/13 indicators at 5% level)
- **Policy Implication**: Euro adoption could substantially reduce Iceland's financial volatility
- **Features**: Professional boxplots, time series visualizations, export functionality
- **Status**: ✅ Complete production implementation

### ✅ Case Study 2: Baltic Euro Adoption
**Question**: How did Euro adoption affect capital flow volatility in Baltic countries?

- **Countries**: Estonia (2011), Latvia (2014), Lithuania (2015)
- **Methodology**: Before/after temporal volatility analysis using F-tests
- **Versions**:
  - **Full Series**: Complete time windows with asymmetric periods
  - **Crisis-Excluded**: Removes Global Financial Crisis (2008-2010) and COVID-19 (2020-2022) periods
- **Features**: Individual country reports + master comparative analysis
- **Status**: ✅ Complete with dual analysis versions

### ✅ Case Study 3: Small Open Economies
**Question**: How does Iceland's capital flow volatility compare to similar small open economies?

- **Countries Analyzed**: Iceland vs 6 comparable small open economies (Aruba, Bahamas, Brunei Darussalam, Malta, Mauritius, Seychelles)
- **Methodology**: Size-adjusted volatility analysis beyond currency union effects
- **Data Limitation**: Bermuda excluded due to missing GDP data (required for % GDP normalization)
- **Focus**: Understanding volatility patterns in small open economies
- **Status**: ✅ Complete implementation

### ✅ Case Study 4: Statistical Analysis Framework
**Question**: What advanced statistical methods reveal about capital flow patterns?

- **Methodology**: 
  - **F-tests**: Variance equality testing with significance stars
  - **AR(4) Models**: Impulse response half-life calculations
  - **RMSE Analysis**: Rolling prediction methodology
- **Data**: 6 indicators (including portfolio investment disaggregation)
- **Comparators**: Eurozone, Small Open Economies, Baltics (weighted & simple averages)
- **Features**: Professional table generation with color coding, ACF analysis
- **Status**: ✅ Complete with advanced methodologies

### ✅ Case Study 5: Capital Controls & Exchange Rate Regimes
**Question**: How do capital controls and exchange rate regimes affect capital flow volatility?

- **Methodology**: 
  - **Capital Controls Analysis (1999-2017)**: Correlation between restrictions and volatility
  - **Exchange Rate Regime Analysis (1999-2019)**: 6-regime classification system
- **External Data Sources**:
  - Fernández et al. (2016) Capital Control Measures Database
  - Ilzetzki, Reinhart, and Rogoff (2019) Exchange Rate Classification
- **Features**: Iceland-highlighted scatter plots, F-test regime comparison table
- **Status**: ✅ Complete implementation

## ✅ Winsorized Analysis Implementation
**Robust Outlier-Adjusted Framework**

- **Methodology**: 5th-95th percentile capping for outlier mitigation
- **Dual Analysis**: All case studies available in both full and outlier-adjusted versions
- **Data Pipeline**: `comprehensive_df_PGDP_labeled_winsorized.csv` and specialized case study datasets
- **R Processing**: `winsorize_datasets.R` for systematic outlier treatment
- **Features**: Complete outlier-adjusted report suite, methodology documentation, export capabilities
- **Quality Assurance**: Comparison between full and winsorized results for statistical robustness
- **Status**: ✅ Complete robust analysis framework

## Data Sources

- **IMF Balance of Payments Statistics**: Quarterly capital flow data by instrument and sector
- **IMF World Economic Outlook**: Annual GDP data for normalization
- **Coverage**: 1999-2024 for developed economies, 2005-2024 for emerging markets
- **Normalization**: All flows converted to "% of GDP (annualized)" for cross-country comparison

## Methodology

### Statistical Framework
- **Primary Test**: F-tests for equality of variances between groups/periods
- **Significance Levels**: 0.1%, 1%, 5%, 10% with multiple testing awareness
- **Effect Sizes**: Cohen's d and Hedges' g for practical significance assessment
- **Robustness**: Crisis period exclusion and outlier sensitivity analysis

### Data Processing Pipeline
1. **Import**: Raw IMF datasets with metadata validation
2. **Clean**: Extract indicators, handle missing values, format time periods
3. **Transform**: Convert to wide format, join BOP and GDP data
4. **Normalize**: Convert to % of GDP, annualize quarterly data
5. **Group**: Create analytical country groups and time periods
6. **Analyze**: Generate descriptive statistics and hypothesis tests

## Key Features

### Interactive Dashboards
- **11-Tab Master Dashboard**: Comprehensive interface integrating all case studies
- **Dual Analysis Modes**: Both full and outlier-adjusted versions for statistical robustness
- **Smart Loading Feedback**: Operation-specific spinners with informative progress messages
- **Prioritized User Experience**: Longest-running operations (CS1-3) get detailed loading feedback
- **Real-time Analysis**: Dynamic parameter adjustment with instant results
- **Statistical Rigor**: Proper hypothesis testing with multiple significance levels
- **Export Capabilities**: Download results, visualizations, PDF reports, and HTML exports
- **Crisis Analysis**: Toggle between full series and crisis-excluded versions

### Advanced Statistical Framework
- **F-tests**: Variance equality testing with significance stars
- **AR(4) Modeling**: Impulse response analysis and half-life calculations
- **RMSE Analysis**: Rolling prediction methodology for forecasting assessment
- **Winsorization**: 5th-95th percentile outlier treatment for robust analysis
- **Multiple Comparisons**: Comprehensive testing across different country groups

### Visualization Suite
- **Professional Quality**: Publication-ready charts optimized for PDF export
- **Time Series Analysis**: Capital flow trends with policy regime indicators
- **Statistical Comparisons**: Boxplots, distribution comparisons, and correlation plots
- **Interactive Charts**: Plotly-based exploration with zoom and filtering capabilities
- **Color-blind Safe**: Consistent professional palette across all visualizations

### Robust Data Pipeline
- **Comprehensive Testing**: 108 automated tests ensuring data integrity
- **Dual Data Access**: Both full and winsorized datasets through consistent API
- **R/Quarto Integration**: Advanced data cleaning and preprocessing pipeline
- **External Data Integration**: Capital controls and exchange rate regime databases
- **Quality Assurance**: Systematic validation and comparison frameworks

## Development

### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-analysis`
3. Make changes with appropriate tests and documentation
4. Submit a pull request with clear description

### Code Style
- **R Code**: tidyverse conventions with 2-space indentation
- **Python Code**: PEP 8 style with comprehensive docstrings
- **Documentation**: Update relevant docs for any functionality changes
- **Testing**: Include data validation and statistical test verification

### Dependencies

**Python Requirements**:
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
openpyxl>=3.1.0
reportlab>=4.0.0
```

**R Requirements**:
```r
# Core data processing
tidyverse, readr, stringr, dplyr, ggplot2

# Document generation
quarto, knitr, gridExtra

# Statistical analysis and data cleaning
boot, psych
```

## Academic Context

This research contributes to the literature on:
- **Optimal Currency Areas**: Empirical evidence for currency union decisions
- **Capital Flow Management**: Understanding volatility patterns across monetary regimes
- **Small Economy Macroeconomics**: Policy implications for small open economies
- **Financial Stability**: Role of monetary frameworks in external stability

### Publications and Presentations
- Research findings suitable for academic publication in international economics journals
- Dashboard provides presentation-ready visualizations for policy audiences
- Methodology designed to meet rigorous peer-review standards

## Support and Documentation

- **Technical Issues**: Review `CLAUDE.md` for detailed development guidance
- **Statistical Methods**: Consult methodology sections in dashboard applications
- **Data Questions**: Reference metadata files and data source documentation
- **Collaboration**: Contact repository maintainers for research collaboration opportunities

## License

This project is developed for academic research purposes. Please cite appropriately if using in your research.

---

**Repository**: Capital Flows Research Analysis  
**Maintainer**: Nicolo Pastrone (Research Assistant)  
**Last Updated**: August 2025  
**Version**: 3.1 (enhanced user experience with comprehensive spinner loading system)

### Recent Updates (v3.1)
- ✅ Complete 5-case-study framework (CS1-CS5)
- ✅ Winsorized analysis implementation with dual data pipeline
- ✅ Advanced statistical methods (AR(4), RMSE, comprehensive F-testing)
- ✅ Professional report structure with PDF export optimization
- ✅ External data integration (capital controls, exchange rate regimes)
- ✅ Comprehensive test suite (108 automated tests)
- ✅ Updated folder structure with full_reports/ and outlier_adjusted_reports/
- ✅ Enhanced documentation and methodology validation
- ✅ **NEW**: Comprehensive spinner loading feedback system with operation-specific progress indicators
- ✅ **NEW**: Smart spinner utilities with informative messages for all CS1-5 case studies
- ✅ **NEW**: Prioritized loading feedback for longest-running operations (CS1-3)

### Project Status: **PRODUCTION READY**
All major functionality operational with comprehensive QA validation and academic-quality statistical framework.
