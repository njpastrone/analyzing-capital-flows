# Capital Flows Research Analysis

A comprehensive research platform for analyzing capital flow volatility across different monetary regimes, with a focus on small open economies and currency union decisions.

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
streamlit run src/dashboard/main_app.py
```

**Individual Case Studies**:
```bash
# Case Study 1: Iceland vs Eurozone
streamlit run src/dashboard/simple_report_app.py

# Case Study 2: Baltic Euro Adoption
streamlit run src/dashboard/case_study_2_euro_adoption.py
```

**R/Quarto Analysis**:
```bash
# Open in RStudio or use command line
quarto render "src/Cleaning Case Study 1.qmd"
```

## Project Structure

```
analyzing-capital-flows/
├── src/                           # Source code and analysis
│   ├── core/                      # Core Python modules
│   │   ├── config.py             # Configuration management
│   │   ├── data_processor.py     # Data pipeline and BOP processing
│   │   ├── statistical_tests.py  # Statistical analysis
│   │   └── visualizer.py         # Visualization and export
│   ├── case_study_one/           # Iceland vs Eurozone analysis
│   │   ├── Cleaning Case Study 1.qmd           # Main R/Quarto analysis
│   │   ├── hypothesis_test_results.csv         # Statistical test results
│   │   └── comprehensive_summary_table.csv     # Summary statistics
│   ├── dashboard/                # Interactive web applications
│   │   ├── main_app.py          # Multi-tab dashboard
│   │   ├── simple_report_app.py # Case Study 1 dashboard
│   │   └── case_study_2_euro_adoption.py      # Case Study 2 dashboard
│   └── data_processor_case_study_2.py         # Euro adoption processor
├── data/                         # Data files
│   ├── case_study_1_data_july_24_2025.csv                    # IMF BOP data
│   ├── dataset_2025-07-24T18_28_31.898465539Z_...csv         # IMF GDP data
│   ├── case_study_2_euro_adoption_data.csv                   # Processed Euro data (full)
│   ├── case_study_2_euro_adoption_data_crisis_excluded.csv   # Crisis-excluded data
│   ├── case_study_2_gdp_data.csv                            # GDP reference data
│   └── Table_DataDefinition_Sources_StijnAndrew_March22_2017.xlsx  # Metadata
├── output/                       # Generated visualizations and reports
├── analyzing-capital-flows.Rproj # RStudio project configuration
├── requirements.txt              # Python dependencies
├── CLAUDE.md                     # Claude Code interaction guide
└── README.md                     # This file
```

## Key Modules and Scripts

### Core Analysis Modules (`/src/core/`)

- **`config.py`**: Central configuration for project paths, statistical parameters, and visualization settings
- **`data_processor.py`**: Complete data pipeline from raw IMF data to analysis-ready datasets
- **`statistical_tests.py`**: F-tests, t-tests, effect size calculations, and comprehensive statistical reporting
- **`visualizer.py`**: Time series plots, boxplots, statistical visualizations with export capabilities

### Dashboard Applications (`/src/dashboard/`)

- **`main_app.py`**: Multi-tab master dashboard integrating all case studies with project overview
- **`simple_report_app.py`**: Complete Case Study 1 implementation with interactive controls and HTML report generation
- **`case_study_2_euro_adoption.py`**: Baltic countries Euro adoption analysis with crisis period options

### Data Processing Scripts

- **`data_processor_case_study_2.py`**: Specialized processor for Euro adoption analysis with dual crisis exclusion
- **`Cleaning Case Study 1.qmd`**: Main R/Quarto analysis document with comprehensive statistical analysis

## Case Studies

### Case Study 1: Iceland vs Eurozone (1999-2024)
**Question**: How would Euro adoption affect Iceland's capital flow volatility?

- **Methodology**: Cross-sectional comparison of capital flow volatility between Iceland and Eurozone countries
- **Key Finding**: Iceland shows significantly higher volatility across 10/13 capital flow indicators
- **Policy Implication**: Euro adoption could substantially reduce Iceland's financial volatility
- **Status**: ✅ Complete with full statistical analysis and interactive dashboard

### Case Study 2: Baltic Euro Adoption
**Question**: How did Euro adoption affect capital flow volatility in Baltic countries?

- **Countries**: Estonia (2011), Latvia (2014), Lithuania (2015)
- **Methodology**: Before/after temporal comparison using F-tests for variance equality
- **Features**:
  - **Full Series**: Uses all available data with asymmetric time windows
  - **Crisis-Excluded**: Removes Global Financial Crisis (2008-2010) and COVID-19 (2020-2022) periods
  - **Adoption Year Inclusion**: Euro adoption years included in post-Euro analysis
- **Status**: ✅ Complete with dual study versions and comprehensive dashboard

### Case Study 3: Emerging Markets (Planned)
**Question**: What factors determine capital flow volatility across emerging market economies?

- **Methodology**: Cross-country panel analysis with institutional and structural variables
- **Status**: 📋 Framework designed, implementation pending

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
- **Real-time Analysis**: Dynamic parameter adjustment with instant results
- **Statistical Rigor**: Proper hypothesis testing with multiple significance levels
- **Export Capabilities**: Download results, visualizations, and HTML reports
- **Crisis Analysis**: Toggle between full series and crisis-excluded versions

### Visualization Suite
- **Time Series Plots**: Capital flow trends with policy regime indicators
- **Statistical Comparisons**: Boxplots and distribution comparisons
- **Interactive Charts**: Plotly-based exploration with zoom and filtering
- **Publication Quality**: High-resolution exports for academic publications

### Data Quality
- **Validation Pipeline**: Comprehensive data integrity checks
- **Missing Value Handling**: Graceful degradation with statistical adjustments
- **Outlier Management**: Luxembourg exclusion and sensitivity analysis
- **Unit Consistency**: Standardized scaling and format validation

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
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
```

**R Requirements**:
```r
tidyverse, readr, stringr, ggplot2, knitr, gridExtra
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
**Maintainer**: [Repository Owner]  
**Last Updated**: [Current Date]  
**Version**: 2.0 (includes dual crisis exclusion and Euro adoption analysis)