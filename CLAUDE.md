# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **capital flows research analysis project** that examines capital flow volatility across different economies, time periods, and policy regimes. The project investigates how monetary policy frameworks, currency unions, and external shocks affect financial stability in small open economies.

### Core Research Questions
1. **Monetary Regime Effects**: How does capital flow volatility vary across different monetary regimes?
2. **Euro Adoption Impact**: What are the effects of joining a currency union on capital flow stability?
3. **External Shock Transmission**: How do global financial crises affect capital flow patterns differently across countries?
4. **Policy Implications**: What do volatility patterns suggest for currency union and monetary policy decisions?

## Project Structure

```
analyzing-capital-flows/
├── src/                           # Source code and analysis
│   ├── core/                      # Core Python modules for data processing and analysis
│   │   ├── config.py             # Central configuration management
│   │   ├── data_processor.py     # Data pipeline and BOP processing
│   │   ├── statistical_tests.py  # Statistical analysis and hypothesis testing
│   │   └── visualizer.py         # Visualization and export functionality
│   ├── case_study_one/           # Iceland vs Eurozone analysis
│   │   └── Cleaning Case Study 1.qmd  # R/Quarto main analysis document
│   ├── dashboard/                # Streamlit web applications
│   │   ├── main_app.py          # Multi-tab master dashboard
│   │   ├── simple_report_app.py # Case Study 1 implementation
│   │   └── case_study_2_euro_adoption.py  # Baltic countries analysis
│   └── data_processor_case_study_2.py    # Specialized processor for Euro adoption analysis
├── data/                         # Raw datasets and processed outputs
│   ├── case_study_1_data_july_24_2025.csv           # IMF BOP data
│   ├── dataset_2025-07-24T18_28_31...csv            # IMF GDP data  
│   ├── case_study_2_euro_adoption_data.csv          # Processed Euro adoption data (full)
│   ├── case_study_2_euro_adoption_data_crisis_excluded.csv  # Crisis-excluded version
│   └── Table_DataDefinition_Sources...xlsx          # Data definitions and metadata
├── output/                       # Generated visualizations and reports
└── Configuration files           # .Rproj, requirements, documentation
```

## Development Environment

This is a **hybrid R/Python project**:

- **R Environment**: RStudio project (`.Rproj` file present) with:
  - 2-space indentation
  - UTF-8 encoding
  - Code indexing enabled
  - Quarto document integration

- **Python Environment**: Streamlit-based dashboards with core analysis modules

## Key Dependencies

### R Packages
- `tidyverse`: Core data manipulation and visualization
- `readr`: CSV file reading and data import
- `stringr`: String manipulation for indicator processing
- `ggplot2`: Data visualization (part of tidyverse)
- `knitr`: Document generation and reporting
- `gridExtra`: Layout utilities for plots and tables

### Python Packages
- `streamlit>=1.28.0`: Interactive web dashboard framework
- `pandas>=2.0.0`: Data manipulation and analysis
- `numpy>=1.24.0`: Numerical computing
- `scipy>=1.10.0`: Statistical analysis and hypothesis testing
- `matplotlib>=3.7.0`: Static plotting and visualization
- `seaborn>=0.12.0`: Statistical data visualization
- `plotly>=5.15.0`: Interactive plotting capabilities

## Data Processing Pipeline

The main analysis workflow follows this pattern:

1. **Data Import**: 
   - Raw IMF Balance of Payments (BOP) data from quarterly reports
   - IMF World Economic Outlook (GDP) data for normalization
   - Metadata files with indicator definitions and data sources

2. **Data Cleaning**:
   - Extract BOP indicator names from accounting entries (first word + indicator type)
   - Separate time periods into year/quarter components
   - Convert data from long to wide format for analysis
   - Handle missing values and data quality issues

3. **Data Integration**:
   - Join BOP and GDP datasets by country and year
   - Validate data consistency and coverage
   - Create analytical country groupings

4. **Normalization**:
   - Convert BOP flows to "% of GDP" for cross-country comparison
   - Annualize quarterly data by multiplying by 4
   - Apply unit scaling corrections for different data formats

5. **Grouping and Analysis**:
   - Create country groups (Iceland vs. Eurozone, Baltic countries)
   - Generate time period classifications (Pre-Euro vs Post-Euro)
   - Exclude outliers (Luxembourg) and handle crisis periods

6. **Statistical Analysis**:
   - Calculate descriptive statistics (means, standard deviations, coefficients of variation)
   - Perform F-tests for variance equality across groups
   - Generate comprehensive hypothesis test results
   - Create statistical visualizations and summary tables

## Case Study Frameworks

### Case Study 1: Iceland vs Eurozone (1999-2024)
- **Status**: ✅ Complete implementation
- **Methodology**: Cross-sectional comparison of capital flow volatility
- **Key Insight**: Iceland shows significantly higher volatility (10/13 indicators at 5% level)
- **Policy Implication**: Euro adoption could reduce financial volatility for Iceland

### Case Study 2: Baltic Euro Adoption 
- **Countries**: Estonia (2011), Latvia (2014), Lithuania (2015)  
- **Status**: ✅ Complete with dual study versions
- **Methodology**: Before/after temporal comparison of volatility changes
- **Features**: 
  - Full Series: Uses all available data with asymmetric time windows
  - Crisis-Excluded: Removes Global Financial Crisis (2008-2010) and COVID-19 (2020-2022) periods
  - Adoption years included in post-Euro periods for complete transition analysis

### Case Study 3: Emerging Markets (Framework)
- **Status**: 📋 Designed but not implemented
- **Scope**: Cross-country panel analysis of volatility determinants
- **Focus**: Institutional quality, exchange rate regimes, external vulnerability measures

## Output Generation

The project generates multiple output formats:

- **Statistical Tables**: CSV exports with comprehensive test results
- **Visualizations**: Time series plots, boxplots, statistical comparisons
- **Interactive Dashboards**: Streamlit web applications for exploration
- **HTML Reports**: Automated report generation with embedded analysis
- **Downloadable Assets**: Charts and data exports in multiple formats

## Working with the Codebase

### Running the Analysis

**R/Quarto Analysis**:
```bash
# Open in RStudio and render
quarto render "src/Cleaning Case Study 1.qmd"
```

**Python Dashboard**:
```bash
# Install dependencies
pip install -r requirements.txt

# Run main dashboard
streamlit run src/dashboard/main_app.py

# Run individual case studies
streamlit run src/dashboard/simple_report_app.py
streamlit run src/dashboard/case_study_2_euro_adoption.py
```

**Data Processing**:
```bash
# Regenerate Case Study 2 datasets
python src/data_processor_case_study_2.py
```

### Data Patterns and Conventions

- **Monetary Values**: All converted to "% of GDP (annualized)" for comparison
- **Time Periods**: BOP data annualized by multiplying quarterly values by 4
- **Country Filtering**: Luxembourg often excluded due to financial center outlier status
- **Missing Data**: Graceful handling with appropriate statistical adjustments
- **Crisis Periods**: Global Financial Crisis (2008-2010) and COVID-19 (2020-2022) tracked for exclusion analysis

### Code Style and Patterns

- **R Code**: tidyverse conventions with 2-space indentation
- **Python Code**: PEP 8 style with object-oriented design patterns
- **Documentation**: Comprehensive docstrings and inline comments
- **Error Handling**: Graceful degradation with informative error messages
- **Testing**: Data validation and statistical test verification throughout pipeline

## Claude Code Interaction Guidelines

When working with this codebase:

1. **Preserve Statistical Rigor**: Maintain proper hypothesis testing procedures and significance levels
2. **Data Integrity**: Always validate data transformations and handle missing values appropriately  
3. **Reproducibility**: Ensure any changes maintain the end-to-end pipeline functionality
4. **Documentation**: Update relevant documentation when adding features or modifying analysis
5. **Visualization Standards**: Follow established chart styling and export format conventions
6. **Performance**: Consider computational efficiency for large dataset operations
7. **User Experience**: Maintain intuitive dashboard interfaces and clear result presentations

### Common Tasks

- **Adding New Countries**: Extend country groupings in `config.py` and update processing logic
- **New Indicators**: Modify BOP indicator extraction and add to analysis frameworks  
- **Additional Statistical Tests**: Extend `statistical_tests.py` with new test methods
- **Dashboard Enhancements**: Add new visualization types or interactive features
- **Crisis Period Updates**: Modify crisis year definitions for different exclusion analyses
- **Export Formats**: Extend visualization export capabilities for different output needs

### Important Notes

- **Data Sources**: Always verify IMF data compatibility and update metadata accordingly
- **Statistical Assumptions**: Check variance equality and normality assumptions for parametric tests
- **Cross-Case Consistency**: Maintain methodological consistency across different case studies
- **Version Control**: Use meaningful commit messages and maintain clean development history
- **Dependency Management**: Keep requirement files updated and test compatibility across versions

This project bridges rigorous academic research with practical policy insights, requiring attention to both methodological soundness and real-world applicability.