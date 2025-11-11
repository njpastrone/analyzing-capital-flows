# Project Structure Documentation

This document provides a comprehensive breakdown of the Capital Flows Research Analysis project structure, explaining the purpose and contents of each directory and major file.

## Directory Overview

```
analyzing-capital-flows/
├── src/                           # Source code and analysis
├── data/                         # Raw datasets and processed outputs  
├── output/                       # Generated visualizations and reports
├── docs/                         # Documentation files (this file)
├── Configuration files           # Project configuration and metadata
└── Documentation files           # README, CLAUDE.md, etc.
```

## Detailed Structure

### `/src/` - Source Code and Analysis

The main source directory contains all analysis code, organized by functionality and case study.

#### `/src/core/` - Core Python Modules

**Purpose**: Foundational modules for data processing, statistical analysis, and visualization that are shared across multiple case studies.

```
src/core/
├── config.py                     # Central configuration management
├── data_processor.py             # Data pipeline and BOP processing  
├── statistical_tests.py          # Statistical analysis and hypothesis testing
└── visualizer.py                 # Visualization and export functionality
```

**Key Files**:

- **`config.py`**: 
  - Project paths and directory structure
  - Statistical test parameters (significance levels, test types)
  - Visualization settings (colors, themes, export formats)
  - Streamlit page configuration

- **`data_processor.py`**:
  - `DataProcessor`: Base class for data operations
  - `BOPDataProcessor`: Specialized Balance of Payments data handling
  - Data validation, cleaning, and transformation pipelines
  - GDP normalization and country grouping logic

- **`statistical_tests.py`**:
  - `StatisticalAnalyzer`: Descriptive statistics and data summarization
  - `VolatilityTester`: F-tests for variance equality, t-tests for means
  - `EffectSizeCalculator`: Cohen's d and Hedges' g calculations
  - Comprehensive statistical reporting with multiple significance levels

- **`visualizer.py`**:
  - `BaseVisualizer`: Common plotting functionality
  - `StatisticalVisualizer`: Specialized plots for statistical comparisons
  - Support for matplotlib and Plotly outputs
  - Export capabilities in PNG, SVG, PDF formats

#### `/src/case_study_one/` - Iceland vs Eurozone Analysis

**Purpose**: Complete analysis comparing Iceland's capital flow volatility to Eurozone countries, examining the potential benefits of Euro adoption.

```
src/case_study_one/
├── Cleaning Case Study 1.qmd                  # Main R/Quarto analysis document
├── Case_Study_1_Report_Template.ipynb         # Jupyter notebook template
├── hypothesis_test_results.csv                # Statistical test results
└── comprehensive_summary_table.csv            # Summary statistics
```

**Key Files**:

- **`Cleaning Case Study 1.qmd`**: 
  - Complete R/Quarto analysis with literate programming
  - Data import, cleaning, and statistical analysis
  - Automated report generation with embedded results
  - Publication-ready tables and visualizations

- **`hypothesis_test_results.csv`**: 
  - F-test results for variance equality across all indicators
  - Multiple significance levels (0.1%, 1%, 5%, 10%)
  - Effect sizes and practical significance measures

- **`comprehensive_summary_table.csv`**: 
  - Descriptive statistics by country group and indicator
  - Means, standard deviations, coefficients of variation
  - Sample sizes and data coverage information

#### `/src/dashboard/` - Interactive Web Applications

**Purpose**: Streamlit-based web applications providing interactive access to analysis results and real-time statistical computation.

```
src/dashboard/
├── main_app.py                   # Multi-tab master dashboard
├── simple_report_app.py          # Case Study 1 dashboard implementation
├── case_study_2_euro_adoption.py # Baltic countries Euro adoption analysis
└── __pycache__/                  # Python bytecode cache
```

**Key Files**:

- **`main_app.py`**: 
  - Master dashboard with project overview
  - Multi-tab interface integrating all case studies
  - Data processing pipeline documentation
  - Navigation between different analyses

- **`simple_report_app.py`**: 
  - Complete Case Study 1 interactive implementation
  - Real-time parameter adjustment and analysis
  - Statistical results tables and visualizations
  - HTML report generation and export capabilities

- **`case_study_2_euro_adoption.py`**: 
  - Baltic countries before/after Euro adoption analysis
  - Interactive country selection and time period options
  - Crisis period inclusion/exclusion toggle
  - Comprehensive statistical testing and visualization

#### `/src/data_processor_case_study_2.py` - Euro Adoption Processor

**Purpose**: Specialized data processor for Case Study 2, handling the complex timeline requirements of Euro adoption analysis.

**Key Features**:
- Asymmetric time window analysis (maximizes available data)
- Dual crisis exclusion (Global Financial Crisis + COVID-19)
- Euro adoption year inclusion in post-adoption periods
- Generates both full series and crisis-excluded datasets

### `/data/` - Data Files

**Purpose**: All raw datasets, processed outputs, and reference data used in the analysis.

```
data/
├── Raw IMF Data
│   ├── case_study_1_data_july_24_2025.csv                    # IMF BOP quarterly data
│   ├── case_study_2_data_july_27_2025.csv                    # Baltic countries BOP data
│   ├── dataset_2025-07-24T18_28_31.898465539Z_...csv         # IMF GDP annual data
│   └── Table_DataDefinition_Sources_StijnAndrew_March22_2017.xlsx  # Metadata/definitions
├── Processed Case Study 1 Data
│   └── [Generated during analysis - various CSV outputs]
├── Processed Case Study 2 Data
│   ├── case_study_2_euro_adoption_data.csv                   # Full series dataset
│   ├── case_study_2_euro_adoption_data_crisis_excluded.csv   # Crisis-excluded dataset
│   └── case_study_2_gdp_data.csv                             # GDP reference data
```

**Data Characteristics**:
- **Coverage**: 1999-2024 for most series, quarterly frequency for BOP, annual for GDP
- **Format**: All monetary values normalized to "% of GDP (annualized)"
- **Quality**: Missing value handling, outlier management, validation checks
- **Sources**: IMF Balance of Payments Statistics, IMF World Economic Outlook

### `/output/` - Generated Content

**Purpose**: Automatically generated visualizations, reports, and analysis outputs.

```
output/
├── charts/                       # Generated visualizations
├── reports/                      # HTML and PDF reports  
├── statistical_tables/           # CSV exports of results
└── presentations/                # Presentation-ready materials
```

**Content Types**:
- **Charts**: Time series plots, boxplots, statistical comparisons
- **Reports**: HTML documents with embedded analysis and results
- **Tables**: Statistical test results, summary statistics, effect sizes
- **Presentations**: Publication-ready figures and summary materials

### Configuration Files

**Purpose**: Project configuration, dependency management, and development environment setup.

- **`analyzing-capital-flows.Rproj`**: RStudio project configuration
  - 2-space indentation, UTF-8 encoding
  - Code indexing and Quarto integration
  - R package management settings

- **`requirements.txt`**: Python dependency specifications
  - Streamlit for web applications
  - Pandas/NumPy for data analysis
  - SciPy for statistical testing
  - Matplotlib/Seaborn/Plotly for visualization

- **`.gitignore`**: Version control exclusions
  - Python cache files, R workspace data
  - Large output files, temporary analysis results

### Documentation Files

**Purpose**: Project documentation, development guidance, and user instructions.

- **`README.md`**: Primary project documentation
  - Installation instructions, quick start guide
  - Project overview and research findings
  - Detailed structure and module descriptions

- **`CLAUDE.md`**: Claude Code interaction guidelines
  - Development environment setup
  - Code patterns and conventions
  - Common tasks and troubleshooting

- **`PROJECT_STRUCTURE.md`**: This file
  - Comprehensive directory breakdown
  - File purposes and relationships
  - Development workflow guidance

## Data Flow Architecture

### Processing Pipeline

1. **Raw Data** (`/data/` IMF files) 
   ↓
2. **Core Processing** (`/src/core/data_processor.py`)
   ↓  
3. **Case-Specific Processing** (`/src/data_processor_case_study_2.py`, R/Quarto)
   ↓
4. **Statistical Analysis** (`/src/core/statistical_tests.py`)
   ↓
5. **Visualization** (`/src/core/visualizer.py`, dashboard apps)
   ↓
6. **Output Generation** (`/output/` directory)

### Analysis Workflow

1. **Interactive Analysis**: Streamlit dashboards (`/src/dashboard/`)
2. **Batch Processing**: R/Quarto documents (`/src/case_study_one/`)
3. **Data Regeneration**: Python processors (`/src/data_processor_case_study_2.py`)
4. **Result Export**: Multiple formats via core modules

## Development Workflow

### Adding New Analysis

1. **Data Processing**: Extend or create new processor in `/src/core/` or case-specific
2. **Statistical Methods**: Add new tests to `/src/core/statistical_tests.py`
3. **Visualization**: Extend `/src/core/visualizer.py` with new chart types
4. **Dashboard Integration**: Create new tab in `/src/dashboard/main_app.py`
5. **Documentation**: Update relevant documentation files

### File Naming Conventions

- **Data Files**: `case_study_[N]_[description]_[date].csv`
- **Processing Scripts**: `data_processor_case_study_[N].py`  
- **Dashboard Apps**: `[case_study_name]_[functionality].py`
- **Output Files**: `[case_study]_[analysis_type]_[timestamp].[ext]`

### Version Control Strategy

- **Main Branch**: Stable, working analysis code
- **Feature Branches**: Individual analysis enhancements
- **Data Versioning**: Date-stamped raw data files
- **Output Exclusion**: Generated files excluded from version control

This structure provides a scalable foundation for expanding the research to additional case studies while maintaining code quality and analytical rigor.