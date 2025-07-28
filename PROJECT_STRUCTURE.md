# Capital Flows Research Project Structure

## 🎯 Project Overview

This research project examines capital flow volatility across different economies, time periods, and policy regimes to understand implications for monetary policy, currency unions, and financial stability.

## 📁 Recommended Project Structure

```
analyzing-capital-flows/
├── README.md                           # Project overview and setup
├── CLAUDE.md                          # Claude AI instructions (existing)
├── PROJECT_STRUCTURE.md               # This file
├── requirements.txt                   # Python dependencies
├── config/
│   ├── data_sources.yaml             # Data source configurations
│   ├── country_groups.yaml           # Country grouping definitions
│   └── analysis_settings.yaml        # Default analysis parameters
├── data/
│   ├── raw/                          # Raw downloaded data
│   │   ├── imf_bop/                  # IMF Balance of Payments
│   │   ├── imf_weo/                  # IMF World Economic Outlook
│   │   ├── oecd_fdi/                 # OECD FDI data
│   │   └── bis_banking/              # BIS Banking data
│   ├── processed/                    # Cleaned and harmonized data
│   │   ├── case_study_1/             # Iceland vs Eurozone
│   │   ├── case_study_2/             # Brexit analysis
│   │   └── case_study_3/             # Emerging markets
│   └── external/                     # External datasets (VIX, policy vars)
├── src/
│   ├── core/                         # Core analysis modules (existing)
│   │   ├── config.py
│   │   ├── data_processor.py
│   │   ├── statistical_tests.py
│   │   └── visualizer.py
│   ├── data_collection/              # Data downloading and APIs
│   │   ├── imf_api.py               # IMF data downloader
│   │   ├── oecd_api.py              # OECD data downloader
│   │   └── data_validator.py        # Data quality checks
│   ├── case_studies/                 # Individual case study modules
│   │   ├── case_study_1/            # Iceland vs Eurozone (existing)
│   │   ├── case_study_2/            # Brexit analysis
│   │   └── case_study_3/            # Emerging markets
│   ├── comparative/                  # Cross-case analysis
│   │   ├── meta_analysis.py         # Effect size comparisons
│   │   └── policy_synthesis.py      # Policy recommendations
│   └── dashboard/                    # Streamlit applications
│       ├── main_app.py              # Multi-tab main dashboard (new)
│       ├── simple_report_app.py     # Case Study 1 (existing, preserved)
│       ├── templates/               # Dashboard templates (existing)
│       └── components/              # Reusable UI components
├── analysis/                         # Analysis notebooks and scripts
│   ├── exploratory/                 # Initial data exploration
│   ├── case_study_1/               # Iceland analysis (existing)
│   ├── case_study_2/               # Brexit analysis
│   └── comparative/                # Cross-case comparisons
├── reports/                         # Generated reports and papers
│   ├── case_study_1/               # Iceland vs Eurozone findings
│   ├── case_study_2/               # Brexit analysis findings
│   ├── working_papers/             # Academic papers
│   └── policy_briefs/              # Policy-focused summaries
├── tests/                          # Unit and integration tests
│   ├── test_data_processing.py
│   ├── test_statistical_methods.py
│   └── test_dashboard_components.py
└── docs/                           # Documentation
    ├── methodology.md              # Statistical methods documentation
    ├── data_dictionary.md          # Variable definitions
    ├── api_documentation.md        # API usage guides
    └── user_guide.md              # Dashboard user instructions
```

## 🚀 Formatting and Organization Suggestions

### 1. **Standardized Naming Conventions**
```
# Case Studies
case_study_1_iceland_eurozone/
case_study_2_brexit_impact/
case_study_3_emerging_markets/

# Files
iceland_vs_eurozone_analysis.py
brexit_volatility_analysis.py
emerging_markets_comparison.py

# Variables
bop_flows_pgdp          # Balance of payments as % of GDP
volatility_test_results # F-test results
policy_timeline_events  # Brexit timeline events
```

### 2. **Modular Dashboard Architecture**
```python
# main_app.py - Master dashboard with tabs
# Each case study as separate module:
from case_studies.iceland_eurozone import run_analysis as iceland_analysis
from case_studies.brexit_impact import run_analysis as brexit_analysis
from case_studies.emerging_markets import run_analysis as emerging_analysis
```

### 3. **Configuration Management**
```yaml
# config/data_sources.yaml
imf_bop:
  base_url: "http://dataservices.imf.org/REST/SDMX_JSON.svc/"
  datasets: ["BOP", "BOPSFSR"]
  frequency: "Q"  # Quarterly

# config/country_groups.yaml
eurozone_original:
  - Austria
  - Belgium
  - Finland
  # ... etc

brexit_analysis:
  treatment: ["United Kingdom"]
  control: ["Germany", "France", "Netherlands"]
```

### 4. **Data Pipeline Automation**
```python
# src/data_collection/data_pipeline.py
class DataPipeline:
    def download_latest_data()
    def clean_and_process()
    def validate_quality()
    def update_case_studies()
```

### 5. **Reproducible Analysis**
```python
# Each case study follows template:
class CaseStudyTemplate:
    def load_data()
    def process_data() 
    def run_analysis()
    def generate_visualizations()
    def export_results()
    def generate_report()
```

## 📊 Suggested Case Study Extensions

### Case Study 2: Brexit Impact Analysis
- **Pre/Post Brexit volatility comparison**
- **Event study around key Brexit dates**
- **Sectoral analysis (banking vs portfolio flows)**
- **Control group: Non-Brexit EU countries**

### Case Study 3: Emerging Markets Panel
- **Cross-country volatility determinants**
- **Institutional quality impacts**
- **Global financial cycle effects**
- **Policy regime comparisons**

### Case Study 4: COVID-19 Impact (Future)
- **Pandemic shock transmission**
- **Policy response effectiveness**
- **Recovery pattern analysis**
- **Structural break identification**

### Case Study 5: Central Bank Digital Currencies (Future)
- **CBDC pilot program impacts**
- **Cross-border payment flows**
- **Financial stability implications**

## 🛠️ Technical Recommendations

### 1. **Version Control Strategy**
```
main branch:     Stable, deployed dashboard
develop branch:  Integration of new features
feature branches: Individual case studies
hotfix branches: Critical bug fixes
```

### 2. **Testing Framework**
```python
# tests/test_case_study_1.py
def test_data_loading()
def test_statistical_calculations() 
def test_visualization_generation()
def test_report_generation()
```

### 3. **Documentation Standards**
- **Methodology docs:** Statistical methods, assumptions, limitations
- **Data dictionary:** Variable definitions, sources, transformations
- **User guides:** Dashboard navigation, interpretation of results
- **API docs:** For programmatic access to analysis functions

### 4. **Deployment Options**
- **Streamlit Cloud:** Easy deployment for public access
- **Docker containers:** Reproducible environments
- **GitHub Pages:** Static documentation hosting
- **Academic servers:** Institution-specific deployment

## 📈 Progressive Development Plan

### Phase 1: Foundation (Current)
- ✅ Case Study 1 complete
- ✅ Core analysis framework
- ✅ Dashboard infrastructure

### Phase 2: Expansion (Q2 2024)
- 🔄 Brexit impact analysis
- 🔄 Multi-tab dashboard integration
- 🔄 Enhanced data pipeline

### Phase 3: Synthesis (Q3 2024)
- 📋 Emerging markets analysis
- 📋 Comparative analysis framework
- 📋 Policy recommendation engine

### Phase 4: Publication (Q4 2024)
- 📋 Academic paper preparation
- 📋 Policy brief generation
- 📋 Public dashboard launch

## 🎯 Success Metrics

- **Academic Impact:** Citations, conference presentations
- **Policy Relevance:** Central bank usage, policy citations
- **Technical Quality:** Code reliability, reproducibility
- **User Engagement:** Dashboard usage statistics
- **Research Output:** Working papers, policy briefs

This structure provides a scalable, maintainable framework for expanding the capital flows research project while preserving the excellent work already completed in Case Study 1.