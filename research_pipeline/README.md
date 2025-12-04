# Research Pipeline - Transparent Analysis Notebooks

## Purpose
This directory contains clean, transparent Jupyter notebooks that reproduce all statistical analyses from the Capital Flows research project. These notebooks are designed for:
- Academic peer review
- Research transparency
- Reproducible results
- Publication supplementary materials

## Structure

```
research_pipeline/
├── notebooks/                 # Jupyter notebooks for each case study
│   ├── 00_Data_Overview.ipynb        # Data sources and cleaning process
│   ├── CS1_Iceland_vs_Eurozone.ipynb # Case Study 1 analysis
│   ├── CS2_Baltic_Euro_Adoption.ipynb # Case Study 2 analysis
│   ├── CS3_Small_Open_Economies.ipynb # Case Study 3 analysis
│   ├── CS4_Statistical_Framework.ipynb # Case Study 4 analysis
│   └── CS5_Capital_Controls_Regimes.ipynb # Case Study 5 analysis
├── data/                      # Symlinks to clean data (no duplication)
│   └── → ../updated_data/Clean/
├── outputs/                   # Results from notebook analyses
│   ├── tables/               # Statistical results tables (CSV)
│   ├── figures/              # Publication-ready figures
│   └── statistics/           # Detailed statistical outputs
├── verification/              # Comparison with dashboard results
│   ├── dashboard_results/    # Exported results from Streamlit dashboard
│   └── comparison_reports/   # Verification that notebook matches dashboard
└── docs/                      # Additional documentation
    ├── methodology.md         # Detailed statistical methodology
    └── data_dictionary.md     # Variable definitions and sources
```

## Key Principles

1. **Transparency First**: Every calculation is shown with intermediate steps
2. **No Hidden Logic**: All formulas and assumptions are explicit
3. **Reproducible**: Anyone can run the notebooks and get the same results
4. **Verified**: Results are compared against the dashboard to ensure accuracy
5. **Clean Code**: Simple, readable Python without UI complexity

## Getting Started

### Prerequisites
```bash
# Create conda environment
conda create -n capital_flows python=3.9
conda activate capital_flows

# Install requirements
pip install jupyter pandas numpy scipy statsmodels matplotlib seaborn
```

### Running the Notebooks

1. Start Jupyter:
   ```bash
   cd research_pipeline
   jupyter notebook
   ```

2. Begin with `00_Data_Overview.ipynb` to understand the data

3. Run case studies in any order (they're independent)

## Verification Process

Each notebook includes a verification section that:
1. Loads corresponding dashboard results
2. Compares statistical outputs
3. Documents any differences with explanations
4. Confirms accuracy of calculations

## Relationship to Main Codebase

- **Data Source**: Uses same cleaned CSV files from `updated_data/Clean/`
- **Logic Source**: Statistical calculations extracted from `src/dashboard/reports/`
- **Verification**: Results compared with Streamlit dashboard outputs
- **Independence**: Notebooks are self-contained and don't import from src/

## For Reviewers

If you're reviewing this research:
1. Each notebook is self-contained with all necessary explanations
2. Run cells sequentially to reproduce results
3. Check the verification section to confirm accuracy
4. Outputs are saved in `outputs/` for reference

## Contact

For questions about the methodology or implementation, see the main project documentation.