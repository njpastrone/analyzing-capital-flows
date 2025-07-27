# Capital Flows Research Dashboard

An interactive Streamlit dashboard for analyzing capital flow volatility between country groups using OOP architecture.

## Features

- **Template-based Analysis**: Reusable templates for different types of research
- **Interactive Interface**: Easy-to-use web interface with file uploads and parameter selection
- **Comprehensive Statistics**: Descriptive statistics, F-tests, volatility measures
- **Rich Visualizations**: Both static (matplotlib) and interactive (plotly) charts
- **Export Capabilities**: Export results in multiple formats (CSV, Excel, PDF)
- **Modular Architecture**: Clean OOP design for extensibility

## Quick Start

### 1. Install Dependencies

```bash
cd src/dashboard
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

### 3. Access the Interface

Open your browser to `http://localhost:8501`

## Usage Guide

### Volatility Analysis

1. **Load Data**:
   - Upload BOP (Balance of Payments) CSV file
   - Upload GDP CSV file
   - Or use default Case Study 1 files

2. **Configure Groups**:
   - Choose preset (Iceland vs Eurozone) or create custom groups
   - Define countries for each group
   - Set processing options

3. **Run Analysis**:
   - Select primary and comparison groups
   - Choose significance level
   - Execute volatility analysis

4. **View Results**:
   - Statistical summary with key findings
   - Detailed test results table
   - Multiple visualization types

5. **Export**:
   - Download results in CSV/Excel format
   - Save visualizations as images

## Dashboard Structure

```
dashboard/
├── app.py                    # Main Streamlit application
├── components/               # UI components (future expansion)
├── templates/
│   └── case_study_template.py   # Base template classes
└── requirements.txt         # Python dependencies

core/
├── config.py                # Configuration settings
├── data_processor.py        # Data loading and processing
├── statistical_tests.py     # Statistical analysis classes
└── visualizer.py           # Visualization classes
```

## Template System

### VolatilityAnalysisTemplate

Specialized template for capital flow volatility analysis:

- **Data Processing**: BOP and GDP data cleaning and normalization
- **Group Creation**: Flexible country grouping system
- **Statistical Tests**: F-tests for variance equality
- **Visualizations**: Boxplots, time series, interactive charts

### Creating New Templates

1. Inherit from `CaseStudyTemplate`
2. Implement required abstract methods:
   - `load_data()`
   - `process_data()`
   - `run_analysis()`
   - `generate_visualizations()`

## Key Features

### 1. Data Processing
- Automatic BOP indicator name creation
- GDP normalization (% of GDP, annualized)
- Flexible country grouping
- Data validation and error handling

### 2. Statistical Analysis
- F-tests for equal variances
- Comprehensive descriptive statistics
- Effect size calculations
- Multiple significance levels

### 3. Visualizations
- Side-by-side boxplots for means and standard deviations
- Time series plots with group averages
- Interactive Plotly charts
- Customizable styling and colors

### 4. Export Options
- CSV and Excel data export
- High-resolution image export
- Comprehensive results packaging

## Configuration

Edit `core/config.py` to customize:

- Default file paths
- Plot styling and colors
- Statistical test parameters
- Export formats

## Example Usage

```python
# Initialize template
template = VolatilityAnalysisTemplate("Iceland vs Eurozone")

# Load data
template.load_data("bop_data.csv", "gdp_data.csv")

# Process with group definitions
groups = {
    "Iceland": ["Iceland"],
    "Eurozone": ["Austria", "Belgium", "Finland", "France", "Germany", 
                "Ireland", "Italy", "Netherlands", "Portugal", "Spain"]
}
template.process_data(group_definitions=groups)

# Run analysis
results = template.run_analysis(group1="Iceland", group2="Eurozone")

# Generate visualizations
visualizations = template.generate_visualizations()

# Export results
template.export_results("output/", formats=['csv', 'excel'])
```

## Future Enhancements

- Additional analysis templates (mean comparison, correlation analysis)
- Real-time data fetching from APIs
- Advanced statistical tests
- Custom report generation
- Multi-language support

## Support

For questions or issues:
1. Check the logs in the Streamlit interface
2. Verify data file formats match expected structure
3. Ensure all dependencies are installed correctly