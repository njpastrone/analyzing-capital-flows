# Capital Flows Research Dashboard

Interactive Streamlit web applications for analyzing capital flow volatility across different monetary regimes, with specialized implementations for multiple case studies.

## Dashboard Applications

### 1. Master Dashboard (`main_app.py`)
**Purpose**: Multi-tab interface integrating all project components

**Features**:
- Project overview with research questions and findings
- Data processing pipeline documentation  
- Navigation between different case studies
- Integrated access to all analysis tools

**Usage**:
```bash
streamlit run src/dashboard/main_app.py
```

### 2. Case Study 1 Dashboard (`simple_report_app.py`)
**Purpose**: Iceland vs Eurozone volatility comparison analysis

**Features**:
- Interactive parameter selection (significance levels, group definitions)
- Real-time statistical computation and hypothesis testing
- Multiple visualization types (boxplots, time series, statistical comparisons)
- HTML report generation with embedded analysis
- CSV/Excel export capabilities

**Key Capabilities**:
- F-tests for variance equality across 13 capital flow indicators
- Descriptive statistics with multiple significance levels
- Interactive chart exploration with Plotly integration
- Publication-ready export formats

### 3. Case Study 2 Dashboard (`case_study_2_euro_adoption.py`)
**Purpose**: Baltic countries Euro adoption before/after analysis

**Features**:
- Country selection (Estonia, Latvia, Lithuania)
- Dual study versions:
  - **Full Series**: All available data with asymmetric time windows
  - **Crisis-Excluded**: Removes GFC (2008-2010) and COVID-19 (2020-2022) periods
- Comprehensive statistical testing and visualization
- Timeline-specific analysis with adoption year inclusion

**Interactive Elements**:
- Study version toggle (Full vs Crisis-Excluded)
- Country-specific analysis selection
- Crisis period visualization with shaded exclusion zones
- Downloadable results in multiple formats

## Architecture and Design

### Modular Structure
```
dashboard/
├── main_app.py                   # Master multi-tab dashboard
├── simple_report_app.py          # Case Study 1 implementation
├── case_study_2_euro_adoption.py # Baltic Euro adoption analysis
├── components/                   # Reusable UI components (future)
├── templates/                    # Analysis templates
│   └── case_study_template.py    # Base template classes
└── __pycache__/                  # Python bytecode cache
```

### Shared Functionality
All dashboards utilize the core modules:
- `../core/config.py`: Configuration management
- `../core/data_processor.py`: Data pipeline and processing
- `../core/statistical_tests.py`: Statistical analysis framework
- `../core/visualizer.py`: Visualization and export utilities

## Key Features

### 1. Statistical Rigor
- **F-Tests**: Variance equality testing across groups/periods
- **Multiple Significance Levels**: 0.1%, 1%, 5%, 10% with proper reporting
- **Effect Sizes**: Cohen's d and Hedges' g for practical significance
- **Robustness Checks**: Crisis period exclusion and sensitivity analysis

### 2. Interactive Analysis
- **Real-Time Computation**: Dynamic parameter adjustment with instant results
- **Parameter Flexibility**: User-controlled significance levels and group definitions
- **Data Upload**: Support for custom dataset analysis
- **Session Management**: Unique widget keys preventing UI conflicts

### 3. Professional Visualization
- **Time Series Plots**: Capital flow trends with policy regime indicators
- **Statistical Comparisons**: Side-by-side boxplots with reference lines
- **Interactive Charts**: Plotly-based exploration with zoom and filtering
- **Crisis Period Visualization**: Shaded exclusion zones for temporal analysis

### 4. Export Capabilities
- **Multiple Formats**: PNG, SVG, PDF for visualizations
- **Data Export**: CSV and Excel formats for statistical results
- **HTML Reports**: Complete analysis reports with embedded results
- **Publication Quality**: High-resolution outputs for academic use

## Technical Implementation

### Session State Management
- Unique widget keys prevent DuplicateWidgetID errors
- Session-based persistence for user selections
- Robust handling of app reruns and version switching

### Data Processing Pipeline
1. **Import**: Raw IMF datasets with validation
2. **Clean**: Extract indicators, handle missing values
3. **Transform**: Wide format conversion, GDP normalization
4. **Group**: Country groupings and time period classifications
5. **Analyze**: Statistical testing and result generation
6. **Visualize**: Chart generation and interactive display

### Widget Key Strategy
```python
# Comprehensive uniqueness system
session_id = f"{base_timestamp}_{random_suffix}_{study_version}_{country}"
widget_key = f"download_{type}_{country}_{indicator}_{version}_{session_id}"
```

## Running the Dashboards

### Prerequisites
```bash
# Install dependencies (from project root)
pip install -r requirements.txt
```

### Individual Applications
```bash
# Master dashboard (recommended entry point)
streamlit run src/dashboard/main_app.py

# Case Study 1 (Iceland vs Eurozone)
streamlit run src/dashboard/simple_report_app.py

# Case Study 2 (Baltic Euro adoption)
streamlit run src/dashboard/case_study_2_euro_adoption.py
```

### Data Requirements
- **Case Study 1**: Uses processed data from R/Quarto analysis
- **Case Study 2**: Requires running `src/data_processor_case_study_2.py` to generate datasets
- **Custom Analysis**: Upload BOP and GDP CSV files in IMF format

## Configuration and Customization

### Styling and Themes
Edit `../core/config.py` to customize:
- Color schemes and visualization themes
- Export formats and resolution settings
- Statistical test parameters
- Default file paths

### Adding New Analysis
1. **Create New Dashboard**: Follow existing pattern in new Python file
2. **Implement Core Functions**: Data loading, processing, analysis, visualization
3. **Integrate**: Add new tab to `main_app.py` for navigation
4. **Document**: Update this README with new functionality

## Data Quality and Validation

### Input Validation
- **Format Checking**: Verify CSV structure and required columns
- **Missing Value Handling**: Graceful degradation with user notifications
- **Data Range Validation**: Logical checks for years, countries, indicators

### Error Handling
- **User-Friendly Messages**: Clear error explanations with suggested fixes
- **Fallback Options**: Default settings when user input is invalid
- **Debug Information**: Detailed logging for development and troubleshooting

## Performance Optimization

### Caching Strategy
- **Data Loading**: Cache processed datasets to avoid recomputation
- **Statistical Results**: Cache test results for parameter changes
- **Visualization**: Efficient chart generation with minimal redrawing

### Memory Management
- **Large Dataset Handling**: Chunked processing for memory efficiency
- **Session Cleanup**: Proper disposal of temporary objects
- **Resource Monitoring**: Optional memory usage tracking

## Future Enhancements

### Planned Features
- **Advanced Statistical Tests**: Panel data analysis, time series tests
- **Real-Time Data**: API integration for live IMF data feeds
- **Multi-Language Support**: Internationalization for broader access
- **Custom Report Templates**: User-defined report generation

### Extensibility
- **Plugin Architecture**: Framework for third-party analysis modules
- **API Endpoints**: RESTful API for programmatic access
- **Batch Processing**: Command-line interface for automated analysis

## Support and Troubleshooting

### Common Issues
1. **Widget Key Errors**: Clear browser cache and refresh application
2. **Data Loading Failures**: Verify file formats match IMF structure
3. **Statistical Test Errors**: Check data completeness and sample sizes
4. **Visualization Problems**: Ensure all required dependencies are installed

### Debug Mode
Enable debug mode by setting `debug=True` in the application configuration for:
- Detailed error messages and stack traces
- Data processing step-by-step logging
- Performance timing information
- Memory usage monitoring

### Performance Monitoring
- **Response Times**: Track analysis computation duration
- **Memory Usage**: Monitor peak memory consumption
- **User Interactions**: Log user behavior for UX improvements

This dashboard suite provides a comprehensive, user-friendly interface for sophisticated capital flows research while maintaining the statistical rigor required for academic publication.