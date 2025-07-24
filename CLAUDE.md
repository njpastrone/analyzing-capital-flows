# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an R-based capital flows research analysis project focused on comparative case studies. The project uses Quarto documents (.qmd) for literate programming, combining R code with analysis documentation.

## Project Structure

- `src/`: Contains analysis documents (Quarto .qmd files)
- `data/`: Raw datasets including:
  - Balance of Payments data from IMF
  - GDP data from World Economic Outlook
  - Excel metadata files
  - Processed CSV outputs

## Development Environment

This is an RStudio project (`.Rproj` file present) configured with:
- 2-space indentation
- UTF-8 encoding
- Code indexing enabled

## Key Dependencies

The analysis relies on these R packages:
- `tidyverse`: Core data manipulation and visualization
- `readr`: CSV file reading
- `stringr`: String manipulation
- `ggplot2`: Data visualization (part of tidyverse)
- `knitr`: Document generation
- `gridExtra`: Layout utilities for plots and tables

## Analysis Workflow

The main analysis is in `src/Cleaning Case Study 1.qmd` which follows this pattern:

1. **Data Import**: Reads raw BOP and GDP CSV files from `data/`
2. **Data Cleaning**: 
   - Extracts indicator names from BOP accounting entries
   - Separates time periods into year/quarter
   - Pivots data to wide format
3. **Data Integration**: Joins BOP and GDP data by country and year
4. **Normalization**: Converts BOP flows to percentage of GDP (annualized)
5. **Grouping**: Creates country groups (Iceland vs. Eurozone)
6. **Analysis**: Generates summary statistics and time series visualizations

## Data Processing Patterns

- All monetary values are converted to "% of GDP" for comparison
- BOP data is annualized by multiplying quarterly values by 4
- Data filtering often excludes Luxembourg due to its outlier status
- Time series plotting uses continuous date representation

## Output Generation

- Cleaned datasets are exported as CSV files to `data/`
- Analysis includes both summary statistics tables and time series plots
- Uses automated report generation functions for multiple indicators

## Working with Quarto Documents

To render the analysis:
- Use RStudio's "Render" button or `quarto render` command
- Output format is HTML by default
- Visual editor is enabled for the .qmd files