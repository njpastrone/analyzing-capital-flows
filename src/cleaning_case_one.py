import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('default')

print("="*60)
print("CASE STUDY 1: ICELAND vs EUROZONE CAPITAL FLOWS ANALYSIS")
print("="*60)

# Load raw BOP data
print("\n1. Loading and cleaning raw BOP data...")
case_one_raw = pd.read_csv("../data/case_study_1_data_july_24_2025.csv")
print(f"Raw BOP data shape: {case_one_raw.shape}")
print("\nFirst few rows of raw BOP data:")
print(case_one_raw.head())

# Clean BOP data - extract first word from BOP_ACCOUNTING_ENTRY and create FULL_INDICATOR
case_one_new_cols = case_one_raw.copy()
case_one_new_cols['ENTRY_FIRST_WORD'] = case_one_new_cols['BOP_ACCOUNTING_ENTRY'].str.extract(r'^([^,]+)')
case_one_new_cols['FULL_INDICATOR'] = case_one_new_cols['ENTRY_FIRST_WORD'] + ' - ' + case_one_new_cols['INDICATOR']

# Drop unnecessary columns and reorder
columns_to_drop = ['BOP_ACCOUNTING_ENTRY', 'INDICATOR', 'ENTRY_FIRST_WORD', 'FREQUENCY', 'SCALE']
case_one_new_cols = case_one_new_cols.drop(columns=columns_to_drop)

# Reorder columns
cols_ordered = ['COUNTRY', 'TIME_PERIOD', 'FULL_INDICATOR'] + [col for col in case_one_new_cols.columns if col not in ['COUNTRY', 'TIME_PERIOD', 'FULL_INDICATOR']]
case_one_new_cols = case_one_new_cols[cols_ordered]

print("\nAfter initial cleaning:")
print(case_one_new_cols.head())

# Separate TIME_PERIOD into YEAR and QUARTER
case_one_final_cols = case_one_new_cols.copy()
case_one_final_cols[['YEAR', 'QUARTER']] = case_one_final_cols['TIME_PERIOD'].str.split('-', expand=True)
case_one_final_cols['YEAR'] = case_one_final_cols['YEAR'].astype(int)
case_one_final_cols['QUARTER'] = case_one_final_cols['QUARTER'].str.extract(r'(\d+)').astype(int)
case_one_final_cols = case_one_final_cols.drop('TIME_PERIOD', axis=1)

print("\nAfter separating time periods:")
print(case_one_final_cols.head())

# Display unique countries and indicators
print(f"\nUnique countries: {sorted(case_one_final_cols['COUNTRY'].unique())}")
print(f"\nNumber of unique indicators: {len(case_one_final_cols['FULL_INDICATOR'].unique())}")
print("\nFull indicators:")
for indicator in sorted(case_one_final_cols['FULL_INDICATOR'].unique()):
    print(f"  - {indicator}")

# Load and clean GDP data
print("\n2. Loading and cleaning raw GDP data...")
gdp_raw = pd.read_csv("../data/dataset_2025-07-24T18_28_31.898465539Z_DEFAULT_INTEGRATION_IMF.RES_WEO_6.0.0.csv")
print(f"Raw GDP data shape: {gdp_raw.shape}")
print("\nFirst few rows of raw GDP data:")
print(gdp_raw.head())

# Clean GDP data - select relevant columns
gdp_cleaned = gdp_raw[['COUNTRY', 'TIME_PERIOD', 'INDICATOR', 'OBS_VALUE']].copy()
print("\nCleaned GDP data:")
print(gdp_cleaned.head())

# Pivot both datasets wider
print("\n3. Pivoting datasets to wide format...")

# Pivot BOP data
case_one_pivoted = case_one_final_cols.pivot_table(
    index=['COUNTRY', 'YEAR', 'QUARTER', 'UNIT'],
    columns='FULL_INDICATOR',
    values='OBS_VALUE',
    aggfunc='first'
).reset_index()

print(f"BOP pivoted shape: {case_one_pivoted.shape}")
print("\nBOP pivoted columns:")
print(case_one_pivoted.columns.tolist())

# Pivot GDP data  
gdp_pivoted = gdp_cleaned.pivot_table(
    index=['COUNTRY', 'TIME_PERIOD'],
    columns='INDICATOR',
    values='OBS_VALUE',
    aggfunc='first'
).reset_index()

print(f"\nGDP pivoted shape: {gdp_pivoted.shape}")
print("\nGDP pivoted columns:")
print(gdp_pivoted.columns.tolist())

# Join BOP and GDP data
print("\n4. Joining BOP and GDP datasets...")
case_one_joined = case_one_pivoted.merge(
    gdp_pivoted,
    left_on=['COUNTRY', 'YEAR'],
    right_on=['COUNTRY', 'TIME_PERIOD'],
    how='left'
)

print(f"Joined data shape: {case_one_joined.shape}")
print("\nJoined data sample:")
print(case_one_joined.head())

# Clean up the unit column and remove TIME_PERIOD
case_one_join_clean = case_one_joined.copy()
case_one_join_clean['UNIT'] = case_one_join_clean['UNIT'] + ", Nominal (Current Prices)"
case_one_join_clean = case_one_join_clean.drop('TIME_PERIOD', axis=1)

print(f"\nFinal cleaned data shape: {case_one_join_clean.shape}")
print("\nColumn names:")
print(case_one_join_clean.columns.tolist())

# Normalize BOP flows as percentage of GDP (annualized)
print("\n5. Converting BOP flows to % of GDP (annualized)...")

# Identify BOP indicator columns (excluding metadata columns)
metadata_cols = ['COUNTRY', 'YEAR', 'QUARTER', 'UNIT']
gdp_col = 'Gross domestic product (GDP), Current prices, US dollar'
bop_indicator_cols = [col for col in case_one_join_clean.columns 
                     if col not in metadata_cols and col != gdp_col]

print(f"Found {len(bop_indicator_cols)} BOP indicators to normalize")

# Create normalized columns (annualized BOP as % of GDP)
full_case_one_df = case_one_join_clean[metadata_cols + [gdp_col]].copy()

for col in bop_indicator_cols:
    # Annualize (multiply by 4) and convert to % of GDP, then multiply by 100 for percentage
    normalized_col = f"{col}_PGDP"
    full_case_one_df[normalized_col] = (case_one_join_clean[col] * 4 / case_one_join_clean[gdp_col]) * 100

# Update unit description
full_case_one_df['UNIT'] = full_case_one_df['UNIT'] + ", % of GDP"

print(f"Normalized data shape: {full_case_one_df.shape}")
print("\nNormalized data sample:")
print(full_case_one_df.head())

# Create country groupings: Iceland vs 10-country Eurozone bloc
print("\n6. Creating country groupings...")

# Define the 10 Euro adoption countries (excluding Luxembourg as outlier)
eurozone_countries = ['Austria', 'Belgium', 'Finland', 'France', 'Germany', 
                     'Ireland', 'Italy', 'Netherlands', 'Portugal', 'Spain']

# Add GROUP column
full_case_one_grouped = full_case_one_df.copy()
full_case_one_grouped['GROUP'] = full_case_one_grouped['COUNTRY'].apply(
    lambda x: 'Iceland' if x == 'Iceland' else 'Eurozone'
)

# Reorder columns to put GROUP after COUNTRY
cols = full_case_one_grouped.columns.tolist()
cols.insert(1, cols.pop(cols.index('GROUP')))
full_case_one_grouped = full_case_one_grouped[cols]

print(f"Grouped data shape: {full_case_one_grouped.shape}")
print("\nCountry distribution:")
print(full_case_one_grouped['GROUP'].value_counts())
print("\nCountries by group:")
for group in ['Iceland', 'Eurozone']:
    countries = full_case_one_grouped[full_case_one_grouped['GROUP'] == group]['COUNTRY'].unique()
    print(f"{group}: {sorted(countries)}")

# Remove Luxembourg (outlier) from analysis
print("\n7. Removing Luxembourg (financial center outlier)...")
full_case_one_grouped_no_lux = full_case_one_grouped[full_case_one_grouped['COUNTRY'] != 'Luxembourg'].copy()

print(f"Data without Luxembourg shape: {full_case_one_grouped_no_lux.shape}")
print("\nFinal country distribution:")
print(full_case_one_grouped_no_lux['GROUP'].value_counts())
print("\nFinal countries by group:")
for group in ['Iceland', 'Eurozone']:
    countries = full_case_one_grouped_no_lux[full_case_one_grouped_no_lux['GROUP'] == group]['COUNTRY'].unique()
    print(f"{group}: {sorted(countries)}")

# Save the cleaned dataset
output_file = "case_one_grouped_python.csv"
full_case_one_grouped_no_lux.to_csv(output_file, index=False)
print(f"\nSaved cleaned dataset to: {output_file}")

# Get list of normalized indicator columns for analysis
indicator_cols = [col for col in full_case_one_grouped_no_lux.columns if col.endswith('_PGDP')]
print(f"\n{len(indicator_cols)} indicators available for analysis:")
for i, col in enumerate(indicator_cols, 1):
    clean_name = col.replace('_PGDP', '')
    print(f"{i:2d}. {clean_name}")

print("\n" + "="*60)
print("STATISTICAL ANALYSIS")
print("="*60)

# Generate comprehensive summary statistics
print("\n8. Generating summary statistics by group...")

def calculate_comprehensive_stats(data, group_col, indicator_cols):
    """Calculate comprehensive statistics including skewness and coefficient of variation"""
    results = []
    
    for group in data[group_col].unique():
        group_data = data[data[group_col] == group]
        
        for indicator in indicator_cols:
            values = group_data[indicator].dropna()
            
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                cv = (std_val / abs(mean_val)) * 100 if mean_val != 0 else np.inf
                skew_val = stats.skew(values)
                
                results.append({
                    'Group': group,
                    'Indicator': indicator.replace('_PGDP', ''),
                    'Count': len(values),
                    'Mean': mean_val,
                    'Std_Dev': std_val,
                    'Min': values.min(),
                    'Max': values.max(),
                    'Skewness': skew_val,
                    'Coeff_of_Variation': cv
                })
    
    return pd.DataFrame(results)

# Calculate statistics by group
summary_stats = calculate_comprehensive_stats(
    full_case_one_grouped_no_lux, 'GROUP', indicator_cols
)

print("Summary Statistics by Group:")
print("="*80)

# Display formatted summary statistics
for indicator in indicator_cols[:3]:  # Show first 3 indicators as example
    clean_name = indicator.replace('_PGDP', '')
    print(f"\n{clean_name}:")
    print("-" * len(clean_name))
    
    indicator_stats = summary_stats[summary_stats['Indicator'] == clean_name]
    for _, row in indicator_stats.iterrows():
        print(f"{row['Group']:10s}: Mean={row['Mean']:8.2f}, SD={row['Std_Dev']:7.2f}, "
              f"CV={row['Coeff_of_Variation']:6.1f}%, Skew={row['Skewness']:6.2f}")

# Calculate statistics by individual country for detailed analysis
print(f"\n9. Generating detailed country-level statistics...")

country_stats = calculate_comprehensive_stats(
    full_case_one_grouped_no_lux, 'COUNTRY', indicator_cols
)

# Display Iceland vs average Eurozone country comparison
print("\nIceland vs Individual Eurozone Countries (first 3 indicators):")
print("="*70)

for indicator in indicator_cols[:3]:
    clean_name = indicator.replace('_PGDP', '')
    print(f"\n{clean_name}:")
    print("-" * len(clean_name))
    
    # Iceland stats
    iceland_stats = country_stats[
        (country_stats['Indicator'] == clean_name) & 
        (country_stats['Group'] == 'Iceland')
    ]
    
    if not iceland_stats.empty:
        row = iceland_stats.iloc[0]
        print(f"{'Iceland':12s}: Mean={row['Mean']:8.2f}, SD={row['Std_Dev']:7.2f}, "
              f"CV={row['Coeff_of_Variation']:6.1f}%, Skew={row['Skewness']:6.2f}")
    
    # Eurozone countries stats
    eurozone_stats = country_stats[
        (country_stats['Indicator'] == clean_name) & 
        (full_case_one_grouped_no_lux[full_case_one_grouped_no_lux['COUNTRY'].isin(
            country_stats['Group']
        )]['GROUP'] == 'Eurozone')
    ]
    
    # Show a few example Eurozone countries
    eurozone_countries_sample = ['Germany', 'France', 'Spain', 'Italy']
    for country in eurozone_countries_sample:
        country_stat = country_stats[
            (country_stats['Indicator'] == clean_name) & 
            (country_stats['Group'] == country)
        ]
        if not country_stat.empty:
            row = country_stat.iloc[0]
            print(f"{country:12s}: Mean={row['Mean']:8.2f}, SD={row['Std_Dev']:7.2f}, "
                  f"CV={row['Coeff_of_Variation']:6.1f}%, Skew={row['Skewness']:6.2f}")

# Test Hypothesis 1: Iceland shows more volatility than Eurozone
print(f"\n10. Testing Hypothesis 1: Iceland's capital flows show more volatility...")
print("="*70)

volatility_comparison = []

for indicator in indicator_cols:
    clean_name = indicator.replace('_PGDP', '')
    
    # Get Iceland data
    iceland_data = full_case_one_grouped_no_lux[
        full_case_one_grouped_no_lux['GROUP'] == 'Iceland'
    ][indicator].dropna()
    
    # Get Eurozone data
    eurozone_data = full_case_one_grouped_no_lux[
        full_case_one_grouped_no_lux['GROUP'] == 'Eurozone'
    ][indicator].dropna()
    
    if len(iceland_data) > 1 and len(eurozone_data) > 1:
        # Calculate variances
        iceland_var = iceland_data.var()
        eurozone_var = eurozone_data.var()
        
        # F-test for equal variances
        f_stat = iceland_var / eurozone_var if eurozone_var != 0 else np.inf
        df1 = len(iceland_data) - 1
        df2 = len(eurozone_data) - 1
        
        # Two-tailed p-value
        p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
        
        volatility_comparison.append({
            'Indicator': clean_name,
            'Iceland_Variance': iceland_var,
            'Eurozone_Variance': eurozone_var,
            'F_Statistic': f_stat,
            'P_Value': p_value,
            'Iceland_More_Volatile': iceland_var > eurozone_var,
            'Significant_5pct': p_value < 0.05
        })

# Convert to DataFrame and display results
volatility_df = pd.DataFrame(volatility_comparison)

print("Volatility Comparison Results:")
print("-" * 30)
print(f"{'Indicator':<25} {'Iceland Var':<12} {'Euro Var':<10} {'F-stat':<8} {'P-value':<8} {'Iceland > Euro':<13} {'Sig(5%)'}")
print("-" * 95)

for _, row in volatility_df.head(10).iterrows():  # Show first 10 indicators
    print(f"{row['Indicator']:<25} {row['Iceland_Variance']:<12.2f} {row['Eurozone_Variance']:<10.2f} "
          f"{row['F_Statistic']:<8.2f} {row['P_Value']:<8.3f} {str(row['Iceland_More_Volatile']):<13} {str(row['Significant_5pct'])}")

# Summary of hypothesis testing
iceland_more_volatile_count = sum(volatility_df['Iceland_More_Volatile'])
total_indicators = len(volatility_df)
significant_differences = sum(volatility_df['Significant_5pct'])

print(f"\nHypothesis 1 Summary:")
print(f"- Iceland shows higher volatility in {iceland_more_volatile_count}/{total_indicators} indicators ({iceland_more_volatile_count/total_indicators*100:.1f}%)")
print(f"- Statistically significant differences (p<0.05): {significant_differences}/{total_indicators} indicators ({significant_differences/total_indicators*100:.1f}%)")

# Save detailed statistics
summary_stats.to_csv('summary_statistics_by_group.csv', index=False)
country_stats.to_csv('summary_statistics_by_country.csv', index=False) 
volatility_df.to_csv('volatility_comparison_results.csv', index=False)

print(f"\nSaved statistical analysis results:")
print(f"- summary_statistics_by_group.csv")
print(f"- summary_statistics_by_country.csv") 
print(f"- volatility_comparison_results.csv")

print("\n" + "="*60)
print("DATA VISUALIZATION")
print("="*60)

# Create visualizations
print("\n11. Creating side-by-side boxplots...")

# Prepare data for boxplot visualization
boxplot_data = []
for group in ['Iceland', 'Eurozone']:
    group_summary = summary_stats[summary_stats['Group'] == group]
    for _, row in group_summary.iterrows():
        boxplot_data.extend([
            {'Group': group, 'Statistic': 'Mean', 'Value': row['Mean'], 'Indicator': row['Indicator']},
            {'Group': group, 'Statistic': 'Std_Dev', 'Value': row['Std_Dev'], 'Indicator': row['Indicator']}
        ])

boxplot_df = pd.DataFrame(boxplot_data)

# Create figure with subplots for mean and standard deviation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Boxplot for means
mean_data = boxplot_df[boxplot_df['Statistic'] == 'Mean']
sns.boxplot(data=mean_data, x='Group', y='Value', ax=ax1)
ax1.set_title('Distribution of Means by Group\n(All Capital Flow Indicators)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Value (% of GDP)')
ax1.grid(True, alpha=0.3)

# Boxplot for standard deviations
std_data = boxplot_df[boxplot_df['Statistic'] == 'Std_Dev']
sns.boxplot(data=std_data, x='Group', y='Value', ax=ax2)
ax2.set_title('Distribution of Standard Deviations by Group\n(All Capital Flow Indicators)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Standard Deviation (% of GDP)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('capital_flows_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

print("Saved boxplot visualization: capital_flows_boxplots.png")

# Time series analysis and visualization
print("\n12. Creating time series charts...")

# Create date column for time series
full_case_one_grouped_no_lux['Date'] = pd.to_datetime(
    full_case_one_grouped_no_lux['YEAR'].astype(str) + '-' + 
    ((full_case_one_grouped_no_lux['QUARTER'] - 1) * 3 + 1).astype(str) + '-01'
)

def create_time_series_plot(data, indicator, figsize=(12, 8)):
    """Create side-by-side time series plots for Iceland vs Eurozone"""
    clean_name = indicator.replace('_PGDP', '')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Iceland plot
    iceland_data = data[data['GROUP'] == 'Iceland']
    ax1.plot(iceland_data['Date'], iceland_data[indicator], color='#d62728', linewidth=2, marker='o', markersize=4)
    ax1.set_title(f'Iceland - {clean_name}', fontsize=12, fontweight='bold', color='#d62728')
    ax1.set_ylabel('% of GDP')
    ax1.grid(True, alpha=0.3)
    
    # Eurozone plot
    eurozone_data = data[data['GROUP'] == 'Eurozone']
    for country in eurozone_data['COUNTRY'].unique():
        country_data = eurozone_data[eurozone_data['COUNTRY'] == country]
        ax2.plot(country_data['Date'], country_data[indicator], alpha=0.7, linewidth=1, label=country)
    
    ax2.set_title(f'Eurozone Countries - {clean_name}', fontsize=12, fontweight='bold', color='#1f77b4')
    ax2.set_ylabel('% of GDP')
    ax2.set_xlabel('Year')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    return fig

# Generate time series plots for first 5 indicators (to avoid too many plots)
selected_indicators = indicator_cols[:5]
print(f"Generating time series plots for {len(selected_indicators)} indicators...")

for i, indicator in enumerate(selected_indicators, 1):
    clean_name = indicator.replace('_PGDP', '')
    print(f"  {i}. Creating plot for: {clean_name}")
    
    fig = create_time_series_plot(full_case_one_grouped_no_lux, indicator)
    # Clean filename by removing problematic characters
    clean_filename = clean_name.replace(" ", "_").replace("-", "_").replace(",", "").replace("/", "_").replace("(", "").replace(")", "").lower()
    filename = f'timeseries_{clean_filename}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

print(f"\nSaved {len(selected_indicators)} time series plots")

# Create a summary comparison plot
print("\n13. Creating volatility comparison visualization...")

# Select most interesting indicators based on volatility differences
top_volatile_indicators = volatility_df.nlargest(6, 'F_Statistic')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (_, row) in enumerate(top_volatile_indicators.iterrows()):
    indicator = row['Indicator'] + '_PGDP'
    
    # Get data for this indicator
    iceland_data = full_case_one_grouped_no_lux[
        full_case_one_grouped_no_lux['GROUP'] == 'Iceland'
    ][indicator].dropna()
    
    eurozone_data = full_case_one_grouped_no_lux[
        full_case_one_grouped_no_lux['GROUP'] == 'Eurozone'  
    ][indicator].dropna()
    
    # Create box plot comparison
    data_for_plot = pd.DataFrame({
        'Values': list(iceland_data) + list(eurozone_data),
        'Group': ['Iceland'] * len(iceland_data) + ['Eurozone'] * len(eurozone_data)
    })
    
    sns.boxplot(data=data_for_plot, x='Group', y='Values', ax=axes[i])
    axes[i].set_title(f'{row["Indicator"]}\n(F-stat: {row["F_Statistic"]:.2f}, p: {row["P_Value"]:.3f})', 
                     fontsize=10, fontweight='bold')
    axes[i].set_ylabel('% of GDP')
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Top 6 Indicators by Volatility Difference\n(Iceland vs Eurozone)', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('top_volatile_indicators_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Saved volatility comparison: top_volatile_indicators_comparison.png")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

print(f"\nFiles generated:")
print(f"1. case_one_grouped_python.csv - Main cleaned dataset")
print(f"2. summary_statistics_by_group.csv - Group-level statistics")
print(f"3. summary_statistics_by_country.csv - Country-level statistics")
print(f"4. volatility_comparison_results.csv - Hypothesis testing results")
print(f"5. capital_flows_boxplots.png - Distribution comparison")
print(f"6. top_volatile_indicators_comparison.png - Volatility analysis")
print(f"7. {len(selected_indicators)} time series plots")

print(f"\nKey Findings:")
print(f"- Dataset contains {len(indicator_cols)} capital flow indicators")
print(f"- Analysis covers Iceland vs {len(full_case_one_grouped_no_lux[full_case_one_grouped_no_lux['GROUP'] == 'Eurozone']['COUNTRY'].unique())} Eurozone countries")
print(f"- Iceland shows higher volatility in {iceland_more_volatile_count}/{total_indicators} indicators")
print(f"- {significant_differences} indicators show statistically significant volatility differences")

print(f"\nFor detailed analysis, review the generated CSV files and visualizations.")