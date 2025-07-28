"""
Case Study 2 Data Processor - Euro Adoption Impact Analysis
Processes IMF BOP and GDP data for Baltic countries (Estonia, Latvia, Lithuania)
Following same methodology as Case Study 1 with temporal comparison framework
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def process_case_study_2_data():
    """
    Process Case Study 2 data for Euro adoption analysis
    Baltic countries: Estonia (2011), Latvia (2014), Lithuania (2015)
    """
    
    print("🔄 Processing Case Study 2 data...")
    
    # Define data paths
    data_dir = Path(__file__).parent.parent / "data"
    bop_file = data_dir / "case_study_2_data_july_27_2025.csv"
    gdp_file = data_dir / "dataset_2025-07-24T18_28_31.898465539Z_DEFAULT_INTEGRATION_IMF.RES_WEO_6.0.0.csv"
    
    # Load raw data
    print("📥 Loading raw BOP and GDP data...")
    case_two_raw = pd.read_csv(bop_file)
    gdp_raw = pd.read_csv(gdp_file)
    
    print(f"   - BOP data shape: {case_two_raw.shape}")
    print(f"   - GDP data shape: {gdp_raw.shape}")
    
    # Filter for Baltic countries only
    baltic_countries = ['Estonia, Republic of', 'Latvia, Republic of', 'Lithuania, Republic of']
    case_two_filtered = case_two_raw[case_two_raw['COUNTRY'].isin(baltic_countries)].copy()
    gdp_filtered = gdp_raw[gdp_raw['COUNTRY'].isin(baltic_countries)].copy()
    
    print(f"📊 Filtered to Baltic countries:")
    print(f"   - BOP observations: {case_two_filtered.shape[0]}")
    print(f"   - GDP observations: {gdp_filtered.shape[0]}")
    print(f"   - Countries: {case_two_filtered['COUNTRY'].unique()}")
    
    # Process BOP data
    print("🔧 Processing BOP data...")
    case_two_clean = case_two_filtered.copy()
    
    # Extract first word from BOP accounting entry
    case_two_clean['ENTRY_FIRST_WORD'] = case_two_clean['BOP_ACCOUNTING_ENTRY'].str.extract(r'^([^,]+)')
    case_two_clean['FULL_INDICATOR'] = case_two_clean['ENTRY_FIRST_WORD'] + ' - ' + case_two_clean['INDICATOR']
    
    # Drop unnecessary columns
    columns_to_drop = ['DATASET', 'SERIES_CODE', 'OBS_MEASURE', 'BOP_ACCOUNTING_ENTRY', 'INDICATOR', 'ENTRY_FIRST_WORD', 'FREQUENCY']
    case_two_clean = case_two_clean.drop(columns=[col for col in columns_to_drop if col in case_two_clean.columns])
    
    # Convert from wide to long format
    print("🔄 Converting from wide to long format...")
    time_columns = [col for col in case_two_clean.columns if col.startswith(('19', '20'))]
    
    case_two_long = pd.melt(
        case_two_clean,
        id_vars=['COUNTRY', 'FULL_INDICATOR', 'UNIT', 'SCALE'],
        value_vars=time_columns,
        var_name='TIME_PERIOD',
        value_name='OBS_VALUE'
    )
    
    # Clean time periods and extract year/quarter
    case_two_long['TIME_PERIOD'] = case_two_long['TIME_PERIOD'].str.strip()
    case_two_long[['YEAR', 'QUARTER']] = case_two_long['TIME_PERIOD'].str.split('-', expand=True)
    case_two_long['YEAR'] = case_two_long['YEAR'].astype(int)
    case_two_long['QUARTER'] = case_two_long['QUARTER'].str.extract(r'(\d+)').astype(int)
    
    # Convert values to numeric
    case_two_long['OBS_VALUE'] = pd.to_numeric(case_two_long['OBS_VALUE'], errors='coerce')
    
    # Remove SCALE column if it exists (following Case Study 1 methodology)
    if 'SCALE' in case_two_long.columns:
        case_two_long = case_two_long.drop('SCALE', axis=1)
    
    # Pivot to wide format by indicator
    print("🔄 Pivoting data by indicator...")
    bop_pivoted = case_two_long.pivot_table(
        index=['COUNTRY', 'YEAR', 'QUARTER', 'UNIT'],
        columns='FULL_INDICATOR',
        values='OBS_VALUE',
        aggfunc='first'
    ).reset_index()
    
    print(f"   - BOP pivoted shape: {bop_pivoted.shape}")
    print(f"   - Available indicators: {len([col for col in bop_pivoted.columns if col not in ['COUNTRY', 'YEAR', 'QUARTER', 'UNIT']])}")
    
    # Process GDP data
    print("🔧 Processing GDP data...")
    gdp_clean = gdp_filtered[['COUNTRY', 'TIME_PERIOD', 'INDICATOR', 'OBS_VALUE']].copy()
    gdp_clean['TIME_PERIOD'] = gdp_clean['TIME_PERIOD'].astype(int)  # GDP is annual
    gdp_clean['OBS_VALUE'] = pd.to_numeric(gdp_clean['OBS_VALUE'], errors='coerce')
    
    # Pivot GDP data
    gdp_pivoted = gdp_clean.pivot_table(
        index=['COUNTRY', 'TIME_PERIOD'],
        columns='INDICATOR',
        values='OBS_VALUE',
        aggfunc='first'
    ).reset_index()
    
    print(f"   - GDP pivoted shape: {gdp_pivoted.shape}")
    
    # Join BOP and GDP data
    print("🔗 Joining BOP and GDP data...")
    merged_data = bop_pivoted.merge(
        gdp_pivoted,
        left_on=['COUNTRY', 'YEAR'],
        right_on=['COUNTRY', 'TIME_PERIOD'],
        how='left'
    )
    
    if 'TIME_PERIOD' in merged_data.columns:
        merged_data = merged_data.drop('TIME_PERIOD', axis=1)
    
    print(f"   - Merged data shape: {merged_data.shape}")
    
    # Identify columns for normalization
    gdp_col = 'Gross domestic product (GDP), Current prices, US dollar'
    metadata_cols = ['COUNTRY', 'YEAR', 'QUARTER', 'UNIT']
    
    if gdp_col not in merged_data.columns:
        available_gdp_cols = [col for col in merged_data.columns if 'GDP' in col.upper()]
        if available_gdp_cols:
            gdp_col = available_gdp_cols[0]
            print(f"   - Using GDP column: {gdp_col}")
        else:
            raise ValueError("No GDP column found in merged data")
    
    indicator_cols = [col for col in merged_data.columns if col not in metadata_cols + [gdp_col]]
    
    print(f"   - Found {len(indicator_cols)} BOP indicators to normalize")
    print(f"   - GDP column: {gdp_col}")
    
    # Normalize to % of GDP (annualized)
    print("📊 Normalizing BOP flows to annualized % of GDP...")
    normalized_data = merged_data[metadata_cols + [gdp_col]].copy()
    
    # Handle scale issues - GDP is in billions, BOP likely in millions
    gdp_values = merged_data[gdp_col]
    print(f"   - GDP range: ${gdp_values.min():.0f} to ${gdp_values.max():.0f}")
    
    for col in indicator_cols:
        if col in merged_data.columns:
            # Convert to % of GDP and annualize (multiply quarterly by 4)
            normalized_data[f"{col}_PGDP"] = (merged_data[col] * 4 / merged_data[gdp_col]) * 100
    
    # Update unit
    normalized_data['UNIT'] = "% of GDP (annualized)"
    
    # Add Euro adoption timeline classification
    print("🗓️ Adding Euro adoption timeline classification...")
    timeline = {
        'Estonia, Republic of': {
            'adoption_date': '2011-01-01',
            'pre_period': (2005, 2010),
            'post_period': (2012, 2017),
            'adoption_year': 2011
        },
        'Latvia, Republic of': {
            'adoption_date': '2014-01-01', 
            'pre_period': (2007, 2012),
            'post_period': (2015, 2020),
            'adoption_year': 2014
        },
        'Lithuania, Republic of': {
            'adoption_date': '2015-01-01',
            'pre_period': (2008, 2013), 
            'post_period': (2016, 2021),
            'adoption_year': 2015
        }
    }
    
    normalized_data['EURO_PERIOD'] = 'Other'
    
    for country, periods in timeline.items():
        country_mask = normalized_data['COUNTRY'] == country
        pre_start, pre_end = periods['pre_period']
        post_start, post_end = periods['post_period']
        
        pre_mask = (normalized_data['YEAR'] >= pre_start) & (normalized_data['YEAR'] <= pre_end)
        post_mask = (normalized_data['YEAR'] >= post_start) & (normalized_data['YEAR'] <= post_end)
        
        normalized_data.loc[country_mask & pre_mask, 'EURO_PERIOD'] = 'Pre-Euro'
        normalized_data.loc[country_mask & post_mask, 'EURO_PERIOD'] = 'Post-Euro'
    
    # Filter to analysis periods only
    final_data = normalized_data[
        normalized_data['EURO_PERIOD'].isin(['Pre-Euro', 'Post-Euro'])
    ].copy()
    
    print(f"📊 Final dataset statistics:")
    print(f"   - Final shape: {final_data.shape}")
    print(f"   - Analysis periods: {final_data['EURO_PERIOD'].value_counts().to_dict()}")
    print(f"   - Time range: {final_data['YEAR'].min()} to {final_data['YEAR'].max()}")
    
    # Get analysis indicators (normalized ones)
    analysis_indicators = [col for col in final_data.columns if col.endswith('_PGDP')]
    print(f"   - Analysis indicators: {len(analysis_indicators)}")
    
    # Save processed data
    output_file = data_dir / "case_study_2_euro_adoption_data.csv"
    gdp_output_file = data_dir / "case_study_2_gdp_data.csv"
    
    final_data.to_csv(output_file, index=False)
    gdp_filtered.to_csv(gdp_output_file, index=False)
    
    print(f"💾 Saved processed data:")
    print(f"   - Main dataset: {output_file}")
    print(f"   - GDP dataset: {gdp_output_file}")
    
    # Summary statistics
    print("\n📈 Summary by country and period:")
    summary = final_data.groupby(['COUNTRY', 'EURO_PERIOD']).agg({
        'YEAR': ['min', 'max', 'count']
    }).round(2)
    print(summary)
    
    print("\n✅ Case Study 2 data processing complete!")
    
    return final_data, analysis_indicators, {
        'bop_shape': case_two_raw.shape,
        'gdp_shape': gdp_raw.shape,
        'final_shape': final_data.shape,
        'n_indicators': len(analysis_indicators),
        'countries': baltic_countries,
        'timeline': timeline
    }

if __name__ == "__main__":
    # Run data processing
    final_data, indicators, metadata = process_case_study_2_data()
    
    print(f"\n🎯 Ready for analysis:")
    print(f"   - Countries: {len(metadata['countries'])}")
    print(f"   - Indicators: {metadata['n_indicators']}")
    print(f"   - Time periods: Pre-Euro vs Post-Euro")
    print(f"   - Observations: {metadata['final_shape'][0]:,}")