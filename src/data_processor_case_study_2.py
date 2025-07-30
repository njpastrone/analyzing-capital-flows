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
    
    # UNIT SCALING CORRECTION: Apply scaling based on SCALE metadata before dropping
    if 'SCALE' in case_two_long.columns:
        print("🔧 Applying BOP unit scaling correction...")
        
        # Check for scaling patterns - Case Study 2 has smaller values already in millions
        sample_values = case_two_long['OBS_VALUE'].dropna().head(1000)
        max_val = abs(sample_values).max() if len(sample_values) > 0 else 0
        
        # Case Study 2 pattern: SCALE="Millions" and values are already scaled (small values)
        if max_val > 1000000:  # Large values need scaling down
            print("   - Converting raw dollar values to millions")
            case_two_long.loc[case_two_long['SCALE'] == 'Millions', 'OBS_VALUE'] = \
                case_two_long.loc[case_two_long['SCALE'] == 'Millions', 'OBS_VALUE'] / 1_000_000
        else:
            print("   - BOP data already in correct scale (millions)")
        
        # Now safe to drop SCALE column
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
    gdp_clean = gdp_filtered[['COUNTRY', 'TIME_PERIOD', 'INDICATOR', 'OBS_VALUE', 'SCALE']].copy()
    gdp_clean['TIME_PERIOD'] = gdp_clean['TIME_PERIOD'].astype(int)  # GDP is annual
    gdp_clean['OBS_VALUE'] = pd.to_numeric(gdp_clean['OBS_VALUE'], errors='coerce')
    
    # UNIT SCALING CORRECTION: Create scaled version for calculations but keep original for display
    gdp_display_values = gdp_clean['OBS_VALUE'].copy()  # Keep original values for final dataset
    
    if 'SCALE' in gdp_clean.columns:
        print("🔧 Applying GDP unit scaling correction...")
        
        # Check for scaling patterns
        sample_values = gdp_clean['OBS_VALUE'].dropna().head(100)
        max_val = abs(sample_values).max() if len(sample_values) > 0 else 0
        
        # GDP pattern: SCALE="Billions" but values are raw dollars (need division)
        if max_val > 1000000000:  # Large values need scaling down
            print("   - Converting raw dollar values to billions for calculations")
            gdp_clean.loc[gdp_clean['SCALE'] == 'Billions', 'OBS_VALUE'] = \
                gdp_clean.loc[gdp_clean['SCALE'] == 'Billions', 'OBS_VALUE'] / 1_000_000_000
        else:
            print("   - GDP data already in correct scale (billions)")
        
        # Drop SCALE column after correction
        gdp_clean = gdp_clean.drop('SCALE', axis=1)
    
    # Pivot GDP data (scaled for calculations)
    gdp_pivoted = gdp_clean.pivot_table(
        index=['COUNTRY', 'TIME_PERIOD'],
        columns='INDICATOR',
        values='OBS_VALUE',
        aggfunc='first'
    ).reset_index()
    
    # Create GDP dataset with original display values
    gdp_display_clean = gdp_filtered[['COUNTRY', 'TIME_PERIOD', 'INDICATOR', 'OBS_VALUE']].copy()
    gdp_display_clean['TIME_PERIOD'] = gdp_display_clean['TIME_PERIOD'].astype(int)
    gdp_display_clean['OBS_VALUE'] = pd.to_numeric(gdp_display_clean['OBS_VALUE'], errors='coerce')
    
    gdp_display_pivoted = gdp_display_clean.pivot_table(
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
    
    # After scaling correction: GDP is in billions, BOP is in millions
    gdp_values = merged_data[gdp_col]
    print(f"   - GDP range (billions): ${gdp_values.min():.2f} to ${gdp_values.max():.2f}")
    
    for col in indicator_cols:
        if col in merged_data.columns:
            # BOP is in millions, GDP is in billions
            # Convert BOP millions to billions: BOP / 1000
            # Then convert to % of GDP and annualize (multiply quarterly by 4)
            normalized_data[f"{col}_PGDP"] = (merged_data[col] / 1000 * 4 / merged_data[gdp_col]) * 100
    
    # Update unit
    normalized_data['UNIT'] = "% of GDP (annualized)"
    
    # Replace scaled GDP values with original display values
    print("🔄 Restoring original GDP values for display...")
    gdp_display_merged = bop_pivoted[['COUNTRY', 'YEAR']].merge(
        gdp_display_pivoted,
        left_on=['COUNTRY', 'YEAR'],
        right_on=['COUNTRY', 'TIME_PERIOD'],
        how='left'
    )
    
    if 'TIME_PERIOD' in gdp_display_merged.columns:
        gdp_display_merged = gdp_display_merged.drop('TIME_PERIOD', axis=1)
    
    # Replace the scaled GDP column with original values
    if gdp_col in gdp_display_merged.columns:
        normalized_data[gdp_col] = gdp_display_merged[gdp_col]
        print(f"   - GDP range (original): ${normalized_data[gdp_col].min():,.0f} to ${normalized_data[gdp_col].max():,.0f}")
    
    # Add Euro adoption timeline classification with maximized data usage
    print("🗓️ Adding Euro adoption timeline classification...")
    
    # Expanded timeline using all available data (including adoption years in post-Euro periods)
    timeline = {
        'Estonia, Republic of': {
            'adoption_date': '2011-01-01',
            'adoption_year': 2011,
            'pre_period_full': (1999, 2010),      # 12 years before
            'post_period_full': (2011, 2024),     # 14 years after (include 2011 adoption year)
            'crisis_years': [2008, 2009, 2010, 2020, 2021, 2022]  # GFC + COVID-19
        },
        'Latvia, Republic of': {
            'adoption_date': '2014-01-01', 
            'adoption_year': 2014,
            'pre_period_full': (1999, 2013),      # 15 years before  
            'post_period_full': (2014, 2024),     # 11 years after (include 2014 adoption year)
            'crisis_years': [2008, 2009, 2010, 2020, 2021, 2022]  # GFC + COVID-19
        },
        'Lithuania, Republic of': {
            'adoption_date': '2015-01-01',
            'adoption_year': 2015,
            'pre_period_full': (1999, 2014),      # 16 years before
            'post_period_full': (2015, 2024),     # 10 years after (include 2015 adoption year)
            'crisis_years': [2008, 2009, 2010, 2020, 2021, 2022]  # GFC + COVID-19
        }
    }
    
    # Initialize period classifications
    normalized_data['EURO_PERIOD_FULL'] = 'Other'
    normalized_data['EURO_PERIOD_CRISIS_EXCLUDED'] = 'Other'
    normalized_data['IS_CRISIS_YEAR'] = False
    
    # Classify periods for each country
    for country, periods in timeline.items():
        country_mask = normalized_data['COUNTRY'] == country
        
        # Full series periods (uses all available data)
        pre_start_full, pre_end_full = periods['pre_period_full']
        post_start_full, post_end_full = periods['post_period_full']
        
        pre_mask_full = (normalized_data['YEAR'] >= pre_start_full) & (normalized_data['YEAR'] <= pre_end_full)
        post_mask_full = (normalized_data['YEAR'] >= post_start_full) & (normalized_data['YEAR'] <= post_end_full)
        
        # Apply full series classification
        normalized_data.loc[country_mask & pre_mask_full, 'EURO_PERIOD_FULL'] = 'Pre-Euro'
        normalized_data.loc[country_mask & post_mask_full, 'EURO_PERIOD_FULL'] = 'Post-Euro'
        
        # Mark crisis years (GFC 2008-2010 + COVID 2020-2022)
        crisis_years = periods['crisis_years']
        crisis_mask = normalized_data['YEAR'].isin(crisis_years)
        normalized_data.loc[country_mask & crisis_mask, 'IS_CRISIS_YEAR'] = True
        
        # Crisis-excluded periods (same as full but excluding crisis years)
        pre_mask_clean = pre_mask_full & (~crisis_mask)
        post_mask_clean = post_mask_full & (~crisis_mask)
        
        normalized_data.loc[country_mask & pre_mask_clean, 'EURO_PERIOD_CRISIS_EXCLUDED'] = 'Pre-Euro'
        normalized_data.loc[country_mask & post_mask_clean, 'EURO_PERIOD_CRISIS_EXCLUDED'] = 'Post-Euro'
    
    # For backward compatibility, default to full series
    normalized_data['EURO_PERIOD'] = normalized_data['EURO_PERIOD_FULL']
    
    # Filter to analysis periods only (using full series by default)
    final_data = normalized_data[
        normalized_data['EURO_PERIOD_FULL'].isin(['Pre-Euro', 'Post-Euro'])
    ].copy()
    
    # Create crisis-excluded dataset
    final_data_crisis_excluded = normalized_data[
        normalized_data['EURO_PERIOD_CRISIS_EXCLUDED'].isin(['Pre-Euro', 'Post-Euro'])
    ].copy()
    
    print(f"📊 Final dataset statistics:")
    print(f"   - Full series shape: {final_data.shape}")
    print(f"   - Crisis-excluded shape: {final_data_crisis_excluded.shape}")
    print(f"   - Full series periods: {final_data['EURO_PERIOD_FULL'].value_counts().to_dict()}")
    print(f"   - Crisis-excluded periods: {final_data_crisis_excluded['EURO_PERIOD_CRISIS_EXCLUDED'].value_counts().to_dict()}")
    print(f"   - Time range: {final_data['YEAR'].min()} to {final_data['YEAR'].max()}")
    
    # Get analysis indicators
    analysis_indicators = [col for col in final_data.columns if col.endswith('_PGDP')]
    print(f"   - Analysis indicators: {len(analysis_indicators)}")
    
    # Save processed data
    output_file = data_dir / "case_study_2_euro_adoption_data.csv"
    output_file_crisis_excluded = data_dir / "case_study_2_euro_adoption_data_crisis_excluded.csv"
    gdp_output_file = data_dir / "case_study_2_gdp_data.csv"
    
    # Save full series (backward compatible)
    final_data.to_csv(output_file, index=False)
    
    # Save crisis-excluded version
    final_data_crisis_excluded.to_csv(output_file_crisis_excluded, index=False)
    
    # Save GDP reference data
    gdp_reference = final_data[['COUNTRY', 'YEAR', gdp_col]].drop_duplicates()
    gdp_reference.to_csv(gdp_output_file, index=False)
    
    print(f"💾 Saved processed data:")
    print(f"   - Full series dataset: {output_file}")
    print(f"   - Crisis-excluded dataset: {output_file_crisis_excluded}")
    print(f"   - GDP dataset: {gdp_output_file}")
    
    # Print detailed summary by country and period
    print(f"\n📈 Full Series Summary by country and period:")
    summary_full = final_data.groupby(['COUNTRY', 'EURO_PERIOD_FULL'])['YEAR'].agg(['min', 'max', 'count'])
    print(summary_full)
    
    print(f"\n📈 Crisis-Excluded Summary by country and period:")
    summary_crisis = final_data_crisis_excluded.groupby(['COUNTRY', 'EURO_PERIOD_CRISIS_EXCLUDED'])['YEAR'].agg(['min', 'max', 'count'])
    print(summary_crisis)
    
    # Show data improvement
    full_obs = len(final_data)
    crisis_obs = len(final_data_crisis_excluded)
    print(f"\n📊 Data maximization achieved:")
    print(f"   - Crisis years excluded: 2008-2010 (GFC), 2020-2022 (COVID-19)")
    for country in baltic_countries:
        country_data_full = final_data[final_data['COUNTRY'] == country]
        country_data_crisis = final_data_crisis_excluded[final_data_crisis_excluded['COUNTRY'] == country]
        
        pre_full = len(country_data_full[country_data_full['EURO_PERIOD_FULL'] == 'Pre-Euro'])
        post_full = len(country_data_full[country_data_full['EURO_PERIOD_FULL'] == 'Post-Euro'])
        pre_crisis = len(country_data_crisis[country_data_crisis['EURO_PERIOD_CRISIS_EXCLUDED'] == 'Pre-Euro'])  
        post_crisis = len(country_data_crisis[country_data_crisis['EURO_PERIOD_CRISIS_EXCLUDED'] == 'Post-Euro'])
        
        country_short = country.split(',')[0]
        print(f"   - {country_short}: Full ({pre_full} pre + {post_full} post), Crisis-excluded ({pre_crisis} pre + {post_crisis} post)")
    
    print(f"\n✅ Case Study 2 data processing complete!")
    
    print(f"\n🎯 Ready for analysis:")
    print(f"   - Countries: {final_data['COUNTRY'].nunique()}")
    print(f"   - Indicators: {len(analysis_indicators)}")
    print(f"   - Study versions: Full series + Crisis-excluded") 
    print(f"   - Full series observations: {full_obs}")
    print(f"   - Crisis-excluded observations: {crisis_obs}")
    
    return final_data, analysis_indicators, {
        'bop_shape': case_two_raw.shape,
        'gdp_shape': gdp_raw.shape,
        'final_shape': final_data.shape,
        'final_shape_crisis_excluded': final_data_crisis_excluded.shape,
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