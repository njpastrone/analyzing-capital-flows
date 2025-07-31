"""
Expanded BOP Dataset Processor - Additional Capital Flow Metrics
Processes new comprehensive BOP data with 5 additional indicators across 159 countries
Following same methodology as Case Study 1 & 2 with transparent debugging pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def process_expanded_bop_data():
    """
    Process expanded BOP dataset with comprehensive capital flow metrics
    
    Returns:
        tuple: (processed_data, processing_metadata, debug_info)
    """
    
    print("🔄 Processing Expanded BOP Dataset...")
    print("=" * 60)
    
    # Define data paths
    data_dir = Path(__file__).parent.parent / "data"
    bop_file = data_dir / "net_flows_july_30_2025.csv"
    gdp_file = data_dir / "dataset_2025-07-24T18_28_31.898465539Z_DEFAULT_INTEGRATION_IMF.RES_WEO_6.0.0.csv"
    
    # Initialize debug info tracking
    debug_info = {}
    
    # Step 1: Load raw data
    print("📥 STEP 1: Loading raw BOP and GDP data...")
    try:
        bop_raw = pd.read_csv(bop_file)
        gdp_raw = pd.read_csv(gdp_file)
        
        print(f"   ✅ BOP data loaded: {bop_raw.shape} ({bop_raw['COUNTRY'].nunique()} countries)")
        print(f"   ✅ GDP data loaded: {gdp_raw.shape} ({gdp_raw['COUNTRY'].nunique()} countries)")
        
        debug_info['raw_bop_shape'] = bop_raw.shape
        debug_info['raw_gdp_shape'] = gdp_raw.shape
        debug_info['bop_countries'] = bop_raw['COUNTRY'].nunique()
        debug_info['gdp_countries'] = gdp_raw['COUNTRY'].nunique()
        
    except Exception as e:
        print(f"   ❌ Error loading data: {str(e)}")
        return None, None, None
    
    # Step 2: Data structure analysis and cleaning
    print(f"\n📋 STEP 2: Data structure analysis and cleaning...")
    
    # Analyze BOP structure
    print("   BOP Structure Analysis:")
    print(f"     • Countries: {bop_raw['COUNTRY'].nunique()}")
    print(f"     • Indicators: {bop_raw['INDICATOR'].nunique()}")
    time_periods = bop_raw['TIME_PERIOD'].dropna().unique()
    print(f"     • Time range: {sorted(time_periods)[0]} to {sorted(time_periods)[-1]}")
    print(f"     • BOP entries: {bop_raw['BOP_ACCOUNTING_ENTRY'].unique()}")
    
    # List all indicators
    indicators = sorted(bop_raw['INDICATOR'].unique())
    print("     • Available indicators:")
    for i, ind in enumerate(indicators, 1):
        obs_count = len(bop_raw[bop_raw['INDICATOR'] == ind])
        print(f"       {i}. {ind} ({obs_count:,} obs)")
    
    debug_info['indicators'] = indicators
    debug_info['indicators_count'] = len(indicators)
    
    # Step 3: BOP data preprocessing (following Case Study methodology)
    print(f"\n🔧 STEP 3: BOP data preprocessing...")
    
    bop_clean = bop_raw.copy()
    
    # Extract BOP accounting entry first word for grouping
    bop_clean['ENTRY_FIRST_WORD'] = bop_clean['BOP_ACCOUNTING_ENTRY'].str.extract(r'^([^,]+)')
    bop_clean['FULL_INDICATOR'] = bop_clean['ENTRY_FIRST_WORD'] + ' - ' + bop_clean['INDICATOR']
    
    print(f"   ✅ Created FULL_INDICATOR combining entry type + indicator")
    
    # Process time periods (handle missing values)
    bop_clean[['YEAR', 'QUARTER']] = bop_clean['TIME_PERIOD'].str.split('-', expand=True)
    
    # Convert to numeric, handling NaN values
    bop_clean['YEAR'] = pd.to_numeric(bop_clean['YEAR'], errors='coerce')
    bop_clean['QUARTER'] = bop_clean['QUARTER'].str.extract(r'(\\d+)', expand=False)
    bop_clean['QUARTER'] = pd.to_numeric(bop_clean['QUARTER'], errors='coerce')
    
    # Remove rows with invalid time periods
    initial_rows = len(bop_clean)
    bop_clean = bop_clean.dropna(subset=['YEAR', 'QUARTER'])
    final_rows = len(bop_clean)
    
    if initial_rows != final_rows:
        print(f"   ⚠️  Removed {initial_rows - final_rows} rows with invalid time periods")
    
    print(f"   ✅ Parsed time periods into YEAR/QUARTER components")
    
    # Drop unnecessary columns
    columns_to_drop = ['BOP_ACCOUNTING_ENTRY', 'INDICATOR', 'ENTRY_FIRST_WORD', 'FREQUENCY', 'SCALE', 'TIME_PERIOD']
    bop_clean = bop_clean.drop(columns=columns_to_drop)
    
    debug_info['bop_clean_shape'] = bop_clean.shape
    debug_info['full_indicators'] = sorted(bop_clean['FULL_INDICATOR'].unique())
    
    # Step 4: Pivot BOP data
    print(f"\n📊 STEP 4: Pivoting BOP data to wide format...")
    
    try:
        bop_pivoted = bop_clean.pivot_table(
            index=['COUNTRY', 'YEAR', 'QUARTER', 'UNIT'],
            columns='FULL_INDICATOR',
            values='OBS_VALUE',
            aggfunc='first'
        ).reset_index()
        
        print(f"   ✅ BOP pivot successful: {bop_pivoted.shape}")
        print(f"   ✅ Indicators after pivot: {len([col for col in bop_pivoted.columns if 'Net -' in col])}")
        
        debug_info['bop_pivoted_shape'] = bop_pivoted.shape
        
    except Exception as e:
        print(f"   ❌ BOP pivot failed: {str(e)}")
        return None, None, None
    
    # Step 5: Process GDP data
    print(f"\n💰 STEP 5: Processing GDP data...")
    
    gdp_clean = gdp_raw[['COUNTRY', 'TIME_PERIOD', 'INDICATOR', 'OBS_VALUE']].copy()
    
    try:
        gdp_pivoted = gdp_clean.pivot_table(
            index=['COUNTRY', 'TIME_PERIOD'],
            columns='INDICATOR',
            values='OBS_VALUE',
            aggfunc='first'
        ).reset_index()
        
        print(f"   ✅ GDP pivot successful: {gdp_pivoted.shape}")
        
        # Check for GDP column
        gdp_col = 'Gross domestic product (GDP), Current prices, US dollar'
        if gdp_col in gdp_pivoted.columns:
            print(f"   ✅ Found GDP column for normalization")
        else:
            print(f"   ⚠️  GDP column not found. Available: {list(gdp_pivoted.columns)}")
        
        debug_info['gdp_pivoted_shape'] = gdp_pivoted.shape
        debug_info['gdp_column_available'] = gdp_col in gdp_pivoted.columns
        
    except Exception as e:
        print(f"   ❌ GDP pivot failed: {str(e)}")
        return None, None, None
    
    # Step 6: Join BOP and GDP data
    print(f"\n🔗 STEP 6: Joining BOP and GDP datasets...")
    
    try:
        merged_data = bop_pivoted.merge(
            gdp_pivoted,
            left_on=['COUNTRY', 'YEAR'],
            right_on=['COUNTRY', 'TIME_PERIOD'],
            how='left'
        ).drop('TIME_PERIOD', axis=1, errors='ignore')
        
        print(f"   ✅ Data merge successful: {merged_data.shape}")
        
        # Check merge quality
        gdp_matches = merged_data[gdp_col].notna().sum()
        total_rows = len(merged_data)
        match_rate = (gdp_matches / total_rows) * 100
        
        print(f"   ✅ GDP match rate: {gdp_matches}/{total_rows} ({match_rate:.1f}%)")
        
        debug_info['merged_shape'] = merged_data.shape
        debug_info['gdp_match_rate'] = match_rate
        
    except Exception as e:
        print(f"   ❌ Data merge failed: {str(e)}")
        return None, None, None
    
    # Step 7: Pre-normalization debugging sample
    print(f"\n🔍 STEP 7: Pre-normalization debugging sample...")
    
    # Show sample data before normalization
    sample_countries = ['Iceland', 'Germany', 'United States']
    available_sample = [c for c in sample_countries if c in merged_data['COUNTRY'].unique()]
    
    if available_sample:
        sample_country = available_sample[0]
        sample_data = merged_data[merged_data['COUNTRY'] == sample_country].head(3)
        
        print(f"   📋 Sample data for {sample_country} (first 3 rows):")
        indicator_cols = [col for col in merged_data.columns if 'Net -' in col]
        display_cols = ['COUNTRY', 'YEAR', 'QUARTER'] + indicator_cols[:2] + [gdp_col]
        
        for _, row in sample_data[display_cols].iterrows():
            print(f"     {row['COUNTRY']} {row['YEAR']}-Q{row['QUARTER']}: GDP=${row[gdp_col]:,.0f}M")
            for col in indicator_cols[:2]:
                if pd.notna(row[col]):
                    print(f"       {col}: ${row[col]:,.0f}M")
    
    # Step 8: GDP normalization
    print(f"\n📈 STEP 8: GDP normalization (% of GDP, annualized)...")
    
    # Identify columns
    metadata_cols = ['COUNTRY', 'YEAR', 'QUARTER', 'UNIT']
    indicator_cols = [col for col in merged_data.columns if col not in metadata_cols + [gdp_col]]
    
    print(f"   📊 Found {len(indicator_cols)} indicators to normalize")
    
    # Create normalized dataset
    normalized_data = merged_data[metadata_cols + [gdp_col]].copy()
    
    normalization_stats = {}
    for col in indicator_cols:
        # GDP normalization: (BOP * 4 / GDP) * 100
        normalized_data[f"{col}_PGDP"] = (merged_data[col] * 4 / merged_data[gdp_col]) * 100
        
        # Track normalization stats
        original_vals = merged_data[col].dropna()
        normalized_vals = normalized_data[f"{col}_PGDP"].dropna()
        
        if len(normalized_vals) > 0:
            normalization_stats[col] = {
                'original_count': len(original_vals),
                'normalized_count': len(normalized_vals),
                'original_range': (original_vals.min(), original_vals.max()) if len(original_vals) > 0 else (0, 0),
                'normalized_range': (normalized_vals.min(), normalized_vals.max())
            }
    
    normalized_data['UNIT'] = "% of GDP (annualized)"
    
    print(f"   ✅ Normalization complete: {normalized_data.shape}")
    print(f"   ✅ Normalized indicators: {len([col for col in normalized_data.columns if col.endswith('_PGDP')])}")
    
    debug_info['normalization_stats'] = normalization_stats
    debug_info['final_shape'] = normalized_data.shape
    
    # Step 9: Post-normalization sample
    print(f"\n🎯 STEP 9: Post-normalization sample validation...")
    
    if available_sample:
        sample_country = available_sample[0]
        sample_normalized = normalized_data[normalized_data['COUNTRY'] == sample_country].head(3)
        
        normalized_cols = [col for col in normalized_data.columns if col.endswith('_PGDP')]
        display_cols_norm = ['COUNTRY', 'YEAR', 'QUARTER'] + normalized_cols[:2]
        
        print(f"   📋 Normalized sample for {sample_country}:")
        for _, row in sample_normalized[display_cols_norm].iterrows():
            print(f"     {row['COUNTRY']} {row['YEAR']}-Q{row['QUARTER']}:")
            for col in normalized_cols[:2]:
                if pd.notna(row[col]):
                    print(f"       {col}: {row[col]:.3f}% of GDP")
    
    # Step 10: Final processing summary
    print(f"\n✅ STEP 10: Processing complete!")
    print("=" * 60)
    
    # Create processing metadata
    processing_metadata = {
        'source_file': 'net_flows_july_30_2025.csv',
        'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'countries_total': normalized_data['COUNTRY'].nunique(),
        'countries_list': sorted(normalized_data['COUNTRY'].unique()),
        'indicators_count': len([col for col in normalized_data.columns if col.endswith('_PGDP')]),
        'time_range': (normalized_data['YEAR'].min(), normalized_data['YEAR'].max()),
        'observations_total': len(normalized_data),
        'gdp_match_rate': debug_info.get('gdp_match_rate', 0)
    }
    
    print(f"📈 Final dataset: {processing_metadata['observations_total']:,} observations")
    print(f"🌍 Countries: {processing_metadata['countries_total']}")
    print(f"📊 Normalized indicators: {processing_metadata['indicators_count']}")
    print(f"📅 Time range: {processing_metadata['time_range'][0]}-{processing_metadata['time_range'][1]}")
    
    return normalized_data, processing_metadata, debug_info

def extract_case_study_subsets(processed_data, processing_metadata):
    """
    Extract relevant country subsets for Case Study 1 and 2
    
    Args:
        processed_data: Normalized BOP dataset
        processing_metadata: Processing information
        
    Returns:
        dict: Case study subsets and their metadata
    """
    
    print("\n🎯 Extracting Case Study Subsets...")
    print("=" * 50)
    
    # Define case study countries
    cs1_countries = ['Iceland', 'Austria', 'Belgium', 'Finland', 'France', 'Germany', 'Ireland', 'Italy', 'Netherlands', 'Portugal', 'Spain']
    cs2_countries = ['Slovenia', 'Slovakia', 'Estonia', 'Latvia', 'Lithuania', 'Cyprus', 'Malta']
    
    available_countries = processed_data['COUNTRY'].unique()
    
    # Extract Case Study 1 subset
    cs1_available = [c for c in cs1_countries if c in available_countries]
    cs1_data = processed_data[processed_data['COUNTRY'].isin(cs1_available)].copy()
    
    print(f"📊 Case Study 1: {len(cs1_available)}/{len(cs1_countries)} countries")
    print(f"   Available: {cs1_available}")
    print(f"   Missing: {[c for c in cs1_countries if c not in available_countries]}")
    print(f"   Data shape: {cs1_data.shape}")
    
    # Extract Case Study 2 subset  
    cs2_available = [c for c in cs2_countries if c in available_countries]
    cs2_data = processed_data[processed_data['COUNTRY'].isin(cs2_available)].copy()
    
    print(f"📊 Case Study 2: {len(cs2_available)}/{len(cs2_countries)} countries")
    print(f"   Available: {cs2_available}")
    print(f"   Missing: {[c for c in cs2_countries if c not in available_countries]}")
    print(f"   Data shape: {cs2_data.shape}")
    
    return {
        'case_study_1': {
            'data': cs1_data,
            'countries_requested': cs1_countries,
            'countries_available': cs1_available,
            'coverage_rate': len(cs1_available) / len(cs1_countries) * 100
        },
        'case_study_2': {
            'data': cs2_data,
            'countries_requested': cs2_countries,
            'countries_available': cs2_available,
            'coverage_rate': len(cs2_available) / len(cs2_countries) * 100
        }
    }

def main():
    """
    Main processing workflow for expanded BOP dataset
    """
    
    # Process the expanded dataset
    processed_data, metadata, debug_info = process_expanded_bop_data()
    
    if processed_data is not None:
        # Extract case study subsets
        case_study_subsets = extract_case_study_subsets(processed_data, metadata)
        
        # Save processed data
        data_dir = Path(__file__).parent.parent / "data"
        output_file = data_dir / "expanded_bop_processed.csv"
        processed_data.to_csv(output_file, index=False)
        print(f"\n💾 Processed data saved to: {output_file}")
        
        return processed_data, metadata, debug_info, case_study_subsets
    
    return None, None, None, None

if __name__ == "__main__":
    main()