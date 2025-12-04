"""
Extract Baseline Results from Dashboard
Run this ONCE to save dashboard calculations for verification

This script extracts the actual results from the existing dashboard
so we can verify our notebooks produce the same values.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add dashboard code to path
dashboard_path = Path(__file__).parent.parent.parent / 'src' / 'dashboard'
sys.path.append(str(dashboard_path))
sys.path.append(str(dashboard_path / 'reports'))
sys.path.append(str(dashboard_path / 'shared_utils'))

# Import dashboard functions
from cs1_report import perform_volatility_tests, load_default_data
from cs2_shared_functions import create_euro_adoption_timeline

print("="*60)
print("EXTRACTING BASELINE RESULTS FROM DASHBOARD")
print("="*60)

# Create output directory
output_dir = Path(__file__).parent / 'baseline_results'
output_dir.mkdir(exist_ok=True)

# ========== CS1: Iceland vs Eurozone ==========
print("\n[CS1] Extracting Iceland vs Eurozone results...")
try:
    # Load data using dashboard function
    data, indicators, metadata = load_default_data()

    # Get volatility test results
    test_results = perform_volatility_tests(data, indicators)

    # Save baseline
    cs1_path = output_dir / 'CS1_baseline.csv'
    test_results.to_csv(cs1_path, index=False)
    print(f"✓ Saved {len(test_results)} CS1 results to {cs1_path.name}")

    # Also save summary statistics for each group
    iceland_stats = []
    eurozone_stats = []

    for indicator in indicators:
        if indicator in data.columns:
            iceland_data = data[data['GROUP'] == 'Iceland'][indicator].dropna()
            eurozone_data = data[data['GROUP'] == 'Eurozone'][indicator].dropna()

            iceland_stats.append({
                'indicator': indicator,
                'mean': iceland_data.mean(),
                'variance': iceland_data.var(),
                'std': iceland_data.std(),
                'n': len(iceland_data)
            })

            eurozone_stats.append({
                'indicator': indicator,
                'mean': eurozone_data.mean(),
                'variance': eurozone_data.var(),
                'std': eurozone_data.std(),
                'n': len(eurozone_data)
            })

    pd.DataFrame(iceland_stats).to_csv(output_dir / 'CS1_iceland_stats.csv', index=False)
    pd.DataFrame(eurozone_stats).to_csv(output_dir / 'CS1_eurozone_stats.csv', index=False)
    print(f"✓ Saved group statistics for verification")

except Exception as e:
    print(f"✗ Error extracting CS1 baseline: {e}")

# ========== CS2: Baltic Euro Adoption ==========
print("\n[CS2] Extracting Baltic Euro adoption results...")
try:
    # Get Euro adoption timeline
    timeline = create_euro_adoption_timeline()

    # Save timeline for reference
    timeline_df = pd.DataFrame([
        {
            'country': country,
            'adoption_year': info['adoption_year'],
            'pre_start': info['pre_period_full'][0],
            'pre_end': info['pre_period_full'][1],
            'post_start': info['post_period_full'][0],
            'post_end': info['post_period_full'][1]
        }
        for country, info in timeline.items()
    ])

    timeline_df.to_csv(output_dir / 'CS2_timeline.csv', index=False)
    print(f"✓ Saved CS2 Euro adoption timeline")

    # Note: Full CS2 baseline extraction would require running the full
    # temporal analysis from cs2_shared_functions.py

except Exception as e:
    print(f"✗ Error extracting CS2 baseline: {e}")

# ========== CS4: Statistical Framework ==========
print("\n[CS4] Extracting statistical framework results...")
try:
    # Import CS4 analysis
    sys.path.append(str(Path(__file__).parent.parent.parent / 'src' / 'core'))
    from cs4_statistical_analysis import CS4AnalysisFramework

    # Note: Would need to load CS4 specific data and run analysis
    # For now, just note that the framework exists
    print("✓ CS4 framework available for extraction")

except Exception as e:
    print(f"✗ Error accessing CS4 framework: {e}")

print("\n" + "="*60)
print("BASELINE EXTRACTION COMPLETE")
print("="*60)
print(f"\nBaseline files saved to: {output_dir}")
print("\nThese files will be used to verify notebook results match the dashboard.")
print("\nNOTE: This script should only be run ONCE before creating notebooks.")