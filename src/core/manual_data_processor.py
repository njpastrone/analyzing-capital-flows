# imf_data_pipeline.py

import pandas as pd
import numpy as np
import re
import sys
import os

# --- Config ---
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

# --- Utility Function to Check & Pivot Timeseries Data ---
def pivot_if_timeseries(df, name="dataset", threshold=3):
    year_cols = [col for col in df.columns if re.match(r"^\d{4}", str(col))]
    
    if len(year_cols) > threshold:
        print(f"'{name}' is in time-series-per-row format. Pivoting longer...")
        df_long = df.melt(
            id_vars=[col for col in df.columns if col not in year_cols],
            value_vars=year_cols,
            var_name="TIME_PERIOD",
            value_name="OBS_VALUE"
        )
    else:
        print(f"'{name}' is NOT in time-series-per-row format. No pivot applied.")
        df_long = df.copy()
    
    return df_long


# === Load and Clean BOP Data ===
bop_raw = pd.read_csv("timeseries_per_row_july_28_2025.csv")

bop = pivot_if_timeseries(bop_raw, name="BOP Data")

# Create FULL_INDICATOR and clean columns
bop["ENTRY_FIRST_WORD"] = bop["BOP_ACCOUNTING_ENTRY"].str.extract(r"^([^,]+)")
bop["FULL_INDICATOR"] = bop["ENTRY_FIRST_WORD"] + " - " + bop["INDICATOR"]

bop = bop.drop(columns=["BOP_ACCOUNTING_ENTRY", "INDICATOR", "ENTRY_FIRST_WORD", "FREQUENCY"], errors="ignore")

# Reorder
cols = ["COUNTRY", "TIME_PERIOD", "FULL_INDICATOR"] + [col for col in bop.columns if col not in ["COUNTRY", "TIME_PERIOD", "FULL_INDICATOR"]]
bop = bop[cols]

# Separate TIME_PERIOD into YEAR and QUARTER
bop[["YEAR", "QUARTER"]] = bop["TIME_PERIOD"].str.split("-", expand=True)
bop["QUARTER"] = bop["QUARTER"].str.extract(r"(\d+)").astype(float)
bop["YEAR"] = pd.to_numeric(bop["YEAR"], errors="coerce")

# View metadata
print("Unique countries:", bop["COUNTRY"].unique())
print("Unique indicators:", bop["FULL_INDICATOR"].unique())

# === Load and Clean GDP Data ===
gdp_raw = pd.read_csv("dataset_2025-07-24T18_28_31.898465539Z_DEFAULT_INTEGRATION_IMF.RES_WEO_6.0.0.csv")

gdp = pivot_if_timeseries(gdp_raw, name="GDP Data")

# Reduce columns
gdp_cleaned = gdp[["COUNTRY", "TIME_PERIOD", "INDICATOR", "OBS_VALUE"]]

# === Pivot Both Wider ===
bop_pivoted = bop.drop(columns=["SCALE", "DATASET", "SERIES_CODE", "OBS_MEASURE"], errors="ignore")

bop_pivoted = bop_pivoted.pivot_table(
    index=["COUNTRY", "YEAR", "QUARTER", "UNIT"],
    columns="FULL_INDICATOR",
    values="OBS_VALUE",
    aggfunc="first"
).reset_index()

gdp_pivoted = gdp_cleaned.pivot_table(
    index=["COUNTRY", "TIME_PERIOD"],
    columns="INDICATOR",
    values="OBS_VALUE",
    aggfunc="first"
).reset_index()

# Rename for join compatibility
gdp_pivoted = gdp_pivoted.rename(columns={"TIME_PERIOD": "YEAR"})
gdp_pivoted["YEAR"] = pd.to_numeric(gdp_pivoted["YEAR"], errors="coerce")

# === Join BOP and GDP ===
joined = pd.merge(
    bop_pivoted,
    gdp_pivoted,
    how="left",
    on=["COUNTRY", "YEAR"]
)

# Clean UNIT
joined["UNIT"] = joined["UNIT"].astype(str) + ", Nominal (Current Prices)"

# Output Preview
print(joined.head())
print(joined.columns.tolist())
