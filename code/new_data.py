import pandas as pd

# Load dataset (adjust path as needed)
df = pd.read_csv("../data/hdb_resale_flat_transactions.csv")

# Convert month to datetime if not already
df["month"] = pd.to_datetime(df["month"])

# Define columns that uniquely identify a flat
id_cols = [
    "town", "blk_no", "road_name", "building", "postal",
    "planning_area_ura", "region_ura", "x", "y",
    "latitude", "longitude", "storey_range",
    "flat_type", "flat_model", "lease_commence_date"
]

# Sort by date and keep latest transaction per flat
df_latest = df.sort_values("month").drop_duplicates(subset=id_cols, keep="last")

# Optional: Reset index for convenience
df_latest = df_latest.reset_index(drop=True)

# Save filtered dataset (optional)
df_latest.to_csv("../output/hdb_resale_latest_transactions.csv", index=False)
