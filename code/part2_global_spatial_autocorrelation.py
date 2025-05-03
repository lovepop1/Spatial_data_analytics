import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import esda
from libpysal.weights import KNN
from esda.moran import Moran
from esda.geary import Geary

# Set folders
DATA_PATH = "../output/hdb_resale_latest_transactions.csv"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Drop rows with missing coordinates or prices
df = df.dropna(subset=["x", "y", "resale_price", "price_per_sqft"])

# Create GeoDataFrame with projected coordinates
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:3414")

# Create spatial weights matrix using 12-nearest neighbors
w = KNN.from_dataframe(gdf, k=40)
w.transform = "r"

# --- Moran's I for resale_price ---
mi_resale = Moran(gdf["resale_price"], w)

# Moran's I for price_per_sqft
mi_pps = Moran(gdf["price_per_sqft"], w)

# --- Geary's C ---
gc_resale = Geary(gdf["resale_price"], w)
gc_pps = Geary(gdf["price_per_sqft"], w)

def save_moran_plot(moran_obj, var_name):
    fig, ax = plt.subplots()
    ax.scatter(moran_obj.z, esda.moran.lag_spatial(moran_obj.w, moran_obj.z), 
               edgecolor='k', facecolor='skyblue')
    ax.axvline(0, color='k', linestyle='--')
    ax.axhline(0, color='k', linestyle='--')
    ax.set_xlabel(f'{var_name} (standardized)')
    ax.set_ylabel(f'Spatial Lag of {var_name}')
    ax.set_title(f"Moran's I Scatterplot for {var_name}")
    output_path = os.path.join(OUTPUT_DIR, f"moran_{var_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# Save Moran scatter plots
save_moran_plot(mi_resale, "resale_price")
save_moran_plot(mi_pps, "price_per_sqft")

# Print Results
print("\n--- Moran’s I ---")
print(f"Resale Price: I = {mi_resale.I:.4f}, p = {mi_resale.p_sim:.4f}")
print(f"Price per Sqft: I = {mi_pps.I:.4f}, p = {mi_pps.p_sim:.4f}")

print("\n--- Geary’s C ---")
print(f"Resale Price: C = {gc_resale.C:.4f}, p = {gc_resale.p_sim:.4f}")
print(f"Price per Sqft: C = {gc_pps.C:.4f}, p = {gc_pps.p_sim:.4f}")


