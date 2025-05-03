import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from libpysal.cg import KDTree
from esda.moran import Moran_BV_matrix
from skgstat import Variogram
from libpysal.weights import DistanceBand
from esda.moran import Moran
import numpy as np

# Set paths
DATA_PATH = "../data/hdb_resale_flat_transactions.csv"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["x", "y", "resale_price", "price_per_sqft"])
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:3414")

coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))

def compute_variogram(values, var_name):
    # Subsample for memory efficiency
    np.random.seed(42)
    sample_idx = np.random.choice(len(values), size=min(2000, len(values)), replace=False)
    sampled_coords = coords[sample_idx]
    sampled_values = values[sample_idx]

    # Compute variogram
    V = Variogram(sampled_coords, sampled_values, normalize=False, n_lags=20)
    fig = V.plot(show=False)
    plt.title(f'Variogram - {var_name} (Subsampled)')
    plt.xlabel("Distance")
    plt.ylabel("Semivariance")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'variogram_{var_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Spatial correlogram using Moran's I by distance band
def compute_correlogram(gdf, var_name, sample_size=2000):

    # Subset & sample for memory efficiency
    clean = gdf.dropna(subset=[var_name]).copy()
    if len(clean) > sample_size:
        clean = clean.sample(n=sample_size, random_state=42)
    
    # Reproject to metric CRS for distance calculation
    clean = clean.to_crs(epsg=3414)  # Singapore SVY21
    values = clean[var_name].values
    coords = np.vstack((clean.geometry.x, clean.geometry.y)).T

    # Compute Moran's I for multiple distance bands
    max_dist = 20000  # meters
    step = 500
    distances = []
    morans = []

    for dist in range(step, max_dist + step, step):
        w = DistanceBand(coords, threshold=dist, binary=True, silence_warnings=True)
        if not w.cardinalities or np.mean(list(w.cardinalities.values())) == 0:
            continue
        mi = Moran(values, w)
        distances.append(dist)
        morans.append(mi.I)

    # Plot and save correlogram
    plt.figure()
    plt.plot(distances, morans, marker='o', color='skyblue')
    plt.title(f"Correlogram (Moran's I) – {var_name}")
    plt.xlabel("Distance Band (m)")
    plt.ylabel("Moran's I")
    plt.grid(True)

    output_path = os.path.join(OUTPUT_DIR, f'correlogram_{var_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f">> Correlogram saved to: {output_path}")


# Run for each variable
for var in ["resale_price", "price_per_sqft"]:
    compute_variogram(gdf[var].values, var)
    


for var in ["resale_price", "price_per_sqft"]:
    compute_correlogram(gdf, var)

print("✅ Variograms and spatial correlograms saved to output directory.")
