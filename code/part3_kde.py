import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import geopandas as gpd
from shapely.geometry import Point

# ─── Setup paths ────────────────────────────────────────────────────────────────
DATA_PATH = "../data/hdb_resale_flat_transactions.csv"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# ─── Create GeoDataFrame ────────────────────────────────────────────────────────
geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:3414")  # SVY21 Singapore

# ─── Prepare Grid ───────────────────────────────────────────────────────────────
minx, miny, maxx, maxy = gdf.total_bounds
xgrid = np.linspace(minx, maxx, 100)
ygrid = np.linspace(miny, maxy, 100)
xx, yy = np.meshgrid(xgrid, ygrid)
grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
coords = np.vstack([gdf.geometry.x, gdf.geometry.y]).T

# ─── KDE Plotting Function ──────────────────────────────────────────────────────
def kde_plot(sample_weights, title, filename, cmap):
    kde = KernelDensity(bandwidth=500, kernel='gaussian')
    kde.fit(coords, sample_weight=sample_weights)
    Z = np.exp(kde.score_samples(grid_points)).reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

# ─── KDE Plots ──────────────────────────────────────────────────────────────────
kde_plot(gdf['resale_price'], 'KDE of Resale Price', 'kde_resale_price.png', 'viridis')
kde_plot(gdf['price_per_sqft'], 'KDE of Price per Sqft', 'kde_price_per_sqft.png', 'magma')
kde_plot(1 / (gdf['distance_to_mrt_meters'].fillna(0) + 1), 'KDE by 1/Distance to MRT', 'kde_distance_to_mrt.png', 'plasma')
kde_plot(1 / (gdf['distance_to_pri_school_meters'].fillna(0) + 1), 'KDE by 1/Distance to Primary School', 'kde_distance_to_school.png', 'cool')
kde_plot(1 / (gdf['distance_to_cbd'].fillna(0) + 1), 'KDE by 1/Distance to CBD', 'kde_distance_to_cbd.png', 'cividis')
