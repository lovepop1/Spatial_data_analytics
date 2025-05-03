import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import geopandas as gpd
from shapely.geometry import Point

# ─── Setup paths ────────────────────────────────────────────────────────────────
DATA_PATH = "../output/hdb_resale_oldest_transactions.csv"
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

kde_plot(gdf['price_per_sqft'], 'KDE of Price per Sqft', 'oldest_kde_price_per_sqft.png', 'magma')
