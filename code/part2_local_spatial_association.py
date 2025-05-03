# code/part1_local_spatial_association.py

import warnings
warnings.filterwarnings("ignore", message=".*not fully connected.*")

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from libpysal.weights import KNN
from esda.moran import Moran_Local
from esda.getisord import G_Local
from splot.esda import lisa_cluster

# ─── Setup paths ────────────────────────────────────────────────────────────────
DATA_PATH  = "../data/hdb_resale_flat_transactions.csv"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Load & clean ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["x", "y", "resale_price", "price_per_sqft"])
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:3414")

# ─── Spatial weights ────────────────────────────────────────────────────────────
w = KNN.from_dataframe(gdf, k=40)
w.transform = "r"

# report islands
print("Number of islands (zero‐neighbor points):", len(w.islands))


# ─── Functions to save maps ─────────────────────────────────────────────────────
def save_lisa_maps(local_moran, gdf, var_name):
    # attach results
    gdf[f"{var_name}_Is"]  = local_moran.Is
    gdf[f"{var_name}_q"]   = local_moran.q
    gdf[f"{var_name}_p"]   = local_moran.p_sim
    gdf[f"{var_name}_sig"] = (local_moran.p_sim < 0.05).astype(int)  # 0/1

    # 1) Cluster map (HH, LL, HL, LH)
    fig, ax = plt.subplots(figsize=(10, 8))
    lisa_cluster(local_moran, gdf, p=0.05, ax=ax, legend=True)
    ax.set_title(f"LISA Cluster Map – {var_name}")
    fig.savefig(os.path.join(OUTPUT_DIR, f"lisa_cluster_{var_name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2) Significance map (0 = non‐sig, 1 = sig)
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(column=f"{var_name}_sig",
             categorical=True,
             cmap="coolwarm",
             legend=True,
             ax=ax)
    ax.set_title(f"LISA Significance (p<0.05) – {var_name}")
    fig.savefig(os.path.join(OUTPUT_DIR, f"lisa_significance_{var_name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_getis_ord_map(g_star, gdf, var_name):
    gdf[f"{var_name}_g"] = g_star.Zs
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(column=f"{var_name}_g",
             cmap="coolwarm",
             scheme="quantiles",
             legend=True,
             ax=ax)
    ax.set_title(f"Getis-Ord G* Z-Score – {var_name}")
    fig.savefig(os.path.join(OUTPUT_DIR, f"getis_ord_{var_name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─── Run for each variable ─────────────────────────────────────────────────────
for var in ["resale_price", "price_per_sqft"]:
    lm = Moran_Local(gdf[var], w)
    save_lisa_maps(lm, gdf, var)

    g_star = G_Local(gdf[var], w)
    save_getis_ord_map(g_star, gdf, var)

print("✅ LISA & Getis-Ord maps saved to output/")
