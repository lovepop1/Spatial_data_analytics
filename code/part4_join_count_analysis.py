import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from esda.join_counts import Join_Counts
from libpysal.weights import Queen

# ─── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = "../data/hdb_resale_flat_transactions.csv"
OUTPUT_DIR = "../output/join_count_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Load and Subsample Data ───────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df_sample = df.sample(n=15000, random_state=42)  # Subsample for speed

# ─── Create GeoDataFrame ───────────────────────────────────────────────────────
geometry = [Point(xy) for xy in zip(df_sample['longitude'], df_sample['latitude'])]
gdf = gpd.GeoDataFrame(df_sample, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3414)

# ─── Queen Weight Matrix ───────────────────────────────────────────────────────
w = Queen.from_dataframe(gdf)
w.transform = 'b'

# ─── Join Count Plot Function ──────────────────────────────────────────────────
def join_count_plot(column, output_prefix):
    cat_series = gdf[column].astype('category')
    cat_codes = cat_series.cat.codes
    categories = cat_series.cat.categories

    for i, label in enumerate(categories):
        gdf['binary'] = (cat_codes == i).astype(int)
        if gdf['binary'].sum() == 0 or gdf['binary'].sum() == len(gdf):
            continue  # Skip if all or none

        jc = Join_Counts(gdf['binary'], w)

        fig, ax = plt.subplots(figsize=(8, 6))
        gdf.plot(column='binary', cmap='coolwarm', edgecolor='k', linewidth=0.05, ax=ax)
        ax.set_title(f"{column} = {label}\nBB: {jc.bb:.2f}, BW: {jc.bw:.2f}, WW: {jc.ww:.2f}")
        ax.axis('off')
        fig.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_{label}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

# ─── Categorical Columns for Analysis ──────────────────────────────────────────
categorical_cols = ['flat_type', 'transport_type']

for col in categorical_cols:
    join_count_plot(col, f"join_count_{col}")
