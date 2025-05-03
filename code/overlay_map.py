import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
DATA_PATH = "../data/hdb_resale_flat_transactions.csv"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Clean and filter data
df = df[
    df['latitude'].notnull() &
    df['longitude'].notnull() &
    df['planning_area_ura'].notnull()
]

# Compute centroid coordinates for each planning area
planning_centroids = df.groupby('planning_area_ura')[['latitude', 'longitude']].mean()

# Create figure with fixed bounds for perfect overlaying
fig, ax = plt.subplots(figsize=(8,6))

# Plot labels exactly at the average coordinate of each planning area
for area, row in planning_centroids.iterrows():
    ax.text(row['longitude'], row['latitude'], area, fontsize=7, fontweight='bold',
            ha='center', va='center', color='black')

# Match KDE plot dimensions: set bounds based on the full data range
min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
min_lat, max_lat = df['latitude'].min(), df['latitude'].max()

# Add a small padding (optional — adjust if needed)
padding = 0.005
ax.set_xlim(min_lon - padding, max_lon + padding)
ax.set_ylim(min_lat - padding, max_lat + padding)
ax.set_aspect('equal')

# No axes, ticks, or borders
ax.axis('off')

# Save with transparent background for overlay
output_path = os.path.join(OUTPUT_DIR, "black_overlay_ready_sg_labels.png")
plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
plt.close()
