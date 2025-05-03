# code/part1_understanding_data.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
input_file = os.path.join("..", "output", "hdb_resale_latest_transactions.csv")
df = pd.read_csv(input_file)

# Create output directory
output_img_dir = os.path.join("..", "output")
os.makedirs(output_img_dir, exist_ok=True)

# 1. Dataset Structure and Data Types
print("Dataset Structure:")
print(f"Shape: {df.shape}")
print("\nData Types:")
print(df.dtypes)

# 2. Check for Duplicates
duplicates = df.duplicated()
print(f"\nNumber of duplicate rows: {duplicates.sum()}")
df = df[~duplicates]  # Drop duplicates
print(f"New shape after removing duplicates: {df.shape}")

# 3. Summary Statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# 4. Identify Spatial and Non-Spatial Attributes
spatial_attrs = ['x', 'y', 'latitude', 'longitude', 'distance_to_mrt_meters',
                 'distance_to_cbd', 'distance_to_pri_school_meters']
non_spatial_attrs = [col for col in df.columns if col not in spatial_attrs + ['transaction_id']]

print("\nSpatial Attributes:")
print(spatial_attrs)
print("\nNon-Spatial Attributes:")
print(non_spatial_attrs)

# 5. Missing Values
missing = df.isnull().sum()
print("\nMissing Values:")
print(missing[missing > 0])

# 6. Coordinate Consistency
print("\nCoordinate Ranges:")
print(f"Latitude: {df['latitude'].min()} to {df['latitude'].max()}")
print(f"Longitude: {df['longitude'].min()} to {df['longitude'].max()}")

# 7. Basic Preprocessing
# Replace missing numerical values with column medians
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Ensure distances are nonzero (to avoid issues in inverse-distance weighting)
for col in ['distance_to_mrt_meters', 'distance_to_cbd', 'distance_to_pri_school_meters']:
    df[col] = df[col].replace(0, 1)

# 8. Encode Categorical Variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print("\nCategorical columns encoded:", categorical_cols)


print(f"\nAverage price_per_sqft: {df['price_per_sqft'].mean():.2f}")


# 9. Correlation Matrix (Heatmap saved as image)
spatial_attrs2 = ['x','y','distance_to_mrt_meters',
                 'distance_to_cbd', 'distance_to_pri_school_meters']
corr_features = spatial_attrs2 + ['resale_price', 'price_per_sqft']
corr = df[corr_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    square=True,
    annot_kws={"size": 12},  # Increase size of annotation text
    cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Matrix of Spatial and Price Features", fontsize=18)  # Increase title size
plt.xticks(fontsize=12, rotation=45, ha='right')
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_img_dir, "correlation_matrix_heatmap.png"))
plt.close()