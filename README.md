# Analysis of Singapore’s HDB Flat Resale Market: Spatial Patterns and Machine Learning

## Project Overview
This project investigates spatial patterns and socio-economic dynamics in Singapore's Housing and Development Board (HDB) resale market through a dual approach of spatial statistical analysis and machine learning. Focusing on resale flat transactions, the analysis uncovers clustering patterns, spatial heterogeneity, and predictive insights into the `price_per_sqft` (average: 540.98 SGD). The study leverages Python libraries to integrate spatial autocorrelation, statistical modeling, and predictive techniques, providing actionable insights for urban planners and policymakers.

## Objectives
- **Identify Spatial Patterns:** Use spatial statistical methods to detect global and local clustering, spatial ranges of influence, and price density gradients in HDB resale transactions.
- **Model Price Dynamics:** Apply machine learning techniques to model and predict `price_per_sqft`, capturing spatial and socio-economic factors.
- **Provide Contextual Insights:** Relate findings to global housing market trends and explore future implications for Singapore’s HDB market.

## Methodology
The project is structured into several key sections:

### 1. Data Preprocessing
- Filtered the dataset to the latest listing per flat for a unique spatial snapshot.
- Handled duplicates and inconsistent spatial coordinates to ensure data integrity.

### 2. Exploratory Statistical Analysis
- **Methods Used:** Moran’s I, LISA, variograms, Kernel Density Estimation (KDE), and Temporal Hotspot Analysis.
- **Findings:** Identified global and local clustering, a spatial range of influence (3,000–5,000 meters), price density gradients (e.g., high densities in Sengkang), and minimal temporal effects on hotspots.

### 3. Spatial Machine Learning
- **Feature Engineering:** Created spatial lags (e.g., `price_per_sqft_lag`) and distance features (e.g., `distance_to_mrt_meters`).
- **OLS Regression:** Established a baseline model (R²: 0.7076) but revealed limitations in capturing spatial heterogeneity.
- **Geographically Weighted Regression (GWR):** Improved performance (R²: 0.7525) by modeling spatial variation, showing stronger effects of MRT proximity in central areas.
- **HDBSCAN Clustering:** Identified three spatially distinct clusters, aligning high-price areas like Sengkang (Cluster 1) with close MRT proximity.
- **Random Forest Modeling:** Achieved the highest predictive accuracy (R²: 0.8681), with `price_per_sqft_lag` as the dominant feature (importance: 0.8083).
- **Conclusions:** Synthesized findings, emphasizing the critical role of spatial factors like MRT proximity and neighborhood effects.


### 5. Challenges Faced
- **Data Preprocessing:** Handling duplicates and inconsistent spatial coordinates.
- **Spatial Analysis:** Disconnected components in Moran’s I weights matrix (72 components with k=40).
- **Machine Learning:** Collinearity in GWR (e.g., correlation of -0.6013 between `distance_to_cbd` and `price_per_sqft_lag`), tuning HDBSCAN parameters, and dispersion in Random Forest predictions at higher price ranges.

## Key Findings
- **Spatial Patterns:** Significant clustering in areas like Sengkang and Punggol, with a spatial range of influence of 3,000–5,000 meters.
- **Price Drivers:** Proximity to MRT and neighborhood price effects (`price_per_sqft_lag`) are critical, with varying impacts across Singapore, as captured by GWR.
- **Predictive Performance:** Random Forest outperformed OLS and GWR, achieving an R² of 0.8681, highlighting the importance of nonlinear modeling.

## Tools and Libraries
- **Python Libraries:** `pandas`, `geopandas`, `matplotlib`, `seaborn`, `sklearn`, `libpysal`, `mgwr`, `hdbscan`.
- **Data Source:** https://www.kaggle.com/datasets/lzytim/hdb-resale-prices


## Future Work
- Explore adaptive bandwidths in GWR for more nuanced spatial modeling.
- Incorporate additional features (e.g., proximity to commercial hubs) in clustering.
- Leverage ensemble methods with spatial constraints to further improve predictive accuracy.

## How to Use This Repository
- **Code Structure:** The repository contains Jupyter notebooks or Python scripts for each analysis section (preprocessing, statistical analysis, machine learning).
- **Requirements:** Install dependencies.
- **Data:** create a data folder and add the dataset into it.
- **Running the Analysis:** Execute the scripts in order (preprocessing → statistical analysis → machine learning) to replicate the results.

