# infrared.city

Urban green-space detection using Sentinel-2 satellite imagery and machine learning. The system classifies pixels as **green** (trees, shrubs, grassland) or **non-green** using multi-month spectral composites, trained across 11 cities worldwide with [ESA WorldCover 2021](https://esa-worldcover.org/) ground truth.

## Architecture

```
Sentinel-2 Raw Bands (B02, B03, B04, B08)
        │
        ▼
┌─ Sentinel-2 Processing App (Flask :5002) ─┐
│  Load & Reproject  →  Clip to AOI          │
│  Compute NDVI, EVI, SAVI                   │
│  Create 21-Band Multi-Month Stack          │
└────────────────────────────────────────────┘
        │
        ▼
   21-Band GeoTIFF ──┬──────────────────────────────────┐
        │            │                                   │
        ▼            ▼                                   ▼
  Balanced Sampling + WorldCover Labels       ┌─ Predictor App ─┐
        │                                     │  (Flask :5004)   │
        ▼                                     │                  │
  Feature Scaling (StandardScaler)            │  Predict Green   │
        │                                     │  Threshold 0.55  │
   ┌────┼────┐                                │  Median Filter   │
   ▼    ▼    ▼                                │  Coverage Stats  │
  RF   SVM  U-Net                             └──────────────────┘
   │    │    │                                        │
   ▼    ▼    ▼                                        ▼
 89.1% 87.7%  —                              Green Space Map +
  F1    F1                                   Coverage Metrics
```

## Project Structure

```
├── apps/
│   ├── sentinel2_multimonth_app.py        # Data processing web app (port 5002)
│   └── greenspace_predictor_app_FINAL.py  # Prediction web app (port 5004)
├── notebooks/
│   ├── training/
│   │   ├── Multi_City_WorldCover_Training.ipynb  # Random Forest (production)
│   │   ├── SVM_WorldCover_Training.ipynb         # SVM baseline
│   │   └── UNet_WorldCover_Training.ipynb        # U-Net deep learning
│   └── evaluation/
│       ├── Model_Evaluation.ipynb                # Single-model eval
│       └── Multi_Model_Evaluation.ipynb          # Comparative eval
├── models/
│   ├── random_forest_model.pkl            # Production model (200 trees)
│   ├── feature_scaler.pkl                 # StandardScaler for RF
│   ├── svm_model.pkl                      # SVM model
│   ├── svm_scaler.pkl                     # StandardScaler for SVM
│   ├── unet_model.keras                   # U-Net model
│   └── unet_normalization_params.json     # U-Net normalization config
├── data/
│   ├── aois/                              # 15 city AOI boundaries (GeoJSON)
│   ├── sentinel_stacks/                   # 12 processed 21-band GeoTIFFs
│   └── worldcover/                        # WorldCover 2021 ground truth labels
├── outputs/                               # Training run results & metrics
└── requirements.txt
```

## Models

| Model | Accuracy | F1 Score | Training Samples | Status |
|-------|----------|----------|-----------------|--------|
| **Random Forest** | 89.07% | 89.07% | 772,120 | Production |
| SVM (Linear) | 87.75% | 87.69% | 741,864 | Trained |
| U-Net (CNN) | — | — | — | Trained |

All models are trained on **11 cities**: Amsterdam, Auckland, Barcelona, London, Melbourne, Paris, San Francisco, Seattle, Sydney, Toronto, Vienna.

Ground truth: WorldCover 2021 — green classes (tree cover, shrubland, grassland, mangroves) vs. non-green.

## 21-Band Feature Stack

Each input is a multi-month GeoTIFF with **7 bands x 3 months** (April, August, November):

| Band | Description |
|------|-------------|
| B02 | Blue (490 nm) |
| B03 | Green (560 nm) |
| B04 | Red (665 nm) |
| B08 | NIR (842 nm) |
| NDVI | (NIR - Red) / (NIR + Red) |
| EVI | 2.5 * (NIR - Red) / (NIR + 6Red - 7.5Blue + 1) |
| SAVI | (NIR - Red) * 1.5 / (NIR + Red + 0.5) |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Data Processing App

Converts raw Sentinel-2 band files into 21-band multi-month stacks:

```bash
python apps/sentinel2_multimonth_app.py
# Open http://localhost:5002
```

Upload JP2/TIFF band files for 1-3 months + a GeoJSON AOI boundary. The app clips, reprojects, computes vegetation indices, and outputs a 21-band GeoTIFF.

### Prediction App

Predicts green space from a 21-band stack using the trained Random Forest model:

```bash
python apps/greenspace_predictor_app_FINAL.py
# Open http://localhost:5004
```

Upload a 21-band GeoTIFF. The app returns a green space probability map, binary prediction mask, and coverage statistics.

### Training

Open the Jupyter notebooks in `notebooks/training/` to retrain models:

```bash
jupyter notebook notebooks/training/Multi_City_WorldCover_Training.ipynb
```

## City Coverage

AOI boundaries available for 15 cities: Amsterdam, Auckland, Barcelona, Berlin, Lisbon, London, Madrid, Melbourne, Paris, San Francisco, Seattle, Sydney, Toronto, Vancouver, Vienna.

Processed Sentinel-2 stacks available for 12 cities (excluding Berlin, Lisbon, Madrid).

## Tech Stack

- **Data processing**: NumPy, xarray, rasterio, rioxarray, GeoPandas
- **ML**: scikit-learn (Random Forest, SVM), TensorFlow/Keras (U-Net)
- **Web**: Flask
- **Visualization**: Matplotlib, Seaborn
