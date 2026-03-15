# 🌬️ AirGuard UK

**Real-time 6-hour and 12-hour urban air quality forecast and health alert system**

[![Live App](https://img.shields.io/badge/Live%20App-airguard--uk.streamlit.app-brightgreen)](https://airguard-uk.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![SDG 13](https://img.shields.io/badge/SDG-13%20Climate%20Action-gold)](https://sdgs.un.org/goals/goal13)

Built for the **AI Tool Development Challenge 2026** on OneEarth, targeting **SDG 13: Climate Action**.

---

## What It Does

AirGuard UK fetches the last 73 hours of real pollution and weather data for five UK cities, builds 150 lag and rolling-average features, and uses two separately trained XGBoost classifiers to predict the DAQI (Daily Air Quality Index) band **6 hours** and **12 hours** into the future.

Rather than showing a simple Low/Moderate/Dangerous label, the app uses a **5-tier alert system** based on the raw probability assigned to the Dangerous class — giving users a more nuanced and honest picture of risk, even when the model's confidence is low.

---

## Live Demo

**[airguard-uk.streamlit.app](https://airguard-uk.streamlit.app)**

---

## Features

- **Dual-horizon forecasting** — toggle between 6h and 12h forecasts, each using its own trained model
- **5-tier alert system** — Low / Moderate / Watch / Elevated Risk / Dangerous, derived from Dangerous class probability to avoid binary over-alerting
- **Forecast All Cities** — one click populates the entire UK map for all 5 cities simultaneously
- **SHAP explainability** — interactive bar chart showing the top 10 features that drove each prediction
- **Pollution trend chart** — last 24h of PM10, PM2.5, NO2, O3 with WHO guideline lines overlaid
- **Health advice** — tailored recommendations for general public, asthma/respiratory, elderly and children
- **Forecast accuracy tracker** — every prediction is logged and automatically verified against real readings after the forecast horizon elapses
- **Folium map** — colour-coded city dots reflecting the current alert tier

---

## Cities

| Site ID | City | Lat | Lon |
|---------|------|-----|-----|
| MY1 | London | 51.5223 | -0.1546 |
| BIRR | Birmingham | 52.4800 | -1.9025 |
| MAN3 | Manchester | 53.4808 | -2.2426 |
| NEWC | Newcastle | 54.9783 | -1.6178 |
| CARD | Cardiff | 51.4816 | -3.1791 |

---

## Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Model | XGBoost (GPU training, CPU inference) |
| Explainability | SHAP TreeExplainer |
| Map | Folium + streamlit-folium |
| Charts | Plotly |
| Training data | DEFRA AURN (2021–2024) |
| Live data | Open-Meteo Air Quality API + Weather API |
| Imbalanced learning | SMOTE (imbalanced-learn) |

---

## Model Architecture

### Target Variable

DAQI is computed using the official DEFRA formula: the maximum sub-index across all five pollutants (NO2, PM2.5, PM10, O3, SO2). Three classes:

- `0` — Low (DAQI 1–3)
- `1` — Moderate (DAQI 4–6)  
- `2` — Dangerous (DAQI 7–10, merges High + Very High due to data scarcity)

### Feature Engineering (150 features)

Applied to all 9 variables (5 pollutants + 4 weather):

- **Lags**: 1h, 2h, 3h, 6h, 12h, 24h, 48h
- **Rolling mean**: 6h, 12h, 24h, 48h, 72h
- **Rolling max**: 6h, 12h

Applied to pollutants only:

- **Trend**: `shift(1) - shift(4)` (3h), `shift(1) - shift(7)` (6h)
- **vs_24h**: current − shift(24)

Applied to PM10, PM2.5, NO2 only:

- **roc_3h**: rate of change over 3 hours

Time features: `city_code`, `hour`, `day_of_week`, `month`, `season`, `is_weekend`

### Training

- **Train set**: 2021–2024 (chronological split — never random)
- **Test set**: 2025
- **Class imbalance**: SMOTE applied to training set only; test set kept at natural distribution
- **Class weights**: `{0:1, 1:5, 2:20}` for the best-performing 6h model
- **Hardware**: NVIDIA RTX 4060 (`device='cuda'`, `tree_method='hist'`)

### Model Performance (best configuration — Iter 1 baseline)

| Metric | 6h Model | 12h Model |
|--------|----------|-----------|
| Overall accuracy | ~88% | ~87% |
| Macro F1 (default threshold) | 0.404 | 0.395 |
| Dangerous recall @ t=0.10 | **51.2%** | — |
| Dangerous recall @ t=0.04 | — | **54.4%** |
| Optimal threshold | 0.10 | 0.04 |

### Tiered Alert Classification

Instead of a hard threshold, probabilities are mapped to tiers:

| Dangerous probability | Alert tier |
|----------------------|-----------|
| ≥ 0.30 | 🔴 Dangerous |
| ≥ 0.10 | 🟠 Elevated Risk |
| ≥ threshold (from `thresholds.json`) | ⚡ Watch |
| proba[1] ≥ 0.45 | 🟡 Moderate |
| Otherwise | 🟢 Low |

This avoids the alert fatigue problem caused by applying a single low threshold (e.g. 0.04) that fires on virtually every hour.

---

## Project Structure

```
airguard-uk/
├── app.py                          # Streamlit application
├── requirements.txt
├── models/
│   ├── xgboost_6h_best.pkl         # Best 6h XGBoost model
│   ├── xgboost_12h_best.pkl        # Best 12h XGBoost model
│   ├── thresholds.json             # Optimal Dangerous thresholds per horizon
│   ├── feature_list.csv            # 150 feature names (used by app)
│   ├── train_median.csv            # Training set medians (NaN fill in app)
│   ├── tuning_results.csv          # All fine-tuning iteration results
│   └── forecast_log.csv            # Auto-generated accuracy tracker log
├── data/
│   ├── London/merged/MY1_merged.csv
│   ├── Birmingham/merged/BIRR_merged.csv
│   ├── Manchester/merged/MAN3_merged.csv
│   ├── Newcastle/merged/NEWC_merged.csv
│   ├── Cardiff/merged/CARD_merged.csv
│   └── test/merged/{SITEID}_merged_2025.csv
└── notebooks/
    ├── phase4_eda.ipynb
    ├── forecast_features.ipynb     # Feature engineering pipeline
    ├── forecast_train.ipynb        # Baseline model training
    └── finetune.ipynb              # 5-iteration hyperparameter tuning
```

> `train_forecast.csv` and `test_forecast.csv` are excluded from the repo via `.gitignore` (>100 MB). Run `forecast_features.ipynb` to regenerate them.

---

## Running Locally

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)

### Setup

```bash
git clone https://github.com/nin-dark/airguard-uk.git
cd airguard-uk

python -m venv aienv
# Windows:
aienv\Scripts\activate
# Linux/macOS:
source aienv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Data Sources

### Training Data — DEFRA AURN

Historical hourly readings (2021–2024) for all five cities downloaded from the [UK Air Information Resource](https://uk-air.defra.gov.uk/data/). CSV files use `skiprows=4`, date format `DD-MM-YYYY`, with a 24:00 boundary fix applied during ingestion.

**Known data gaps:**
- Birmingham + Newcastle: SO2 = all NaN (no monitoring equipment) → imputed as 0
- Manchester: PM2.5/PM10 gap of ~1,861 rows (monitor offline 2022–2023) → forward-filled

### Live Data — Open-Meteo

| API | Variables |
|-----|-----------|
| [Air Quality API](https://open-meteo.com/en/docs/air-quality-api) | NO2, PM2.5, PM10, O3, SO2 |
| [Weather Forecast API](https://open-meteo.com/en/docs) | Temperature, humidity, wind speed, surface pressure |

The app fetches the last 73 hours of data on each forecast run. 73 hours is the minimum needed to compute 72h rolling window features without NaN propagation on the final row.

---

## Reproducing the Models

Run notebooks in order:

1. **`forecast_features.ipynb`** — loads merged city CSVs, engineers all 150 features, computes DAQI targets for both horizons, saves `train_forecast.csv` and `test_forecast.csv`
2. **`forecast_train.ipynb`** — trains the original baseline model (`xgboost_forecast_model.pkl`)
3. **`finetune.ipynb`** — runs 5 hyperparameter iterations × 2 horizons with SMOTE, early stopping, and threshold sweep; saves `xgboost_6h_best.pkl`, `xgboost_12h_best.pkl`, and `thresholds.json`

### Fine-tuning Results

| Experiment | Macro F1 | Dangerous recall | Best threshold |
|---|---|---|---|
| 6h_iter1_baseline | 0.387 | **51.2%** | 0.10 |
| 6h_iter2_deep | 0.438 | 27.2% | 0.10 |
| 6h_iter3_smotetomek | 0.416 | 31.2% | 0.10 |
| 6h_iter4_high_reg | 0.410 | 36.8% | 0.10 |
| 6h_iter5_targeted | 0.449 | 19.2% | 0.04 |
| 12h_iter1_baseline | 0.380 | 43.2% | 0.10 |
| 12h_iter5_targeted | 0.370 | **54.4%** | 0.04 |

Iter 1 baseline was selected as best 6h model (highest Dangerous recall at a usable precision). Iter 5 targeted was selected as best 12h model.

---

## Key Engineering Decisions

**Chronological split** — data is always split by year (≤2024 train, 2025 test). Random shuffling would cause data leakage via lag features.

**SMOTE on training set only** — the test set is kept at its natural distribution (98% Low, 1.7% Moderate, 0.2% Dangerous) to ensure evaluation reflects real-world conditions.

**Dangerous class merging** — DAQI bands 7–10 (High, Very High, Hazardous) are merged into a single "Dangerous" class. Fewer than 0.2% of hours breach band 7, making separate classes untrainable.

**Tiered alerts over binary threshold** — applying a single low threshold (e.g. 0.04) produces 99 false alarms per real Dangerous event. The 5-tier system surfaces low-probability signals as "Watch" or "Elevated Risk" rather than silently discarding or falsely alarming.

**73-hour fetch window** — the 72h rolling window feature requires 72 prior hours of data. Fetching only 24–25 hours (as in earlier versions) caused all 48h and 72h features to be filled with training medians rather than real observed values.

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
shap
imbalanced-learn
streamlit>=1.32.0
altair>=5.0.0
folium
streamlit-folium
plotly
requests
joblib
matplotlib
seaborn
```

---

## SDG 13 Alignment

AirGuard UK directly supports **SDG 13: Climate Action** by addressing one of the most immediate health consequences of climate change — worsening urban air quality. Rising temperatures, increased frequency of heatwaves, and altered wind patterns all intensify ground-level ozone and particulate matter concentrations in cities.

By providing accessible, real-time, explainable air quality forecasts with health-specific advice for vulnerable groups (elderly, children, asthma sufferers), the system enables individuals and public health bodies to take preventative action ahead of pollution events — rather than reacting after the fact. The dual 6h/12h horizon gives both immediate and short-range warning, supporting climate adaptation at the individual and community level.

---

## Limitations

- Open-Meteo AQ data is model-derived (CAMS reanalysis), not direct sensor readings, which introduces uncertainty in live forecasts
- Dangerous class precision is low (~1–2% at the operating threshold) — the tiered display is designed to communicate this uncertainty rather than hide it
- Models are trained on 5 UK cities and may not generalise well to other urban environments without retraining
- The forecast accuracy tracker requires the app to remain running for 6–12 hours to verify predictions

---

## Author

**Nikesh** — AI Tool Development Challenge 2026  
GitHub: [github.com/nin-dark/airguard-uk](https://github.com/nin-dark/airguard-uk)

---

## Licence

MIT