# LA County Public Safety Risk Prediction API

## Overview

This project builds an end-to-end machine learning system to predict **next-day incident volume** for LA County Sheriff reporting districts.

The system ingests historical crime data, engineers time-series features, trains a predictive model, and exposes results through a **FastAPI service** that returns ranked district recommendations.

The goal is to demonstrate how an analytic model can be turned into a **decision-support tool**.

---

## Problem

Public safety resources are limited. Agencies need a way to identify **which geographic areas are most likely to experience higher incident volume tomorrow**.

This project answers:

Which reporting districts are expected to have the highest incident activity on the next day?

---

## Data

- Source: LA County Sheriff Part I & II Crimes (Historical)
- Grain: reporting_district × day
- Size: ~3M+ records after aggregation

**Note:** Data is not included in this repository due to size.  
Use the ingestion scripts in `src/` to recreate datasets.

---

## Pipeline

### 1. Data Ingestion
- Pulls crime data from LA County API
- Stores raw data as parquet

### 2. Processing
- Cleans and standardizes fields
- Converts timestamps
- Aggregates to daily district-level counts

### 3. Feature Engineering
- Calendar features (day of week, month, etc.)
- Lag features (1, 7, 14 days)
- Rolling windows (7-day, 14-day mean/sum)
- District baseline feature:
  - expanding historical mean incidents per district

### 4. Modeling
- Model: Random Forest Regressor
- Target: next-day incident count
- Time-based train/test split
- Trained on ~2.5M rows

### 5. Prediction

The model is trained to predict:

    incident_count at day T + 1

given features from day T.

The API generates predictions using the **most recent available data**, meaning:

- Latest feature date: 2026-01-17  
- Predicted date: 2026-01-18  

This simulates a **real-time forecasting system**, where the model always predicts the next day relative to the latest data.

### 6. API Layer
- FastAPI service exposes predictions and recommendations

---

## Model Performance

- MAE: ~0.27  
- RMSE: ~0.63  
- R²: ~0.23  

---

## API

### Run the API

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

Open:
- http://127.0.0.1:8000/docs

---

### Endpoints

#### GET /health

Returns API status and data freshness.

#### GET /recommend

Returns top N districts by predicted incident volume.

Example:

```bash
/recommend?top_n=5
```

#### GET /predict

Returns predictions for all districts.

---

## Project Structure

```
data/
  raw/
  processed/

artifacts/
  model_100_trees.pkl
  metrics.json

src/
  ingestion/
  processing/
  features/
  modeling/
  api/
    app.py
```

---

## Key Insight

Initial predictions produced identical outputs across districts due to similar recent activity patterns.

This was resolved by introducing a district-level historical baseline feature, allowing the model to differentiate between consistently high-activity and low-activity districts.

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- FastAPI
- Uvicorn
- Parquet

---

## Notes

- API is intended for **local use only**
- No authentication or deployment configuration included
- Focus is on modeling and serving predictions

---

## Summary

This project demonstrates:

- Time-series feature engineering
- Predictive modeling on large-scale real-world data
- Translating model outputs into ranked recommendations
- Serving an ML model through a simple API
