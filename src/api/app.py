from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_PATH = Path("artifacts/model_100_trees.pkl")
DATA_PATH = Path("data/processed/model_dataset.parquet")

FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "incidents_lag_1",
    "incidents_lag_7",
    "incidents_lag_14",
    "incidents_rolling_mean_7",
    "incidents_rolling_mean_14",
    "incidents_rolling_sum_7",
    "incidents_rolling_sum_14",
    "district_expanding_mean_incidents",
]

# ── Pydantic Models ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    latest_feature_date: str
    prediction_for_day: str


class DistrictPrediction(BaseModel):
    reporting_district: str
    prediction_for_day: str
    predicted_incident_count_next_day: float


class RecommendResponse(BaseModel):
    top_n: int
    results: List[DistrictPrediction]


class PredictResponse(BaseModel):
    total_districts: int
    results: List[DistrictPrediction]


# ── Helper ─────────────────────────────────────────────────────────────────────

def build_predictions(model, df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["reporting_district", "incident_day"])
    latest = df.groupby("reporting_district").tail(1).copy()

    latest["prediction_for_day"] = latest["incident_day"] + pd.Timedelta(days=1)

    X = latest[FEATURE_COLUMNS]
    latest["predicted_incident_count_next_day"] = model.predict(X).round(3)

    latest["reporting_district"] = latest["reporting_district"].astype(str)
    latest["incident_day"] = latest["incident_day"].dt.strftime("%Y-%m-%d")
    latest["prediction_for_day"] = latest["prediction_for_day"].dt.strftime("%Y-%m-%d")

    return latest[
        ["reporting_district", "incident_day", "prediction_for_day", "predicted_incident_count_next_day"]
    ].reset_index(drop=True)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictions = None
    app.state.model_loaded = False

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(DATA_PATH)

    app.state.predictions = build_predictions(model, df)
    app.state.model_loaded = True
    yield

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LA County Sheriff — Incident Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    preds = getattr(app.state, "predictions", None)
    model_loaded = getattr(app.state, "model_loaded", False)

    if preds is None or preds.empty:
        raise HTTPException(status_code=503, detail="Predictions not available.")

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        latest_feature_date=preds["incident_day"].max(),
        prediction_for_day=preds["prediction_for_day"].max(),
    )


@app.get("/recommend", response_model=RecommendResponse)
def recommend(top_n: int = Query(default=10, ge=1, le=100)):
    preds = app.state.predictions
    if preds is None:
        raise HTTPException(status_code=503, detail="Predictions not available.")

    top = (
        preds.sort_values("predicted_incident_count_next_day", ascending=False)
        .head(top_n)
    )

    return RecommendResponse(
        top_n=top_n,
        results=[DistrictPrediction(**row) for row in top.to_dict(orient="records")],
    )


@app.get("/predict", response_model=PredictResponse)
def predict():
    preds = app.state.predictions
    if preds is None:
        raise HTTPException(status_code=503, detail="Predictions not available.")

    sorted_preds = preds.sort_values("predicted_incident_count_next_day", ascending=False)

    return PredictResponse(
        total_districts=len(sorted_preds),
        results=[DistrictPrediction(**row) for row in sorted_preds.to_dict(orient="records")],
    )
