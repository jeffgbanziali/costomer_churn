from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm

from auth import authenticate_user, create_access_token, get_current_user
from schemas import (
    CustomerFeatures,
    ChurnPredictionResponse,
    ChurnBatchResponse,
    RevenueRiskResponse,
    RevenueRiskBatchResponse,
    HealthResponse,
)

MODELS_DIR = Path(__file__).parent.parent / "models"

# ---------------------------------------------------------------------------
# Feature engineering (identique au notebook 02)
# ---------------------------------------------------------------------------

def _normalize(series: pd.Series, lo: float, hi: float) -> pd.Series:
    return ((series - lo) / (hi - lo + 1e-8)).clip(0, 1)


def engineer_features(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    df = df.copy()
    df["tickets_per_tenure"] = df["support_tickets"] / (df["tenure_months"] + 1)
    df["fee_per_tenure"]     = df["monthly_fee"]      / (df["tenure_months"] + 1)
    df["engagement_score"] = (
        _normalize(df["monthly_logins"],     *bounds["monthly_logins"])     * 0.30 +
        _normalize(df["weekly_active_days"], *bounds["weekly_active_days"]) * 0.25 +
        _normalize(df["avg_session_time"],   *bounds["avg_session_time"])   * 0.25 +
        _normalize(df["features_used"],      *bounds["features_used"])      * 0.20
    )
    return df


# ---------------------------------------------------------------------------
# État global du service
# ---------------------------------------------------------------------------

_churn_model       = None
_preprocessor      = None
_engagement_bounds = {}
_feature_names     = []
_numeric_all       = []
_categorical       = []
_churn_model_name  = "unknown"
_churn_threshold   = 0.5


def _load_models():
    global _churn_model, _preprocessor, _engagement_bounds
    global _feature_names, _numeric_all, _categorical
    global _churn_model_name, _churn_threshold

    # Charger les artefacts de prétraitement (notebook 02)
    arts_path = MODELS_DIR / "preprocessing_artifacts.joblib"
    if arts_path.exists():
        arts = joblib.load(arts_path)
        _preprocessor      = arts["preprocessor"]
        _engagement_bounds = arts["engagement_bounds"]
        _feature_names     = arts["feature_names"]
        _numeric_all       = arts["NUMERIC_ALL"]
        _categorical       = arts["CATEGORICAL"]

    # Charger le modèle de churn (notebook 03)
    sklearn_path = MODELS_DIR / "best_model_churn.joblib"
    keras_path   = MODELS_DIR / "best_model_churn.keras"

    if sklearn_path.exists():
        _churn_model      = joblib.load(sklearn_path)
        _churn_model_name = type(_churn_model).__name__
    elif keras_path.exists():
        try:
            from tensorflow import keras
            _churn_model      = keras.models.load_model(keras_path)
            _churn_model_name = "MLP (Keras)"
        except Exception as exc:
            print(f"Erreur chargement modèle Keras : {exc}")

    # Charger le seuil optimal (notebook 03)
    mod_arts_path = MODELS_DIR / "modeling_artifacts.joblib"
    if mod_arts_path.exists():
        mod_arts         = joblib.load(mod_arts_path)
        _churn_threshold = mod_arts.get("best_threshold", 0.5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Churn Intelligence API",
    description="API de prédiction du churn — EFREI M1 Data Engineering 2025-26",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _features_to_matrix(customers: list[CustomerFeatures]) -> np.ndarray:
    rows = [c.model_dump() for c in customers]
    df   = pd.DataFrame(rows)
    df   = engineer_features(df, _engagement_bounds)
    return _preprocessor.transform(df[_numeric_all + _categorical])


def _risk_level(proba: float) -> str:
    if proba < 0.3:
        return "Faible"
    if proba < 0.6:
        return "Moyen"
    return "Élevé"


def _predict_churn_batch(customers: list[CustomerFeatures]) -> np.ndarray:
    if _churn_model is None or _preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Exécutez d'abord les notebooks 02 et 03.",
        )
    X = _features_to_matrix(customers)
    if _churn_model_name == "MLP (Keras)":
        return _churn_model.predict(X, verbose=0).flatten()
    return _churn_model.predict_proba(X)[:, 1]


def _churn_response(proba: float) -> ChurnPredictionResponse:
    return ChurnPredictionResponse(
        churn_prediction=int(proba >= _churn_threshold),
        churn_probability=round(float(proba), 4),
        risk_level=_risk_level(proba),
        model_used=_churn_model_name,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/token", summary="Obtenir un token JWT")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not authenticate_user(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants incorrects",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": create_access_token({"sub": form_data.username}), "token_type": "bearer"}


@app.get("/health", response_model=HealthResponse, summary="Santé du service")
def health():
    return HealthResponse(
        status="ok",
        churn_model_loaded=_churn_model is not None,
        revenue_model_loaded=False,
        preprocessor_loaded=_preprocessor is not None,
    )


@app.post("/predict/churn", response_model=ChurnPredictionResponse,
          summary="Prédire la probabilité de churn d'un client")
def predict_churn(customer: CustomerFeatures, _: str = Depends(get_current_user)):
    proba = _predict_churn_batch([customer])[0]
    return _churn_response(float(proba))


@app.post("/predict/churn/batch", response_model=ChurnBatchResponse,
          summary="Prédire le churn pour une liste de clients")
def predict_churn_batch(customers: list[CustomerFeatures], _: str = Depends(get_current_user)):
    if not customers:
        raise HTTPException(status_code=400, detail="Liste de clients vide.")
    if len(customers) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 clients par requête.")
    probas = _predict_churn_batch(customers)
    return ChurnBatchResponse(
        count=len(probas),
        predictions=[_churn_response(float(p)) for p in probas],
    )


@app.post("/predict/revenue-risk", response_model=RevenueRiskResponse,
          summary="Estimer le revenu à risque (fallback : total_revenue × P(churn))")
def predict_revenue_risk(customer: CustomerFeatures, _: str = Depends(get_current_user)):
    proba   = float(_predict_churn_batch([customer])[0])
    revenue = float(customer.total_revenue) * proba
    return RevenueRiskResponse(
        revenue_at_risk=round(revenue, 2),
        churn_probability=round(proba, 4),
        model_used=f"Fallback ({_churn_model_name} × total_revenue)",
    )


@app.post("/predict/revenue-risk/batch", response_model=RevenueRiskBatchResponse,
          summary="Estimer le revenu à risque pour une liste de clients")
def predict_revenue_risk_batch(customers: list[CustomerFeatures], _: str = Depends(get_current_user)):
    if not customers:
        raise HTTPException(status_code=400, detail="Liste de clients vide.")
    if len(customers) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 clients par requête.")
    probas = _predict_churn_batch(customers)
    model  = f"Fallback ({_churn_model_name} × total_revenue)"
    return RevenueRiskBatchResponse(
        count=len(probas),
        churn_model_used=_churn_model_name,
        predictions=[
            RevenueRiskResponse(
                revenue_at_risk=round(float(c.total_revenue) * float(p), 2),
                churn_probability=round(float(p), 4),
                model_used=model,
            )
            for c, p in zip(customers, probas)
        ],
    )
