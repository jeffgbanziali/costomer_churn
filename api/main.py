import sys
from pathlib import Path

# Permettre l'import de src/ depuis l'API
sys.path.append(str(Path(__file__).parent.parent / "src"))

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from auth import authenticate_user, create_access_token, get_current_user
from schemas import (
    CustomerFeatures,
    ChurnPredictionResponse,
    RevenueRiskResponse,
    HealthResponse,
)
from preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES

MODELS_DIR  = Path(__file__).parent.parent / "models"
ENGINEERED  = ["tickets_per_tenure", "fee_per_tenure", "engagement_score"]
NUMERIC_ALL = NUMERIC_FEATURES + ENGINEERED

app = FastAPI(
    title="Customer Churn Intelligence API",
    description="API de prédiction du churn et du revenu à risque — EFREI M1 Data Engineering 2025-26",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Chargement des modèles au démarrage (une seule fois)
# ---------------------------------------------------------------------------

_churn_model       = None
_revenue_model     = None
_preprocessor_churn   = None
_preprocessor_revenue = None
_churn_model_name  = "unknown"
_revenue_model_name = "unknown"


def _load_models():
    global _churn_model, _revenue_model, _preprocessor_churn, _preprocessor_revenue
    global _churn_model_name, _revenue_model_name

    try:
        _preprocessor_churn = joblib.load(MODELS_DIR / "preprocessor_churn.joblib")
    except FileNotFoundError:
        pass

    try:
        _preprocessor_revenue = joblib.load(MODELS_DIR / "preprocessor_revenue.joblib")
    except FileNotFoundError:
        pass

    # Churn model (sklearn ou Keras)
    sklearn_path = MODELS_DIR / "best_model_churn.joblib"
    keras_path   = MODELS_DIR / "best_model_churn.keras"
    if sklearn_path.exists():
        _churn_model = joblib.load(sklearn_path)
        _churn_model_name = type(_churn_model).__name__
    elif keras_path.exists():
        try:
            from tensorflow import keras

            print("Chargement du modèle Keras...")
            _churn_model = keras.models.load_model(keras_path)
            _churn_model_name = "MLP (Keras)"
            print("Modèle chargé :", _churn_model_name)

        except Exception as e:
            print("ERREUR :", e)

    # Revenue model
    sklearn_path_r = MODELS_DIR / "best_model_revenue.joblib"
    keras_path_r   = MODELS_DIR / "best_model_revenue.keras"
    if sklearn_path_r.exists():
        _revenue_model = joblib.load(sklearn_path_r)
        _revenue_model_name = type(_revenue_model).__name__
    elif keras_path_r.exists():
        from tensorflow import keras
        _revenue_model = keras.models.load_model(keras_path_r)
        _revenue_model_name = "MLP Regressor (Keras)"


@app.on_event("startup")
def startup_event():
    _load_models()


# ---------------------------------------------------------------------------
# Utilitaire : convertir CustomerFeatures → array numpy
# ---------------------------------------------------------------------------

def _minmax_scalar(val: float, col_min: float, col_max: float) -> float:
    rng = col_max - col_min
    return (val - col_min) / rng if rng > 0 else 0.0


def _features_to_df(customer: CustomerFeatures) -> pd.DataFrame:
    data = customer.model_dump()
    # Engineered features — mêmes formules que preprocessing.engineer_features
    data["tickets_per_tenure"] = data["support_tickets"] / (data["tenure_months"] + 1)
    data["fee_per_tenure"]     = data["monthly_fee"]     / (data["tenure_months"] + 1)
    # engagement_score — approximation avec des bornes fixes issues du dataset
    # (en prod, les bornes seraient sauvegardées lors du fit)
    data["engagement_score"] = (
        0.25 * _minmax_scalar(data["monthly_logins"],      0, 60)
        + 0.20 * _minmax_scalar(data["weekly_active_days"], 0, 7)
        + 0.20 * _minmax_scalar(data["avg_session_time"],  0, 120)
        + 0.10 * _minmax_scalar(data["features_used"],     0, 20)
        + 0.10 * _minmax_scalar(data["usage_growth_rate"], -1, 1)
        + 0.10 * (1 - _minmax_scalar(data["last_login_days_ago"], 0, 90))
    )
    return pd.DataFrame([data])


def _predict_churn_proba(customer: CustomerFeatures) -> tuple[float, str]:
    if _churn_model is None or _preprocessor_churn is None:
        raise HTTPException(status_code=503, detail="Modèle churn non chargé. Lancez d'abord train_churn.py.")
    df = _features_to_df(customer)
    X  = _preprocessor_churn.transform(df[NUMERIC_ALL + CATEGORICAL_FEATURES])
    if _churn_model_name == "MLP (Keras)":
        proba = float(_churn_model.predict(X, verbose=0).flatten()[0])
    else:
        proba = float(_churn_model.predict_proba(X)[0, 1])
    return proba, _churn_model_name


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
    token = create_access_token(data={"sub": form_data.username})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/health", response_model=HealthResponse, summary="Santé du service")
def health():
    return HealthResponse(
        status="ok",
        churn_model_loaded=_churn_model is not None,
        revenue_model_loaded=_revenue_model is not None,
        preprocessor_loaded=_preprocessor_churn is not None,
    )


@app.post(
    "/predict/churn",
    response_model=ChurnPredictionResponse,
    summary="Prédire la probabilité de churn d'un client",
)
def predict_churn(
    customer: CustomerFeatures,
    current_user: str = Depends(get_current_user),
):
    proba, model_name = _predict_churn_proba(customer)
    pred = int(proba >= 0.5)

    if proba < 0.3:
        risk_level = "Faible"
    elif proba < 0.6:
        risk_level = "Moyen"
    else:
        risk_level = "Élevé"

    return ChurnPredictionResponse(
        churn_prediction=pred,
        churn_probability=round(proba, 4),
        risk_level=risk_level,
        model_used=model_name,
    )


@app.post(
    "/predict/revenue-risk",
    response_model=RevenueRiskResponse,
    summary="Estimer le revenu à risque associé à un client",
)
def predict_revenue_risk(
    customer: CustomerFeatures,
    current_user: str = Depends(get_current_user),
):
    # Proba de churn toujours calculée depuis le modèle Tâche A
    churn_proba, _ = _predict_churn_proba(customer)

    if _revenue_model is None or _preprocessor_revenue is None:
        # Fallback : estimation directe sans modèle de régression
        revenue_at_risk = round(customer.total_revenue * churn_proba, 2)
        model_name = "Fallback (total_revenue × P(churn))"
    else:
        df = _features_to_df(customer)
        X  = _preprocessor_revenue.transform(df[NUMERIC_ALL + CATEGORICAL_FEATURES])
        if _revenue_model_name == "MLP Regressor (Keras)":
            revenue_at_risk = float(_revenue_model.predict(X, verbose=0).flatten()[0])
        else:
            revenue_at_risk = float(_revenue_model.predict(X)[0])
        model_name = _revenue_model_name

    return RevenueRiskResponse(
        revenue_at_risk=round(max(revenue_at_risk, 0), 2),
        churn_probability=round(churn_proba, 4),
        model_used=model_name,
    )
