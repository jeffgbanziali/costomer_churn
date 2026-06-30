import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras

from preprocessing import (
    load_raw_data, engineer_features,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_CHURN,
)
from evaluation import regression_metrics, compare_regressors

DATA_PATH  = Path(__file__).parent.parent / "data" / "customer_churn_business_dataset.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"
RANDOM_STATE = 42

ENGINEERED_FEATURES = ["tickets_per_tenure", "fee_per_tenure", "engagement_score"]
NUMERIC_ALL = NUMERIC_FEATURES + ENGINEERED_FEATURES


def build_mlp_regressor(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def train_and_evaluate():
    proba_df = pd.read_parquet(MODELS_DIR / "churn_proba_test.parquet")
    # La cible n'est pas binaire : c'est la perte financière attendue (valeur continue)
    y_reg    = proba_df["total_revenue"] * proba_df["churn_proba"]

    df = load_raw_data(DATA_PATH)
    df = engineer_features(df)

    # Même seed que Task A : indispensable pour aligner churn_proba_test.parquet avec les features
    _, X_test_full, _, _ = train_test_split(
        df, df[TARGET_CHURN], test_size=0.2, stratify=df[TARGET_CHURN], random_state=RANDOM_STATE
    )
    X_reg = X_test_full[NUMERIC_ALL + CATEGORICAL_FEATURES].reset_index(drop=True)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg.reset_index(drop=True), test_size=0.25, random_state=RANDOM_STATE
    )

    preprocessor_r = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), NUMERIC_ALL),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("enc", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
    ], remainder="drop")

    X_train_rp = preprocessor_r.fit_transform(X_train_r)
    X_test_rp  = preprocessor_r.transform(X_test_r)

    results = []
    trained_models = {}

    for name, model in [
        ("Linear Regression",       LinearRegression()),
        ("Random Forest Regressor", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ("XGBoost Regressor",       xgb.XGBRegressor(n_estimators=200, random_state=RANDOM_STATE, verbosity=0)),
    ]:
        model.fit(X_train_rp, y_train_r)
        m = regression_metrics(y_test_r, model.predict(X_test_rp), model_name=name)
        results.append(m)
        trained_models[name] = model
        print(f"[{name}] RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R²={m['r2']:.4f}")

    mlp_r = build_mlp_regressor(X_train_rp.shape[1])
    mlp_r.fit(X_train_rp, y_train_r, validation_split=0.15, epochs=100, batch_size=64,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                                       restore_best_weights=True)],
              verbose=0)
    m = regression_metrics(y_test_r, mlp_r.predict(X_test_rp, verbose=0).flatten(), "MLP Regressor")
    results.append(m)
    trained_models["MLP Regressor"] = mlp_r
    print(f"[MLP Regressor] RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R²={m['r2']:.4f}")

    comparison = compare_regressors(results)
    print("\n=== Comparaison — Tâche B (revenu à risque) ===")
    print(comparison.to_string())

    best_name = comparison.index[0]
    joblib.dump(preprocessor_r, MODELS_DIR / "preprocessor_revenue.joblib")
    if best_name == "MLP Regressor":
        trained_models[best_name].save(MODELS_DIR / "best_model_revenue.keras")
    else:
        joblib.dump(trained_models[best_name], MODELS_DIR / "best_model_revenue.joblib")

    print(f"Meilleur modèle revenu : {best_name}")
    return comparison, preprocessor_r, trained_models


if __name__ == "__main__":
    train_and_evaluate()
