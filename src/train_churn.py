"""
Task A — Binary churn classification.

Train 4 models (Logistic Regression, Random Forest, XGBoost, MLP),
compare them, and persist the best one with its preprocessing pipeline.

Run: python src/train_churn.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras

from preprocessing import (
    load_raw_data,
    engineer_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_CHURN,
)
from evaluation import classification_metrics, compare_classifiers

DATA_PATH  = Path(__file__).parent.parent / "data" / "customer_churn_business_dataset.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"
RANDOM_STATE = 42

# Engineered feature names (added by engineer_features)
ENGINEERED_FEATURES = ["tickets_per_tenure", "fee_per_tenure", "engagement_score"]
NUMERIC_ALL = NUMERIC_FEATURES + ENGINEERED_FEATURES


def build_mlp(input_dim: int) -> keras.Model:
    """
    MLP for binary churn classification.
    2 hidden layers with Dropout + BatchNorm to limit overfitting on 10k rows.
    class_weight passed in model.fit() to handle 10% churn imbalance.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )
    return model


def train_and_evaluate():
    # ------------------------------------------------------------------ #
    # 1. Load + engineer features                                          #
    # ------------------------------------------------------------------ #
    df = load_raw_data(DATA_PATH)
    df = engineer_features(df)

    X = df[NUMERIC_ALL + CATEGORICAL_FEATURES]
    y = df[TARGET_CHURN].astype(int)

    print(f"Dataset: {df.shape} | Churn rate: {y.mean():.3f}")

    # ------------------------------------------------------------------ #
    # 2. Stratified split FIRST — before any fit                          #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Churn rate — train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

    # ------------------------------------------------------------------ #
    # 3. Preprocessing pipeline (fit on X_train only)                     #
    # ------------------------------------------------------------------ #
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_ALL),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ], remainder="drop")

    X_train_proc = preprocessor.fit_transform(X_train)   # fit ONLY on train
    X_test_proc  = preprocessor.transform(X_test)        # no fit on test

    # ------------------------------------------------------------------ #
    # 4. Class weight for imbalance (~10% churn)                          #
    # ------------------------------------------------------------------ #
    # Ratio majoritaire/minoritaire ≈ 8.8 pour 10% de churn
    scale_pw = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pw:.2f}")

    # ------------------------------------------------------------------ #
    # 5. Train models                                                      #
    # ------------------------------------------------------------------ #
    results = []
    trained_models = {}

    # Logistic Regression (baseline)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_proc, y_train)
    m = classification_metrics(y_test, lr.predict(X_test_proc),
                               lr.predict_proba(X_test_proc)[:, 1], "Logistic Regression")
    results.append(m)
    trained_models["Logistic Regression"] = lr
    print(f"[LR]  F1={m['f1']:.4f}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_proc, y_train)
    m = classification_metrics(y_test, rf.predict(X_test_proc),
                               rf.predict_proba(X_test_proc)[:, 1], "Random Forest")
    results.append(m)
    trained_models["Random Forest"] = rf
    print(f"[RF]  F1={m['f1']:.4f}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}")

    # XGBoost
    xgb_m = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=scale_pw,
                               random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0)
    xgb_m.fit(X_train_proc, y_train, eval_set=[(X_test_proc, y_test)], verbose=False)
    m = classification_metrics(y_test, xgb_m.predict(X_test_proc),
                               xgb_m.predict_proba(X_test_proc)[:, 1], "XGBoost")
    results.append(m)
    trained_models["XGBoost"] = xgb_m
    print(f"[XGB] F1={m['f1']:.4f}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}")

    # MLP
    mlp = build_mlp(X_train_proc.shape[1])
    mlp.fit(
        X_train_proc, y_train,
        validation_split=0.15, epochs=100, batch_size=64,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                                  restore_best_weights=True)],
        class_weight={0: 1.0, 1: scale_pw},  # Keras attend un dict, pas 'balanced'
        verbose=0,
    )
    mlp_proba = mlp.predict(X_test_proc, verbose=0).flatten()
    mlp_pred  = (mlp_proba >= 0.5).astype(int)
    m = classification_metrics(y_test, mlp_pred, mlp_proba, "MLP")
    results.append(m)
    trained_models["MLP"] = mlp
    print(f"[MLP] F1={m['f1']:.4f}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}")

    # ------------------------------------------------------------------ #
    # 6. Comparison & save                                                 #
    # ------------------------------------------------------------------ #
    comparison = compare_classifiers(results)
    print("\n=== Comparaison — Tâche A (churn) ===")
    print(comparison.to_string())

    best_name  = comparison.index[0]
    best_model = trained_models[best_name]
    print(f"\nMeilleur modèle : {best_name}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor_churn.joblib")
    joblib.dump(comparison,   MODELS_DIR / "churn_comparison.joblib")

    if best_name == "MLP":
        best_model.save(MODELS_DIR / "best_model_churn.keras")
    else:
        joblib.dump(best_model, MODELS_DIR / "best_model_churn.joblib")

    # Save test probabilities for Task B
    best_proba = (
        best_model.predict(X_test_proc, verbose=0).flatten()
        if best_name == "MLP"
        else best_model.predict_proba(X_test_proc)[:, 1]
    )
    pd.DataFrame({
        "churn_proba":   best_proba,
        "total_revenue": X_test["total_revenue"].values,
    }).to_parquet(MODELS_DIR / "churn_proba_test.parquet", index=False)

    print("Modèles sauvegardés dans models/")
    return comparison, preprocessor, trained_models


if __name__ == "__main__":
    train_and_evaluate()
