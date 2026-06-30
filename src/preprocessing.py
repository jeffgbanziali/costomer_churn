import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DROP_COLS = ["customer_id"]

NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_logins",
    "weekly_active_days",
    "avg_session_time",
    "features_used",
    "usage_growth_rate",
    "last_login_days_ago",
    "monthly_fee",
    "total_revenue",
    "payment_failures",
    "support_tickets",
    "avg_resolution_time",
    "csat_score",
    "escalations",
    "email_open_rate",
    "marketing_click_rate",
    "nps_score",
    "referral_count",
]

# Features engineered 
ENGINEERED_NUMERIC = ["tickets_per_tenure", "fee_per_tenure", "engagement_score"]
NUMERIC_ALL = NUMERIC_FEATURES + ENGINEERED_NUMERIC  # 

CATEGORICAL_FEATURES = [
    "gender",
    "country",
    "city",
    "customer_segment",
    "signup_channel",
    "contract_type",
    "payment_method",
    "discount_applied",
    "price_increase_last_3m",
    "complaint_type",
    "survey_response",
]

TARGET_CHURN = "churn"
TARGET_REVENUE_RISK = "revenue_at_risk"


def build_preprocessing_pipeline() -> Pipeline:
    """Return a fitted-ready sklearn Pipeline."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_ALL),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline([("preprocessor", preprocessor)])


def get_feature_names_out(pipeline: Pipeline) -> list[str]:
    """Return feature names after transformation."""
    ct: ColumnTransformer = pipeline.named_steps["preprocessor"]
    ohe: OneHotEncoder = ct.named_transformers_["cat"].named_steps["encoder"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERIC_ALL.copy() + cat_names


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV and normalise column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df


def engineer_features_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features plus sophistiquées"""
    df = df.copy()
    
    # Features existantes
    df["tickets_per_tenure"] = df["support_tickets"] / (df["tenure_months"] + 1)
    df["fee_per_tenure"] = df["monthly_fee"] / (df["tenure_months"] + 1)
    
    # NOUVELLES FEATURES
    
    # 1. Ratio support/revenue
    df["support_cost_ratio"] = df["support_tickets"] / (df["total_revenue"] + 1)
    
    # 2. Interaction engagement × tenure
    df["engagement_x_tenure"] = df["engagement_score"] * df["tenure_months"]
    
    # 3. Détérioration de l'engagement (derniers mois)
    df["engagement_decline"] = df["weekly_active_days"] - df["monthly_logins"] / 4
    
    # 4. Ratio paiements échoués
    df["payment_failure_rate"] = df["payment_failures"] / (df["tenure_months"] + 1)
    
    # 5. Score de risque combiné
    df["risk_score"] = (
        0.3 * (1 - df["engagement_score"]) +
        0.2 * df["payment_failures"] +
        0.25 * df["support_tickets"] +
        0.15 * (1 - df["csat_score"]/5) +
        0.1 * df["escalations"]
    )
    
    # 6. Ancienneté avec contrat mensuel
    df["monthly_contract_risk"] = (
        (df["contract_type"] == "Monthly").astype(int) * 
        (1 / (df["tenure_months"] + 1))
    )
    
    return df


def build_revenue_at_risk(df: pd.DataFrame, churn_proba: np.ndarray) -> pd.Series:
    """Compute revenue_at_risk = total_revenue * P(churn) for Task B."""
    return pd.Series(df["total_revenue"].values * churn_proba, name=TARGET_REVENUE_RISK)