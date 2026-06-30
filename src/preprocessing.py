import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Column names (verified against the actual CSV)
# ---------------------------------------------------------------------------

# Colonnes à exclure : identifiant pur, pas de signal prédictif
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
    "complaint_type",     # 20% de NaN → imputation "Unknown" dans le pipeline
    "survey_response",
]

TARGET_CHURN = "churn"
TARGET_REVENUE_RISK = "revenue_at_risk"

# Churn rate : ~10.2 % → déséquilibre → F1/PR-AUC prioritaires sur l'accuracy


def build_preprocessing_pipeline() -> Pipeline:
    """
    Return a fitted-ready sklearn Pipeline.
    Fit ONLY on X_train to prevent data leakage.

    Numeric  : median imputation → StandardScaler
    Categorical : 'Unknown' imputation (gère complaint_type 20% NaN)
                  → OneHotEncoder (drop first, handle_unknown='ignore')
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        # 'constant' + fill_value='Unknown' est plus adapté que 'most_frequent'
        # pour complaint_type qui a 20% de NaN (valeur manquante = information)
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        # drop='first' évite le piège des variables muettes (multicolinéarité)
        # handle_unknown='ignore' → pas d'erreur si l'API reçoit une ville inconnue
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline([("preprocessor", preprocessor)])


def get_feature_names_out(pipeline: Pipeline) -> list[str]:
    """Return feature names after transformation (for SHAP / permutation importance)."""
    ct: ColumnTransformer = pipeline.named_steps["preprocessor"]
    ohe: OneHotEncoder = ct.named_transformers_["cat"].named_steps["encoder"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERIC_FEATURES.copy() + cat_names


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV and normalise column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features with business signal.
    Applied BEFORE the train/test split so derivation is identical on both sets.
    Pure arithmetic — no look-ahead, no use of the target.
    """
    df = df.copy()

    # +1 évite la division par zéro pour les clients avec tenure_months=0
    df["tickets_per_tenure"] = df["support_tickets"] / (df["tenure_months"] + 1)
    df["fee_per_tenure"] = df["monthly_fee"] / (df["tenure_months"] + 1)

    # Score d'engagement (formule du sujet, toutes variables déjà présentes)
    # Normalisation MinMax locale (sera re-scalée par le StandardScaler ensuite)
    def _minmax(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    df["engagement_score"] = (
        0.25 * _minmax(df["monthly_logins"])
        + 0.20 * _minmax(df["weekly_active_days"])
        + 0.20 * _minmax(df["avg_session_time"])
        + 0.10 * _minmax(df["features_used"])
        + 0.10 * _minmax(df["usage_growth_rate"])
        + 0.10 * (1 - _minmax(df["last_login_days_ago"]))  # inversion : + c'est élevé, - d'engagement
    )

    return df


def build_revenue_at_risk(df: pd.DataFrame, churn_proba: np.ndarray) -> pd.Series:
    """
    Compute revenue_at_risk = total_revenue * P(churn) for Task B.
    Must be called AFTER Task A model is trained — proba comes from TEST set inference.
    """
    return pd.Series(df["total_revenue"].values * churn_proba, name=TARGET_REVENUE_RISK)
