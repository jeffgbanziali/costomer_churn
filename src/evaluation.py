"""
Evaluation utilities: metrics, comparison tables, error analysis.
Shared between notebooks and API — do not import anything from notebooks here.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.inspection import permutation_importance


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classification_metrics(y_true, y_pred, y_proba=None, model_name: str = "") -> dict:
    """
    Compute all relevant classification metrics.
    Always include PR-AUC alongside ROC-AUC — churn datasets are imbalanced,
    accuracy alone is misleading.
    """
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        # zero_division=0 : retourne 0 (pas d'erreur) quand le modèle ne prédit aucun positif
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    return metrics


def compare_classifiers(results: list[dict]) -> pd.DataFrame:
    """
    Turn a list of metric dicts (from classification_metrics) into a
    sorted comparison DataFrame, best F1 first.
    """
    df = pd.DataFrame(results).set_index("model")
    return df.sort_values("f1", ascending=False).round(4)


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matrice de confusion — {model_name}")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def regression_metrics(y_true, y_pred, model_name: str = "") -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "model": model_name,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
    }


def compare_regressors(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results).set_index("model")
    return df.sort_values("r2", ascending=False).round(4)


# ---------------------------------------------------------------------------
# Permutation Importance (model-agnostic, recommended level)
# ---------------------------------------------------------------------------

def compute_permutation_importance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    scoring: str = "f1",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance on the test set.
    Returns a DataFrame sorted by mean importance descending.
    """
    result = permutation_importance(
        model, X_test, y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


def plot_permutation_importance(df: pd.DataFrame, model_name: str, top_n: int = 15, save_path: str = None):
    top = df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"][::-1], top["importance_mean"][::-1],
            xerr=top["importance_std"][::-1], color="steelblue", alpha=0.85)
    ax.set_xlabel("Baisse de performance (permutation)")
    ax.set_title(f"Permutation Importance — {model_name} (top {top_n})")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Error analysis helper
# ---------------------------------------------------------------------------

def error_analysis_classification(
    X_test_raw: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_worst: int = 10,
) -> pd.DataFrame:
    """
    Return the n_worst false negatives (churners missed by the model).
    False negatives are the most costly errors in a churn context.
    """
    df = X_test_raw.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["churn_proba"] = y_proba
    # Faux négatifs seulement : un churner non détecté coûte plus cher qu'une fausse alarme
    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)]
    return fn.sort_values("churn_proba", ascending=False).head(n_worst)
