# Plateforme Intelligence Client — Prédiction du Churn

Système de prédiction du churn client combinant machine learning classique et deep learning, avec une API REST sécurisée et un dashboard interactif. Le projet traite 10 000 clients SaaS avec un taux de churn de ~10%.

---

## Résultats (seuil optimisé par courbe PR)

| Modèle | F1 | Recall | Précision | ROC-AUC | PR-AUC | Seuil |
|---|---|---|---|---|---|---|
| **Random Forest** | **0.43** | **0.76** | 0.30 | **0.84** | **0.36** | 0.63 |
| XGBoost | 0.42 | 0.59 | 0.33 | 0.84 | 0.35 | 0.66 |
| MLP (Keras) | 0.42 | 0.59 | 0.32 | 0.80 | 0.26 | 0.58 |
| Logistic Regression | 0.39 | 0.56 | 0.30 | 0.79 | 0.28 | 0.57 |

Le **Random Forest** est retenu : meilleur F1 (0.43) et meilleur rappel (76%) sur validation. Contexte métier : rater un churner coûte 238 EUR, déclencher une action inutile 30 EUR — maximiser le rappel prime.

Test final (Random Forest, seuil 0.63) : F1=0.376, Recall=0.680, ROC-AUC=0.793.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard (5 pages)           │
│    EDA · Analyse Risque · Simulateur · Batch · Modeles  │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP (JWT Bearer)
┌────────────────────────▼────────────────────────────────┐
│                    FastAPI :8000                          │
│  POST /token  GET /health                                │
│  POST /predict/churn  POST /predict/churn/batch          │
│  POST /predict/revenue-risk                              │
└────────────────────────┬────────────────────────────────┘
                         │
              preprocessing_artifacts.joblib
              best_model_churn.joblib (Random Forest)
              modeling_artifacts.joblib
```

---

## Structure du projet

```
projet_data_science/
│
├── data/                              # Dataset brut (non versionné)
│   └── customer_churn_business_dataset.csv
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Analyse exploratoire
│   ├── 02_preprocessing.ipynb         # Pipeline anti-leakage (70/15/15)
│   ├── 03_modeling.ipynb              # 4 modèles de classification
│   └── 04_interpretability.ipynb      # SHAP beeswarm, waterfall, dependence
│
├── api/
│   ├── main.py                        # FastAPI — endpoints JWT
│   ├── auth.py                        # python-jose, bcrypt
│   ├── schemas.py                     # Validation Pydantic v2
│   ├── .env.example                   # Template de configuration
│   └── .env                           # Secrets (non versionné)
│
├── dashboard/
│   └── app.py                         # Streamlit — 5 pages
│
├── models/                            # Artefacts ML (non versionnés)
│   ├── preprocessing_artifacts.joblib # Preprocessor + splits (7.4 MB)
│   ├── best_model_churn.joblib        # Random Forest final (1.5 MB)
│   ├── churn_comparison.joblib        # Tableau comparatif
│   └── modeling_artifacts.joblib      # Seuils, métriques, probas (25 KB)
│
├── reports/figures/                   # Visualisations exportées (PNG)
├── docker-compose.yml
└── requirements.txt
```

---

## Installation

### Prérequis
- Python 3.11 (TensorFlow 2.x incompatible avec Python 3.12+)

### Environnement local

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configurer les secrets

```powershell
copy api\.env.example api\.env
# Éditer api\.env et remplacer SECRET_KEY par une valeur aléatoire
```

---

## Lancer le projet

```powershell
# Terminal 1 : API
cd api
..\\.venv\Scripts\uvicorn.exe main:app --reload --port 8000

# Terminal 2 : Dashboard
.\.venv\Scripts\streamlit.exe run dashboard/app.py
```

- Swagger : http://localhost:8000/docs
- Dashboard : http://localhost:8501

### Régénérer les modèles

Exécuter les notebooks dans l'ordre :

```
01_eda → 02_preprocessing → 03_modeling → 04_interpretability
```

---

## Utiliser l'API

```powershell
# 1. Authentification
$token = (Invoke-RestMethod -Uri "http://localhost:8000/token" `
  -Method POST `
  -Body "username=admin&password=churn2025!" `
  -ContentType "application/x-www-form-urlencoded").access_token

# 2. Prédiction churn
Invoke-RestMethod -Uri "http://localhost:8000/predict/churn" `
  -Method POST `
  -Headers @{Authorization="Bearer $token"} `
  -ContentType "application/json" `
  -Body '{
    "age": 32, "gender": "Male", "country": "France", "city": "Paris",
    "customer_segment": "SME", "tenure_months": 2, "signup_channel": "Web",
    "contract_type": "Monthly", "monthly_logins": 4, "weekly_active_days": 1,
    "avg_session_time": 5.0, "features_used": 2, "usage_growth_rate": -0.2,
    "last_login_days_ago": 15, "monthly_fee": 50, "total_revenue": 100,
    "payment_method": "Card", "payment_failures": 2, "discount_applied": "No",
    "price_increase_last_3m": "Yes", "support_tickets": 4,
    "avg_resolution_time": 35.0, "complaint_type": null, "csat_score": 2.0,
    "escalations": 1, "email_open_rate": 0.1, "marketing_click_rate": 0.05,
    "nps_score": -40, "survey_response": "Unsatisfied", "referral_count": 0
  }'
```

Réponse :
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8932,
  "risk_level": "Elevé",
  "model_used": "RandomForestClassifier"
}
```

---

## Stack technique

| Composant | Technologie |
|---|---|
| Langage | Python 3.11 |
| ML | scikit-learn 1.9, XGBoost 3.2 |
| Deep Learning | TensorFlow / Keras 2.21 |
| Rééquilibrage classes | imbalanced-learn (SMOTE comparatif) |
| Interprétabilité | SHAP 0.51 |
| API | FastAPI 0.138, Uvicorn |
| Auth | python-jose (JWT), passlib (bcrypt) |
| Dashboard | Streamlit 1.58, Plotly |
| Containerisation | Docker, Docker Compose |
