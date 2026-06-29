# Plateforme Intelligence Client — Prédiction du Churn & Risque de Revenus

Système de prédiction du churn client combinant machine learning classique et deep learning, avec une API REST sécurisée et un dashboard interactif. Le projet traite un cas réel d'une base de 10 000 clients SaaS présentant un taux de churn de ~10%.

---

## Aperçu des résultats

| Modèle | F1 (churn) | ROC-AUC | PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.311 | 0.748 | 0.271 |
| Random Forest | 0.193 | 0.802 | 0.271 |
| XGBoost | 0.174 | 0.741 | 0.219 |
| **MLP (sélectionné)** | **0.335** | **0.742** | **0.219** |

Le MLP est retenu pour son meilleur **rappel (53%)** — dans un contexte churn, rater un churner coûte plus cher que de déclencher une action de rétention inutile.

Revenu à risque moyen estimé : **238 €/client** (modèle régression Random Forest, R²=0.65).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                   │
│         (4 pages : EDA · Prédiction · Batch · API)       │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP (JWT Bearer)
┌────────────────────────▼────────────────────────────────┐
│                     FastAPI  :8000                       │
│   POST /token   GET /health                              │
│   POST /predict/churn   POST /predict/revenue-risk       │
└────────────────────────┬────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
     MLP Keras (.keras)     Random Forest (.joblib)
      Tâche A : churn        Tâche B : revenu à risque
```

---

## Structure du projet

```
projet_data_science/
│
├── data/                          # Dataset brut (non versionné)
│   └── customer_churn_business_dataset.csv
│
├── notebooks/
│   ├── 01_eda.ipynb               # Analyse exploratoire
│   ├── 02_preprocessing.ipynb    # Pipeline anti-leakage
│   ├── 03_modeling_churn.ipynb   # 4 modèles de classification
│   ├── 04_modeling_revenue_risk.ipynb  # Régression revenu à risque
│   └── 05_interpretability.ipynb # SHAP + Permutation Importance
│
├── src/
│   ├── preprocessing.py          # Pipeline sklearn (ColumnTransformer)
│   ├── evaluation.py             # Métriques, SHAP helpers
│   ├── train_churn.py            # Script d'entraînement standalone
│   └── train_revenue_risk.py     # Script pour la Tâche B
│
├── api/
│   ├── Dockerfile
│   ├── main.py                   # FastAPI — endpoints JWT
│   ├── auth.py                   # python-jose, bcrypt
│   ├── schemas.py                # Validation Pydantic v2
│   ├── requirements.txt
│   ├── .env.example              # Template de configuration
│   └── .env                      # Secrets (non versionné)
│
├── dashboard/
│   ├── Dockerfile
│   └── app.py                    # Streamlit — 4 pages
│
├── models/                       # Artefacts ML (non versionnés)
│   ├── best_model_churn.keras
│   ├── best_model_revenue.joblib
│   ├── preprocessor_churn.joblib
│   └── preprocessor_revenue.joblib
│
├── reports/figures/              # Visualisations exportées
├── docker-compose.yml
├── .dockerignore
└── requirements.txt
```

---

## Installation

### Prérequis
- Python 3.11 (TensorFlow 2.x n'est pas compatible avec Python 3.12+)
- Docker Desktop (pour l'option conteneurs)

### Environnement local

```powershell
# Créer et activer le venv Python 3.11
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Installer toutes les dépendances
pip install -r requirements.txt
pip install -r api/requirements.txt
pip install streamlit
```

### Configurer les secrets

```powershell
copy api\.env.example api\.env
# Éditer api\.env avec une vraie SECRET_KEY
```

---

## Lancer le projet

### Option 1 — En local

```powershell
# Terminal 1 : API
cd api
..\\.venv\Scripts\uvicorn.exe main:app --reload --port 8000

# Terminal 2 : Dashboard
.\.venv\Scripts\streamlit.exe run dashboard/app.py
```

- Swagger : http://localhost:8000/docs
- Dashboard : http://localhost:8501

### Option 2 — Docker Compose

```bash
docker-compose up --build
```

Les deux services démarrent automatiquement. Le dashboard attend que l'API soit prête (healthcheck).

### Régénérer les modèles

Si les fichiers `models/` ne sont pas présents, exécuter les notebooks dans l'ordre :

```
01_eda → 02_preprocessing → 03_modeling_churn → 04_modeling_revenue_risk → 05_interpretability
```

Ou via les scripts standalone :
```powershell
.\.venv\Scripts\python.exe src/train_churn.py
.\.venv\Scripts\python.exe src/train_revenue_risk.py
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

Réponse attendue :
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8932,
  "risk_level": "Élevé",
  "model_used": "MLP (Keras)"
}
```

---

## Stack technique

| Composant | Technologie |
|---|---|
| Langage | Python 3.11 |
| ML | scikit-learn 1.9, XGBoost 3.2 |
| Deep Learning | TensorFlow / Keras 2.21 |
| Interprétabilité | SHAP 0.51, Permutation Importance |
| API | FastAPI 0.138, Uvicorn |
| Auth | python-jose (JWT), passlib (bcrypt) |
| Dashboard | Streamlit 1.58, Plotly |
| Containerisation | Docker, Docker Compose |
