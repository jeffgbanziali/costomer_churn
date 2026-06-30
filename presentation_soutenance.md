---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 22px;
  }
  h1 { color: #1a3c5e; font-size: 36px; }
  h2 { color: #1a3c5e; border-bottom: 2px solid #e74c3c; padding-bottom: 6px; }
  table { font-size: 18px; }
  code { background: #f0f4f8; padding: 2px 6px; border-radius: 4px; }
  .highlight { color: #e74c3c; font-weight: bold; }
---

# Customer Churn Intelligence Platform
### Prédiction du churn & estimation du revenu à risque

**Projet RNCP40875 Bloc 2 — EFREI M1 Data Engineering 2025-2026**

Jeff GBANZIALI WISSANGA

---

## 1. Problématique Business

**Contexte :** un opérateur télécom perd des clients sans anticiper leur départ.

| Métrique | Valeur |
|---|---|
| Portefeuille clients | 10 000 clients, 32 features |
| Taux de churn actuel | **10,2 %** du portefeuille |
| Revenu à risque identifié | **862 640 €** (8,2 % du CA) |
| Impact d'une réduction de 5 pts | ~5 176 000 € de revenu sauvegardé |

**Objectif :** construire un système d'alerte précoce qui :
1. Prédit la probabilité de churn par client (Tâche A — classification)
2. Estime le revenu à risque associé (Tâche B — régression)
3. Explique chaque décision pour l'équipe métier (interprétabilité)

**KPI de succès :** F1-score ≥ 0,75 sur la classe churn=1 (minimiser les faux négatifs)

---

## 2. Dataset & Insights EDA Clés

**32 features** regroupées en 5 domaines : démographie, comportement usage, facturation, support client, engagement marketing.

**Insights critiques identifiés :**

- `csat_score` — gradient quasi-monotone : CSAT=1 → churn **~25 %**, CSAT=5 → churn **~4 %**
- **Clients < 12 mois** : taux de churn 2× supérieur à la moyenne → signal d'échec d'onboarding
- `complaint_type` : **20 % de NaN** — absence de réclamation ≠ satisfaction (désengagement silencieux)
- `monthly_fee` : **6 valeurs distinctes** uniquement (variable ordinale, pas continue)
- `total_revenue` corrélée à `monthly_fee × tenure_months` → multicolinéarité modérée conservée
- `payment_failures` + `price_increase_last_3m` : signal combiné de stress financier

**Features dérivées créées :**
- `tickets_per_tenure` (r = +0,12 avec churn)
- `fee_per_tenure` (r = +0,14 avec churn)

---

## 3. Pipeline ML — Architecture Anti-Fuite

**Principe fondamental :** `train_test_split` AVANT tout `.fit()` pour éviter la fuite de données.

```
Raw data (10 000 lignes)
      │
      ├─ train (80 %) ──→ ColumnTransformer.fit_transform()
      │                        ├── StandardScaler  (numériques)
      │                        └── OneHotEncoder   (catégorielles)
      └─ test  (20 %) ──→ .transform() uniquement
```

**Gestion du déséquilibre (10,2 % churners) :**
| Modèle | Technique |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight ≈ 8.8` (ratio maj/min) |
| MLP Keras | `class_weight={0: 1.0, 1: 8.8}` |

**Imputation :** `complaint_type` → `SimpleImputer(strategy='constant', fill_value='Unknown')`
**Encodage :** `OneHotEncoder(drop='first', handle_unknown='ignore')` — évite le dummy trap

---

## 4. Résultats des Modèles

### Tâche A — Classification Churn

| Modèle | F1-score (churn=1) | PR-AUC | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0,71 | 0,68 | 0,79 |
| Random Forest | 0,78 | 0,74 | 0,84 |
| **XGBoost** | **0,81** | **0,78** | **0,87** |
| MLP (Keras) | 0,79 | 0,76 | 0,85 |

**XGBoost sélectionné** comme modèle de production — meilleur F1 et PR-AUC.

### Tâche B — Régression Revenu à Risque

| Modèle | MAE (€) | RMSE (€) | R² |
|---|---|---|---|
| Ridge Regression | 185 | 312 | 0,71 |
| Random Forest | 142 | 268 | 0,79 |
| **XGBoost Regressor** | **128** | **241** | **0,83** |

---

## 5. Interprétabilité — 3 Niveaux

**Niveau 1 — Feature Importance** (rapide, global) :
Top features XGBoost : `csat_score` > `nps_score` > `tenure_months` > `payment_failures` > `monthly_fee`

**Niveau 2 — Permutation Importance** (modèle-agnostique, sur test set) :
`csat_score` : Δ F1 = −0,076 si permutée → variable la plus prédictive confirmée

**Niveau 3 — SHAP KernelExplainer** (explicabilité individuelle) :
- 200 échantillons de background (k-means, 50 clusters) pour le MLP
- Waterfall plot par client → raison précise de l'alerte

**Cas réel analysé — churner manqué (P(churn) = 0,001, vrai churn = 1) :**
- `tenure_months = 24` → SHAP très négatif (signal de fidélité fort)
- `nps_score = −25`, `support_tickets = 3` → signaux de détresse ignorés
- **Conclusion :** le MLP sur-pondère l'ancienneté et rate la détérioration récente

> L'interprétabilité permet de corriger les biais du modèle en production.

---

## 6. Architecture Déployée

```
┌─────────────────────────────────────────────┐
│              Docker Compose                  │
│                                             │
│  ┌──────────────────┐  ┌─────────────────┐  │
│  │   API FastAPI     │  │  Dashboard      │  │
│  │   Port 8000       │  │  Streamlit      │  │
│  │                   │  │  Port 8501      │  │
│  │  POST /token      │  │                 │  │
│  │  GET  /health     │◄─┤  Simulateur     │  │
│  │  POST /predict/   │  │  EDA Charts     │  │
│  │    churn          │  │  Top clients    │  │
│  │    revenue-risk   │  │  à risque       │  │
│  │                   │  │                 │  │
│  │  JWT Auth         │  │  API_BASE_URL   │  │
│  │  Pydantic v2      │  │  = http://api   │  │
│  └──────────────────┘  └─────────────────┘  │
│         │ healthcheck                        │
│         └── depends_on ──────────────────►  │
└─────────────────────────────────────────────┘
```

**Sécurité :** JWT (python-jose) + bcrypt (passlib) + `.env` exclu du git
**Modèles :** montés en volume read-only (`./models:/app/models:ro`)
**Reproductibilité :** `docker compose up --build` → stack complète opérationnelle
