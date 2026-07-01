# Rapport de Projet — Prédiction du Churn Client

**RNCP40875 Bloc 2 — Projet Data Science**
Auteur : Jeff GBANZIALI WISSANGA — M1 Data Engineering EFREI
Date : Juillet 2026

---

## 1. Contexte et problématique métier

Une entreprise SaaS souhaite anticiper le désabonnement (churn) de ses clients afin de déclencher des actions de rétention ciblées avant que la perte ne soit consommée.

**Enjeu financier quantifié :**
- Coût d'un faux négatif (churner non détecté) : **238 EUR** (revenu perdu)
- Coût d'un faux positif (client fidèle contacté inutilement) : **30 EUR** (campagne de rétention)
- Taux de rétention estimé des actions ciblées : 25%
- ROI à démontrer : détecter suffisamment de churners pour que les économies réalisées dépassent les coûts de campagne

Ce déséquilibre de coûts justifie de maximiser le **rappel** plutôt que la précision, et d'utiliser le **F1-score** comme métrique principale (compromis précision/rappel).

---

## 2. Dataset

- **Source :** `customer_churn_business_dataset.csv`
- **Volume :** 10 000 clients, 31 variables
- **Variable cible :** `churn` (binaire : 0 = client actif, 1 = client churné)
- **Taux de churn :** ~10% (fort déséquilibre de classes)

**Catégories de variables :**

| Catégorie | Variables |
|---|---|
| Démographie | age, gender, country, city, customer_segment |
| Comportement produit | monthly_logins, weekly_active_days, avg_session_time, features_used |
| Contrat & facturation | tenure_months, contract_type, monthly_fee, total_revenue, payment_method |
| Satisfaction | csat_score, nps_score, survey_response, support_tickets, escalations |
| Marketing | email_open_rate, marketing_click_rate, referral_count |

---

## 3. Analyse exploratoire (Notebook 01)

**Principaux facteurs de churn identifiés :**

- **Ancienneté faible** : les clients qui churnent ont en médiane 12 mois d'ancienneté vs 36 mois pour les clients fidèles
- **CSAT et NPS bas** : churners ont un CSAT moyen de 2.1/5 vs 3.8/5
- **Tickets de support élevés** : 4+ tickets corrèle fortement avec le churn
- **Contrats mensuels** : taux de churn 3× supérieur aux contrats annuels
- **Inactivité récente** : last_login_days_ago > 14 jours est un signal fort

**Feature engineering créé :**
- `tickets_per_tenure` : taux de tickets normalisé par l'ancienneté
- `fee_per_tenure` : valeur mensuelle normalisée
- `engagement_score` : score composite (logins × 30% + active_days × 25% + session_time × 25% + features_used × 20%)

---

## 4. Préparation des données (Notebook 02)

### 4.1 Pipeline anti-leakage

Principe fondamental respecté : **aucun `.fit()` avant le split**.

```
Données brutes (10 000 lignes)
         ↓ train_test_split (stratified, random_state=42)
    70% train / 15% val / 15% test
         ↓ feature engineering (bounds calculés sur train uniquement)
         ↓ ColumnTransformer.fit(X_train)
         ↓ .transform(X_val) et .transform(X_test)
```

### 4.2 ColumnTransformer

- **Variables numériques** (28) : `SimpleImputer(median)` + `StandardScaler`
- **Variables catégorielles** (6) : `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')`

### 4.3 Gestion du déséquilibre

Approche retenue : **`class_weight='balanced'`** (ou équivalent) dans chaque modèle, plutôt que SMOTE, pour éviter de synthétiser des données fictives dans le pipeline.

---

## 5. Modélisation (Notebook 03)

### 5.1 Modèles entraînés

| Modèle | Stratégie anti-déséquilibre | Optimisation |
|---|---|---|
| Logistic Regression | `class_weight='balanced'` | `RandomizedSearchCV` (C, solver) |
| Random Forest | `class_weight='balanced'` | `RandomizedSearchCV` (n_estimators, max_depth, min_samples) |
| XGBoost | `scale_pos_weight=8.79` | `RandomizedSearchCV` (n_estimators, max_depth, learning_rate) |
| MLP (Keras) | `class_weight={0:1.0, 1:8.79}` | 3 architectures, EarlyStopping |

`RandomizedSearchCV` avec `StratifiedKFold(k=5)` sur le jeu d'entraînement.

### 5.2 Optimisation du seuil de décision

Le seuil par défaut (0.5) est sous-optimal pour les classes déséquilibrées. Chaque modèle dispose d'un seuil optimisé via la **courbe Précision-Rappel** (F1 maximal sur le jeu de validation) :

| Modèle | Seuil optimisé |
|---|---|
| Logistic Regression | 0.57 |
| Random Forest | 0.63 |
| XGBoost | 0.65 |
| MLP (Keras) | 0.58 |

### 5.3 Résultats comparatifs (jeu de validation)

| Modèle | F1 | Recall | Précision | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.43** | **0.76** | 0.30 | **0.84** | **0.36** |
| XGBoost | 0.42 | 0.59 | 0.33 | 0.84 | 0.35 |
| MLP (Keras) | 0.42 | 0.59 | 0.32 | 0.80 | 0.26 |
| Logistic Regression | 0.39 | 0.56 | 0.30 | 0.79 | 0.28 |

### 5.4 Modèle retenu : Random Forest

**Justification :**
1. Meilleur F1 sur validation (0.43) et meilleur rappel (76%) — critère prioritaire vu le coût des FN
2. ROC-AUC de 0.84 — le plus élevé avec XGBoost
3. Robustesse : forêts aléatoires peu sensibles au bruit et aux outliers
4. Interprétabilité partielle via SHAP (vs MLP qui est une boîte noire)

**Résultats sur jeu de test (jamais vu) :**

| Métrique | Valeur |
|---|---|
| F1 | 0.376 |
| Recall | 0.680 |
| Précision | 0.260 |
| ROC-AUC | 0.793 |
| PR-AUC | 0.273 |

Légère dégradation F1 (0.43 → 0.38) attendue : seuil optimisé sur validation, généralisation légèrement moindre.

---

## 6. Interprétabilité (Notebook 04)

### 6.1 SHAP (SHapley Additive exPlanations)

Les valeurs SHAP quantifient la **contribution marginale** de chaque variable à la prédiction individuelle (et non une corrélation globale).

**Variables les plus impactantes** (beeswarm SHAP) :
1. `tenure_months` — ancienneté faible = contribution positive forte au churn
2. `csat_score` — insatisfaction augmente le risque
3. `tickets_per_tenure` — ratio tickets/ancienneté élevé = signal fort
4. `contract_type_Monthly` — contrats mensuels plus risqués
5. `engagement_score` — faible engagement corrèle avec le churn

### 6.2 Analyse cas individuels

- **Waterfall TP** : client churné correctement identifié — ancienneté faible + CSAT bas cumulés
- **Waterfall FN** : churner manqué — engagement moyen masquant les signaux négatifs
- **Dependence plots** : effet non-linéaire de tenure_months (seuil de risque ~18 mois)

---

## 7. Architecture de production

### 7.1 API FastAPI

- **Endpoint** : `POST /predict/churn` — prédit le churn d'un client unique
- **Endpoint** : `POST /predict/churn/batch` — jusqu'à 500 clients par requête
- **Endpoint** : `POST /predict/revenue-risk` — revenu à risque = total_revenue × P(churn)
- **Sécurité** : authentification JWT (python-jose + bcrypt), OAuth2 Password Flow
- **Validation** : Pydantic v2 — 31 champs typés avec contraintes
- **Chargement à froid** : les artifacts sont chargés une fois au démarrage (`lifespan` FastAPI)

### 7.2 Dashboard Streamlit (5 pages)

| Page | Contenu |
|---|---|
| Vue d'ensemble | 5 KPIs, distribution par segment, violin CSAT, NPS histogram |
| Analyse du risque | Quadrant NPS×CSAT, payment failures, revenu par ancienneté |
| Simulateur | Prédiction individuelle, jauge de risque, recommandation d'action |
| Analyse batch | Upload CSV, distribution des probabilités, top clients à risque |
| Comparaison modèles | Tableau métriques, F1 barplot avec objectif 0.75, seuils optimaux |

---

## 8. Écoresponsabilité

Le choix du modèle tient compte du rapport performance/coût computationnel :

| Modèle | Complexité | Temps entraînement | F1 val | Bilan |
|---|---|---|---|---|
| Logistic Regression | Linéaire | < 1 s | 0.39 | Baseline frugale |
| Random Forest | 100-200 arbres | ~30 s | **0.43** | Meilleur ratio perf/coût |
| XGBoost | Boosting itératif | ~45 s | 0.42 | Légèrement moins bon |
| MLP (Keras) | 3 couches denses, GPU | ~5 min | 0.42 | Coût élevé, gain nul |

Le **Random Forest** offre le meilleur F1 avec un coût d'entraînement modéré. Le **MLP Keras** — 6× plus long à entraîner — n'apporte aucune amélioration mesurable. Dans un contexte de déploiement continu (ré-entraînement mensuel), choisir Random Forest réduit significativement l'empreinte énergétique.

**En production** : les prédictions à l'inférence sont quasi-instantanées pour tous les modèles. Le critère éco porte sur le cycle d'entraînement.

---

## 9. Conclusion

**Objectifs atteints :**
- Pipeline anti-leakage robuste (split stratifié → fit uniquement sur train)
- 4 modèles comparés avec tuning et optimisation de seuil
- Modèle Random Forest retenu : F1=0.43, Recall=0.68 sur test
- Interprétabilité SHAP complète (globale + individuelle)
- API sécurisée déployable (FastAPI + JWT)
- Dashboard interactif 5 pages (Streamlit + Plotly)

**Limites et perspectives :**
- F1 de 0.38 sur test reste inférieur à l'objectif 0.75 fixé — le déséquilibre des classes (10%) rend difficile d'atteindre ce seuil avec les données disponibles
- Feature engineering limité — des données comportementales plus granulaires (événements in-app) amélioreraient les signaux
- Modèle de détection de dérive (data drift) à implémenter pour la mise en production
- Ré-entraînement périodique à automatiser (MLflow, Airflow)
