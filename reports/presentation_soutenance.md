---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 22px;
    background: #ffffff;
  }
  section.lead {
    background: #1a2b4a;
    color: white;
  }
  section.lead h1 { color: #4fc3f7; font-size: 2em; }
  section.lead h2 { color: #b0bec5; }
  h2 { color: #1a2b4a; border-bottom: 3px solid #4fc3f7; padding-bottom: 6px; }
  table { font-size: 0.85em; width: 100%; }
  th { background: #1a2b4a; color: white; }
  td, th { padding: 6px 10px; }
  .highlight { color: #e74c3c; font-weight: bold; }
  .good { color: #27ae60; font-weight: bold; }
  .tag { background: #4fc3f7; color: white; border-radius: 4px; padding: 2px 8px; font-size: 0.8em; }
---

<!-- _class: lead -->

# Plateforme Intelligence Client
## Prédiction du Churn — Projet Data Science

**Jeff GBANZIALI WISSANGA**
M1 Data Engineering — EFREI
RNCP40875 Bloc 2 · Juillet 2026

---

<!-- Slide 1 — Bloc 1 placeholder (ne pas présenter) -->

---

<!-- Slides 2–9 : Projet 1 (à compléter par le binôme) -->

---

<!-- Slide 10 : Question métier -->

## Question métier

**Problème :** Une entreprise SaaS perd ~10% de ses clients par an.
Peut-on anticiper le churn pour déclencher des actions de rétention ciblées ?

### Enjeu financier

| Scénario | Coût |
|---|---|
| Client churné non détecté (Faux Négatif) | **238 EUR** perdu |
| Client fidèle contacté inutilement (Faux Positif) | 30 EUR campagne |
| Rétention réussie (taux ~25%) | **60 EUR économisés** |

> Rater un churner coûte 8× plus cher qu'une action inutile.
> → Priorité au **rappel**, pas à la précision.

**Dataset :** 10 000 clients, 31 variables (comportement, satisfaction, contrat)

---

<!-- Slide 11 : Préparation et nettoyage -->

## Préparation des données

### Pipeline anti-leakage

```
10 000 clients
    ↓  train_test_split stratifié (random_state=42)
  70% train │ 15% validation │ 15% test
    ↓  Feature engineering (bounds sur train uniquement)
    ↓  ColumnTransformer.fit(X_train) uniquement
    ↓  .transform(X_val) et .transform(X_test)
```

**Règle fondamentale :** Aucune information du val/test ne pollue l'entraînement.

### Transformations

- **Numériques (28 vars) :** `SimpleImputer(median)` → `StandardScaler`
- **Catégorielles (6 vars) :** `SimpleImputer(mode)` → `OneHotEncoder`
- **Features engineerées :** `tickets_per_tenure`, `fee_per_tenure`, `engagement_score` (score composite logins/activité/session/features)

### Déséquilibre de classes

Taux de churn = 10% → `class_weight='balanced'` dans chaque modèle
(ratio négatifs/positifs = 8.79)

---

<!-- Slide 12 : Analyse exploratoire -->

## Analyse exploratoire — Principaux insights

### Facteurs de churn identifiés

| Facteur | Churners | Non-churners |
|---|---|---|
| Ancienneté médiane | 12 mois | 36 mois |
| CSAT moyen | 2.1 / 5 | 3.8 / 5 |
| Tickets support | 4+ | < 2 |
| Contrat mensuel | 85% | 40% |
| Last login (jours) | > 14 | < 7 |

### Visualisations clés
- Distribution du churn par segment (Enterprise, SME, Consumer)
- Violin plot CSAT — séparation nette des distributions
- Quadrant NPS × CSAT — zone à risque visible en bas à gauche
- Matrice de corrélation — colinéarités contrôlées

> Les clients Enterprise churnent moins (contrats pluriannuels)
> Les contrats mensuels ont un taux de churn 3× supérieur

---

<!-- Slide 13 : Modélisation prédictive -->

## Modélisation prédictive

### 4 modèles comparés

| Modèle | Tuning | Anti-déséquilibre |
|---|---|---|
| Logistic Regression | RandomizedSearchCV (C) | `class_weight='balanced'` |
| Random Forest | RandomizedSearchCV (n_est, depth) | `class_weight='balanced'` |
| XGBoost | RandomizedSearchCV (lr, depth) | `scale_pos_weight=8.79` |
| MLP Keras | 3 architectures, EarlyStopping | `class_weight={0:1.0, 1:8.79}` |

### Optimisation du seuil de décision

Seuil par défaut 0.5 → sous-optimal. Chaque modèle dispose d'un **seuil optimisé** via la courbe Précision-Rappel (F1 maximal sur validation).

### Résultats (validation)

| Modèle | F1 | Recall | ROC-AUC |
|---|---|---|---|
| **Random Forest ✓** | **0.43** | **0.76** | **0.84** |
| XGBoost | 0.42 | 0.59 | 0.84 |
| MLP (Keras) | 0.42 | 0.59 | 0.80 |
| Logistic Regression | 0.39 | 0.56 | 0.79 |

---

<!-- Slide 14 : Comparaison et écoresponsabilité -->

## Comparaison et écoresponsabilité

### Modèle retenu : Random Forest (seuil = 0.63)

**Pourquoi Random Forest ?**
1. Meilleur F1 (0.43) et rappel (76%) sur validation
2. Test (jamais vu) : F1=0.376, Recall=0.680, ROC-AUC=0.793
3. Interprétabilité via SHAP — variables clés : ancienneté, CSAT, tickets/tenure, contrat
4. Coût computationnel modéré

### Empreinte écologique comparée

| Modèle | Durée entraînement | F1 val | Bilan éco |
|---|---|---|---|
| Logistic Regression | < 1 s | 0.39 | Baseline frugale |
| **Random Forest** | ~30 s | **0.43** | **Meilleur ratio** |
| XGBoost | ~45 s | 0.42 | Légèrement moins bon |
| MLP (Keras) | ~5 min + GPU | 0.42 | Coût élevé, gain nul |

> Le MLP est 10× plus coûteux pour des performances identiques.
> **Décision :** Random Forest — performance maximale, empreinte minimale.

### Architecture de production
API FastAPI sécurisée (JWT) + Dashboard Streamlit 5 pages + Containerisation Docker

---

<!-- Slide 15 : Réponses jury (notes — ne pas projeter) -->

## Notes jury (usage personnel)

**"Pourquoi ce modèle final ?"**
→ Random Forest : meilleur F1 ET meilleur rappel sur validation. Contexte churn = FN coûteux (238 EUR), donc rappel prioritaire. Ecologiquement plus sobre que MLP à performances égales.

**"Quelles métriques avez-vous utilisées ?"**
→ F1-score (compromis précision/rappel), Recall (coût FN), ROC-AUC (discrimination), PR-AUC (adapté aux classes déséquilibrées). Pas uniquement l'accuracy (trompeuse à 10% positifs).

**"Comment avez-vous évité le surapprentissage ?"**
→ Split 70/15/15 stratifié AVANT tout fit. ColumnTransformer fitté uniquement sur train. Seuil optimisé sur val, évaluation finale sur test (résultats légèrement différents — preuve de non-leakage).

**"Quelles limites présente votre modèle ?"**
→ F1=0.38 sur test < objectif 0.75 (difficile avec 10% positifs). Données statiques — pas d'historique temporel. Pas de détection de dérive en production. Régression revenu non finale (fallback P(churn)×total_revenue).

**"Quels sont les facteurs les plus importants ?"**
→ SHAP : tenure_months, csat_score, tickets_per_tenure, contract_type_Monthly, engagement_score.
