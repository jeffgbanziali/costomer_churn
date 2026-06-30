import os
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import joblib



# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_USER = os.getenv("DEMO_USERNAME", "admin")
API_PASS = os.getenv("DEMO_PASSWORD", "churn2025!")

st.set_page_config(
    page_title="Rétention Client — Dashboard IA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Authentification (token JWT mis en cache dans la session)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3000, show_spinner=False)
def get_jwt_token() -> str | None:
    try:
        resp = requests.post(
            f"{API_BASE}/token",
            data={"username": API_USER, "password": API_PASS},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()["access_token"]
    except requests.exceptions.ConnectionError:
        pass
    return None


def api_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Chargement des données (lecture locale du CSV)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Chargement des données...")
def load_data() -> pd.DataFrame:
    data_path = Path("/app/data") / "customer_churn_business_dataset.csv"
    if not data_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


# ---------------------------------------------------------------------------
# Sidebar — navigation
# ---------------------------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Section",
    ["Vue Globale (KPIs)", "Analyse du Risque", "Simulateur Client", "Comparaison Modèles"],
)

# Statut de l'API
try:
    health = requests.get(f"{API_BASE}/health", timeout=3).json()
    api_ok = health.get("status") == "ok"
except Exception:
    api_ok = False
    health = {}

status_color = "green" if api_ok else "red"
st.sidebar.markdown(
    f"**Statut API :** :{status_color}[{'En ligne' if api_ok else 'Hors ligne'}]"
)
if api_ok:
    st.sidebar.caption(
        f"Churn model: {'✅' if health.get('churn_model_loaded') else '⚠️ Non chargé'}  |  "
        f"Revenue model: {'✅' if health.get('revenue_model_loaded') else '⚠️ Non chargé'}"
    )

df = load_data()
token = get_jwt_token()

# ---------------------------------------------------------------------------
# PAGE 1 — Vue Globale (KPIs)
# ---------------------------------------------------------------------------

if page == "Vue Globale (KPIs)":
    st.title("Vue Globale — Indicateurs Clés de Performance")

    if df.empty:
        st.warning("Données non disponibles. Placez `customer_churn.csv` dans `data/`.")
        st.stop()

    churners   = df[df["churn"] == 1]
    n_total    = len(df)
    n_churn    = len(churners)
    churn_rate = n_churn / n_total

    rev_at_risk = churners["total_revenue"].sum() if "total_revenue" in df.columns else 0
    rev_total   = df["total_revenue"].sum() if "total_revenue" in df.columns else 0
    pct_rev_at_risk = rev_at_risk / rev_total if rev_total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clients totaux",      f"{n_total:,}")
    col2.metric("Clients churners",    f"{n_churn:,}",       delta=f"{churn_rate:.1%} du portefeuille", delta_color="inverse")
    col3.metric("Revenu total",        f"{rev_total:,.0f} €")
    col4.metric("Revenu à risque",     f"{rev_at_risk:,.0f} €", delta=f"{pct_rev_at_risk:.1%} du CA", delta_color="inverse")

    st.markdown("---")
    st.subheader("Distribution du churn par type de contrat")

    if "contract_type" in df.columns:
        churn_by_contract = (
            df.groupby("contract_type")["churn"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "Taux de churn", "count": "Nb clients"})
            .reset_index()
        )
        churn_by_contract["Taux de churn (%)"] = (churn_by_contract["Taux de churn"] * 100).round(1)
        fig = px.bar(
            churn_by_contract, x="contract_type",
            y="Taux de churn (%)", color="Taux de churn (%)",
            color_continuous_scale="Reds",
            labels={"contract_type": "Type de contrat"},
            title="Taux de churn par type de contrat",
        )
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribution NPS — Churners vs Rétentionnés")
        if "nps_score" in df.columns:
            fig2 = px.histogram(
                df, x="nps_score", color=df["churn"].map({0: "Rétentionné", 1: "Churner"}),
                barmode="overlay", opacity=0.7,
                color_discrete_map={"Churner": "#e74c3c", "Rétentionné": "#2ecc71"},
                labels={"nps_score": "Score NPS", "color": "Statut"},
            )
            st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.subheader("Ancienneté vs. Churn")
        if "tenure_months" in df.columns:
            fig3 = px.box(
                df, x=df["churn"].map({0: "Rétentionné", 1: "Churner"}),
                y="tenure_months", color=df["churn"].map({0: "Rétentionné", 1: "Churner"}),
                color_discrete_map={"Churner": "#e74c3c", "Rétentionné": "#2ecc71"},
                labels={"x": "Statut", "tenure_months": "Ancienneté (mois)"},
            )
            st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 2 — Analyse du Risque
# ---------------------------------------------------------------------------

elif page == "Analyse du Risque":
    st.title("Analyse du Risque — Segments Clients Prioritaires")

    if df.empty:
        st.warning("Données non disponibles.")
        st.stop()

    st.subheader("Top 10 clients par revenu mensuel à risque (estimation)")
    if "monthly_fee" in df.columns and "payment_failures" in df.columns:
        at_risk = df[df["churn"] == 1].copy()
        at_risk["revenue_mensuel_risque"] = at_risk["monthly_fee"]
        top10 = at_risk.sort_values("revenue_mensuel_risque", ascending=False).head(10)
        cols_to_show = [c for c in ["age", "tenure_months", "contract_type", "monthly_fee",
                                    "payment_failures", "nps_score"] if c in top10.columns]
        st.dataframe(top10[cols_to_show].reset_index(drop=True), use_container_width=True)

        total_mensuel_risque = at_risk["monthly_fee"].sum()
        st.info(
            f"**Insight business :** Les clients churners représentent "
            f"**{total_mensuel_risque:,.0f} €/mois** de charges mensuelles à risque. "
            f"Une campagne de rétention ciblant les 10% les plus à risque permettrait de "
            f"sauvegarder ~{total_mensuel_risque * 0.1:,.0f} €/mois si le taux de succès atteint 30%."
        )

    st.subheader("Revenu à risque par segment d'ancienneté")
    if "tenure_months" in df.columns:
        df_copy = df.copy()
        df_copy["segment_tenure"] = pd.cut(
            df_copy["tenure_months"], bins=[0, 12, 24, 60, 999],
            labels=["< 1 an", "1-2 ans", "2-5 ans", "> 5 ans"]
        )
        risk_by_seg = (
            df_copy[df_copy["churn"] == 1]
            .groupby("segment_tenure", observed=True)["total_revenue"]
            .sum()
            .reset_index()
        )
        fig4 = px.bar(
            risk_by_seg, x="segment_tenure", y="total_revenue",
            labels={"segment_tenure": "Ancienneté", "total_revenue": "Revenu à risque (€)"},
            title="Revenu à risque cumulé par segment d'ancienneté",
            color="total_revenue", color_continuous_scale="Oranges",
        )
        st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 3 — Simulateur Client
# ---------------------------------------------------------------------------

elif page == "Simulateur Client":
    st.title("Simulateur — Prédiction en Temps Réel")
    st.caption("Les prédictions sont obtenues via l'API (architecture Front / API / Modèle).")

    if not api_ok:
        st.error("L'API est hors ligne. Lancez : `uvicorn api.main:app --reload`")
        st.stop()
    if token is None:
        st.error("Impossible d'obtenir un token JWT. Vérifiez les variables d'environnement.")
        st.stop()
    if not health.get("churn_model_loaded"):
        st.warning("Modèle de churn non chargé. Lancez d'abord `python src/train_churn.py`.")

    with st.form("customer_form"):
        st.subheader("Profil du client")
        col1, col2, col3 = st.columns(3)
        age              = col1.number_input("Âge", 18, 74, 35)
        gender           = col1.selectbox("Genre", ["Male", "Female"])
        tenure_months    = col2.number_input("Ancienneté (mois)", 0, 300, 24)
        contract_type    = col2.selectbox("Contrat", ["Monthly", "Quarterly", "Yearly"])
        monthly_fee      = col3.number_input("Abonnement mensuel (€)", 0.0, 500.0, 65.0)
        total_revenue    = col3.number_input("Revenu total (€)", 0.0, 50000.0, 1500.0)

        col4, col5, col6 = st.columns(3)
        payment_failures = col4.number_input("Échecs de paiement", 0, 20, 1)
        support_tickets  = col5.number_input("Tickets support", 0, 50, 2)
        avg_session_time = col6.number_input("Durée session moy. (min)", 0.0, 300.0, 45.0)
        monthly_logins   = col4.number_input("Connexions/mois", 0, 200, 15)
        nps_score        = col5.slider("Score NPS", -100, 100, 20)
        csat_score       = col6.slider("CSAT (1-5)", 1.0, 5.0, 3.5, step=0.5)

        submitted = st.form_submit_button("Prédire", type="primary")

    if submitted:
        payload = {
            "age": age,
            "gender": gender,
            "country": "France",
            "city": "Paris",
            "customer_segment": "Individual",
            "tenure_months": tenure_months,
            "signup_channel": "Web",
            "contract_type": contract_type,
            "monthly_logins": monthly_logins,
            "weekly_active_days": 3,
            "avg_session_time": avg_session_time,
            "features_used": 5,
            "usage_growth_rate": 0.0,
            "last_login_days_ago": 7,
            "monthly_fee": monthly_fee,
            "total_revenue": total_revenue,
            "payment_method": "Card",
            "payment_failures": payment_failures,
            "discount_applied": "No",
            "price_increase_last_3m": "No",
            "support_tickets": support_tickets,
            "avg_resolution_time": 24.0,
            "complaint_type": None,
            "csat_score": csat_score,
            "escalations": 0,
            "email_open_rate": 0.5,
            "marketing_click_rate": 0.25,
            "nps_score": nps_score,
            "survey_response": "Neutral",
            "referral_count": 0,
        }
        with st.spinner("Interrogation de l'API..."):
            try:
                r_churn = requests.post(f"{API_BASE}/predict/churn",
                                        json=payload, headers=api_headers(token), timeout=10)
                r_rev   = requests.post(f"{API_BASE}/predict/revenue-risk",
                                        json=payload, headers=api_headers(token), timeout=10)
            except requests.exceptions.ConnectionError:
                st.error("API injoignable.")
                st.stop()

        if r_churn.status_code == 200:
            churn_res = r_churn.json()
            rev_res   = r_rev.json() if r_rev.status_code == 200 else {}

            col_r1, col_r2, col_r3 = st.columns(3)
            risk_color = {"Faible": "green", "Moyen": "orange", "Élevé": "red"}
            rl = churn_res["risk_level"]
            col_r1.metric("Risque de churn", f"{churn_res['churn_probability']:.1%}")
            col_r1.markdown(f"Niveau de risque : **:{risk_color[rl]}[{rl}]**")
            col_r2.metric("Prédiction", "Churner" if churn_res["churn_prediction"] == 1 else "Rétentionné")
            if rev_res:
                col_r3.metric("Revenu à risque", f"{rev_res['revenue_at_risk']:,.0f} €")

            # Jauge visuelle
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_res["churn_probability"] * 100,
                title={"text": "Probabilité de churn (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkred"},
                    "steps": [
                        {"range": [0, 30],  "color": "#2ecc71"},
                        {"range": [30, 60], "color": "#f39c12"},
                        {"range": [60, 100],"color": "#e74c3c"},
                    ],
                },
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.caption(f"Modèle utilisé : {churn_res['model_used']}")
        else:
            st.error(f"Erreur API ({r_churn.status_code}) : {r_churn.text}")

# ---------------------------------------------------------------------------
# PAGE 4 — Comparaison Modèles
# ---------------------------------------------------------------------------

elif page == "Comparaison Modèles":
    st.title("Comparaison des Modèles")

    models_dir = Path("/app/models")
    comparison_path = models_dir / "churn_comparison.joblib"

    if not comparison_path.exists():
        st.info("Lancez d'abord `python src/train_churn.py` pour générer la comparaison.")
        st.stop()

    import joblib
    comparison = joblib.load(comparison_path)

    st.subheader("Tâche A — Classification Churn")
    st.dataframe(comparison.style.highlight_max(color="#d4edda", axis=0), use_container_width=True)

    # Radar chart
    metrics_to_plot = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
                       if c in comparison.columns]
    if metrics_to_plot:
        fig_radar = go.Figure()
        for model_name in comparison.index:
            values = comparison.loc[model_name, metrics_to_plot].tolist()
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=metrics_to_plot + [metrics_to_plot[0]],
                fill="toself", name=model_name,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Comparaison radar des métriques (test set)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.subheader("Justification du modèle retenu")
    best = comparison.index[0]
    st.success(
        f"**Modèle final : {best}**  \n"
        f"Sélectionné sur la base du meilleur F1 (rappel des churners prioritaire dans un "
        f"contexte de déséquilibre de classes), de la stabilité en cross-validation, et de "
        f"la facilité de déploiement via API.  \n"
        f"Le MLP a été comparé pour justifier (ou non) l'apport du Deep Learning sur ce "
        f"type de données tabulaires."
    )
