import os
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
import joblib


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_FILENAME = "customer_churn_business_dataset.csv"

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_USER = os.getenv("DEMO_USERNAME", "admin")
API_PASS = os.getenv("DEMO_PASSWORD", "churn2025!")

COLORS = {
    "churn":    "#e74c3c",
    "retain":   "#2ecc71",
    "neutral":  "#3498db",
    "warning":  "#f39c12",
    "dark":     "#2c3e50",
}


def _resolve_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def data_file_path() -> Path:
    return _resolve_path([Path("/app/data") / DATA_FILENAME, PROJECT_ROOT / "data" / DATA_FILENAME])


def models_dir() -> Path:
    return _resolve_path([Path("/app/models"), PROJECT_ROOT / "models"])


st.set_page_config(
    page_title="Rétention Client — Dashboard IA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Authentification
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
# Chargement des données
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Chargement des données...")
def load_data() -> pd.DataFrame:
    data_path = data_file_path()
    if not data_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    # Features engineered pour les analyses
    df["tickets_per_tenure"] = df["support_tickets"] / (df["tenure_months"] + 1)
    df["segment_tenure"] = pd.cut(
        df["tenure_months"], bins=[0, 12, 24, 60, 999],
        labels=["< 1 an", "1–2 ans", "2–5 ans", "> 5 ans"]
    )
    return df


# ---------------------------------------------------------------------------
# Sidebar — navigation
# ---------------------------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Section",
    ["Vue Globale (KPIs)", "Analyse du Risque", "Simulateur Client", "Prédiction Batch", "Comparaison Modèles"],
)

try:
    health = requests.get(f"{API_BASE}/health", timeout=3).json()
    api_ok = health.get("status") == "ok"
except Exception:
    api_ok = False
    health = {}

status_color = "green" if api_ok else "red"
st.sidebar.markdown(f"**Statut API :** :{status_color}[{'En ligne' if api_ok else 'Hors ligne'}]")
if api_ok:
    st.sidebar.caption(
        f"Churn model: {'OK' if health.get('churn_model_loaded') else 'Non chargé'}  |  "
        f"Preprocessor: {'OK' if health.get('preprocessor_loaded') else 'Erreur'}"
    )
st.sidebar.markdown("---")
st.sidebar.caption("EFREI M1 Data Engineering — Projet Rétention Client")

df    = load_data()
token = get_jwt_token()

# ---------------------------------------------------------------------------
# PAGE 1 — Vue Globale (KPIs)
# ---------------------------------------------------------------------------

if page == "Vue Globale (KPIs)":
    st.title("Vue Globale — Indicateurs Clés de Performance")

    if df.empty:
        st.warning(f"Données introuvables. Placez `{DATA_FILENAME}` dans `data/`.")
        st.stop()

    churners   = df[df["churn"] == 1]
    retained   = df[df["churn"] == 0]
    n_total    = len(df)
    n_churn    = len(churners)
    churn_rate = n_churn / n_total
    rev_at_risk = churners["total_revenue"].sum()
    rev_total   = df["total_revenue"].sum()
    mrr_risque  = churners["monthly_fee"].sum() if "monthly_fee" in df.columns else 0
    nps_moyen   = df["nps_score"].mean() if "nps_score" in df.columns else 0

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Clients totaux",     f"{n_total:,}")
    c2.metric("Churners",           f"{n_churn:,}", delta=f"{churn_rate:.1%}", delta_color="inverse")
    c3.metric("Revenu à risque",    f"{rev_at_risk:,.0f} €", delta=f"{rev_at_risk/rev_total:.1%} du CA", delta_color="inverse")
    c4.metric("MRR à risque (€/mois)", f"{mrr_risque:,.0f} €")
    if "nps_score" in df.columns:
        nps_churn   = churners["nps_score"].mean()
        nps_retain  = retained["nps_score"].mean()
        c5.metric("NPS moyen", f"{nps_moyen:.0f}",
                  delta=f"Churners {nps_churn:.0f} vs Fidèles {nps_retain:.0f}",
                  delta_color="off")
    else:
        c5.metric("NPS moyen", f"{nps_moyen:.0f}")

    st.markdown("---")

    # Ligne 1 : Taux de churn par segment + CSAT distribution
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Taux de churn par segment")
        seg_col = st.selectbox("Segmenter par :", ["contract_type", "customer_segment", "signup_channel", "payment_method"], key="seg1")
        if seg_col in df.columns:
            seg_df = (
                df.groupby(seg_col)["churn"]
                .agg(taux="mean", effectif="count")
                .assign(taux_pct=lambda x: (x["taux"] * 100).round(1))
                .sort_values("taux_pct", ascending=True)
                .reset_index()
            )
            fig = px.bar(
                seg_df, x="taux_pct", y=seg_col, orientation="h",
                text="taux_pct",
                color="taux_pct", color_continuous_scale=[[0, "#2ecc71"], [0.5, "#f39c12"], [1, "#e74c3c"]],
                labels={seg_col: "", "taux_pct": "Taux de churn (%)"},
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=30, t=20, b=0), height=280)
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Score CSAT — Churners vs Rétentionnés")
        if "csat_score" in df.columns:
            csat_df = pd.DataFrame({
                "CSAT": pd.concat([churners["csat_score"], retained["csat_score"]]),
                "Statut": (["Churner"] * len(churners)) + (["Rétentionné"] * len(retained)),
            })
            fig_csat = px.violin(
                csat_df, x="Statut", y="CSAT", color="Statut",
                color_discrete_map={"Churner": COLORS["churn"], "Rétentionné": COLORS["retain"]},
                box=True, points=False,
            )
            fig_csat.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=0), height=280)
            st.plotly_chart(fig_csat, use_container_width=True)

    # Ligne 2 : NPS distribution + Ancienneté
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribution NPS")
        if "nps_score" in df.columns:
            fig_nps = px.histogram(
                df, x="nps_score",
                color=df["churn"].map({0: "Rétentionné", 1: "Churner"}),
                barmode="overlay", opacity=0.7, nbins=30,
                color_discrete_map={"Churner": COLORS["churn"], "Rétentionné": COLORS["retain"]},
                labels={"nps_score": "Score NPS", "color": "Statut"},
            )
            fig_nps.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Neutre")
            fig_nps.update_layout(margin=dict(t=20, b=0), height=280, legend_title="")
            st.plotly_chart(fig_nps, use_container_width=True)

    with col_b:
        st.subheader("Ancienneté vs Churn")
        if "tenure_months" in df.columns:
            fig_tenure = px.box(
                df, x=df["churn"].map({0: "Rétentionné", 1: "Churner"}),
                y="tenure_months", color=df["churn"].map({0: "Rétentionné", 1: "Churner"}),
                color_discrete_map={"Churner": COLORS["churn"], "Rétentionné": COLORS["retain"]},
                labels={"x": "", "tenure_months": "Ancienneté (mois)"},
                points="outliers",
            )
            fig_tenure.update_layout(showlegend=False, margin=dict(t=20, b=0), height=280)
            st.plotly_chart(fig_tenure, use_container_width=True)

    # Ligne 3 : Top corrélations avec churn
    st.subheader("Top corrélations numériques avec le churn")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excl = ["churn", "customer_id"]
    num_cols = [c for c in num_cols if c not in excl]
    corr = df[num_cols + ["churn"]].corr()["churn"].drop("churn").sort_values(key=abs, ascending=False).head(10)
    fig_corr = px.bar(
        x=corr.values, y=corr.index,
        orientation="h",
        color=corr.values,
        color_continuous_scale=[[0, COLORS["retain"]], [0.5, "lightgray"], [1, COLORS["churn"]]],
        color_continuous_midpoint=0,
        labels={"x": "Corrélation de Pearson avec churn", "y": ""},
    )
    fig_corr.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0), height=320)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Corrélation linéaire — utile pour orienter les features, mais les modèles ML capturent les non-linéarités.")

# ---------------------------------------------------------------------------
# PAGE 2 — Analyse du Risque
# ---------------------------------------------------------------------------

elif page == "Analyse du Risque":
    st.title("Analyse du Risque — Segments Clients Prioritaires")

    if df.empty:
        st.warning("Données non disponibles.")
        st.stop()

    churners = df[df["churn"] == 1].copy()

    # NPS × CSAT quadrant
    st.subheader("Quadrant NPS × CSAT — Identification des zones à risque")
    if "nps_score" in df.columns and "csat_score" in df.columns:
        sample = df.sample(min(2000, len(df)), random_state=42)
        fig_quad = px.scatter(
            sample, x="nps_score", y="csat_score",
            color=sample["churn"].map({0: "Rétentionné", 1: "Churner"}),
            color_discrete_map={"Churner": COLORS["churn"], "Rétentionné": COLORS["retain"]},
            opacity=0.5, size_max=6,
            labels={"nps_score": "Score NPS", "csat_score": "Score CSAT"},
        )
        nps_med  = df["nps_score"].median()
        csat_med = df["csat_score"].median()
        fig_quad.add_vline(x=nps_med,  line_dash="dash", line_color="gray", annotation_text=f"Médiane NPS={nps_med:.0f}", annotation_position="top right")
        fig_quad.add_hline(y=csat_med, line_dash="dash", line_color="gray", annotation_text=f"Médiane CSAT={csat_med:.1f}", annotation_position="bottom right")
        fig_quad.add_annotation(x=-80, y=1.2, text="Zone rouge\n(NPS bas + CSAT bas)", showarrow=False,
                                bgcolor="#e74c3c", font=dict(color="white"), opacity=0.8)
        fig_quad.update_layout(height=420, legend_title="")
        st.plotly_chart(fig_quad, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Échecs de paiement — distribution")
        if "payment_failures" in df.columns:
            pay_df = pd.DataFrame({
                "Échecs": pd.concat([churners["payment_failures"], df[df["churn"]==0]["payment_failures"]]),
                "Statut": (["Churner"] * len(churners)) + (["Rétentionné"] * (len(df) - len(churners))),
            })
            fig_pay = px.histogram(
                pay_df, x="Échecs", color="Statut", barmode="overlay",
                opacity=0.75, nbins=6,
                color_discrete_map={"Churner": COLORS["churn"], "Rétentionné": COLORS["retain"]},
            )
            fig_pay.update_layout(legend_title="", margin=dict(t=20, b=0), height=280)
            st.plotly_chart(fig_pay, use_container_width=True)

    with col_r:
        st.subheader("Revenu à risque par ancienneté")
        if "segment_tenure" in df.columns:
            risk_seg = (
                churners.groupby("segment_tenure", observed=True)["total_revenue"]
                .sum().reset_index()
                .rename(columns={"total_revenue": "Revenu à risque (€)"})
            )
            fig_seg = px.bar(
                risk_seg, x="segment_tenure", y="Revenu à risque (€)",
                color="Revenu à risque (€)", color_continuous_scale="Oranges",
                labels={"segment_tenure": "Ancienneté"},
            )
            fig_seg.update_layout(coloraxis_showscale=False, margin=dict(t=20, b=0), height=280)
            st.plotly_chart(fig_seg, use_container_width=True)

    # Top 10 à risque
    st.subheader("Top 10 churners par revenu mensuel à risque")
    if "monthly_fee" in df.columns:
        cols_show = [c for c in ["tenure_months", "contract_type", "customer_segment",
                                  "monthly_fee", "payment_failures", "nps_score", "csat_score"] if c in churners.columns]
        top10 = churners.sort_values("monthly_fee", ascending=False).head(10)[cols_show].reset_index(drop=True)
        st.dataframe(
            top10.style
            .background_gradient(subset=["monthly_fee"] if "monthly_fee" in cols_show else [], cmap="Reds")
            .background_gradient(subset=["payment_failures"] if "payment_failures" in cols_show else [], cmap="OrRd"),
            use_container_width=True,
        )
        mrr = churners["monthly_fee"].sum()
        st.info(
            f"**{len(churners)} churners** représentent **{mrr:,.0f} €/mois** de MRR à risque. "
            f"Une campagne de rétention à 25% de succès permettrait de sauver "
            f"**{mrr * 0.25:,.0f} €/mois**."
        )

# ---------------------------------------------------------------------------
# PAGE 3 — Simulateur Client
# ---------------------------------------------------------------------------

elif page == "Simulateur Client":
    st.title("Simulateur — Prédiction en Temps Réel")
    st.caption("Prédictions via l'API FastAPI (architecture Front / API / Modèle ML).")

    if not api_ok:
        st.error("L'API est hors ligne. Lancez : `uvicorn api.main:app --reload` depuis la racine du projet.")
        st.stop()
    if token is None:
        st.error("Impossible d'obtenir un token JWT. Vérifiez les variables d'environnement.")
        st.stop()
    if not health.get("churn_model_loaded"):
        st.warning("Modèle non chargé. Exécutez d'abord les notebooks 02 et 03.")

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
        avg_session_time = col6.number_input("Durée session moy. (min)", 0.0, 300.0, 15.0)
        monthly_logins   = col4.number_input("Connexions/mois", 0, 200, 15)
        nps_score        = col5.slider("Score NPS", -100, 100, 20)
        csat_score       = col6.slider("CSAT (1–5)", 1.0, 5.0, 3.5, step=0.5)

        submitted = st.form_submit_button("Prédire", type="primary")

    if submitted:
        payload = {
            "age": age, "gender": gender, "country": "France", "city": "Paris",
            "customer_segment": "Individual", "tenure_months": tenure_months,
            "signup_channel": "Web", "contract_type": contract_type,
            "monthly_logins": monthly_logins, "weekly_active_days": 3,
            "avg_session_time": avg_session_time, "features_used": 5,
            "usage_growth_rate": 0.0, "last_login_days_ago": 7,
            "monthly_fee": monthly_fee, "total_revenue": total_revenue,
            "payment_method": "Card", "payment_failures": payment_failures,
            "discount_applied": "No", "price_increase_last_3m": "No",
            "support_tickets": support_tickets, "avg_resolution_time": 24.0,
            "complaint_type": None, "csat_score": csat_score, "escalations": 0,
            "email_open_rate": 0.5, "marketing_click_rate": 0.25,
            "nps_score": nps_score, "survey_response": "Neutral", "referral_count": 0,
        }
        with st.spinner("Interrogation de l'API..."):
            try:
                r = requests.post(f"{API_BASE}/predict/churn", json=payload, headers=api_headers(token), timeout=10)
            except requests.exceptions.ConnectionError:
                st.error("API injoignable.")
                st.stop()

        if r.status_code == 200:
            res = r.json()
            proba = res["churn_probability"]
            rl    = res["risk_level"]
            risk_color = {"Faible": "green", "Moyen": "orange", "Élevé": "red"}

            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Probabilité de churn", f"{proba:.1%}")
            col_r1.markdown(f"Niveau de risque : **:{risk_color[rl]}[{rl}]**")
            col_r2.metric("Prédiction", "Churner" if res["churn_prediction"] == 1 else "Rétentionné")
            col_r3.metric("Revenu à risque estimé", f"{total_revenue * proba:,.0f} €")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=proba * 100,
                delta={"reference": 10.2, "valueformat": ".1f", "suffix": "% vs base 10.2%"},
                title={"text": "Probabilité de churn (%)"},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar": {"color": COLORS["churn"] if proba > 0.5 else COLORS["warning"]},
                    "steps": [
                        {"range": [0, 30],   "color": "#d5f5e3"},
                        {"range": [30, 60],  "color": "#fdebd0"},
                        {"range": [60, 100], "color": "#fadbd8"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.75, "value": 50},
                },
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=50, b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.caption(f"Modèle utilisé : **{res['model_used']}**")

            if res["churn_prediction"] == 1:
                st.warning(
                    "**Action recommandée :** Ce client présente un risque élevé. "
                    "Planifier un contact proactif (offre de fidélité, appel conseiller) "
                    f"pour tenter de le retenir (valeur estimée : {total_revenue * proba:,.0f} €)."
                )
        else:
            st.error(f"Erreur API ({r.status_code}) : {r.text}")

# ---------------------------------------------------------------------------
# PAGE 4 — Prédiction Batch
# ---------------------------------------------------------------------------

elif page == "Prédiction Batch":
    st.title("Prédiction Batch — Scoring CSV")
    st.caption("Uploadez un CSV pour scorer l'ensemble de vos clients via l'API.")

    if not api_ok:
        st.error("L'API est hors ligne. Lancez : `uvicorn api.main:app --reload`")
        st.stop()
    if token is None:
        st.error("Impossible d'obtenir un token JWT.")
        st.stop()

    uploaded = st.file_uploader("Fichier CSV", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        batch_df.columns = batch_df.columns.str.strip().str.lower().str.replace(" ", "_")
        st.write(f"**{len(batch_df)}** lignes chargées.")
        st.dataframe(batch_df.head(5), use_container_width=True)

        if st.button("Lancer le scoring", type="primary"):
            drop_cols  = [c for c in ("customer_id", "churn") if c in batch_df.columns]
            records    = batch_df.drop(columns=drop_cols, errors="ignore").to_dict(orient="records")

            with st.spinner(f"Scoring de {len(records)} clients..."):
                try:
                    r = requests.post(f"{API_BASE}/predict/churn/batch", json=records,
                                      headers=api_headers(token), timeout=120)
                except requests.exceptions.ConnectionError:
                    st.error("API injoignable.")
                    st.stop()

            if r.status_code != 200:
                st.error(f"Erreur ({r.status_code}) : {r.text}")
                st.stop()

            preds   = r.json()["predictions"]
            results = batch_df.copy()
            results["churn_probability"] = [p["churn_probability"] for p in preds]
            results["churn_prediction"]  = [p["churn_prediction"]  for p in preds]
            results["risk_level"]        = [p["risk_level"]        for p in preds]

            st.success(f"Scoring terminé — {len(results)} clients traités.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Clients à risque élevé",  (results["risk_level"] == "Élevé").sum())
            col2.metric("Clients à risque moyen",   (results["risk_level"] == "Moyen").sum())
            col3.metric("Probabilité moyenne",       f"{results['churn_probability'].mean():.1%}")

            fig_dist = px.histogram(
                results, x="churn_probability", nbins=30,
                color="risk_level",
                color_discrete_map={"Élevé": COLORS["churn"], "Moyen": COLORS["warning"], "Faible": COLORS["retain"]},
                labels={"churn_probability": "P(churn)", "risk_level": "Risque"},
                title="Distribution des probabilités de churn",
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            st.dataframe(results.sort_values("churn_probability", ascending=False), use_container_width=True)
            st.download_button(
                "Télécharger les résultats (CSV)",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="churn_scoring_results.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# PAGE 5 — Comparaison Modèles
# ---------------------------------------------------------------------------

elif page == "Comparaison Modèles":
    st.title("Comparaison des Modèles — Classification Churn")

    comp_path = models_dir() / "churn_comparison.joblib"
    if not comp_path.exists():
        st.info(f"Exécutez d'abord le notebook `03_modeling.ipynb` pour générer la comparaison.")
        st.stop()

    comp = joblib.load(comp_path)

    # Tableau avec highlight
    st.subheader("Métriques sur le val set (seuils optimisés)")
    display_cols = [c for c in ["f1", "recall", "precision", "roc_auc", "pr_auc"] if c in comp.columns]
    st.dataframe(
        comp[display_cols].style
        .highlight_max(color="#d5f5e3", subset=display_cols)
        .highlight_min(color="#fadbd8", subset=display_cols)
        .format("{:.4f}"),
        use_container_width=True,
    )

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Radar chart — comparaison globale")
        metrics_radar = [c for c in ["f1", "recall", "precision", "roc_auc", "pr_auc"] if c in comp.columns]
        palette = [COLORS["neutral"], COLORS["retain"], COLORS["churn"], COLORS["warning"]]
        fig_radar = go.Figure()
        for i, (model_name, row) in enumerate(comp.iterrows()):
            vals = [row[m] for m in metrics_radar]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=metrics_radar + [metrics_radar[0]],
                fill="toself", name=model_name,
                line_color=palette[i % len(palette)],
                opacity=0.8,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9))),
            legend=dict(font=dict(size=10)),
            margin=dict(t=30, b=30),
            height=380,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_r:
        st.subheader("F1-score — objectif 0.75")
        if "f1" in comp.columns:
            f1_df = comp["f1"].sort_values().reset_index()
            f1_df.columns = ["Modèle", "F1"]
            colors_bar = [COLORS["retain"] if f >= 0.75 else COLORS["churn"] if f < 0.5 else COLORS["warning"]
                          for f in f1_df["F1"]]
            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Bar(
                x=f1_df["F1"], y=f1_df["Modèle"], orientation="h",
                marker_color=colors_bar, text=f1_df["F1"].round(3),
                textposition="outside",
            ))
            fig_f1.add_vline(x=0.75, line_dash="dash", line_color="gray",
                              annotation_text="Objectif 0.75", annotation_position="top right")
            fig_f1.update_layout(
                xaxis=dict(range=[0, 1.05]),
                margin=dict(l=0, r=60, t=30, b=0),
                height=380,
            )
            st.plotly_chart(fig_f1, use_container_width=True)

    # Seuils optimaux
    if "seuil_optimal" in comp.columns:
        st.subheader("Seuils de décision optimisés (courbe PR)")
        seuil_df = comp["seuil_optimal"].reset_index()
        seuil_df.columns = ["Modèle", "Seuil optimal"]
        st.dataframe(seuil_df.style.format({"Seuil optimal": "{:.3f}"}), use_container_width=True)
        st.caption(
            "Le seuil 0.5 n'est jamais optimal sur données déséquilibrées (10% de churners). "
            "Le seuil optimal est calculé en maximisant le F1-score sur la courbe Precision-Recall du val set."
        )

    # Modèle retenu
    best = comp.index[0]
    best_f1 = comp.loc[best, "f1"] if "f1" in comp.columns else "N/A"
    st.success(
        f"**Modèle final retenu : {best}**  \n"
        f"F1-score test : **{best_f1:.4f}**  \n"
        f"Sélectionné sur le meilleur F1 (classe minoritaire) + stabilité cross-validation. "
        f"Le MLP Keras est inclus pour justifier (ou non) l'apport du Deep Learning "
        f"sur données tabulaires."
    )
