from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional


class CustomerFeatures(BaseModel):
    """Input schema shared by both prediction tasks. Column names match customer_churn_business_dataset.csv."""

    age: int = Field(..., ge=18, le=74, description="Age du client")
    gender: Literal["Male", "Female"] = Field(..., description="Genre")
    country: str = Field(..., description="Pays")
    city: str = Field(..., description="Ville")
    customer_segment: Literal["SME", "Individual", "Enterprise"] = Field(..., description="Segment client")
    tenure_months: int = Field(..., ge=0, description="Ancienneté (mois)")
    signup_channel: Literal["Web", "Mobile", "Referral"] = Field(..., description="Canal d'inscription")
    contract_type: Literal["Monthly", "Quarterly", "Yearly"] = Field(..., description="Type de contrat")
    monthly_logins: int = Field(..., ge=0, description="Connexions par mois")
    weekly_active_days: int = Field(..., ge=0, le=7, description="Jours actifs par semaine")
    avg_session_time: float = Field(..., ge=0, description="Durée moyenne de session (min)")
    features_used: int = Field(..., ge=0, description="Nombre de fonctionnalités utilisées")
    usage_growth_rate: float = Field(..., description="Taux de croissance d'usage")
    last_login_days_ago: int = Field(..., ge=0, description="Jours depuis dernière connexion")
    monthly_fee: float = Field(..., ge=0, description="Abonnement mensuel (€)")
    total_revenue: float = Field(..., ge=0, description="Revenu total historique (€)")
    payment_method: Literal["PayPal", "Card", "Bank Transfer"] = Field(..., description="Moyen de paiement")
    payment_failures: int = Field(..., ge=0, description="Nombre d'échecs de paiement")
    discount_applied: Literal["Yes", "No"] = Field(..., description="Remise appliquée")
    price_increase_last_3m: Literal["Yes", "No"] = Field(..., description="Hausse de prix (3 derniers mois)")
    support_tickets: int = Field(..., ge=0, description="Tickets support ouverts")
    avg_resolution_time: float = Field(..., ge=0, description="Temps moyen de résolution (h)")
    complaint_type: Optional[str] = Field(None, description="Type de réclamation (optionnel)")
    csat_score: float = Field(..., ge=1, le=5, description="Score de satisfaction CSAT")
    escalations: int = Field(..., ge=0, description="Nombre d'escalades")
    email_open_rate: float = Field(..., ge=0, le=1, description="Taux d'ouverture des emails")
    marketing_click_rate: float = Field(..., ge=0, le=1, description="Taux de clic marketing")
    nps_score: int = Field(..., ge=-100, le=100, description="Score NPS")
    survey_response: Literal["Satisfied", "Neutral", "Unsatisfied"] = Field(..., description="Réponse au sondage")
    referral_count: int = Field(..., ge=0, description="Nombre de parrainages")


class ChurnPredictionResponse(BaseModel):
    churn_prediction: int = Field(..., description="0 = Pas de churn, 1 = Churn prédit")
    churn_probability: float = Field(..., description="Probabilité de churn (0-1)")
    risk_level: Literal["Faible", "Moyen", "Élevé"] = Field(..., description="Niveau de risque")
    model_used: str


class ChurnBatchResponse(BaseModel):
    count: int
    predictions: list[ChurnPredictionResponse]


class RevenueRiskResponse(BaseModel):
    revenue_at_risk: float = Field(..., description="Revenu estimé à risque (€)")
    churn_probability: float = Field(..., description="Probabilité de churn sous-jacente (0-1)")
    model_used: str


class RevenueRiskBatchResponse(BaseModel):
    count: int
    churn_model_used: str
    predictions: list[RevenueRiskResponse]


class HealthResponse(BaseModel):
    status: str
    churn_model_loaded: bool
    revenue_model_loaded: bool
    preprocessor_loaded: bool
