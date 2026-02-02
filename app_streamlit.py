import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard Financier - SVM",
    layout="wide",
    page_icon="📊"
)

sns.set_style("whitegrid")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_financier.csv")

df = load_data()

# =========================
# OPTIMISATION DU MODELE (une seule fois)
# =========================
@st.cache_resource
def train_best_model(df):
    features = ["actifs", "revenu", "taux_interet", "flux_tresorerie", "capital"]
    X = df[features]
    y = df["depenses"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    params = {
        "C": [1, 10, 100],
        "gamma": ["scale", "auto"],
        "epsilon": [0.1, 0.2],
        "kernel": ["rbf", "linear"]
    }

    grid = GridSearchCV(SVR(), params, cv=3, scoring="r2")
    grid.fit(X_scaled, y)

    best_model = grid.best_estimator_
    best_r2 = grid.best_score_

    return scaler, best_model, best_r2

scaler_global, best_model_global, best_r2_global = train_best_model(df)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Aller vers :",
    ["🏠 Accueil", "📊 Dashboard Financier", "🤖 Prédiction SVM"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Filtres")

agence_sel = st.sidebar.multiselect(
    "Agence",
    df["agence"].unique(),
    default=df["agence"].unique()
)

banque_sel = st.sidebar.multiselect(
    "Banque",
    df["banque"].unique(),
    default=df["banque"].unique()
)

lieu_sel = st.sidebar.multiselect(
    "Lieu",
    df["lieu"].unique(),
    default=df["lieu"].unique()
)

df_f = df[
    (df["agence"].isin(agence_sel)) &
    (df["banque"].isin(banque_sel)) &
    (df["lieu"].isin(lieu_sel))
]

# =========================
# ACCUEIL
# =========================
if page == "🏠 Accueil":
    st.title("📊 Application d’Analyse Financière & Prédiction")

    st.markdown("""
    ## 🎯 Objectif
    Cette application aide à **analyser la performance financière**
    des agences et banques, et à **prédire les dépenses**
    grâce au Machine Learning (SVR optimisé).

    ## ⚙️ Fonctionnalités principales
    - Analyse financière multi-agences
    - Visualisation des revenus & dépenses
    - Analyse de la part de marché
    - Prédiction des dépenses via **SVR optimisé**

    ## 🧭 Comment utiliser
    1. Sélectionnez les filtres dans la sidebar
    2. Consultez le **Dashboard**
    3. Testez une **prédiction personnalisée**
    """)

# =========================
# DASHBOARD
# =========================
elif page == "📊 Dashboard Financier":
    st.title("📊 Dashboard Financier")

    if df_f.empty:
        st.error("⚠️ Aucun enregistrement disponible avec les filtres sélectionnés.")
    else:
        # KPIs
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("💰 Revenu total", f"{df_f['revenu'].sum():,.0f}")
        col2.metric("💸 Dépenses totales", f"{df_f['depenses'].sum():,.0f}")
        col3.metric("🏢 Agences", df_f["agence"].nunique())
        col4.metric("🏦 Banques", df_f["banque"].nunique())

        st.markdown("---")

        # Graphiques
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            sns.barplot(data=df_f, x="agence", y="revenu", ax=ax)
            ax.set_title("Revenus par agence")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            best_agence = df_f.groupby("agence")["revenu"].sum().idxmax()
            st.success(f"💡 L'agence **{best_agence}** génère le plus de revenus.")

        with col2:
            fig, ax = plt.subplots()
            sns.barplot(data=df_f, x="agence", y="depenses", ax=ax, color="orange")
            ax.set_title("Dépenses par agence")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            worst_agence = df_f.groupby("agence")["depenses"].sum().idxmax()
            st.warning(f"⚠️ L'agence **{worst_agence}** a les dépenses les plus élevées.")

        st.markdown("---")

        # Analyse marché
        st.subheader("🏦 Analyse de la Part de Marché")

        market_share = df_f.groupby("banque")["revenu"].sum()

        fig, ax = plt.subplots()
        ax.pie(
            market_share,
            labels=market_share.index,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.set_title("Part de marché par banque")
        st.pyplot(fig)

        leader = market_share.idxmax()
        percent = market_share.max() / market_share.sum() * 100
        st.success(f"🏆 **{leader}** détient **{percent:.1f}%** du marché.")

        st.markdown("---")

        # Bilan financier
        st.subheader("📑 Bilan Financier")
        bilan = df_f.groupby("agence")[["revenu", "depenses"]].sum()
        bilan["bilan"] = bilan["revenu"] - bilan["depenses"]
        st.dataframe(bilan)

# =========================
# PREDICTION SVR OPTIMISÉ
# =========================
elif page == "🤖 Prédiction SVM":
    st.title("🤖 Prédiction des Dépenses (SVR optimisé)")

    if df_f.empty:
        st.error("⚠️ Veuillez sélectionner au moins une agence, une banque et un lieu pour lancer la prédiction.")
    else:
        st.info("ℹ️ Sélection unique = prédiction ciblée, sélection multiple = prédiction globale.")

        features = ["actifs", "revenu", "taux_interet", "flux_tresorerie", "capital"]
        
        st.subheader("📥 Entrer les données")

        user_inputs = []
        for col in features:
            value = st.number_input(
                col,
                min_value=float(df_f[col].min()),
                max_value=float(df_f[col].max()),
                value=float(df_f[col].mean())
            )
            user_inputs.append(value)

        if st.button("🔮 Prédire les dépenses"):
            scaled_input = scaler_global.transform([user_inputs])
            prediction = best_model_global.predict(scaled_input)[0]

            st.success(f"💰 Dépenses prédites : **{prediction:,.2f}**")

            if prediction > df_f["depenses"].mean():
                st.warning("⚠️ Dépenses supérieures à la moyenne.")
            else:
                st.info("✅ Dépenses maîtrisées.")
