import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ✅ Streamlit Config
st.set_page_config(page_title="ARG Mobility Analyzer", layout="wide")

st.title("🧬 Antibiotic Resistance Gene Mobility Analyzer")
st.write("""
Upload CARD metadata + sequences to perform:
- EDA  
- Sequence length analysis  
- PCA visualization  
- KMeans clustering  
- Mobility risk scoring  
""")

# ----------------------------------------------------
# ✅ FILE UPLOAD
# ----------------------------------------------------
uploaded_df = st.file_uploader("Upload Final Merged Dataset (CSV)", type=["csv"])

if uploaded_df is not None:
    df = pd.read_csv(uploaded_df)
    st.success("✅ File Loaded Successfully")

    st.subheader("📌 Preview of Dataset")
    st.dataframe(df.head())

    # ----------------------------------------------------
    # ✅ EDA SECTION
    # ----------------------------------------------------
    st.header("📊 Exploratory Data Analysis")

    if "seq_len" not in df.columns:
        df["seq_len"] = df["sequence"].str.len()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Drug Class Distribution")
        fig = px.bar(df["Drug Class"].value_counts().head(15))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Resistance Mechanisms")
        fig = px.bar(df["Resistance Mechanism"].value_counts().head(15))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sequence Length Histogram")
    fig = px.histogram(df, x="seq_len", nbins=50)
    st.plotly_chart(fig)

    # ----------------------------------------------------
    # ✅ PCA + CLUSTERING
    # ----------------------------------------------------
    st.header("🧩 PCA + Clustering")

    # Label encode categorical features
    enc = LabelEncoder()
    df["Family_enc"] = enc.fit_transform(df["AMR Gene Family"].astype(str))
    df["Drug_enc"] = enc.fit_transform(df["Drug Class"].astype(str))
    df["Mech_enc"] = enc.fit_transform(df["Resistance Mechanism"].astype(str))

    X = df[["seq_len", "Family_enc", "Drug_enc", "Mech_enc"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    st.subheader("PCA 2D Visualization")
    fig = px.scatter(
        df, x="PC1", y="PC2",
        color="cluster",
        hover_data=["ARO Name", "Drug Class", "Resistance Mechanism"],
        title="PCA Clustering of ARGs"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------
    # ✅ MOBILITY SCORE
    # ----------------------------------------------------
    st.header("🚨 Mobility Risk Scoring")

    def mobility_score(row):
        score = 0

        # Short sequences → more mobile
        if row["seq_len"] < df["seq_len"].median():
            score += 1

        # Efflux / inactivation mechanisms
        mech = row["Resistance Mechanism"].lower()
        if "efflux" in mech:
            score += 2
        if "inactivation" in mech:
            score += 1

        # Known mobile families
        mobile_families = ["beta-lactamase", "aminoglycoside", "multidrug"]
        fam = row["AMR Gene Family"].lower()
        for m in mobile_families:
            if m in fam:
                score += 2

        return score

    df["mobility_score"] = df.apply(mobility_score, axis=1)

    st.subheader("Mobility Score Distribution")
    fig = px.histogram(df, x="mobility_score")
    st.plotly_chart(fig)

    # ----------------------------------------------------
    # ✅ TOP HIGH RISK ARGs
    # ----------------------------------------------------
    st.header("🔥 Top 20 High-Mobility ARGs")

    top_20 = df.sort_values("mobility_score", ascending=False).head(20)

    st.dataframe(top_20[[
        "ARO Accession",
        "ARO Name",
        "Drug Class",
        "AMR Gene Family",
        "Resistance Mechanism",
        "mobility_score"
    ]])

    # Download button
    csv = top_20.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download High-Risk ARGs (CSV)",
        data=csv,
        file_name="high_risk_ARGs.csv",
        mime="text/csv"
    )

else:
    st.info("👉 Please upload your merged CARD dataset (with metadata + sequences).")
