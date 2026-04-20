# ==========================================================
# FINAL CORRECTED V2
# REAL ESTATE MARKET INTELLIGENCE DASHBOARD
# Run:
# streamlit run final_real_estate_market_intelligence_v2.py
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Market Intelligence",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Real Estate Buyer Segmentation Dashboard")
st.markdown("Machine Learning Based Buyer Segmentation + Investment Profiling")

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data():
   clients = pd.read_csv("clients.csv")
   properties = pd.read_csv("properties.csv")
   return clients, properties

clients, properties = load_data()

# ----------------------------------------------------------
# CLEAN DATA
# ----------------------------------------------------------
clients.drop_duplicates(inplace=True)
properties.drop_duplicates(inplace=True)

clients.fillna("Unknown", inplace=True)
properties.fillna("Unknown", inplace=True)

# ----------------------------------------------------------
# DATE HANDLING
# ----------------------------------------------------------
clients["date_of_birth"] = pd.to_datetime(
    clients["date_of_birth"],
    errors="coerce",
    dayfirst=False
)

properties["transaction_date"] = pd.to_datetime(
    properties["transaction_date"],
    errors="coerce",
    dayfirst=True
)

# Create Age
current_year = 2026
clients["Age"] = current_year - clients["date_of_birth"].dt.year

# ----------------------------------------------------------
# CLEAN SALE PRICE
# ----------------------------------------------------------
properties["sale_price"] = (
    properties["sale_price"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

properties["sale_price"] = pd.to_numeric(
    properties["sale_price"],
    errors="coerce"
)

# Numeric Conversion
properties["tower_number"] = pd.to_numeric(
    properties["tower_number"],
    errors="coerce"
)

properties["floor_area_sqft"] = pd.to_numeric(
    properties["floor_area_sqft"],
    errors="coerce"
)

# ----------------------------------------------------------
# MERGE DATA
# ----------------------------------------------------------
df = pd.merge(
    clients,
    properties,
    left_on="client_id",
    right_on="client_ref",
    how="inner"
)

# ----------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------
st.sidebar.header("🔍 Filters")

country = st.sidebar.multiselect(
    "Country",
    sorted(df["country"].dropna().unique()),
    default=sorted(df["country"].dropna().unique())
)

region = st.sidebar.multiselect(
    "Region",
    sorted(df["region"].dropna().unique()),
    default=sorted(df["region"].dropna().unique())
)

purpose = st.sidebar.multiselect(
    "Purpose",
    sorted(df["acquisition_purpose"].dropna().unique()),
    default=sorted(df["acquisition_purpose"].dropna().unique())
)

unit = st.sidebar.multiselect(
    "Unit Category",
    sorted(df["unit_category"].dropna().unique()),
    default=sorted(df["unit_category"].dropna().unique())
)

df = df[
    (df["country"].isin(country)) &
    (df["region"].isin(region)) &
    (df["acquisition_purpose"].isin(purpose)) &
    (df["unit_category"].isin(unit))
]

# ----------------------------------------------------------
# FEATURE SELECTION
# ----------------------------------------------------------
model_df = df[[
    "client_type",
    "gender",
    "country",
    "region",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel",
    "satisfaction_score",
    "Age",
    "tower_number",
    "floor_area_sqft",
    "sale_price"
]].copy()

# ----------------------------------------------------------
# FORCE NUMERIC COLUMNS
# ----------------------------------------------------------
numeric_cols_force = [
    "satisfaction_score",
    "Age",
    "tower_number",
    "floor_area_sqft",
    "sale_price"
]

for col in numeric_cols_force:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

# ----------------------------------------------------------
# LABEL ENCODE TEXT COLUMNS
# ----------------------------------------------------------
text_cols = model_df.select_dtypes(include="object").columns

for col in text_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))

# ----------------------------------------------------------
# HANDLE MISSING VALUES SAFELY
# ----------------------------------------------------------
numeric_cols = model_df.select_dtypes(include=np.number).columns

model_df[numeric_cols] = model_df[numeric_cols].fillna(
    model_df[numeric_cols].mean()
)

# ----------------------------------------------------------
# SCALE DATA
# ----------------------------------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(model_df)

# ----------------------------------------------------------
# CLUSTER SETTINGS
# ----------------------------------------------------------
st.sidebar.subheader("⚙ ML Settings")

cluster_num = st.sidebar.slider(
    "Number of Segments",
    2,
    6,
    4
)

# ----------------------------------------------------------
# ELBOW METHOD
# ----------------------------------------------------------
inertia = []

for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_data)
    inertia.append(km.inertia_)

# ----------------------------------------------------------
# KMEANS
# ----------------------------------------------------------
kmeans = KMeans(
    n_clusters=cluster_num,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(scaled_data)

# ----------------------------------------------------------
# HIERARCHICAL
# ----------------------------------------------------------
agg = AgglomerativeClustering(n_clusters=cluster_num)
df["Hierarchical_Cluster"] = agg.fit_predict(scaled_data)

# ----------------------------------------------------------
# SCORE
# ----------------------------------------------------------
score = silhouette_score(scaled_data, df["Cluster"])

# ----------------------------------------------------------
# SEGMENT NAMES
# ----------------------------------------------------------
segment_names = {
    0: "Global Investors",
    1: "First-Time Buyers",
    2: "Corporate Buyers",
    3: "Luxury Investors",
    4: "Strategic Buyers",
    5: "Premium Buyers"
}

df["Segment_Name"] = df["Cluster"].map(segment_names)

# ----------------------------------------------------------
# KPI CARDS
# ----------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Buyers", len(df))
c2.metric("Revenue", f"${df['sale_price'].sum():,.0f}")
c3.metric("Avg Price", f"${df['sale_price'].mean():,.0f}")
c4.metric("Silhouette", round(score, 3))

st.markdown("---")

# ----------------------------------------------------------
# ELBOW CHART
# ----------------------------------------------------------
st.subheader("📈 Elbow Method")

fig1, ax1 = plt.subplots(figsize=(8,4))
ax1.plot(range(2,7), inertia, marker="o")
ax1.set_xlabel("Clusters")
ax1.set_ylabel("Inertia")
st.pyplot(fig1)

# ----------------------------------------------------------
# SEGMENT DISTRIBUTION
# ----------------------------------------------------------
st.subheader("📊 Segment Distribution")

fig2, ax2 = plt.subplots(figsize=(10,4))
sns.countplot(x="Segment_Name", data=df, ax=ax2)
plt.xticks(rotation=30)
st.pyplot(fig2)

# ----------------------------------------------------------
# REVENUE BY SEGMENT
# ----------------------------------------------------------
st.subheader("💰 Revenue by Segment")

fig3, ax3 = plt.subplots(figsize=(10,4))
sns.barplot(x="Segment_Name", y="sale_price", data=df, ax=ax3)
plt.xticks(rotation=30)
st.pyplot(fig3)

# ----------------------------------------------------------
# AGE BY SEGMENT
# ----------------------------------------------------------
st.subheader("👤 Average Age by Segment")

fig4, ax4 = plt.subplots(figsize=(10,4))
sns.barplot(x="Segment_Name", y="Age", data=df, ax=ax4)
plt.xticks(rotation=30)
st.pyplot(fig4)

# ----------------------------------------------------------
# LOAN TABLE
# ----------------------------------------------------------
st.subheader("🏦 Loan Dependency")

loan_table = pd.crosstab(df["Segment_Name"], df["loan_applied"])
st.dataframe(loan_table)

# ----------------------------------------------------------
# REGION REVENUE
# ----------------------------------------------------------
st.subheader("🌍 Region Revenue")

region_rev = df.groupby("region")["sale_price"].sum().sort_values(ascending=False)
st.bar_chart(region_rev)

# ----------------------------------------------------------
# UNIT CATEGORY
# ----------------------------------------------------------
st.subheader("🏢 Unit Preference")

unit_pref = df["unit_category"].value_counts()
st.bar_chart(unit_pref)

# ----------------------------------------------------------
# SEGMENT SUMMARY
# ----------------------------------------------------------
st.subheader("🧠 Segment Insights")

summary = df.groupby("Segment_Name")[[
    "sale_price",
    "floor_area_sqft",
    "Age",
    "satisfaction_score"
]].mean()

st.dataframe(summary)

# ----------------------------------------------------------
# RAW DATA
# ----------------------------------------------------------
st.subheader("📄 Data Preview")
st.dataframe(df.head(100))

# ----------------------------------------------------------
# DOWNLOAD
# ----------------------------------------------------------
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Final CSV",
    data=csv,
    file_name="real_estate_output.csv",
    mime="text/csv"
)

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.markdown("---")
st.markdown("Built with Python • Streamlit • Machine Learning")