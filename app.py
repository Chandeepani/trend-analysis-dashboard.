import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Trend Analysis Dashboard",
    layout="wide"
)

st.title("📊 Trend Analysis Dashboard")
st.subheader("Machine Learning Based Industry Trend Prediction")

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_trend_dataset.csv")
    return df

df = load_data()

# -----------------------------
# LOAD TRAINED MODEL (.pkl)
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("trend_prediction_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("🔮 Predict Future Trend")

industry = st.sidebar.selectbox(
    "Select Industry",
    df["Industry"].unique()
)

month = st.sidebar.slider("Select Month", 1, 36, 12)
revenue = st.sidebar.number_input("Revenue", value=50000)
customers = st.sidebar.number_input("Customers", value=2000)
growth_rate = st.sidebar.slider("Growth Rate", 0.0, 1.0, 0.2)

# -----------------------------
# ENCODE INDUSTRY (Same as Training)
# -----------------------------
df_encoded = df.copy()
df_encoded = pd.get_dummies(df_encoded, columns=["Industry"])

# Create input dataframe
input_data = pd.DataFrame({
    "Month": [month],
    "Revenue": [revenue],
    "Customers": [customers],
    "GrowthRate": [growth_rate],
    "Industry_Finance": [1 if industry == "Finance" else 0],
    "Industry_Healthcare": [1 if industry == "Healthcare" else 0],
    "Industry_Retail": [1 if industry == "Retail" else 0],
    "Industry_Technology": [1 if industry == "Technology" else 0],
    "Industry_Transportation": [1 if industry == "Transportation" else 0],
})

# -----------------------------
# MAKE PREDICTION
# -----------------------------
prediction = model.predict(input_data)[0]

# -----------------------------
# TREND CATEGORY LOGIC
# -----------------------------
def trend_category(score):
    if score >= 0.6:
        return "📈 Rising"
    elif score >= 0.4:
        return "➖ Stable"
    else:
        return "📉 Declining"

trend_status = trend_category(prediction)

# -----------------------------
# DISPLAY PREDICTION
# -----------------------------
st.subheader("🔍 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted Trend Score", round(prediction, 3))

with col2:
    st.metric("Trend Status", trend_status)

# -----------------------------
# VISUALIZATION 1: Trend Over Time
# -----------------------------
st.subheader("📅 Industry Trend Over Time")

filtered_df = df[df["Industry"] == industry]

fig1 = px.line(
    filtered_df,
    x="Month",
    y="Customers",
    title=f"{industry} Customer Trend (36 Months)",
    markers=True
)
st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# VISUALIZATION 2: Revenue vs Customers
# -----------------------------
st.subheader("💰 Revenue vs Customers Analysis")

fig2 = px.scatter(
    df,
    x="Revenue",
    y="Customers",
    color="Industry",
    size="GrowthRate",
    title="Revenue vs Customers by Industry"
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# VISUALIZATION 3: Growth Rate Distribution
# -----------------------------
st.subheader("📊 Growth Rate Distribution")

fig3 = px.box(
    df,
    x="Industry",
    y="GrowthRate",
    title="Growth Rate Comparison Across Industries"
)
st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# SHOW DATA TABLE
# -----------------------------
st.subheader("📁 Dataset Preview")
st.dataframe(df.head(20))
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Trend Analysis Dashboard",
    layout="wide"
)

st.title("📊 Trend Analysis Dashboard")
st.subheader("Machine Learning Based Industry Trend Prediction")

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_trend_dataset.csv")
    return df

df = load_data()

# -----------------------------
# LOAD TRAINED MODEL (.pkl)
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("trend_prediction_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("🔮 Predict Future Trend")

industry = st.sidebar.selectbox(
    "Select Industry",
    df["Industry"].unique()
)

month = st.sidebar.slider("Select Month", 1, 36, 12)
revenue = st.sidebar.number_input("Revenue", value=50000)
customers = st.sidebar.number_input("Customers", value=2000)
growth_rate = st.sidebar.slider("Growth Rate", 0.0, 1.0, 0.2)

# -----------------------------
# ENCODE INDUSTRY (Same as Training)
# -----------------------------
df_encoded = df.copy()
df_encoded = pd.get_dummies(df_encoded, columns=["Industry"])

# Create input dataframe
input_data = pd.DataFrame({
    "Month": [month],
    "Revenue": [revenue],
    "Customers": [customers],
    "GrowthRate": [growth_rate],
    "Industry_Finance": [1 if industry == "Finance" else 0],
    "Industry_Healthcare": [1 if industry == "Healthcare" else 0],
    "Industry_Retail": [1 if industry == "Retail" else 0],
    "Industry_Technology": [1 if industry == "Technology" else 0],
    "Industry_Transportation": [1 if industry == "Transportation" else 0],
})

# -----------------------------
# MAKE PREDICTION
# -----------------------------
# Scale the input data (same as training)
scaled_input = scaler.transform(input_data)

# Make prediction
prediction = model.predict(scaled_input)[0]

# -----------------------------
# TREND CATEGORY LOGIC
# -----------------------------
def trend_category(score):
    if score >= 0.6:
        return "📈 Rising"
    elif score >= 0.4:
        return "➖ Stable"
    else:
        return "📉 Declining"

trend_status = trend_category(prediction)

# -----------------------------
# DISPLAY PREDICTION
# -----------------------------
st.subheader("🔍 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted Trend Score", round(prediction, 3))

with col2:
    st.metric("Trend Status", trend_status)

# -----------------------------
# VISUALIZATION 1: Trend Over Time
# -----------------------------
st.subheader("📅 Industry Trend Over Time")

filtered_df = df[df["Industry"] == industry]

fig1 = px.line(
    filtered_df,
    x="Month",
    y="Customers",
    title=f"{industry} Customer Trend (36 Months)",
    markers=True
)
st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# VISUALIZATION 2: Revenue vs Customers
# -----------------------------
st.subheader("💰 Revenue vs Customers Analysis")

fig2 = px.scatter(
    df,
    x="Revenue",
    y="Customers",
    color="Industry",
    size="GrowthRate",
    title="Revenue vs Customers by Industry"
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# VISUALIZATION 3: Growth Rate Distribution
# -----------------------------
st.subheader("📊 Growth Rate Distribution")

fig3 = px.box(
    df,
    x="Industry",
    y="GrowthRate",
    title="Growth Rate Comparison Across Industries"
)
st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# SHOW DATA TABLE
# -----------------------------
st.subheader("📁 Dataset Preview")
st.dataframe(df.head(20))