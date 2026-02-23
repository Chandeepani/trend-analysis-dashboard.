import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Trend Analysis Dashboard",
    layout="wide"
)

st.title("📊 Trend Analysis Dashboard")
st.subheader("Expected Customers Prediction (ML Model)")

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("trend_prediction_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# LOAD DATASET (for encoder reference)
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_trend_dataset.csv")

df = load_data()

# -----------------------------
# RECREATE LABEL ENCODER (SAME AS TRAINING)
# -----------------------------
industry_encoder = LabelEncoder()
industry_encoder.fit(df["Industry"].astype(str))

# -----------------------------
# USER INPUTS (MATCH TRAINING FEATURES)
# -----------------------------
st.sidebar.header("🔮 Enter Prediction Inputs")

industry = st.sidebar.selectbox(
    "Select Industry",
    df["Industry"].unique()
)

month = st.sidebar.slider("Month", 1, 36, 12)
revenue = st.sidebar.number_input("Revenue", value=50000.0)
customers = st.sidebar.number_input("Current Customers", value=2000.0)
growth_rate = st.sidebar.slider("Growth Rate", 0.0, 1.0, 0.2)
trend_score = st.sidebar.slider("Trend Score", 0.0, 1.0, 0.5)

# -----------------------------
# ENCODE INDUSTRY (IMPORTANT)
# -----------------------------
industry_encoded = industry_encoder.transform([str(industry)])[0]

# -----------------------------
# CREATE INPUT DATAFRAME (EXACT SAME ORDER AS TRAINING)
# -----------------------------
input_data = pd.DataFrame({
    "Industry": [industry_encoded],
    "Month": [month],
    "Revenue": [revenue],
    "Customers": [customers],
    "GrowthRate": [growth_rate],
    "TrendScore": [trend_score]
})

# Ensure correct column order (VERY IMPORTANT)
input_data = input_data[[
    "Industry",
    "Month",
    "Revenue",
    "Customers",
    "GrowthRate",
    "TrendScore"
]]

# -----------------------------
# SCALE INPUT (SAME SCALER AS TRAINING)
# -----------------------------
scaled_input = scaler.transform(input_data)

# -----------------------------
# MAKE PREDICTION
# -----------------------------
if st.sidebar.button("Predict Expected Customers"):
    prediction = model.predict(scaled_input)[0]

    st.subheader("📊 Prediction Result")
    st.metric("Predicted Expected Customers", f"{prediction:.2f}")

# -----------------------------
# VISUALIZATIONS (Optional Dashboard)
# -----------------------------
st.subheader("📈 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("Sample Data")
    st.dataframe(df.head(10))

with col2:
    st.write("Industry Distribution")
    st.bar_chart(df["Industry"].value_counts())