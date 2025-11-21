import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="Vehicle Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ---------- FULL UI CSS ----------
st.markdown("""
<style>

body {
    margin: 0;
    padding: 0;
}

/* Animated gradient background */
.block-container {
    padding-top: 2rem;
    background: linear-gradient(135deg, #020617, #0f172a, #1e293b, #0b1120);
    background-size: 400% 400%;
    animation: gradientMove 18s ease infinite;
    max-width: 85% !important;
}
@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ===== SPORTS CAR THEME TITLE ===== */
.app-title {
    font-size: 3.4rem;
    text-align: center;
    font-weight: 900;
    text-transform: uppercase;
    margin-top: 10px;
    margin-bottom: 18px;

    background: linear-gradient(180deg, #ffffff, #d6d6d6, #9b9b9b);
    -webkit-background-clip: text;
    color: transparent;

    text-shadow:
        0 0 10px rgba(255, 0, 50, 0.8),
        0 0 25px rgba(255, 20, 60, 0.9),
        0 0 45px rgba(255, 0, 0, 1),
        0 0 80px rgba(255, 0, 0, 0.8);

    animation: pulseGlow 2.5s infinite ease-in-out;
}
@keyframes pulseGlow {
    0% { text-shadow: 0 0 10px rgba(255,0,50,0.7); }
    50% { text-shadow: 0 0 35px rgba(255,0,50,1); }
    100% { text-shadow: 0 0 10px rgba(255,0,50,0.7); }
}

/* Subtitle */
.app-subtitle {
    text-align: center;
    font-size: 1.1rem;
    font-weight: 450;
    color: #d1d5db;
    margin-bottom: 1.6rem;
}

/* Animated car emoji */
.vehicle-anim {
    text-align: center;
    font-size: 48px;
    margin-bottom: 25px;
    animation: drive 3.5s linear infinite;
}
@keyframes drive {
    0% {transform: translateX(-200px);}
    50% {transform: translateX(200px);}
    100% {transform: translateX(-200px);}
}

/* ===== VEHICLE CAROUSEL ===== */
.carousel-strip {
    overflow: hidden;
    white-space: nowrap;
    margin: 15px auto 10px auto;
    padding: 8px 0;
}
.carousel-inner {
    display: inline-block;
    animation: slideCars 16s linear infinite;
    font-size: 1.5rem;
    letter-spacing: 2px;
    color: #e5e7eb;
}
@keyframes slideCars {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* ===== CARD ===== */
.form-card {
    background: rgba(15, 23, 42, 0.96);
    border-radius: 22px;
    padding: 26px 28px;
    border: 1px solid rgba(148,163,184,0.4);
    box-shadow: 0 18px 45px rgba(15,23,42,0.85);
    margin-top: 24px;
    margin-bottom: 30px;
    width: 80%;
    margin-left: auto;
    margin-right: auto;
}

/* Section title */
.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 16px;
    color: #e5e7eb;
}

/* Bigger labels */
label, span {
    font-size: 1.02rem !important;
}

/* Equal column padding */
div[data-testid="column"] {
    padding: 6px 10px !important;
}

/* Predict button */
.stButton > button {
    width: 100%;
    border-radius: 999px;
    padding: 0.8rem;
    font-size: 1.12rem;
    font-weight: 700;
    background: linear-gradient(90deg, #22c55e, #3b82f6);
    border: none;
    margin-top: 10px;
    margin-bottom: 5px;
    color: white;
    box-shadow: 0 12px 30px rgba(37, 99, 235, 0.65);
    transition: all 0.18s ease-in-out;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.03);
}

/* Price box */
.pred-box {
    margin-top: 30px;
    padding: 24px;
    border-radius: 20px;
    background: radial-gradient(circle at top left, rgba(45, 212, 191, 0.3), transparent),
                radial-gradient(circle at bottom right, rgba(59, 130, 246, 0.35), transparent);
    border: 2px solid rgba(148, 163, 184, 0.7);
}
.pred-value {
    font-size: 2rem;
    font-weight: 900;
    text-align: center;
    color: #bbf7d0;
}

</style>
""", unsafe_allow_html=True)

# -------- TITLE + ANIMATION --------
st.markdown('<div class="app-title">VEHICLE PRICE PREDICTION</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Smart valuation using Machine Learning</div>', unsafe_allow_html=True)
st.markdown('<div class="vehicle-anim">🚗💨</div>', unsafe_allow_html=True)

# -------- VEHICLE CAROUSEL --------
st.markdown("""
<div class="carousel-strip">
  <div class="carousel-inner">
    🚗 Swift &nbsp; 🏎️ Ferrari &nbsp; 🚙 Creta &nbsp; 🚐 Innova &nbsp; 🏍️ Pulsar &nbsp; 🛵 Activa &nbsp; 🚗 Baleno &nbsp; 🚙 Ertiga &nbsp; 🏎️ Mustang &nbsp; 🚗 City &nbsp; 🚙 Venue &nbsp; 🚐 XUV500 &nbsp; 🏍️ Apache &nbsp; 🛵 Dio &nbsp; 🚗 I20 &nbsp; 🚙 Harrier &nbsp;
    🚗 Swift &nbsp; 🏎️ Ferrari &nbsp; 🚙 Creta &nbsp; 🚐 Innova &nbsp; 🏍️ Pulsar &nbsp; 🛵 Activa &nbsp; 🚗 Baleno &nbsp; 🚙 Ertiga &nbsp; 🏎️ Mustang &nbsp; 🚗 City &nbsp; 🚙 Venue &nbsp; 🚐 XUV500 &nbsp; 🏍️ Apache &nbsp; 🛵 Dio &nbsp; 🚗 I20 &nbsp; 🚙 Harrier
  </div>
</div>
""", unsafe_allow_html=True)

# -------- LOAD DATA + TRAIN MODEL FROM CSV --------
@st.cache_resource
def load_data(csv_path="car data.csv"):
    df = pd.read_csv(csv_path)
    # Clean/standardize car names
    df["Car_Name"] = df["Car_Name"].astype(str).str.strip().str.title()
    return df

@st.cache_resource
def train_model():
    df = load_data()

    feature_cols = [
        "Car_Name",
        "Year",
        "Present_Price",
        "Kms_Driven",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Owner",
    ]
    X = df[feature_cols]
    y = df["Selling_Price"]

    cat_cols = ["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"]
    num_cols = ["Year", "Present_Price", "Kms_Driven", "Owner"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X, y)

    # Vehicle names for dropdown
    vehicle_names = sorted(df["Car_Name"].unique().tolist())
    return pipe, vehicle_names

model, vehicle_names = train_model()

# -------- FORM UI: 4–4 SPLIT --------
with st.container():
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🧾 Enter Vehicle Details</div>', unsafe_allow_html=True)

    with st.form("predict_form"):

        col1, col2 = st.columns(2, gap="large")

        with col1:
            vehicle_name = st.selectbox("Vehicle Name", vehicle_names)
            year = st.number_input("Year of Purchase", 1990, 2025, 2015)
            present_price = st.number_input("Present Price (Lakhs)", 0.0, 200.0, 5.0)
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

        with col2:
            kms_driven = st.number_input("KMs Driven", 0, 500000, 50000)
            seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            owner = st.number_input("Previous Owners", 0, 5, 0)

        submitted = st.form_submit_button("Predict Price 💰")

    st.markdown("</div>", unsafe_allow_html=True)

# -------- PREDICT --------
if submitted:
    input_df = pd.DataFrame(
        [
            {
                "Car_Name": vehicle_name,
                "Year": year,
                "Present_Price": present_price,
                "Kms_Driven": kms_driven,
                "Fuel_Type": fuel_type,
                "Seller_Type": seller_type,
                "Transmission": transmission,
                "Owner": owner,
            }
        ]
    )

    price = model.predict(input_df)[0]

    st.markdown('<div class="pred-box">', unsafe_allow_html=True)
    st.markdown(
        '<div class="pred-value">₹ {:,.2f}</div>'.format(price * 100000),
        unsafe_allow_html=True,
    )
    st.caption("*(Model output is in Lakhs, multiplied by 1,00,000)*")
    st.markdown("</div>", unsafe_allow_html=True)
