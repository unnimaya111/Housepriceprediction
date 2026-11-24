
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="EstiNest | House Price Prediction",
    layout="wide"
)

# CSS Styling
st.markdown("""
<style>

.stApp {
    background-color: #EAF2F8;
}

h1, h2, h3 {
    color: #2F4156;
}

p, label {
    color: #2F4156;
    font-size: 24px;
}

/* Dropdown */
.stSelectbox div[data-baseweb="select"] {
    background-color: #DCEAF4;
    border-radius: 12px;
}

/* Number input */
input {
    background-color: #DCEAF4 !important;
    color: #2F4156 !important;
    border-radius: 10px !important;
}

/* Number input buttons */
div[data-testid="stNumberInput"] button {
    background-color: #2F4156 !important;
    color: white !important;
    border-radius: 8px !important;
    min-width: 32px !important;
    height: 32px !important;
    padding: 0px !important;
    margin: 2px !important;
    border: none !important;
}

div[data-testid="stNumberInput"] button:hover {
    background-color: #567C8D !important;
}

/* Predict button only */
div.stButton > button {
    background-color: #2F4156 !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 12px !important;
    font-size: 18px !important;
    border: none !important;
}

div.stButton > button p {
    color: white !important;
}

/* Predict button hover */
div.stButton > button:hover {
    background-color: #567C8D !important;
    color: white !important;
}

/* Slider */
div[data-testid="stSlider"] {
    color: #567C8D;
}

</style>
""", unsafe_allow_html=True)


# Load Model
data = joblib.load("data/house_price_model.pkl")

model = data["model"]
features = data["features"]

df = pd.read_csv("data/cleaned_house_prices.csv")


# Heading
st.markdown("""
<h1 style="text-align:center; font-size:70px; color:#2F4156;">EstiNest</h1>
<p style="text-align:center; font-size:22px; color:#567C8D;">Discover the estimated value of your property</p>
<hr>
""", unsafe_allow_html=True)


# House Details
st.markdown("## Enter House Details")

st.write("Provide the details of your property to get a price estimate.")


# Inputs

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox(
        "Location",
        sorted(df["Location"].dropna().unique())
    )

with col2:
    property_type = st.selectbox(
        "Property Type",
        sorted(df["Property_Type"].dropna().unique())
    )


col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "Area (sq.ft)",
        min_value=100,
        max_value=10000,
        value=1200
    )

with col2:
    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=2
    )


col1, col2 = st.columns(2)

with col1:
    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

with col2:
    age = st.number_input(
        "Age of House (Years)",
        min_value=0,
        max_value=100,
        value=5
    )


col1, col2 = st.columns(2)

with col1:
    floor = st.number_input(
        "Floor",
        min_value=0,
        max_value=50,
        value=1
    )

with col2:
    parking = st.number_input(
        "Parking Spaces",
        min_value=0,
        max_value=10,
        value=1
    )


col1, col2 = st.columns(2)

with col1:
    quality = st.slider(
        "Construction Quality",
        min_value=1,
        max_value=10,
        value=5
    )

with col2:
    furnishing = st.selectbox(
        "Furnishing",
        sorted(df["Furnishing"].dropna().unique())
    )


# Predict Price

if st.button("Predict House Price", use_container_width=True):

    house = pd.DataFrame([{
        "Location": location,
        "Property_Type": property_type,
        "Area_sqft": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age_Years": age,
        "Floor": floor,
        "Parking_Spaces": parking,
        "Construction_Quality": quality,
        "Furnishing": furnishing
    }])

    house = pd.get_dummies(house)

    house = house.reindex(columns=features, fill_value=0)

    price = model.predict(house)

    # Display Prediction
    st.markdown(
        f"<div style='background-color:#DCEBE4; padding:25px; border-radius:12px; text-align:center;'>"
        f"<p style='font-size:22px; color:#2F4156; margin-bottom:5px;'>Estimated House Price</p>"
        f"<h1 style='font-size:42px; color:#2F4156; margin:0;'>₹{price[0]:,.0f}</h1>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Disclaimer
    st.markdown(
        "<p style='text-align:center; color:#8295A7; margin-top:10px;'>"
        "This is a model-based estimate and not a formal property valuation."
        "</p>",
        unsafe_allow_html=True
    )
