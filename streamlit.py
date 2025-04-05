import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st

# Set the page title and description
st.title("House Price Predictor")
st.write("""
This app predicts the price of a house based on various features 
such as square footage, number of bedrooms, property age, and more.
""")



# Load the pre-trained model
rf_pickle = open("models/RFmodel.pkl", "rb")
rf_model = pickle.load(rf_pickle)
rf_pickle.close()


# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("House Details")
    
   # Year Sold
    year_sold = st.number_input("Year Sold", min_value=1900, max_value=2025, value=2013, step=1)

    # Property Tax
    property_tax = st.number_input("Annual Property Tax ($)", min_value=0, step=10)

    # Insurance
    insurance = st.number_input("Annual Insurance Cost ($)", min_value=0, step=10)

    # Bedrooms
    beds = st.number_input("Number of Bedrooms", min_value=0, step=1)

    # Bathrooms
    baths = st.number_input("Number of Bathrooms", min_value=0, step=1)

    # Square Footage
    sqft = st.number_input("House Square Footage (sqft)", min_value=100, step=10)

    # Year Built
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025, step=1)

    # Lot Size
    lot_size = st.number_input("Lot Size (sqft)", min_value=0, step=1)

    # Basement
    basement = st.selectbox("Has Basement?", options=["Yes", "No"])
    basement = 1 if basement == "Yes" else 0

    # Popular
    popular = st.selectbox("Popular Area?", options=["Yes", "No"])
    popular = 1 if popular == "Yes" else 0

    # Recession
    recession = st.selectbox("Sold During Recession?", options=["Yes", "No"])
    recession = 1 if recession == "Yes" else 0

    # Property Age Calculation
    property_age = year_sold - year_built

    # Property Type
    property_type = st.selectbox("Property Type", options=["Bungalow", "Condo"])
    property_type_Bungalow = 1 if property_type == "Bungalow" else 0
    property_type_Condo = 1 if property_type == "Condo" else 0

    # Submit button
    submitted = st.form_submit_button("Predict House Price")


# Handle the input for the regression model
if submitted:
    # Prepare the input in the correct order
    prediction_input = [[
        year_sold, property_tax, insurance, beds, baths, sqft, year_built, 
        lot_size, basement, popular, recession, property_age, 
        property_type_Bungalow, property_type_Condo
    ]]

    # Make prediction
    predicted_price = rf_model.predict(prediction_input)[0]

    # Display result
    st.subheader("Predicted House Price:")
    st.write(f"**${predicted_price:,.2f}**")

st.write(
    """We used a machine learning (Random Forest Regressor) model to predict the house price."""
)
