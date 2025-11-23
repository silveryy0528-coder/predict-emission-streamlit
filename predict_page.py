import streamlit as st
import joblib
import numpy as np
import pandas as pd


def load_model():
    model = joblib.load('emission_model_pipeline.joblib')
    return model


def show_predict_page():
    st.title("Vehicle CO2 Emissions Prediction")

    st.write("""### Enter the vehicle details to predict CO2 emissions""")

    make_names = (
        "Ford", "Chevrolet", "Mercedes-Benz", "Porsche", "GMC", "BMW", "Audi", "Toyota", "Honda", "Nissan"
    )
    vehicle_names = (
        "SUV: Small", "SUV: Standard", "SUV", "Pickup truck: Standard", "Mid-size",
        "Compact", "Subcompact", "Full-size", "Two-seater", "Minicompact"
    )
    transmission_names = (
        "AS8", "A9", "A8", "AS10", "AM7", "M6", "A10", "AS6", "AM8", "AV", "AV8", "AS9", "AV7"
    )

    # Example input features - adjust based on actual model features
    make = st.selectbox("Make", options=make_names)
    vehicle_class = st.selectbox("Vehicle Class", options=vehicle_names)
    transmission = st.selectbox("Transmission", options=transmission_names)
    engine_size = st.slider("Engine Size (L)", min_value=1.0, max_value=8.0, value=2.0)
    cylinders = st.slider("Cylinders", min_value=1, max_value=16, value=4)
    fuel_comsum_in_city = st.slider("Fuel Consumption in City (L/100 km)", min_value=1.0, max_value=30.0, value=12.0)
    fuel_comsum_in_city_hwy = st.slider("Fuel Consumption in City Hwy (L/100 km)", min_value=1.0, max_value=30.0, value=8.0)
    fuel_comsum_comb = st.slider("Fuel Consumption comb (L/100 km)", min_value=1.0, max_value=30.0, value=10.0)
    smog_level = st.slider("Smog Level", min_value=1, max_value=7, value=5)

    ok = st.button("Predict CO2 Emissions")
    if ok:
        model = load_model()
        input_data = np.array([[make, vehicle_class, engine_size, cylinders, transmission,
                                fuel_comsum_in_city, fuel_comsum_in_city_hwy,
                                fuel_comsum_comb, smog_level]])
        input_df = pd.DataFrame(input_data, columns=[
            'Make', 'Vehicle_Class', 'Engine_Size', 'Cylinders', 'Transmission',
            'Fuel_Consumption_in_City(L/100 km)', 'Fuel_Consumption_in_City_Hwy(L/100 km)',
            'Fuel_Consumption_comb(L/100km)', 'Smog_Level'
        ])
        pred_emission = model.predict(input_df)
        st.subheader("The Predicted CO2 Emissions: {:.2f} g/km".format(pred_emission[0]))


