import numpy as np
import pickle
import streamlit as st

# Model Loaded
scaler = pickle.load(open('scaler.sav','rb'))
model = pickle.load(open('Calorie_Burnt_Prediction.sav','rb'))

st.title("🔥 Calorie Burnt Prediction")
st.write("Enter Details")

# Form Inputs
gender_map = {"Male": 1, "Female": 0}
Gender = st.selectbox("Gender", ("Male", "Female"),index=0)
Gender = gender_map[Gender]

Age = st.number_input("Age", min_value=10, max_value=80, value=23)
Height = st.number_input("Height (cm)", min_value=120, max_value=210, value=198)
Weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=100)
Duration = st.number_input("Duration (min)", min_value=0, max_value=300, value=12)
Heart_Rate = st.number_input("Heart Rate (bpm)", min_value=50, max_value=200, value=90)
Body_Temp = st.number_input("Body Temp (°C)", min_value=35.0, max_value=41.0, value=40.4)

# Prepare input
input_data = np.array([[Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]], dtype=float)

scaler_input=scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(scaler_input)
    st.success(f"Estimated Calorie Burnt : {prediction[0]:.2f} calorie")
