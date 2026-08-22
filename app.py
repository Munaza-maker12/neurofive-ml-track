import streamlit as st
import joblib
import numpy as np

model = joblib.load('best_model.joblib')

st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸")
st.title("🌸 Iris Flower Species Predictor")
st.write("Enter the flower measurements below and click Predict.")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)

species_names = ["Setosa", "Versicolor", "Virginica"]

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Species: 🌼 {species_names[prediction[0]]}")