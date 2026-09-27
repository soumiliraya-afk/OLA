
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("ola_driver_churn.sav")

st.title("Ola Driver Churn Prediction")
st.write("Enter the driver's details to estimate their churn risk:")

input_data = {}
input_data["Age"] = st.number_input("Age", min_value=18, max_value=65, value=30)
input_data["Gender"] = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
input_data["Education_Level"] = st.selectbox("Education Level", options=[0, 1, 2], format_func=lambda x: ["10+", "12+", "Graduate"][x])
input_data["Joining_Designation"] = st.slider("Joining Designation", min_value=1, max_value=5, value=1)
input_data["Grade"] = st.slider("Current Grade", min_value=1, max_value=5, value=1)
input_data["Total_Business_Value"] = st.number_input("Total Business Value generated so far", value=0, step=10000)
input_data["Tenure_Months"] = st.number_input("Months with the company so far", min_value=1, max_value=60, value=6)
input_data["Income_Increased"] = st.selectbox("Has income increased since joining?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
input_data["Grade_Increased"] = st.selectbox("Has grade increased since joining?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
input_data["Rating_Increased"] = st.selectbox("Has quarterly rating increased since joining?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

if st.button("Predict Churn Risk"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.error(f"Prediction: **Likely to churn** (Probability: {prediction_proba[0][1]:.2f})")
    else:
        st.success(f"Prediction: **Likely to stay** (Probability: {prediction_proba[0][0]:.2f})")

st.write("\n\n---")
st.write("To run this Streamlit app, save this code as `app.py` and then run `streamlit run app.py` in your terminal.")
