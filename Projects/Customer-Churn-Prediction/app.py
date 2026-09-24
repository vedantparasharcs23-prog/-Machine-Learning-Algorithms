'''### Load the Trained Model and Preprocessor

The saved Logistic Regression model and preprocessing pipeline are loaded
into the Streamlit application using Joblib.

The loaded artifacts will be used to transform customer input and generate
churn predictions.

The Streamlit interface is initialized with a title and a short description
of the Customer Churn Prediction application.'''
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and preprocessor
model = joblib.load("final_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("Customer Churn Prediction")

st.write("Predict whether a customer is likely to churn.")

'''### Create Customer Input Fields

The Streamlit application collects customer information through interactive
input widgets.

The input fields correspond to the features used by the trained machine
learning model.'''

# Customer information
gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

'''### Add Service Information Fields

The application collects information about the customer's phone service,
internet service, and additional subscribed services.

These features are used by the trained model to predict customer churn.'''

# Service information
phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)


'''### Add Contract and Billing Information

The application collects contract, billing, and payment information from
the customer.

These features are important inputs for the churn prediction model and
complete the required customer feature set.'''


# Contract and billing information
contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)


'''### Create the Customer DataFrame

The values collected from the Streamlit input fields are combined into a
Pandas DataFrame.

The DataFrame follows the same feature structure used during model training,
allowing it to be passed through the saved preprocessing pipeline.'''


# Create a DataFrame from user inputs
customer_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])


'''### Preprocess Customer Data

The customer DataFrame is transformed using the saved preprocessing pipeline.

The same preprocessing steps used during model training are applied to the
new customer data to ensure that the input format is compatible with the
trained machine learning model.'''


# Preprocess the customer data

customer_encoded = preprocessor.transform(customer_data)


'''### Generate Churn Prediction

The preprocessed customer data is passed to the trained Logistic Regression
model to generate a churn prediction and churn probability.

The prediction is displayed as either "Churn" or "No Churn".'''


# Generate prediction and churn probability
if st.button("Predict Churn"):
    prediction = model.predict(customer_encoded)[0]
    probability = model.predict_proba(customer_encoded)[0][1]

    if prediction == 1:
        st.error("Customer is predicted to churn.")
    else:
        st.success("Customer is predicted to stay.")

    st.write(f"Churn Probability: {probability:.2%}")