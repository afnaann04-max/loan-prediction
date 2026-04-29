import streamlit as st
import joblib
import pandas as pd

# Load the saved model, scaler, and ordinal encoder
loaded_model = joblib.load('loan_prediction.pkl')
loaded_scaler = joblib.load('standard_scaler.pkl')
loaded_ordinal_encoder = joblib.load('ordinal_encoder.pkl')

def predict_Loan_status(
    Gender, Married, Dependents, Education,
    Self_Employed, ApplicantIncome, CoapplicantIncome,
    LoanAmount, Loan_Amount_Term, Credit_History, Property_Area,
    model, scaler, ordinal_encoder
):
    # Create input DataFrame
    new_data = pd.DataFrame([{
        'Gender':Gender,
        'Married':Married,
        'Dependents':Dependents,
        'Education':Education,
        'Self_Employed':Self_Employed,
        'ApplicantIncome':ApplicantIncome,
        'CoapplicantIncome':CoapplicantIncome,
        'LoanAmount':LoanAmount,
        'Loan_Amount_Term':Loan_Amount_Term,
        'Credit_History':Credit_History,
        'Property_Area':Property_Area,
    }])

    # Ensure column order matches training data
    # The original `x` was defined as: x = df[['Gender', 'Married', 'Dependents', 'Education',
    #    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    #    'Loan_Amount_Term', 'Credit_History', 'Property_Area']]
    # We need to replicate this column order for the new_data
    columns_order = ['Gender', 'Married', 'Dependents', 'Education',
                     'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                     'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    new_data = new_data[columns_order]

    # Encoding categorical features
    new_data['Gender'] = new_data['Gender'].map({'Male':0,'Female':1})
    new_data['Married'] = new_data['Married'].map({'No':0,'Yes':1})
    new_data['Education'] = new_data['Education'].map({'Not Graduate':0,'Graduate':1})
    new_data['Self_Employed'] = new_data['Self_Employed'].map({'No':0,'Yes':1})
    new_data['Property_Area'] = new_data['Property_Area'].map({'Urban':0,'Semiurban':1,'Rural':2})

    # Ordinal encoding for Dependents
    new_data['Dependents'] = ordinal_encoder.transform(new_data[['Dependents']])

    # Scale numerical features
    new_data_scaled = scaler.transform(new_data)

    # Predict probability and class
    prob = model.predict_proba(new_data_scaled)[0][1]
    pred_class = model.predict(new_data_scaled)[0]

    return prob, pred_class

# Streamlit UI
st.title('Loan Prediction Application')

st.write("Enter the applicant's details to predict loan status.")

# Input fields
gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Married', ['No', 'Yes'])
dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['No', 'Yes'])
applicant_income = st.number_input('Applicant Income', min_value=0, value=5000)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0, value=0)
loan_amount = st.number_input('Loan Amount', min_value=0, value=150)
loan_amount_term = st.selectbox('Loan Amount Term (months)', [12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0], index=8)
credit_history = st.selectbox('Credit History (1.0 for Yes, 0.0 for No)', [0.0, 1.0])
property_area = st.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'])

if st.button('Predict Loan Status'):
    probability, prediction = predict_Loan_status(
        Gender=gender,
        Married=married,
        Dependents=dependents,
        Education=education,
        Self_Employed=self_employed,
        ApplicantIncome=applicant_income,
        CoapplicantIncome=coapplicant_income,
        LoanAmount=loan_amount,
        Loan_Amount_Term=loan_amount_term,
        Credit_History=credit_history,
        Property_Area=property_area,
        model=loaded_model,
        scaler=loaded_scaler,
        ordinal_encoder=loaded_ordinal_encoder
    )

    if prediction == 1:
        st.success(f"Prediction: Loan will be APPROVED (Probability: {probability:.2f})")
    else:
        st.error(f"Prediction: Loan will be REJECTED (Probability: {1-probability:.2f})")