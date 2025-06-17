import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# --------------------------
# Load Model and Transformers
# --------------------------
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pk1', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('label_encoder_geo.pk1', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('onehot_encoder_geo.pk1', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pk1', 'rb') as file:
    scaler = pickle.load(file)

# --------------------------
# Streamlit UI
# --------------------------
st.title('Customer Churn Prediction App')

# Input widgets
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
credit_score = st.number_input('Credit Score', min_value=0)
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure (Years)', 0, 10)
balance = st.number_input('Account Balance', min_value=0.0)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card?', [0, 1])
is_active_member = st.selectbox('Is Active Member?', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)

# --------------------------
# Data Preprocessing
# --------------------------
# Encode categorical variables
gender_encoded = label_encoder_gender.transform([gender])[0]
geo_onehot = onehot_encoder_geo.transform([[geography]]).toarray()
geo_columns = onehot_encoder_geo.get_feature_names_out(['Geography'])
geo_df = pd.DataFrame(geo_onehot, columns=geo_columns)

# Create main input DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Combine with one-hot encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

# Scale input
input_scaled = scaler.transform(input_data)

# --------------------------
# Prediction
# --------------------------
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

st.subheader(f'Churn Probability: {prediction_prob:.2f}')

if prediction_prob > 0.5:
    st.error('⚠️ The customer is likely to churn.')
else:
    st.success('✅ The customer is not likely to churn.')
