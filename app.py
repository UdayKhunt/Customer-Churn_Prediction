import streamlit as st
import pandas as pd
import pickle
import tensorflow

model = tensorflow.keras.models.load_model('saved_models/model.h5')

with open('saved_models/le_gender.pkl' , 'rb') as file:
    le = pickle.load(file)

with open('saved_models/ohe_geography.pkl' , 'rb') as file:
    ohe = pickle.load(file)

with open('saved_models/scaler.pkl' , 'rb') as file:
    scaler = pickle.load(file)

st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', le.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Geography' : [geography]
})

ohe_geo = ohe.transform(df[['Geography']])
df_ohe = pd.DataFrame(ohe_geo.toarray() , columns=ohe.get_feature_names_out())
df = pd.concat([df.drop('Geography' , axis = 1) , df_ohe] , axis = 1)

df = scaler.transform(df)

prediction = model.predict(df)[0][0]

st.write(f'Churn Probability: {prediction:.2f}')

if prediction > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
