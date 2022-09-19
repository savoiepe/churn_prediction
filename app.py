from shlex import join
from sre_parse import State
import streamlit as st

import numpy as np
import pandas as pd
import joblib

model = joblib.load('model.joblib')

unique_states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
unique_area_codes = ['area_code415', 'area_code408', 'area_code510']

st.title(":fire: Awesome Churn Predictor :fire:")

st.markdown("Welcome to the Super Awesome Churn Predictor ! Are you a telecommunications provider ? Well you came to the right place ! We trained a model to predict whether a customer will change telecommunication service. It's super simple, enter the following details of the customer : ")

with st.form("form key", clear_on_submit=True):
    state = st.selectbox("State :", unique_states)
    account_length = st.number_input("Number of months the customer has been with current provider : ", 0) 
    area_code = st.selectbox("Area Code : ", ['415', '408', '510'])
    internationnal_plan = st.selectbox("Does the customer have an internationnal plan : ", ['yes', 'no'])
    voice_mail_plan = st.selectbox("Does the customer have an voicemail plan : ", ['yes', 'no'])
    number_vmail_messages = st.number_input("Number of voicemail messages : ", 0) 

    total_day_minutes = st.number_input("Total number of minutes on the phone during the day : ", 0.00) 
    total_day_calls = st.number_input("Total number of calls during the day : ", 0) 
    total_day_charge = st.number_input("Total charge of day calls: ", 0.00) 

    total_eve_minutes = st.number_input("Total number of minutes on the phone during the evening : ", 0.00) 
    total_eve_calls = st.number_input("Total number of calls during the evening : ", 0) 
    total_eve_charge = st.number_input("Total charge of evening calls: ", 0.00) 

    total_night_minutes = st.number_input("Total number of minutes on the phone during the night : ", 0.00) 
    total_night_calls = st.number_input("Total number of calls during the night : ", 0) 
    total_night_charge = st.number_input("Total charge of night calls: ", 0.00) 

    total_intl_minutes = st.number_input("Total number of minutes on the internationnal calls : ", 0.00) 
    total_intl_calls = st.number_input("Total number of internationnal calls : ", 0) 
    total_intl_charge = st.number_input("Total charge of internationnal calls: ", 0.00) 

    number_customer_service_calls = st.number_input("Number of calls to customer service : ", 0) 

    def preprocess(df):
        # Change categorical data to one hot encoding
        df['state'] = df['state'].astype(
            pd.CategoricalDtype(categories=unique_states)
            )
        state_one_hot = pd.get_dummies(df['state'])
        df = df.drop('state', axis = 1)
        df = df.join(state_one_hot)

        df['area_code'] = df['area_code'].astype(
            pd.CategoricalDtype(categories=unique_area_codes)
            )
        area_code_one_hot = pd.get_dummies(df['area_code'])
        df = df.drop('area_code', axis = 1)
        df = df.join(area_code_one_hot)

        # Change no/yes to 0/1
        df['international_plan'] = df['international_plan'].map({'yes': 1, 'no': 0})
        df['voice_mail_plan'] = df['voice_mail_plan'].map({'yes': 1, 'no': 0})

        return df

    def predict():
        x = np.array([state, account_length, "area_code_"+str(area_code), internationnal_plan, voice_mail_plan, number_vmail_messages,
        total_day_minutes, total_day_calls, total_day_charge, total_eve_minutes, total_eve_calls, total_eve_charge, total_night_minutes,
        total_night_calls, total_night_charge, total_intl_minutes, total_intl_calls, total_intl_charge, number_customer_service_calls])

        df = pd.DataFrame([x], columns=['state', 'account_length', 'area_code', 'international_plan',
        'voice_mail_plan', 'number_vmail_messages', 'total_day_minutes',
        'total_day_calls', 'total_day_charge', 'total_eve_minutes',
        'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
        'total_night_calls', 'total_night_charge', 'total_intl_minutes',
        'total_intl_calls', 'total_intl_charge', 'number_customer_service_calls'])

        preprocessed_df = preprocess(df)

        prediction = model.predict(preprocessed_df)[0]

        if prediction == 1:
            st.error("Customer is going to change provider :sad:")
        else:
            st.success("Customer is staying with current provider :smile:")

    st.form_submit_button("Predict", on_click= predict)