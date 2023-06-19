import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
import joblib

Model = joblib.load("Model.pkl")

# Load the preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Define the input fields
input_fields = {
    'Airline': data['Airline'].unique(),
    'Source': data['Source'].unique(),
    'Destination': data['Destination'].unique(),
    'Total_Stops': data['Total_Stops'].unique(),
    'Journey_Date': (data['Journey_Day'].min(), data['Journey_Day'].max()),
    'Dep_Hour': (data['Dep_Hour'].min(), data['Dep_Hour'].max()),
    'Arrival_Hour': (data['Arrival_Hour'].min(), data['Arrival_Hour'].max()),
    'Duration_Hours': (data['Duration_Hours'].min(), data['Duration_Hours'].max())
}

def main():
    st.title('Flight Price Prediction')
    st.write('Please enter the following details for flight price prediction:')

    # Create input fields
    inputs = {}
    col1, col2 = st.columns(2)  # Split the fields into two columns

    for i, (field, values) in enumerate(input_fields.items()):
        if field == 'Journey_Date':  # Journey_Day selector
            with col1 if i % 2 == 0 else col2:
                selected_date = st.date_input(field)
                inputs['Journey_Day'] = selected_date.day
                inputs['Journey_Month'] = selected_date.month
        elif field == 'Dep_Hour': 
            with col1 if i % 2 == 0 else col2:
                selected_time = st.time_input(field)
                inputs[field] = selected_time.strftime('%H')
        elif field == 'Arrival_Hour':
            with col1 if i % 2 == 0 else col2:
                inputs[field] = selected_time.strftime('%H')
        elif isinstance(values, tuple):  # Numerical variable
            with col1 if i % 2 == 0 else col2:
                inputs[field] = st.slider(field, 0, 24, 4)
        else:  # Categorical variable
            with col1 if i % 2 == 0 else col2:
                inputs[field] = st.selectbox(field, values)

    inputs['Arrival_Hour'] = str(int(inputs['Arrival_Hour']) + int(inputs['Duration_Hours']))

    # Predict button
    with col1:
        if st.button('Predict Price'):
            input_data = pd.DataFrame(inputs, index=[0])
            prediction = predict_price(input_data)
            st.success(f'The predicted price is {prediction:.2f} $')


def predict_price(input_data):
    # Perform prediction using the trained model
    prediction = Model.predict(input_data)
    return prediction[0]


if __name__ == '__main__':
    main()
