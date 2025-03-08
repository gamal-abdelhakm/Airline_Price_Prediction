import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime

# Load model and data
model = joblib.load("Model.pkl")
data = pd.read_csv('preprocessed_data.csv')

# Define input fields with actual data ranges
input_fields = {
    'Airline': data['Airline'].unique(),
    'Source': data['Source'].unique(),
    'Destination': data['Destination'].unique(),
    'Total_Stops': data['Total_Stops'].unique(),
    'Journey_Date': (data['Journey_Day'].min(), data['Journey_Day'].max()),
    'Dep_Hour': (data['Dep_Hour'].min(), data['Dep_Hour'].max()),
    'Duration_Hours': (data['Duration_Hours'].min(), data['Duration_Hours'].max())
}

# Set page configuration
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide"
)

def calculate_arrival_hour(dep_hour, duration):
    """Calculate arrival hour considering 24-hour format"""
    return (dep_hour + duration) % 24

def main():
    st.title('Flight Price Prediction')
    st.write('Please enter the following details for flight price prediction:')

    inputs = {}
    col1, col2 = st.columns(2)

    for i, (field, values) in enumerate(input_fields.items()):
        with col1 if i % 2 == 0 else col2:
            if field == 'Journey_Date':
                selected_date = st.date_input(field)
                inputs['Journey_Day'] = selected_date.day
                inputs['Journey_Month'] = selected_date.month
            elif field == 'Dep_Hour':
                # Time input with actual data range
                dep_time = st.time_input(field, datetime.strptime(str(data['Dep_Hour'].min()), 
                                        datetime.strptime(str(data['Dep_Hour'].max()))
                inputs[field] = dep_time.hour + dep_time.minute/60
            elif isinstance(values, tuple):  # Numerical input
                min_val, max_val = values
                step = 0.5 if field == 'Duration_Hours' else 1
                inputs[field] = st.slider(
                    field,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val)/2),
                    step=step
                )
            else:  # Categorical input
                inputs[field] = st.selectbox(field, values)

    # Calculate arrival hour based on departure and duration
    inputs['Arrival_Hour'] = calculate_arrival_hour(inputs['Dep_Hour'], inputs['Duration_Hours'])

    # Prediction
    if st.button('Predict Price'):
        try:
            input_df = pd.DataFrame([inputs])
            prediction = model.predict(input_df)
            st.success(f'Predicted Flight Price: ${prediction[0]:.2f}')
        except Exception as e:
            st.error(f'Error making prediction: {str(e)}')

if __name__ == '__main__':
    main()
