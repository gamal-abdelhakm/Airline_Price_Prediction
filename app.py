import pandas as pd
import streamlit as st
import joblib
import datetime

# Load the model and preprocessed data
try:
    Model = joblib.load("Model.pkl")
    data = pd.read_csv('preprocessed_data.csv')
    
    # Define the input fields
    input_fields = {
        'Airline': data['Airline'].unique(),
        'Source': data['Source'].unique(),
        'Destination': data['Destination'].unique(),
        'Total_Stops': data['Total_Stops'].unique(),
        'Journey_Date': None,  # Will be handled with date_input
        'Dep_Time': None,      # Will be handled with time_input
        'Arrival_Time': None,  # Will be handled with time_input
        'Duration_Hours': (max(1, data['Duration_Hours'].min()), min(24, data['Duration_Hours'].max()))
    }
    
    # Add page configuration
    st.set_page_config(
        page_title="Flight Price Predictor",
        page_icon="✈️",
        layout="wide"
    )

except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

def main():
    # Title and description
    st.title('✈️ Flight Price Prediction')
    st.write('Enter your flight details below to get an estimated price.')
    
    with st.container():
        # Create two columns for the form
        col1, col2 = st.columns(2)
        
        # Initialize inputs dictionary
        inputs = {}
        
        # Route information
        with col1:
            st.markdown("### Route Information")
            inputs['Airline'] = st.selectbox('Airline', input_fields['Airline'])
            inputs['Source'] = st.selectbox('From', input_fields['Source'])
            
            # Add validation to prevent same source and destination
            remaining_destinations = [dest for dest in input_fields['Destination'] if dest != inputs['Source']]
            inputs['Destination'] = st.selectbox('To', remaining_destinations)
            
            inputs['Total_Stops'] = st.selectbox('Number of Stops', input_fields['Total_Stops'])
        
        # Time information
        with col2:
            st.markdown("### Time Information")
            
            # Journey date
            selected_date = st.date_input('Journey Date', 
                                        min_value=datetime.date.today(),
                                        value=datetime.date.today())
            
            inputs['Journey_Day'] = selected_date.day
            inputs['Journey_Month'] = selected_date.month
            
            # Departure time
            dep_time = st.time_input('Departure Time', datetime.time(9, 0))
            inputs['Dep_Hour'] = dep_time.hour
            
            # Duration slider
            duration = st.slider('Flight Duration (hours)', 
                                float(input_fields['Duration_Hours'][0]), 
                                float(input_fields['Duration_Hours'][1]), 
                                step=0.5)
            inputs['Duration_Hours'] = duration
            
            # Calculate arrival time
            arrival_hour = (dep_time.hour + int(duration)) % 24
            arrival_minute = dep_time.minute + int((duration % 1) * 60)
            if arrival_minute >= 60:
                arrival_hour = (arrival_hour + 1) % 24
                arrival_minute = arrival_minute % 60
                
            arrival_time = datetime.time(arrival_hour, arrival_minute)
            st.info(f"Calculated Arrival Time: {arrival_time.strftime('%H:%M')}")
            
            inputs['Arrival_Hour'] = arrival_hour
        
        # Predict button with improved styling
        if st.button('Predict Price', type="primary", use_container_width=True):
            with st.spinner('Calculating price...'):
                try:
                    input_data = pd.DataFrame(inputs, index=[0])
                    
                    # Add debug information in expander
                    with st.expander("Debug Information"):
                        st.write("Input data for prediction:")
                        st.write(input_data)
                    
                    prediction = predict_price(input_data)
                    
                    # Display prediction with more emphasis
                    st.markdown("### Price Prediction")
                    st.markdown(f"<h2 style='text-align: center; color: #1E88E5;'>${prediction:.2f}</h2>", unsafe_allow_html=True)
                    
                    # Add confidence disclaimer
                    st.info("This is an estimated price based on historical data. Actual prices may vary.")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

def predict_price(input_data):
    # Perform prediction using the trained model
    prediction = Model.predict(input_data)
    return prediction[0]

if __name__ == '__main__':
    main()
