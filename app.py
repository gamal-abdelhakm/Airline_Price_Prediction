import pandas as pd
import streamlit as st
import joblib
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS to improve appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .title {
        font-size: 2.5rem;
        color: #2c3e50;
        margin-bottom: 1.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .prediction {
        background-color: #f1f8ff;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("Model.pkl")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('preprocessed_data.csv')

try:
    Model = load_model()
    data = load_data()
    
    # Define the input fields
    input_fields = {
        'Airline': data['Airline'].unique(),
        'Source': data['Source'].unique(),
        'Destination': data['Destination'].unique(),
        'Total_Stops': data['Total_Stops'].unique(),
        'Journey_Date': (data['Journey_Day'].min(), data['Journey_Day'].max()),
        'Dep_Hour': (data['Dep_Hour'].min(), data['Dep_Hour'].max()),
        'Duration_Hours': (data['Duration_Hours'].min(), data['Duration_Hours'].max())
    }
    
    def main():
        # Header
        st.markdown("<h1 class='title'>✈️ Flight Price Prediction</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Enter flight details below to get an estimated price</p>", unsafe_allow_html=True)
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Predict Price", "Data Insights", "About"])
        
        with tab1:
            # Sidebar for inputs
            with st.container():
                st.subheader("Flight Details")
                
                col1, col2 = st.columns(2)
                
                # Store inputs
                inputs = {}
                
                # Airline selection with airline logos/icons
                with col1:
                    airline = st.selectbox('Airline', input_fields['Airline'], help="Select the airline")
                    inputs['Airline'] = airline
                
                # Route selection
                with col2:
                    source = st.selectbox('Source', input_fields['Source'], help="Select departure city")
                    inputs['Source'] = source
                
                with col1:
                    destination = st.selectbox('Destination', input_fields['Destination'], help="Select arrival city")
                    if destination == source:
                        st.warning("Source and destination cannot be the same!")
                    inputs['Destination'] = destination
                
                # Date and time selection
                with col2:
                    journey_date = st.date_input('Journey Date', datetime.date.today())
                    inputs['Journey_Day'] = journey_date.day
                    inputs['Journey_Month'] = journey_date.month
                
                with col1:
                    departure_time = st.time_input('Departure Time', datetime.time(9, 0))
                    inputs['Dep_Hour'] = int(departure_time.strftime('%H'))
                
                # Stop selection with clearer options
                with col2:
                    stop_options = {str(int(stop)): f"{int(stop)} Stop{'s' if int(stop) > 1 else ''}" for stop in input_fields['Total_Stops']}
                    selected_stop = st.selectbox('Number of Stops', options=list(stop_options.keys()), format_func=lambda x: stop_options[x])
                    inputs['Total_Stops'] = float(selected_stop)
                
                # Duration slider with better formatting
                with col1:
                    duration = st.slider('Flight Duration (Hours)', 
                                     min_value=1, 
                                     max_value=24, 
                                     value=3,
                                     help="Estimated flight duration in hours")
                    inputs['Duration_Hours'] = duration
                
                # Calculate arrival hour based on departure and duration
                inputs['Arrival_Hour'] = (inputs['Dep_Hour'] + inputs['Duration_Hours']) % 24
                
                # Add a divider
                st.divider()
                
                # Prediction section
                predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
                
                with predict_col2:
                    if st.button('Predict Price', use_container_width=True):
                        # Show loading spinner
                        with st.spinner('Calculating price...'):
                            input_data = pd.DataFrame(inputs, index=[0])
                            prediction = predict_price(input_data)
                            
                            # Display prediction with formatting
                            st.markdown(f"""
                            <div class='prediction'>
                                <h2>Estimated Flight Price</h2>
                                <h1 style='color: #3498db; font-size: 3rem;'>${prediction:.2f}</h1>
                                <p>Based on your selected flight details</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Additional price info
                            st.info(f"Price per hour: ${prediction/inputs['Duration_Hours']:.2f}")
        
        with tab2:
            st.subheader("Flight Data Insights")
            
            # Show some basic statistics and visualizations
            if st.checkbox("Show popular routes"):
                route_data = data.groupby(['Source', 'Destination']).size().reset_index(name='Count')
                route_data = route_data.sort_values('Count', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Count', y=route_data['Source'] + ' to ' + route_data['Destination'], data=route_data)
                plt.title('Most Popular Routes')
                st.pyplot(fig)
            
            if st.checkbox("Show price distribution by airline"):
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x='Airline', y='Price', data=data)
                plt.xticks(rotation=45)
                plt.title('Price Distribution by Airline')
                st.pyplot(fig)
        
        with tab3:
            st.subheader("About This App")
            st.write("""
            This flight price prediction application uses machine learning to estimate 
            flight prices based on various factors such as airline, route, number of stops, 
            and flight duration.
            
            The model has been trained on historical flight data and can provide reasonably 
            accurate price estimates for planning your travel budget.
            
            **Note:** Actual prices may vary based on factors not considered in this model, 
            such as seasonal demand, special events, and last-minute bookings.
            """)
    
    def predict_price(input_data):
        # Perform prediction using the trained model
        try:
            prediction = Model.predict(input_data)
            return prediction[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return 0

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.info("Please make sure 'Model.pkl' and 'preprocessed_data.csv' files are in the same directory as this script.")

if __name__ == '__main__':
    main()
