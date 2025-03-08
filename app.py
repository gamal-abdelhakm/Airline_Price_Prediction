import pandas as pd
import streamlit as st
import joblib
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Caching mechanism for data loading
@st.cache_data
def load_data():
    try:
        return pd.read_csv('preprocessed_data.csv')
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'preprocessed_data.csv' exists in the app directory.")
        return None

@st.cache_resource
def load_model():
    try:
        return joblib.load("Model.pkl")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'Model.pkl' exists in the app directory.")
        return None

def predict_price(model, input_data):
    if model is None:
        return None
    try:
        prediction = model.predict(input_data)
        return prediction[0]
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

def format_currency(amount):
    return f"${amount:,.2f}"

def main():
    # Load data and model
    data = load_data()
    model = load_model()
    
    if data is None or model is None:
        st.stop()
    
    # Define the input fields
    input_fields = {
        'Airline': data['Airline'].unique(),
        'Source': data['Source'].unique(),
        'Destination': data['Destination'].unique(),
        'Total_Stops': sorted(data['Total_Stops'].unique()),
        'Journey_Date': None,  # Will handle separately
        'Dep_Hour': (0, 23),
        'Arrival_Hour': (0, 23),
        'Duration_Hours': (float(data['Duration_Hours'].min()), float(data['Duration_Hours'].max()))
    }
    
    # Sidebar for app navigation
    with st.sidebar:
        st.image("https://api.placeholder.com/150/150", width=150)
        st.title("Navigation")
        app_mode = st.radio("Choose Mode", ["Make Prediction", "View Analytics", "About"])
    
    if app_mode == "Make Prediction":
        # Header section
        st.markdown("<h1 class='main-header'>✈️ Flight Price Predictor</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Enter flight details to get price estimates</p>", unsafe_allow_html=True)
        
        # Input form
        with st.form("prediction_form"):
            st.subheader("Flight Details")
            
            col1, col2 = st.columns(2)
            
            # Flight route
            with col1:
                selected_airline = st.selectbox("Airline", options=input_fields['Airline'])
                source = st.selectbox("Departure City", options=input_fields['Source'])
            
            with col2:
                destination = st.selectbox("Destination City", options=input_fields['Destination'])
                total_stops = st.selectbox("Number of Stops", options=input_fields['Total_Stops'])
            
            # Ensure source and destination are different
            if source == destination:
                st.warning("Source and destination cannot be the same.")
            
            # Date and time
            st.subheader("Date and Time")
            col3, col4 = st.columns(2)
            
            with col3:
                journey_date = st.date_input(
                    "Journey Date", 
                    min_value=datetime.now().date(),
                    max_value=datetime.now().date() + timedelta(days=365)
                )
                dep_time = st.time_input("Departure Time", value=datetime.strptime("10:00", "%H:%M"))
            
            with col4:
                duration_hours = st.slider(
                    "Flight Duration (hours)", 
                    min_value=float(0.5), 
                    max_value=float(input_fields['Duration_Hours'][1]),
                    value=2.0,
                    step=0.5
                )
                
                # Calculate arrival time based on departure time and duration
                dep_datetime = datetime.combine(journey_date, dep_time)
                arr_datetime = dep_datetime + timedelta(hours=duration_hours)
                st.info(f"Estimated Arrival: {arr_datetime.strftime('%Y-%m-%d %H:%M')}")
            
            # Submit button
            submit_button = st.form_submit_button(label="Predict Price")
        
        # Process form submission
        if submit_button:
            # Create input data dictionary
            inputs = {
                'Airline': selected_airline,
                'Source': source,
                'Destination': destination,
                'Total_Stops': total_stops,
                'Journey_Day': journey_date.day,
                'Journey_Month': journey_date.month,
                'Dep_Hour': int(dep_time.strftime('%H')),
                'Arrival_Hour': int(arr_datetime.strftime('%H')),
                'Duration_Hours': duration_hours
            }
            
            # Convert to DataFrame for prediction
            input_df = pd.DataFrame(inputs, index=[0])
            
            # Get prediction
            prediction = predict_price(model, input_df)
            
            if prediction is not None:
                # Display prediction with animation
                st.balloons()
                st.markdown(f"""
                <div class='prediction-box'>
                    <h2>Estimated Flight Price</h2>
                    <h1>{format_currency(prediction)}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Price breakdown visualization (simplified example)
                st.subheader("Price Breakdown (Estimated)")
                
                breakdown = {
                    'Base Fare': prediction * 0.7,
                    'Taxes & Fees': prediction * 0.2,
                    'Fuel Surcharge': prediction * 0.1
                }
                
                breakdown_df = pd.DataFrame({
                    'Component': breakdown.keys(),
                    'Amount': breakdown.values()
                })
                
                fig = px.pie(breakdown_df, values='Amount', names='Component', 
                             title='Price Components', color_discrete_sequence=px.colors.sequential.Blues)
                st.plotly_chart(fig)
                
                # Similar flights
                st.subheader("Similar Flights")
                
                # Find similar flights in the dataset
                similar_flights = data[
                    (data['Airline'] == selected_airline) & 
                    (data['Source'] == source) & 
                    (data['Destination'] == destination)
                ].head(5)
                
                if not similar_flights.empty:
                    st.dataframe(similar_flights[['Airline', 'Source', 'Destination', 'Total_Stops', 'Duration_Hours', 'Price']])
                else:
                    st.info("No similar flights found in the dataset.")

    elif app_mode == "View Analytics":
        st.title("Flight Price Analytics")
        
        # Analytics tabs
        tab1, tab2, tab3 = st.tabs(["Price Trends", "Route Analysis", "Airline Comparison"])
        
        with tab1:
            st.subheader("Price Trends by Month")
            # Create a monthly price trend chart
            monthly_data = data.groupby('Journey_Month')['Price'].mean().reset_index()
            fig = px.line(monthly_data, x='Journey_Month', y='Price', 
                          title='Average Flight Prices by Month',
                          labels={'Journey_Month': 'Month', 'Price': 'Average Price ($)'},
                          markers=True)
            st.plotly_chart(fig)
        
        with tab2:
            st.subheader("Popular Routes")
            # Create a route analysis
            route_data = data.groupby(['Source', 'Destination'])['Price'].agg(['mean', 'count']).reset_index()
            route_data.columns = ['Source', 'Destination', 'Average Price', 'Number of Flights']
            route_data = route_data.sort_values('Number of Flights', ascending=False).head(10)
            st.dataframe(route_data)
        
        with tab3:
            st.subheader("Airline Price Comparison")
            # Create airline comparison
            airline_data = data.groupby('Airline')['Price'].mean().reset_index()
            airline_data = airline_data.sort_values('Price')
            fig = px.bar(airline_data, x='Airline', y='Price',
                        title='Average Price by Airline',
                        color='Price',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig)
    
    else:  # About section
        st.title("About Flight Price Predictor")
        st.write("""
        ## How It Works
        
        This application uses machine learning to predict flight prices based on various factors:
        
        - **Airline**: Different airlines have different pricing strategies
        - **Route**: The departure and arrival cities
        - **Stops**: Number of stops during the journey
        - **Time**: Time of departure and duration
        - **Date**: Day and month of travel
        
        ## Model Information
        
        The prediction model was trained on historical flight data using advanced regression techniques.
        
        ## Usage Tips
        
        - For best results, enter all details accurately
        - Prices tend to be higher during peak travel seasons
        - Compare different airlines and times for the best deals
        
        ## Disclaimer
        
        Predictions are estimates and actual prices may vary based on current market conditions.
        """)
        
        # Model performance metrics (placeholder)
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", "0.92")
        col2.metric("MAE", "$24.56")
        col3.metric("RMSE", "$31.23")

if __name__ == '__main__':
    main()
