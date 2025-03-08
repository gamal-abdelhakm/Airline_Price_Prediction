import pandas as pd
import streamlit as st
import joblib
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .subheader {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 20px;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .info-text {
        color: #616161;
        font-size: 0.9rem;
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
        'Airline': sorted(data['Airline'].unique()),
        'Source': sorted(data['Source'].unique()),
        'Destination': sorted(data['Destination'].unique()),
        'Total_Stops': sorted(data['Total_Stops'].unique()),
        'Journey_Date': None,  # Will be handled separately
        'Dep_Time': None,      # Will be handled separately
        'Duration_Hours': (max(0, data['Duration_Hours'].min()), 
                          min(24, data['Duration_Hours'].max()))
    }
    
    # Extract popular routes for suggestions
    popular_routes = data.groupby(['Source', 'Destination']).size().reset_index(name='count')
    popular_routes = popular_routes.sort_values('count', ascending=False).head(5)
    
except Exception as e:
    st.error(f"Error loading data or model: {str(e)}")
    st.stop()

def main():
    # Header
    st.markdown("<h1 class='main-header'>✈️ Flight Price Prediction</h1>", unsafe_allow_html=True)
    
    # Add tabs
    tab1, tab2, tab3 = st.tabs(["Predict Price", "Price Trends", "About"])
    
    with tab1:
        st.markdown("<h2 class='subheader'>Enter Flight Details</h2>", unsafe_allow_html=True)
        
        # Create columns for the form
        col1, col2 = st.columns(2)
        
        with col1:
            # Source and Destination with validation
            source = st.selectbox("Source", input_fields['Airline'])
            destination = st.selectbox("Destination", input_fields['Destination'])
            
            if source == destination:
                st.warning("Source and destination cannot be the same.")
            
            # Date picker with validation
            today = datetime.now().date()
            journey_date = st.date_input("Journey Date", 
                                         min_value=today,
                                         max_value=today + timedelta(days=365),
                                         value=today + timedelta(days=7))
            
            # Airline selection
            airline = st.selectbox("Airline", input_fields['Airline'])
            
        with col2:
            # Time picker
            dep_time = st.time_input("Departure Time", datetime.strptime("10:00", "%H:%M").time())
            
            # Number of stops
            stops = st.selectbox("Number of Stops", input_fields['Total_Stops'])
            
            # Duration with slider
            duration = st.slider("Flight Duration (hours)", 
                                min_value=float(input_fields['Duration_Hours'][0]),
                                max_value=float(input_fields['Duration_Hours'][1]),
                                value=2.5,
                                step=0.5)
            
            # Calculate arrival time based on departure and duration
            dep_datetime = datetime.combine(journey_date, dep_time)
            arrival_datetime = dep_datetime + timedelta(hours=duration)
            
            # Display calculated arrival time
            st.info(f"Calculated Arrival Time: {arrival_datetime.strftime('%H:%M')}")
        
        # Create input dictionary for prediction
        inputs = {
            'Airline': airline,
            'Source': source,
            'Destination': destination,
            'Total_Stops': stops,
            'Journey_Day': journey_date.day,
            'Journey_Month': journey_date.month,
            'Dep_Hour': int(dep_time.strftime('%H')),
            'Arrival_Hour': int(arrival_datetime.strftime('%H')),
            'Duration_Hours': duration
        }
        
        # Predict button in a centered column
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button('Predict Price', use_container_width=True):
                with st.spinner('Calculating price...'):
                    try:
                        input_data = pd.DataFrame(inputs, index=[0])
                        prediction = predict_price(input_data)
                        
                        # Display prediction with formatting
                        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='text-align: center; color: #1E88E5;'>Estimated Price</h2>", unsafe_allow_html=True)
                        st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>${prediction:.2f}</h1>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Add confidence interval
                        st.markdown("<p class='info-text'>*Price estimate may vary based on market conditions.</p>", unsafe_allow_html=True)
                        
                        # Show similar flights
                        st.subheader("Similar Flights:")
                        show_similar_flights(inputs)
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
        
        # Quick selection for popular routes
        st.markdown("### Popular Routes")
        route_cols = st.columns(len(popular_routes))
        for i, (col, (source, dest, _)) in enumerate(zip(route_cols, popular_routes.itertuples(index=False))):
            with col:
                if st.button(f"{source} → {dest}"):
                    # Set values in the form (note: this would require session state in actual implementation)
                    st.session_state.source = source
                    st.session_state.destination = dest
                    st.experimental_rerun()
    
    with tab2:
        st.subheader("Price Trends Analysis")
        
        # Create a sample price trend based on the data
        # In a real app, you would calculate this from historical data
        dates = pd.date_range(start=datetime.now().date(), periods=30, freq='D')
        prices = np.random.normal(loc=250, scale=50, size=30) + np.sin(np.arange(30)/5) * 30
        
        trend_data = pd.DataFrame({
            'Date': dates,
            'Price': prices
        })
        
        fig = px.line(trend_data, x='Date', y='Price', title='30-Day Price Trend')
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Average Price ($)',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 Tip: Book your tickets 4-6 weeks in advance for the best prices!")
    
    with tab3:
        st.subheader("About this App")
        st.write("""
        This flight price prediction app uses machine learning to estimate flight prices based on various factors:
        
        - **Airlines**: Different airlines have different pricing strategies
        - **Routes**: Popular routes might have more competitive pricing
        - **Time of travel**: Prices vary based on day of week and time of day
        - **Booking time**: How far in advance you're booking
        - **Stops**: Direct flights often cost more than flights with stops
        
        The prediction model was trained on historical flight data and provides estimates based on patterns identified in this data.
        """)
        
        st.write("### How to use")
        st.write("""
        1. Enter your flight details in the 'Predict Price' tab
        2. Click the 'Predict Price' button
        3. View your estimated price
        4. Check the 'Price Trends' tab for historical price patterns
        """)

def predict_price(input_data):
    """
    Predict flight price using the trained model
    """
    try:
        # Perform prediction using the trained model
        prediction = Model.predict(input_data)
        return max(0, prediction[0])  # Ensure non-negative predictions
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0

def show_similar_flights(inputs):
    """
    Display similar flights from the dataset
    """
    # Filter data for similar flights
    similar = data[
        (data['Source'] == inputs['Source']) & 
        (data['Destination'] == inputs['Destination']) &
        (data['Airline'] == inputs['Airline'])
    ]
    
    if len(similar) > 0:
        # Take a sample of up to 3 similar flights
        sample = similar.sample(min(3, len(similar)))
        
        # Display the similar flights
        for _, flight in sample.iterrows():
            with st.expander(f"Flight: {flight['Airline']} - {flight['Source']} to {flight['Destination']}"):
                st.write(f"**Total Stops:** {flight['Total_Stops']}")
                st.write(f"**Duration:** {flight['Duration_Hours']:.1f} hours")
                # Add more fields as needed
    else:
        st.info("No similar flights found in the dataset.")

if __name__ == '__main__':
    main()
