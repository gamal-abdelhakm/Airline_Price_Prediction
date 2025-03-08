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

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #757575;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    return joblib.load("Model.pkl")

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
        'Total_Stops': sorted(data['Total_Stops'].unique()),
        'Journey_Date': (data['Journey_Day'].min(), data['Journey_Day'].max()),
        'Dep_Hour': (data['Dep_Hour'].min(), data['Dep_Hour'].max()),
        'Duration_Hours': (data['Duration_Hours'].min(), data['Duration_Hours'].max())
    }
    
    # Calculate min/max price for reference
    min_price = data['Price'].min()
    max_price = data['Price'].max()
    avg_price = data['Price'].mean()
    
except Exception as e:
    st.error(f"Error loading data or model: {str(e)}")
    st.stop()

def predict_price(input_data):
    try:
        # Ensure all required columns are present
        required_cols = Model.feature_names_in_
        for col in required_cols:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match model expectations
        input_data = input_data[required_cols]
        
        # Perform prediction using the trained model
        prediction = Model.predict(input_data)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def generate_similar_flights(base_inputs, num_flights=5):
    similar_flights = []
    
    for _ in range(num_flights):
        flight = base_inputs.copy()
        
        # Vary departure time slightly
        dep_hour_variation = np.random.choice([-1, 0, 1, 2])
        new_dep_hour = int(flight['Dep_Hour']) + dep_hour_variation
        if new_dep_hour < 0:
            new_dep_hour = 0
        if new_dep_hour > 23:
            new_dep_hour = 23
        flight['Dep_Hour'] = str(new_dep_hour)
        
        # Vary duration slightly
        duration_variation = np.random.choice([-0.5, 0, 0.5, 1])
        new_duration = flight['Duration_Hours'] + duration_variation
        if new_duration < 1:
            new_duration = 1
        flight['Duration_Hours'] = new_duration
        
        # Calculate new arrival hour
        new_arrival_hour = (new_dep_hour + int(new_duration)) % 24
        flight['Arrival_Hour'] = str(new_arrival_hour)
        
        # Randomly change airline for some flights
        if np.random.random() < 0.3:
            flight['Airline'] = np.random.choice(input_fields['Airline'])
            
        # Randomly change number of stops for some flights
        if np.random.random() < 0.3:
            flight['Total_Stops'] = np.random.choice(input_fields['Total_Stops'])
        
        # Predict price
        flight_df = pd.DataFrame(flight, index=[0])
        price = predict_price(flight_df)
        
        similar_flights.append({
            'Airline': flight['Airline'],
            'Departure': f"{flight['Dep_Hour']}:00",
            'Duration': f"{flight['Duration_Hours']} hrs",
            'Stops': flight['Total_Stops'],
            'Price': price
        })
    
    return similar_flights

def main():
    # Sidebar for about information
    with st.sidebar:
        st.title("✈️ Flight Price Predictor")
        st.markdown("---")
        st.subheader("About")
        st.write("This app predicts flight prices based on various parameters using machine learning.")
        st.markdown("---")
        
        # Show data statistics
        st.subheader("Data Statistics")
        st.write(f"Number of airlines: {len(input_fields['Airline'])}")
        st.write(f"Number of routes: {len(input_fields['Source']) * len(input_fields['Destination'])}")
        st.write(f"Price range: ${min_price:.2f} - ${max_price:.2f}")
        st.write(f"Average price: ${avg_price:.2f}")
        
        st.markdown("---")
        show_advanced = st.checkbox("Show Advanced Options", value=False)
    
    # Main content
    st.markdown("<h1 class='main-header'>✈️ Flight Price Prediction</h1>", unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Predict Price", "Explore Data"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>Enter Flight Details</h2>", unsafe_allow_html=True)
        
        # Create input fields
        inputs = {}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            inputs['Airline'] = st.selectbox("Airline", input_fields['Airline'])
            inputs['Source'] = st.selectbox("Source", input_fields['Source'])
            inputs['Destination'] = st.selectbox("Destination", input_fields['Destination'])
        
        with col2:
            selected_date = st.date_input("Journey Date", min_value=datetime.now(), max_value=datetime.now() + timedelta(days=365))
            inputs['Journey_Day'] = selected_date.day
            inputs['Journey_Month'] = selected_date.month
            
            inputs['Total_Stops'] = st.selectbox("Total Stops", input_fields['Total_Stops'])
        
        with col3:
            selected_dep_time = st.time_input("Departure Time", value=datetime.strptime("10:00", "%H:%M"))
            inputs['Dep_Hour'] = selected_dep_time.strftime('%H')
            
            inputs['Duration_Hours'] = st.slider("Flight Duration (Hours)", 
                                                float(input_fields['Duration_Hours'][0]), 
                                                float(input_fields['Duration_Hours'][1]), 
                                                step=0.5, 
                                                value=2.5)
        
        # Calculate arrival hour
        dep_hour = int(inputs['Dep_Hour'])
        duration = inputs['Duration_Hours']
        arrival_hour = (dep_hour + int(duration)) % 24
        inputs['Arrival_Hour'] = str(arrival_hour)
        
        # Advanced options
        if show_advanced:
            st.markdown("<h3>Advanced Options</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                # Additional features if your model uses them
                pass
            
            with col2:
                # Additional features if your model uses them
                pass
        
        # Predict button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button('Predict Price', type="primary", use_container_width=True)
        
        # Make prediction
        if predict_button:
            with st.spinner('Calculating flight price...'):
                input_data = pd.DataFrame(inputs, index=[0])
                prediction = predict_price(input_data)
                
                if prediction is not None:
                    # Display prediction result
                    st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='text-align: center; color: #1E88E5;'>Predicted Price: ${prediction:.2f}</h2>", unsafe_allow_html=True)
                    
                    # Price evaluation
                    if prediction < avg_price * 0.8:
                        st.success("This is a good deal! The price is below average.")
                    elif prediction > avg_price * 1.2:
                        st.warning("This price is above average for this route.")
                    else:
                        st.info("This price is around the average for this route.")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Generate similar flight options
                    st.markdown("<h3>Similar Flight Options</h3>", unsafe_allow_html=True)
                    similar_flights = generate_similar_flights(inputs)
                    
                    # Create a DataFrame for display
                    similar_df = pd.DataFrame(similar_flights)
                    similar_df = similar_df.sort_values('Price')
                    
                    # Display similar flights
                    st.dataframe(similar_df, use_container_width=True)
                    
                    # Visualize options
                    fig = px.bar(similar_df, x='Airline', y='Price', color='Stops',
                                hover_data=['Departure', 'Duration'],
                                labels={'Price': 'Predicted Price ($)'},
                                title='Similar Flight Options Comparison')
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("<h2 class='sub-header'>Data Exploration</h2>", unsafe_allow_html=True)
        
        # Data exploration options
        analysis_type = st.selectbox("Choose Analysis", [
            "Price by Airline", 
            "Price by Source-Destination", 
            "Price vs Duration",
            "Price vs Stops",
            "Price by Month"
        ])
        
        # Display different visualizations based on selection
        if analysis_type == "Price by Airline":
            airline_prices = data.groupby('Airline')['Price'].mean().reset_index()
            fig = px.bar(airline_prices, x='Airline', y='Price', 
                        title='Average Ticket Price by Airline',
                        labels={'Price': 'Average Price ($)'})
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Price by Source-Destination":
            route_prices = data.groupby(['Source', 'Destination'])['Price'].mean().reset_index()
            route_prices = route_prices.sort_values('Price', ascending=False).head(10)
            fig = px.bar(route_prices, x='Price', y='Source', color='Destination',
                        title='Most Expensive Routes',
                        labels={'Price': 'Average Price ($)'})
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Price vs Duration":
            fig = px.scatter(data, x='Duration_Hours', y='Price', color='Airline',
                            title='Price vs. Flight Duration',
                            labels={'Duration_Hours': 'Duration (Hours)', 'Price': 'Price ($)'})
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Price vs Stops":
            stop_prices = data.groupby('Total_Stops')['Price'].mean().reset_index()
            fig = px.bar(stop_prices, x='Total_Stops', y='Price',
                        title='Average Price by Number of Stops',
                        labels={'Total_Stops': 'Number of Stops', 'Price': 'Average Price ($)'})
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Price by Month":
            month_prices = data.groupby('Journey_Month')['Price'].mean().reset_index()
            fig = px.line(month_prices, x='Journey_Month', y='Price',
                        title='Average Price by Month',
                        labels={'Journey_Month': 'Month', 'Price': 'Average Price ($)'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("<div class='footer'>Flight Price Predictor • Developed with Streamlit and ML</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
