import pandas as pd
import streamlit as st
import joblib
import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #004D40;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .info-text {
        font-size: 16px;
        color: #555;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #E3F2FD;
        text-align: center;
        margin-top: 20px;
    }
    .sidebar-content {
        padding: 20px;
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
        'Arrival_Hour': (data['Arrival_Hour'].min(), data['Arrival_Hour'].max()),
        'Duration_Hours': (data['Duration_Hours'].min(), data['Duration_Hours'].max())
    }
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    st.stop()

def create_feature_importance_chart(model, features):
    try:
        # Get feature importance from the model (assuming it's a tree-based model)
        importances = model.feature_importances_
        # Sort features by importance
        indices = np.argsort(importances)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance for Flight Price Prediction')
        
        return fig
    except:
        # If the model doesn't have feature_importances_ attribute
        return None

def predict_price(input_data):
    try:
        # Perform prediction using the trained model
        prediction = Model.predict(input_data)
        return prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

def main():
    # Sidebar for app navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=✈️", width=150)
        st.markdown("## Navigation")
        page = st.radio("", ["Make Prediction", "About", "Data Insights"])
    
    if page == "Make Prediction":
        # Main prediction page
        st.markdown('<div class="main-header">✈️ Flight Price Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-text">Enter flight details below to get a price estimate.</div>', unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            # Create input fields
            inputs = {}
            
            st.markdown('<div class="sub-header">Flight Details</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            # Airline, Source, Destination selection
            with col1:
                inputs['Airline'] = st.selectbox("Airline", input_fields['Airline'])
                inputs['Source'] = st.selectbox("Source", input_fields['Source'])
            
            with col2:
                inputs['Destination'] = st.selectbox("Destination", input_fields['Source'], 
                                                     index=min(1, len(input_fields['Source'])-1))
                inputs['Total_Stops'] = st.selectbox("Number of Stops", input_fields['Total_Stops'])
            
            st.markdown('<div class="sub-header">Date and Time</div>', unsafe_allow_html=True)
            col3, col4 = st.columns(2)
            
            # Date and time selectors
            with col3:
                selected_date = st.date_input("Journey Date", 
                                             min_value=datetime.date.today(),
                                             max_value=datetime.date.today() + datetime.timedelta(days=365))
                inputs['Journey_Day'] = selected_date.day
                inputs['Journey_Month'] = selected_date.month
                
                dep_time = st.time_input("Departure Time", datetime.time(9, 0))
                inputs['Dep_Hour'] = int(dep_time.strftime('%H'))
                
            with col4:
                # Duration slider with better formatting
                duration = st.slider("Flight Duration (hours)", 
                                    min_value=float(max(1.0, data['Duration_Hours'].min())), 
                                    max_value=float(min(24.0, data['Duration_Hours'].max())), 
                                    value=float(2.5),
                                    step=0.5)
                inputs['Duration_Hours'] = duration
                
                # Calculate and display arrival time
                arrival_hour = (dep_time.hour + int(duration)) % 24
                arrival_min = dep_time.minute + int((duration % 1) * 60)
                if arrival_min >= 60:
                    arrival_hour += 1
                    arrival_min -= 60
                arrival_hour = arrival_hour % 24
                
                arrival_time = datetime.time(arrival_hour, arrival_min)
                st.write(f"Estimated Arrival: {arrival_time.strftime('%H:%M')}")
                inputs['Arrival_Hour'] = arrival_hour
            
            # Submit button
            submitted = st.form_submit_button("Predict Price")
        
        # Make prediction when form is submitted
        if submitted:
            input_data = pd.DataFrame(inputs, index=[0])
            
            # Show a spinner while predicting
            with st.spinner('Calculating price...'):
                prediction = predict_price(input_data)
                
            if prediction is not None:
                # Display prediction with nice formatting
                st.markdown(f'<div class="prediction-box"><h2>Estimated Fare</h2><h1>${prediction:.2f}</h1></div>', 
                           unsafe_allow_html=True)
                
                # Add some context to the prediction
                st.info(f"This prediction is based on historical data for {inputs['Airline']} flights from {inputs['Source']} to {inputs['Destination']} with {inputs['Total_Stops']} stops.")
    
    elif page == "About":
        st.markdown('<div class="main-header">About This App</div>', unsafe_allow_html=True)
        st.markdown("""
        This application predicts flight prices based on various parameters like:
        - Airline carrier
        - Source and destination airports
        - Number of stops
        - Journey date and time
        - Flight duration
        
        The prediction model was trained on historical flight data and uses machine learning to estimate prices.
        
        **How to use:**
        1. Enter your flight details in the form
        2. Click "Predict Price" to get an estimate
        3. Explore different combinations to find the best deals
        
        **Note:** This is a prediction tool and actual prices may vary.
        """)
        
    elif page == "Data Insights":
        st.markdown('<div class="main-header">Flight Price Insights</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sub-header">What affects flight prices?</div>', unsafe_allow_html=True)
        
        # Create and display feature importance chart
        try:
            features = list(data.columns)
            fig = create_feature_importance_chart(Model, features)
            if fig:
                st.pyplot(fig)
            else:
                st.info("Feature importance visualization is not available for this model type.")
        except:
            st.warning("Could not generate insights visualization.")
        
        # Display some statistics
        st.markdown('<div class="sub-header">Price Statistics by Airline</div>', unsafe_allow_html=True)
        if 'Price' in data.columns and 'Airline' in data.columns:
            airline_stats = data.groupby('Airline')['Price'].agg(['mean', 'min', 'max']).reset_index()
            airline_stats.columns = ['Airline', 'Average Price', 'Minimum Price', 'Maximum Price']
            airline_stats = airline_stats.sort_values('Average Price')
            st.dataframe(airline_stats.style.format({'Average Price': '${:.2f}', 
                                                   'Minimum Price': '${:.2f}', 
                                                   'Maximum Price': '${:.2f}'}),
                        use_container_width=True)
        else:
            st.info("Price statistics are not available in the dataset.")

if __name__ == '__main__':
    main()
