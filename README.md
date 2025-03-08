# Airline Price Prediction

This repository implements a machine learning model for predicting airline ticket prices based on various features such as airline, source, destination, total stops, journey date, departure time, and duration.

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-brightgreen)](https://share.streamlit.io/gamal-abdelhakm/Airline_Price_Prediction/main/app.py)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Data Sources](#data-sources)
- [Technologies Used](#technologies-used)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/gamal-abdelhakm/Airline_Price_Prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Airline_Price_Prediction
    ```
3. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
4. Activate the virtual environment:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On MacOS/Linux:
        ```bash
        source venv/bin/activate
        ```
5. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to:
    ```
    http://localhost:8501
    ```

## Features

### Data Preprocessing and Exploration
- Conduct exploratory data analysis (EDA) on the dataset to understand the features and their distributions.
- Preprocess the data by handling missing values, duplicates, and feature engineering.

### Airline Price Prediction
- Enter flight details such as airline, source, destination, total stops, journey date, departure time, and duration to get an estimated price.
- Interactive visualizations to understand the relationship between different features and the price.

### Visualizations
- Distribution plots for numerical features.
- Box plots for outlier detection.
- Bar charts and pie charts for categorical features.
- Correlation heatmap and scatter plots for bivariate analysis.

## Data Sources

This system uses the following datasets:
- **Data_Train.xlsx**: Training data with flight details and prices.

## Technologies Used

- **Streamlit**: Web interface framework.
- **Scikit-learn**: Machine learning algorithms for training and prediction.
- **Pandas**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: Data visualizations.
