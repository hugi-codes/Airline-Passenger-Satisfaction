import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Path to the trained model, scaler, and vectorizer
model_path = 'deployment/web-deployment/final_trained_model.pkl'
scaler_path = 'deployment/web-deployment/scaler.pkl'
vectorizer_path = 'deployment/web-deployment/vectorizer.pkl'

# Load the trained model, scaler, and vectorizer
with open(model_path, 'rb') as f_in:
    model = pickle.load(f_in)

with open(scaler_path, 'rb') as f_in:
    scaler = pickle.load(f_in)

with open(vectorizer_path, 'rb') as f_in:
    vectorizer = pickle.load(f_in)

# Define the Streamlit App
st.set_page_config(
    page_title="Airline Passenger Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("✈️ Airline Passenger Satisfaction Predictor")
st.markdown("""
This app predicts passenger satisfaction based on flight experience.
Simply enter the details below and get insights into the predicted satisfaction level.
""")

# Sidebar Input Fields
st.sidebar.header("Passenger Details")
st.sidebar.markdown("Provide the following details for prediction:")

def get_user_input():
    input_data = {
        "Gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
        "Customer Type": st.sidebar.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"]),
        "Age": st.sidebar.number_input("Age", min_value=1, max_value=100, value=30),
        "Type of Travel": st.sidebar.selectbox("Type of Travel", ["Personal Travel", "Business Travel"]),
        "Class": st.sidebar.selectbox("Class", ["Business", "Eco", "Eco Plus"]),
        "Flight Distance": st.sidebar.number_input("Flight Distance", min_value=0, value=1000),
        "Inflight wifi service": st.sidebar.slider("Inflight WiFi Service", 1, 5, 3),  # Changed slider range from 0-5 to 1-5
        "Departure/Arrival time convenient": st.sidebar.slider("Departure/Arrival Time Convenient", 1, 5, 3),  # Changed slider range from 0-5 to 1-5
        "Ease of Online booking": st.sidebar.slider("Ease of Online Booking", 1, 5, 3),  # Changed slider range from 0-5 to 1-5
        "Gate location": st.sidebar.slider("Gate Location", 1, 5, 3),  # Changed slider range from 0-5 to 1-5
        "Food and drink": st.sidebar.slider("Food and Drink", 1, 5, 3),  # Changed slider range from 0-5 to 1-5
        "Online boarding": st.sidebar.slider("Online Boarding", 1, 5, 3),  # Changed slider range from 0-5 to 1-5
        "Seat comfort": st.sidebar.slider("Seat Comfort", 1, 5, 3),  # Changed slider range from 0-5 to 1-5
        "Inflight entertainment": st.sidebar.slider("Inflight Entertainment", 1, 5, 3),  # Changed slider range from 0-5 to 1-5
        "On-board service": st.sidebar.slider("On-board Service", 1, 5, 3),  # Changed slider range from 0-5 to 1-5
        "Leg room service": st.sidebar.slider("Leg Room Service", 1, 5, 3),  # Changed slider range from 0-5 to 1-5
        "Baggage handling": st.sidebar.slider("Baggage Handling", 1, 5, 3),  # Changed slider range from 0-5 to 1-5
        "Checkin service": st.sidebar.slider("Check-in Service", 1, 5, 3),  # Changed slider range from 0-5 to 1-5
        "Inflight service": st.sidebar.slider("Inflight Service", 1, 5, 3),  # Changed slider range from 0-5 to 1-5
        "Cleanliness": st.sidebar.slider("Cleanliness", 1, 5, 3),  # Changed slider range from 0-5 to 1-5
        "Departure Delay in Minutes": st.sidebar.number_input("Departure Delay (Minutes)", min_value=0, value=0),
        "Arrival Delay in Minutes": st.sidebar.number_input("Arrival Delay (Minutes)", min_value=0, value=0)
    }
    return input_data

user_input = get_user_input()


# Preprocessing function
def preprocess_input(data):
    """
    Preprocess incoming data to align with the model's expectations.
    Includes:
    - Vectorization of categorical features
    - Scaling of numerical features
    """
    # Convert incoming data to a DataFrame
    df = pd.DataFrame([data])

    # Define the numerical and categorical columns
    numerical_columns = [
        'Age', 'Flight Distance', 'Inflight wifi service', 
        'Departure/Arrival time convenient', 'Ease of Online booking', 
        'Gate location', 'Food and drink', 'Online boarding', 
        'Seat comfort', 'Inflight entertainment', 'On-board service', 
        'Leg room service', 'Baggage handling', 'Checkin service', 
        'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 
        'Arrival Delay in Minutes'
    ]

    categorical_columns = [
        'Gender', 'Customer Type', 'Type of Travel', 'Class'
    ]

    # Converting numerical columns to numeric dtype
    df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')

    # Convert categorical columns to 'category' dtype
    df[categorical_columns] = df[categorical_columns].astype('category')

    # Split the data into numerical and categorical features
    numerical_data = df[numerical_columns]
    categorical_data = df[categorical_columns]

    # Preprocess the numerical data (apply scaling)
    numerical_data_scaled = scaler.transform(numerical_data)

    # Preprocess the categorical data (apply dictionary vectorization)
    categorical_data_dict = categorical_data.to_dict(orient='records')  # Convert to list of dictionaries
    categorical_data_vectorized = vectorizer.transform(categorical_data_dict)

    # Stack the preprocessed data (numerical + categorical) horizontally
    X_preprocessed = np.hstack([numerical_data_scaled, categorical_data_vectorized])

    return X_preprocessed

# Predict Button
if st.sidebar.button("Predict"):
    # Preprocess the user input
    sample_array = preprocess_input(user_input)
    
    # Check the shape of the input array
    st.write("Shape of input array:", sample_array.shape)
    
    # Make prediction
    y_pred = model.predict(sample_array)
    
    # Determine satisfaction
    satisfaction_label = "Satisfied" if y_pred[0] == 1 else "Not Satisfied"
    
    st.success(f"The predicted passenger satisfaction is: **{satisfaction_label}**")
    st.balloons()
else:
    st.markdown("Enter passenger details in the sidebar and click 'Predict' to see results.")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit. @hugi-codes")
