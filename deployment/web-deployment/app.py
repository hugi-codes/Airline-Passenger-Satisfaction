import streamlit as st
import pickle
import numpy as np

# Path to the trained model
model_path = 'deployment/web_deployment/final_trained_model.pkl'

# Load the trained model
with open(model_path, 'rb') as f_in:
    model = pickle.load(f_in)

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
        "Inflight wifi service": st.sidebar.slider("Inflight WiFi Service", 0, 5, 3),
        "Departure/Arrival time convenient": st.sidebar.slider("Departure/Arrival Time Convenient", 0, 5, 3),
        "Ease of Online booking": st.sidebar.slider("Ease of Online Booking", 0, 5, 3),
        "Gate location": st.sidebar.slider("Gate Location", 0, 5, 3),
        "Food and drink": st.sidebar.slider("Food and Drink", 0, 5, 3),
        "Online boarding": st.sidebar.slider("Online Boarding", 0, 5, 3),
        "Seat comfort": st.sidebar.slider("Seat Comfort", 0, 5, 3),
        "Inflight entertainment": st.sidebar.slider("Inflight Entertainment", 0, 5, 3),
        "On-board service": st.sidebar.slider("On-board Service", 0, 5, 3),
        "Leg room service": st.sidebar.slider("Leg Room Service", 0, 5, 3),
        "Baggage handling": st.sidebar.slider("Baggage Handling", 0, 5, 3),
        "Checkin service": st.sidebar.slider("Check-in Service", 0, 5, 3),
        "Inflight service": st.sidebar.slider("Inflight Service", 0, 5, 3),
        "Cleanliness": st.sidebar.slider("Cleanliness", 0, 5, 3),
        "Departure Delay in Minutes": st.sidebar.number_input("Departure Delay (Minutes)", min_value=0, value=0),
        "Arrival Delay in Minutes": st.sidebar.number_input("Arrival Delay (Minutes)", min_value=0, value=0)
    }
    return input_data

user_input = get_user_input()

# Predict Button
if st.sidebar.button("Predict"):
    sample_array = np.array([list(user_input.values())])
    y_pred = model.predict(sample_array)
    
    satisfaction_label = "Satisfied" if y_pred[0] == 1 else "Not Satisfied"
    
    st.success(f"The predicted passenger satisfaction is: **{satisfaction_label}**")
    st.balloons()
else:
    st.markdown("Enter passenger details in the sidebar and click 'Predict' to see results.")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit. @hugi-codes")
