import requests

# Define the URL of the Flask app endpoint
url = "http://127.0.0.1:9696/predict"

# Create a sample JSON payload with input data

airline_passenger_data = {
    "Gender": "Male",
    "Customer Type": "Loyal Customer",  
    "Age": 32,
    "Type of Travel": "Business travel",
    "Class": "Business",
    "Flight Distance": 2000,
    "Inflight wifi service": 5,
    "Departure/Arrival time convenient": 5,
    "Ease of Online booking": 5,
    "Gate location": 5,
    "Food and drink": 5,
    "Online boarding": 5,
    "Seat comfort": 5,
    "Inflight entertainment": 5,
    "On-board service": 5,
    "Leg room service": 5,
    "Baggage handling": 5,
    "Checkin service": 5,
    "Inflight service": 5,
    "Cleanliness": 5,
    "Departure Delay in Minutes": 0,
    "Arrival Delay in Minutes": 0.0
}

airline_passenger_data_terrible = {
    "Gender": "Female",
    "Customer Type": "Disloyal Customer",  
    "Age": 45,
    "Type of Travel": "Personal Travel",
    "Class": "Economy",
    "Flight Distance": 3500,
    "Inflight wifi service": 1,
    "Departure/Arrival time convenient": 1,
    "Ease of Online booking": 1,
    "Gate location": 1,
    "Food and drink": 1,
    "Online boarding": 1,
    "Seat comfort": 1,
    "Inflight entertainment": 1,
    "On-board service": 1,
    "Leg room service": 1,
    "Baggage handling": 1,
    "Checkin service": 1,
    "Inflight service": 1,
    "Cleanliness": 1,
    "Departure Delay in Minutes": 120,
    "Arrival Delay in Minutes": 90.0
}



# Send a POST request to the Flask app
try:
    response = requests.post(url, json=airline_passenger_data_terrible)
    # Check if the request was successful
    if response.status_code == 200:
        print("Response from Flask app:")
        print(response.json())
    else:
        print(f"Error: Received status code {response.status_code}")
        print(response.json())
except requests.exceptions.RequestException as e:
    print(f"An error occurred while sending the request: {e}")
