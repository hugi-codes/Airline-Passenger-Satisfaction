import requests

# Define the URL of the Flask app endpoint
url = "http://127.0.0.1:9696/predict"

# Create a sample JSON payload with input data
# Updated to include categorical columns

airline_passenger_data = {
    "Gender": "Male",  # Gender doesn't impact prediction but included for consistency.
    "Customer Type": "Loyal Customer",  # Loyal customer is more likely to have a perfect experience.
    "Type of Travel": "Business travel",  # Business travel generally correlates with better experiences.
    "Class": "Business",  # Business class passengers typically experience better service.
    "Age": 40,
    "Flight Distance": 8000,  # Long-haul flight, implying a more comfortable and premium service.
    "Inflight wifi service": 5,  # Excellent wifi service (perfect experience).
    "Departure/Arrival time convenient": 5,  # Extremely convenient departure and arrival times.
    "Ease of Online booking": 5,  # Very easy and smooth online booking process.
    "Gate location": 5,  # Very convenient gate location.
    "Food and drink": 5,  # Excellent food and drink service.
    "Online boarding": 5,  # Very smooth and easy online boarding.
    "Seat comfort": 5,  # Very comfortable seating.
    "Inflight entertainment": 5,  # Excellent inflight entertainment options.
    "On-board service": 5,  # Top-tier on-board service.
    "Leg room service": 5,  # Plenty of leg room.
    "Baggage handling": 5,  # Fast and efficient baggage handling.
    "Checkin service": 5,  # Very smooth and fast check-in process.
    "Inflight service": 5,  # Excellent inflight service.
    "Cleanliness": 5,  # Immaculately clean environment.
    "Departure Delay in Minutes": 0,  # No delays, ideal scenario.
    "Arrival Delay in Minutes": 0  # No delays, ideal scenario.
}





# Send a POST request to the Flask app
try:
    response = requests.post(url, json=airline_passenger_data)
    # Check if the request was successful
    if response.status_code == 200:
        print("Response from Flask app:")
        print(response.json())
    else:
        print(f"Error: Received status code {response.status_code}")
        print(response.json())
except requests.exceptions.RequestException as e:
    print(f"An error occurred while sending the request: {e}")
