# This script loads the model pickle file and the preprocessors pickle files.
# Further, this script contains the Flask App, via which the trained model is served.
# The idea is that when running this script, the Flask App runs and awaits requests.
# If new data is incoming, this data will run through the preprocessing pipeline and then is fed to the model.
# This model then makes a prediction.
import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model and preprocessors
try:
    # Load the scaler
    scaler_path = os.path.join(script_dir, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Load the vectorizer
    vectorizer_path = os.path.join(script_dir, 'vectorizer.pkl')
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    # Load the trained model
    model_path = os.path.join(script_dir, 'final_trained_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print("Model and preprocessors loaded successfully.")


except FileNotFoundError as e:
    print(f"File not found: {e}")
    raise

# Preprocessing function
def preprocess_input(data):
    """
    Preprocess incoming JSON data to align with the model's expectations.
    Includes:
    - Vectorization of categorical features
    - Scaling of numerical features
    """
    # Convert incoming data to a DataFrame
    df = pd.DataFrame([data])

    # Vectorize categorical features
    vectorized_data = vectorizer.transform(df.to_dict(orient='records'))
    vectorized_df = pd.DataFrame(vectorized_data, columns=vectorizer.get_feature_names_out())

    # Scale numerical features
    numerical_columns =  ['Age', 'Flight Distance', 'Inflight wifi service', 
                     'Departure/Arrival time convenient', 'Ease of Online booking', 
                     'Gate location', 'Food and drink', 'Online boarding', 
                     'Seat comfort', 'Inflight entertainment', 'On-board service', 
                     'Leg room service', 'Baggage handling', 'Checkin service', 
                     'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 
                     'Arrival Delay in Minutes']

    if len(numerical_columns) > 0:
        try:
            vectorized_df[numerical_columns] = scaler.transform(df[numerical_columns])
        except Exception as e:
            print("Error during scaling numerical columns:", e)
            print("Numerical columns in input:", df[numerical_columns].columns.tolist())
            print("Expected numerical columns:", numerical_columns)


    # Debug: Print preprocessed data
    print("Preprocessed DataFrame:")
    print(vectorized_df.head())

    return vectorized_df

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict using the loaded model.
    Accepts JSON input with features, preprocesses it, and returns predictions.
    """
    input_data = request.json
    if not input_data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        print("Expected columns by the vectorizer:", vectorizer.get_feature_names_out())

        # Preprocess the input
        preprocessed_data = preprocess_input(input_data)

        # Make a prediction
        prediction = model.predict(preprocessed_data)
        prediction_proba = model.predict_proba(preprocessed_data)[:, 1][0]

        # Define a threshold (0.5 is the common default threshold for classification problems)
        threshold = 0.4

        # Map probability to satisfaction status based on the threshold
        if prediction_proba >= threshold:
            result = "Passenger is satisfied"
        else:
            result = "Passenger is neutral or dissatisfied"

        # Return the result
        return jsonify({
            'prediction': result,
            'probability': prediction_proba,
            'threshold': threshold
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=9696)