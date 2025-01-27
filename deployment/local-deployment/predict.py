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



# ================================================================
# Loading the pkl files: 
# --> Preprocessors (dict vectorizer and scaler preprocessors)
# --> Trained model

import pickle
import os

# Define paths to the pickle files inside the container
VECTORIZER_PATH = os.path.join(os.getcwd(), "vectorizer.pkl")
SCALER_PATH = os.path.join(os.getcwd(), "scaler.pkl")
MODEL_PATH = os.path.join(os.getcwd(), "final_trained_model.pkl")

# Function to load a pickle file
def load_pickle(file_path):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")
    except pickle.UnpicklingError:
        raise Exception(f"Error loading pickle file: {file_path}")

# Load the pickle files
try:
    vectorizer = load_pickle(VECTORIZER_PATH)
    scaler = load_pickle(SCALER_PATH)
    model = load_pickle(MODEL_PATH)
    print("All pickle files loaded successfully.")
except Exception as e:
    print(f"Error loading pickle files: {e}")
# ================================================================

# Uncomment the following code, to read the pkl files from local
# instead of from inside Docker container. But as the Flask app
# in this file is intended to be run as a Docker container (therefore,
# the files are loaded from inside the Docker container instead of from 
# local) the following commented code does not need to be executed.
# The code above loads the desired files from inside the container.

# Get the directory of the current script
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Load the model and preprocessors
# try:
#     # Load the scaler
#     scaler_path = os.path.join(script_dir, 'scaler.pkl')
#     with open(scaler_path, 'rb') as f:
#         scaler = pickle.load(f)

#     # Load the vectorizer
#     vectorizer_path = os.path.join(script_dir, 'vectorizer.pkl')
#     with open(vectorizer_path, 'rb') as f:
#         vectorizer = pickle.load(f)

#     # Load the trained model
#     model_path = os.path.join(script_dir, 'final_trained_model.pkl')  # 'final_trained_model.pkl', best_rf_model
#     with open(model_path, 'rb') as f:
#         model = pickle.load(f)

#     print("Model and preprocessors loaded successfully.")


# except FileNotFoundError as e:
#     print(f"File not found: {e}")
#     raise

# ================================================================

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

        print(preprocessed_data)

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