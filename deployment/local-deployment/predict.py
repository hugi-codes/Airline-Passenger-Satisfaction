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
    numerical_columns = ['numerical_feature1', 'numerical_feature2']  # Replace with actual column names
    if len(numerical_columns) > 0:
        vectorized_df[numerical_columns] = scaler.transform(vectorized_df[numerical_columns])

    # Ensure alignment with the model's expected input
    vectorized_df = vectorized_df.reindex(columns=model.feature_names_in_, fill_value=0)

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
        # Preprocess the input
        preprocessed_data = preprocess_input(input_data)

        # Make a prediction
        prediction = model.predict(preprocessed_data)
        prediction_proba = model.predict_proba(preprocessed_data)[:, 1][0]

        # Map prediction to passenger satisfaction status
        if prediction[0] == 1:
            result = "Passenger is satisfied"
        else:
            result = "Passenger is neutral or dissatisfied"

        # Return the result
        return jsonify({
            'prediction': result,
            'probability': prediction_proba
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
