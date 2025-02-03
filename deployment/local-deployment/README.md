# Dockerized Flask Prediction App

This repository contains a Flask application that serves a machine learning model for predictions. The application is packaged into a Docker container for easy local deployment.

## Getting Started

### 1. Build the Docker Image
Ensure your terminal is in the directory containing the `Dockerfile` and run:

```bash
docker build -t flask-predictor:v1 .
```

### 2. Run the Docker Container
Once the Docker image is built, start a container with:

```bash
docker run -it --rm -p 9696:9696 flask-predictor:v1
```

This command runs the Flask app inside a container and exposes it on port `9696`. You can access it at:

```
http://localhost:9696
```

### 3. Make Predictions
Use `score_passenger.py` to send a request to the Flask app and receive a model prediction.

Run the script with:

```bash
python score_passenger.py
```

You can modify `score_passenger.py` to test different inputs.

---

## Repository Contents
- `Dockerfile` - Instructions to build the Docker image.
- `final_trained_model.pkl` - Serialized trained model.
- `poetry.lock` - Dependency lock file.
- `predict.py` - Flask app handling predictions.
- `pyproject.toml` - Project dependencies and configurations.
- `scaler.pkl` - Scaler for input data transformation.
- `score_passenger.py` - Script for sending test requests.
- `vectorizer.pkl` - Vectorizer for feature processing.

For more details, check `predict.py` to understand how the Flask app processes requests.

Happy deploying! 🚀
