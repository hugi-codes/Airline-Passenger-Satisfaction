# Use a Python 3.12 base image 
FROM python:3.12-slim

# Set working directory in the container
WORKDIR /app

# Install dependencies for Poetry (and some system dependencies for Flask)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to the PATH
ENV PATH="/root/.local/bin:$PATH"

# Verify if Poetry is installed correctly 
RUN poetry --version

# Copy pyproject.toml and poetry.lock to the container
COPY pyproject.toml poetry.lock /app/

# Install dependencies using Poetry
RUN poetry install --only=main --no-interaction

# Copy the Flask app and pickle files into the container
COPY predict.py /app/ 

COPY vectorizer.pkl /app/
COPY scaler.pkl /app/
COPY final_trained_model.pkl /app/

# Expose the port that the Flask app will run on
EXPOSE 9696

# Set the command to run the Flask app inside the Poetry virtual environment
CMD ["poetry", "run", "python", "predict.py"] 
