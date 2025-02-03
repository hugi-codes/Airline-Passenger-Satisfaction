# ✈️ Airline-Passenger-Satisfaction
This repo represents the final project of the [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) offered by [Data Talks Club](https://datatalks.club/). For more information on the expected deliverables, please see [this file](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/projects). Please see my Machine Learning Zoomcamp [repo](https://github.com/hugi-codes/Machine-Learning-Zoomcamp/tree/main) for information on the course syllabus and course homework.

🎉 Live Streamlit App

You can check out the live version of the Streamlit app here: [Streamlit App](https://airline-passenger-satisfaction-h9xpcyhyboangktpbakp9g.streamlit.app/)


## Problem Description 
This project focuses on analyzing airline passenger satisfaction based on a dataset derived from a customer satisfaction survey. The dataset includes various attributes such as demographic details, travel characteristics, and passenger ratings on different aspects of their journey (e.g., inflight services, seat comfort, and cleanliness).

The primary objective is to identify the key factors influencing passenger satisfaction and develop a classification model to predict whether a passenger is satisfied or neutral / dissatisfied. Insights from this analysis can help airlines improve service quality and enhance customer experiences.

## 📊 Dataset
The Data is from Kaggle ([link](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data)) and can be downloaded in Python using the `kagglehub` library. For an example of how do this, please check [notebook.ipynb](https://github.com/hugi-codes/Airline-Passenger-Satisfaction/blob/main/notebook.ipynb) or [training.py](https://github.com/hugi-codes/Airline-Passenger-Satisfaction/blob/main/training.py). 

Info on the features (copied from Kaggle):
- **Gender**: Gender of the passengers (**Female**, **Male**)  
- **Customer Type**: The customer type (**Loyal customer**, **Disloyal customer**)  
- **Age**: The actual age of the passengers  
- **Type of Travel**: Purpose of the flight of the passengers (**Personal Travel**, **Business Travel**)  
- **Class**: Travel class in the plane of the passengers (**Business**, **Eco**, **Eco Plus**)  
- **Flight Distance**: The flight distance of this journey  
- **Inflight Wifi Service**: Satisfaction level of the inflight Wi-Fi service (**0**: Not Applicable, **1-5**)  
- **Departure/Arrival Time Convenient**: Satisfaction level of departure/arrival time convenience (**1-5**)  
- **Ease of Online Booking**: Satisfaction level of online booking (**1-5**)  
- **Gate Location**: Satisfaction level of gate location (**1-5**)  
- **Food and Drink**: Satisfaction level of food and drink (**1-5**)  
- **Online Boarding**: Satisfaction level of online boarding (**1-5**)  
- **Seat Comfort**: Satisfaction level of seat comfort (**1-5**)  
- **Inflight Entertainment**: Satisfaction level of inflight entertainment (**1-5**)  
- **On-Board Service**: Satisfaction level of on-board service (**1-5**)  
- **Leg Room Service**: Satisfaction level of leg room service (**1-5**)  
- **Baggage Handling**: Satisfaction level of baggage handling (**1-5**)  
- **Check-In Service**: Satisfaction level of check-in service (**1-5**)  
- **Inflight Service**: Satisfaction level of inflight service (**1-5**)  
- **Cleanliness**: Satisfaction level of cleanliness (**1-5**)  
- **Departure Delay in Minutes**: Minutes delayed during departure  
- **Arrival Delay in Minutes**: Minutes delayed during arrival  
- **Satisfaction**: Airline satisfaction level. This is the target feature. (**Satisfaction**, **Neutral or Dissatisfaction**)  


## Setup to reproduce the project

#### Clone this Git Repository
```sh
git clone <repository-url>
cd <repository-name>
```

#### Dependency Management with Poetry

Install Poetry if you haven't already:
```sh
curl -sSL https://install.python-poetry.org | python3 -
```

Create a virtual environment and install dependencies:
```sh
poetry install
```

Activate the virtual environment:
```sh
poetry shell
```

#### Install Docker
Docker is required for deployment. Install it using the official instructions:

For Linux:
```sh
sudo apt-get update
sudo apt-get install -y docker.io
```

For macOS (using Homebrew):
```sh
brew install --cask docker
```

For Windows:
Download and install Docker Desktop from [Docker's official website](https://www.docker.com/products/docker-desktop).


## Deployment
For this project I decided to explore two different approaches to deployment:
1) Local deployment with Docker and Flask
2) Web deployment with Docker and Streamlit (Free Streamlit Community Cloud)

More details are available in the respective folders, inside the `deployment` directory.