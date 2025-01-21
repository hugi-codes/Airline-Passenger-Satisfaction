# This script loads the model pickle file and the preprocessors pickle files.
# Further, this script contains the Flask App, via which the trained model is served.
# The idea is that when running this script, the Flask App runs and awaits requests.
# If new data is incoming, this data will run through the preprocessing pipeline and then is fed to the model.
# This model then makes a prediction.