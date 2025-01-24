# The following code is notebook.ipynb exported as a Python script


# Note: code for EDA has been omitted. Only code necessary for model training has been kept from notebook.ipynb
# Exporting notebook.ipynb to train.py is a requirement for mid-term project. 
# For more info on the project requirements, please check the links provided in README.md



# ## Importing packages

import numpy as np
import kagglehub
import seaborn as sns
import matplotlib.pyplot as plt


# ## Loading the dataset


# Download latest version
path = kagglehub.dataset_download("teejmahal20/airline-passenger-satisfaction")

print("Path to dataset files:", path)


import pandas as pd

# Path to the extracted dataset
file_path = '/home/timhug/.cache/kagglehub/datasets/teejmahal20/airline-passenger-satisfaction/versions/1/train.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)



# ## Data Cleaning
# 1)  Delete rows with missing values in Arrival Delay in Minutes column
# 2)  Impute mean where 0_5 cols have value 0
# 3)  Omitting outlier values 
# 4)  Omitting columns not needed for modeling


# 1 Delete rows with missing values in Arrival Delay in Minutes column

# This approach is chosen, as this leads only to a drop in 0.3% of the rows, which is acceptable. 
# If a significant proportion of the data was lost with this approach, it would distort the data and possibly impact model performance.
# ======================================================================================================================


df_cleaned_1 = df.dropna(subset=['Arrival Delay in Minutes'])
# Calculate the number of rows before and after the cleaning
original_row_count = len(df)
cleaned_row_count = len(df_cleaned_1)

# Calculate the number of rows dropped
dropped_row_count = original_row_count - cleaned_row_count

# Calculate the percentage of dropped rows
dropped_percentage = (dropped_row_count / original_row_count) * 100

# Print the result
print(f"Percentage of rows dropped: {dropped_percentage:.2f}%")


# 2. Impute mean where 0_5 cols have value 0

# This approach has been chosen, as in Kaggle discussions about this dataset it is mentioned that 0 values for the
# columns which represent ratings of 1-5 (5 being the best) are likely to signify a missing value.
# I could not find further information on this, so I assume this information from Kaggle discussions is True. 
# ======================================================================================================================

# List of columns to process
columns_to_process = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling", "Checkin service", 
    "Inflight service", "Cleanliness"
]

# Create a copy of df_cleaned_1 to avoid modifying the original DataFrame
df_cleaned_2 = df_cleaned_1.copy()

# Convert the selected columns to float
df_cleaned_2[columns_to_process] = df_cleaned_2[columns_to_process].astype(float)

# Calculate the mean for each column and round it to 2 decimal places
means = df_cleaned_2[columns_to_process].mean().round(2)

# Impute 0's with the rounded mean of the respective column
for column in columns_to_process:
    df_cleaned_2[column] = df_cleaned_2[column].replace(0, means[column])
    


# Check whether the means are imputed correctly:
# Filter rows where 'Inflight wifi service' is 2.73 (which is the mean for this column)
filtered_rows = df_cleaned_2[df_cleaned_2["Inflight wifi service"] == 2.73]

# Display the filtered rows
filtered_rows = filtered_rows[["id", "Inflight wifi service"]]
filtered_rows
# This output just serves to see that the mean rounded to two decimals was imputed correctly, where previously the value was 0. 


# 3. Omitting Outliers in the columns "Departure Delay in Minutes" and "Arrival Delay in Minutes"

# Only Outliers of these columns are omitted, as these are the only columns 
# where the density plots and boxplots show the presence of outliers.
# ======================================================================================================================

# Increase the multiplier for IQR to define outliers more strictly (e.g., 2.0 or 2.5 instead of 1.5)
IQR_multiplier = 7  # 7 is extremely strict, in order to avoid the removal of too many rows. 

# Calculate the IQR for both columns using df_cleaned_2
Q1_departure = df_cleaned_2["Departure Delay in Minutes"].quantile(0.25)
Q3_departure = df_cleaned_2["Departure Delay in Minutes"].quantile(0.75)
IQR_departure = Q3_departure - Q1_departure

Q1_arrival = df_cleaned_2["Arrival Delay in Minutes"].quantile(0.25)
Q3_arrival = df_cleaned_2["Arrival Delay in Minutes"].quantile(0.75)
IQR_arrival = Q3_arrival - Q1_arrival

# Calculate the lower and upper bounds for both columns using the increased multiplier
lower_bound_departure = Q1_departure - IQR_multiplier * IQR_departure
upper_bound_departure = Q3_departure + IQR_multiplier * IQR_departure

lower_bound_arrival = Q1_arrival - IQR_multiplier * IQR_arrival
upper_bound_arrival = Q3_arrival + IQR_multiplier * IQR_arrival

# Filter the DataFrame to omit the outliers and save the result in df_cleaned_3
df_cleaned_3 = df_cleaned_2[
    (df_cleaned_2["Departure Delay in Minutes"] >= lower_bound_departure) &
    (df_cleaned_2["Departure Delay in Minutes"] <= upper_bound_departure) &
    (df_cleaned_2["Arrival Delay in Minutes"] >= lower_bound_arrival) &
    (df_cleaned_2["Arrival Delay in Minutes"] <= upper_bound_arrival)
]

# Calculate the number of rows before and after filtering
initial_rows = len(df_cleaned_2)
final_rows = len(df_cleaned_3)

# Calculate the number of rows removed
rows_deleted = initial_rows - final_rows

# Calculate the percentage of rows deleted
percentage_deleted = (rows_deleted / initial_rows) * 100

# Print the results
print(f"Absolute number of rows deleted: {rows_deleted}")
print(f"Percentage of rows deleted: {percentage_deleted:.2f}%")


# 4. Omitting columns not needed for modeling

df_cleaned_4 = df_cleaned_3.drop(columns=['Unnamed: 0', 'id'])


# ## Preprocessing
# * Min Max Scaling (for later use of distance-based algos)
# * One-hot encoding 
# 


df_cleaned_4.info()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import DictVectorizer

# Define the target column
target_column = 'satisfaction'

# Separate numerical and categorical columns
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

# Step 1: Split data into features and target
X = df_cleaned_4.drop(columns=[target_column])
y = df_cleaned_4[target_column]

# Step 2: Split data into train+validation and test sets (stratified by target)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 3: Split train+validation into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# Step 4: Preprocess numerical columns with MinMaxScaler
scaler = MinMaxScaler()

# Fit scaler on training data
X_train_numerical_scaled = scaler.fit_transform(X_train[numerical_columns])

# Transform validation and test numerical data
X_val_numerical_scaled = scaler.transform(X_val[numerical_columns])
X_test_numerical_scaled = scaler.transform(X_test[numerical_columns])

# Step 5: Preprocess categorical columns with DictVectorizer
vectorizer = DictVectorizer(sparse=False)

# Convert categorical data to a dictionary format
X_train_categorical_dict = X_train[categorical_columns].to_dict(orient='records')
X_val_categorical_dict = X_val[categorical_columns].to_dict(orient='records')
X_test_categorical_dict = X_test[categorical_columns].to_dict(orient='records')

# Fit vectorizer on training data
X_train_categorical_encoded = vectorizer.fit_transform(X_train_categorical_dict)

# Transform validation and test categorical data
X_val_categorical_encoded = vectorizer.transform(X_val_categorical_dict)
X_test_categorical_encoded = vectorizer.transform(X_test_categorical_dict)

# Step 6: Combine processed numerical and categorical data
import numpy as np

X_train_preprocessed = np.hstack([X_train_numerical_scaled, X_train_categorical_encoded])
X_val_preprocessed = np.hstack([X_val_numerical_scaled, X_val_categorical_encoded])
X_test_preprocessed = np.hstack([X_test_numerical_scaled, X_test_categorical_encoded])

# Print the shapes of the original and split DataFrames
print("Original DataFrame:")
print("X:", X.shape, "y:", y.shape)

print("\nSplit DataFrames:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# Print the shapes of the preprocessed data
print("\nPreprocessed Data Shapes:")
print("X_train_preprocessed:", X_train_preprocessed.shape)
print("X_val_preprocessed:", X_val_preprocessed.shape)
print("X_test_preprocessed:", X_test_preprocessed.shape)

print(X_train_preprocessed)

# ## Modeling

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Step 1: Define the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Step 2: Set up hyperparameter grid for tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [5, 10]
}

# Step 3: Set up GridSearchCV
rf_grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all processors
    verbose=2
)


# Step 4: Fit GridSearchCV on the training data
rf_grid_search.fit(X_train_preprocessed, y_train)

# Step 5: Get the best parameters and best model
best_rf_params  = rf_grid_search.best_params_
best_rf_model = rf_grid_search.best_estimator_

print("Best Parameters:", best_rf_params )

# Step 6: Evaluate the model on the validation set
y_val_pred = best_rf_model.predict(X_val_preprocessed)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation Accuracy: {val_accuracy:.4f}")

# Step 7: (Optional) Evaluate on the test set
y_test_pred = best_rf_model.predict(X_test_preprocessed)
test_accuracy = accuracy_score(y_test, y_test_pred)

# checking predict proba values
predict_proba_values = best_rf_model.predict_proba(X_test_preprocessed)
# Set the print options to avoid scientific notation
np.set_printoptions(suppress=True)

# Format the values to be more readable (as percentages, rounded to 2 decimal places)
formatted_proba_values = np.round(predict_proba_values * 100, 2)

# Print the formatted probability values
print(formatted_proba_values)

print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the model to a .pkl file
with open('best_rf_model.pkl', 'wb') as file:
    pickle.dump(best_rf_model, file)

print("Model has been successfully saved as 'best_rf_model.pkl'.")




print("Now training logistic regression")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Step 1: Define the Logistic Regression model
logreg_model = LogisticRegression(random_state=42, solver='liblinear')  # `liblinear` is suitable for small to medium-sized datasets

# Step 2: Set up hyperparameter grid for tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2']         # Regularization type: L1 or L2
}

# Step 3: Set up GridSearchCV
logreg_grid_search = GridSearchCV(
    estimator=logreg_model,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all processors
    verbose=2
)

# Step 4: Fit GridSearchCV on the training data
logreg_grid_search.fit(X_train_preprocessed, y_train)

# Step 5: Get the best parameters and best model
best_logreg_params  = logreg_grid_search.best_params_
best_logreg_model = logreg_grid_search.best_estimator_

print("Best Parameters:", best_logreg_params)

# Step 6: Evaluate the model on the validation set
y_val_pred = best_logreg_model.predict(X_val_preprocessed)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation Accuracy: {val_accuracy:.4f}")

# Step 7: (Optional) Evaluate on the test set
y_test_pred = best_logreg_model.predict(X_test_preprocessed)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")


# Getting the best model
# Two algorithms were trained for which hyper parameters were tuned, respectively:
# --> Random Forest
# --> Logistic Regression

# Now we want to select the best model out of all the models that were trained

models = [
    {
        'name': 'Random Forest',
        'model': best_rf_model,
        'best_params': best_rf_params, 
        'test_accuracy': accuracy_score(y_test, best_rf_model.predict(X_test_preprocessed))
    },
    {
        'name': 'Logistic Regression',
        'model': best_logreg_model,
        'best_params': best_logreg_params,  
        'test_accuracy': accuracy_score(y_test, best_logreg_model.predict(X_test_preprocessed))
    }
]

# Step 2: Identify the best model based on test accuracy
best_model = max(models, key=lambda x: x['test_accuracy'])

# Step 3: Print the best model details
print("Best Overall Model:")
print(f"Algorithm: {best_model['name']}")
print(f"Best Hyperparameters: {best_model['best_params']}")
print(f"Test Accuracy: {best_model['test_accuracy']:.4f}")



# To make the best out of the entire available data (df_cleaned_4): 
# * Split entire available data into feature matrix X and target y
# * Refit min max scaler on entire data (not only on training split), save as pkl for later reuse (for new incoming data)
# * Refit the one-hot encoder (dict vectorizer) on the entire data, save as pkl for later reuse (for new incoming data)
# * Apply the fitted preprocessors to the entire data 
# * Train the best model on the entire available(and preprocessed) data 
# * Save the final model (trained on entire data) as pkl for later reuse

import pickle
import numpy as np

# To make the best out of the entire available data (df_cleaned_4):
# Step 1: Split entire available data into feature matrix X and target y
X_all = df_cleaned_4.drop(columns=[target_column])  # Feature matrix
y_all = df_cleaned_4[target_column]  # Target variable

# Step 2: Fit the MinMaxScaler on the entire data (fit only)
# Fit the scaler on the entire data (numerical columns only)
scaler.fit(X_all[numerical_columns])

# Save the fitted scaler for later reuse
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Scaler fitted on entire data and saved as 'scaler.pkl'.")

# Step 3: Transform the data using the fitted MinMaxScaler
X_all_numerical_scaled = scaler.transform(X_all[numerical_columns])

# Step 4: Fit the DictVectorizer on the entire data (fit only)
# Convert categorical data to dictionary format for the vectorizer
X_all_categorical_dict = X_all[categorical_columns].to_dict(orient='records')

# Fit the vectorizer on the entire data
vectorizer.fit(X_all_categorical_dict)

# Save the fitted vectorizer for later reuse
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("DictVectorizer fitted on entire data and saved as 'vectorizer.pkl'.")

# Step 5: Transform the data using the fitted DictVectorizer
X_all_categorical_encoded = vectorizer.transform(X_all_categorical_dict)

# Step 6: Apply the fitted preprocessors to the entire data
# Combine the scaled numerical data with the encoded categorical data
X_all_preprocessed = np.hstack([X_all_numerical_scaled, X_all_categorical_encoded])

# Step 7: Train the best model on the entire available (and preprocessed) data
# Train the best model (selected earlier in the script) on the preprocessed data
best_model['model'].fit(X_all_preprocessed, y_all)

# Step 8: Save the final model (trained on entire data) as pkl for later reuse
with open('final_trained_model.pkl', 'wb') as f:
    pickle.dump(best_model['model'], f)

print("Final model trained on entire data and saved as 'final_trained_model.pkl'.")

