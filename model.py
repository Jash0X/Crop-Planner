import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Define necessary functions
def preprocess_data(df):
    # Convert columns with non-numeric data to numeric if possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass  # Ignore columns that cannot be converted
    
    # Handle missing values
    df.dropna(inplace=True)  # Drop rows with missing values

def read_in_and_split_data(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Load Dataset
df = pd.read_csv('Crop_recommendation.csv')

# Preprocess data
preprocess_data(df)

# Split Data to Training and Validation set
target ='label'
X_train, X_test, y_train, y_test = read_in_and_split_data(df, target)

# Train model
model = train_model(X_train, y_train)

# Save model
save_model(model, 'model.pkl')
