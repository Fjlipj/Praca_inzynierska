"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.5
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def parse_time_duration(time_str):
    """
    Convert time string (HH:MM:SS) to total seconds
    """
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def load_and_preprocess_data(dataset):
    """
    Load and preprocess the training data
    """
    # Convert Czas treningu to seconds
    dataset['Czas treningu (seconds)'] = dataset['Czas treningu'].apply(parse_time_duration)
    
    # Convert Data to datetime and extract additional features
    dataset['Data'] = pd.to_datetime(dataset['Data'])
    dataset['month'] = dataset['Data'].dt.month
    dataset['day_of_week'] = dataset['Data'].dt.dayofweek
    
    return dataset

def prepare_features_and_target(preprocessed_data):
    """
    Prepare features and target variable for model training
    """
    # Select features
    features = [
        'Dystans', 'Kcal (aktywnosc)', 'Przewyzszenie (w metrach)', 
        'Srednia szybkosc', 'Srednie tetno', 'Temperatura', 
        'Wilgotnosc', 'Predkosc wiatru', 'Cisnienie',
        'month', 'day_of_week'
    ]
    
    X = preprocessed_data[features]
    y = preprocessed_data['Czas treningu (seconds)']

    # create charts for the model performance
 

    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler