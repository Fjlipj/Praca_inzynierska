"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.5
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    # Create correlation matrix
    """
    Create and save a correlation matrix heatmap
    """
    dataset_for_matrix = dataset
    dataset_for_matrix.drop('Czas treningu' , axis=1, inplace=True)
    plt.figure(figsize=(12, 8))
    correlation_matrix = dataset_for_matrix.corr()
    sns.heatmap(correlation_matrix, annot=True, vmax=1, vmin=-1, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.close()


    print(dataset)
    return dataset

def prepare_features_and_target(preprocessed_data):
    """
    Prepare features and target variable for model training without BPM
    """
    # Select features
    features = [
        'Dystans', 'Kcal (aktywnosc)', 'Przewyzszenie (w metrach)', 
        'Srednia szybkosc', 'Srednie tetno',
        'Temperatura', 'Wilgotnosc', 'Predkosc wiatru', 'Cisnienie', 'month', 'day_of_week'
    ]
    
    X = preprocessed_data[features]
    y = preprocessed_data['Czas treningu (seconds)']
 
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    Path("data/06_models").mkdir(parents=True, exist_ok=True) 
    joblib.dump(scaler, "data/06_models/scaler.pkl")
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler, X.columns

def prepare_features_and_target_for_refined_model(preprocessed_data):
    """
    Prepare features and target variable for model training with BPM
    """
    # Select features
    features = [
        'Dystans', 'Kcal (aktywnosc)', 'Przewyzszenie (w metrach)', 
        'Srednia szybkosc', 'Srednie tetno',
        'Czas <135BPM' ,'Czas 136-149BPM', 'Czas 150-163BPM', 'Czas 164-177BPM','Czas > 178BPM',
        'Temperatura', 'Wilgotnosc', 'Predkosc wiatru', 'Cisnienie', 'month', 'day_of_week'
    ]

    XX = preprocessed_data[features]
    yy = preprocessed_data['Czas treningu (seconds)']
 
    # Split the data
    XX_train, XX_test, yy_train, yy_test = train_test_split(
        XX, yy, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    XX_train_scaled = scaler.fit_transform(XX_train)
    XX_test_scaled = scaler.transform(XX_test)

    Path("data/06_models").mkdir(parents=True, exist_ok=True) 
    joblib.dump(scaler, "data/06_models/scaler_with_BPM.pkl")
    
    return XX_train_scaled, XX_test_scaled, yy_train.values, yy_test.values, scaler, XX.columns