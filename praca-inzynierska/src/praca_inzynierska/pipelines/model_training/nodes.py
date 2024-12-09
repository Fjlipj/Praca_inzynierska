"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.5
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import numpy as np
import wandb
import logging

def train_random_forest(X_train, y_train):
    """
    Train Random Forest Regressor
    """
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        max_depth=10
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model
    """
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    #score = accuracy_score(y_test, y_pred)
    #logger = logging.getLogger(__name__)
    #logger.info("Model has a coefficient prediction of %.3f on test data.", score)

    
    metrics = {
        'Mean Absolute Error (seconds)': mae,
        'Root Mean Squared Error (seconds)': rmse,
        'R-squared': r2
    }
    print('\nMetrics:\n')
    print(metrics, y_pred)

    errors = y_test - y_pred
    print('\nErrors:\n')
    print(errors)

    wandb.login(key="7cedcc8572677253cbaf3974533bf4979bb5e496")

    wandb.init(
        # set the wandb project where this run will be logged
        project="praca-inzynierska"
    )

    wandb.log(
        {
            'Mean Absolute Error (seconds)': mae,
            'Root Mean Squared Error (seconds)': rmse,
            'R-squared': r2
        }
    )

    return metrics, y_pred

def convert_seconds_to_time_str(seconds):
    """
    Convert seconds back to HH:MM:SS format
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
