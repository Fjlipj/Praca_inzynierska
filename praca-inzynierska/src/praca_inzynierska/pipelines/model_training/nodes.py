"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.5
"""
import numpy as np
import wandb
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def train_random_forest(X_train, y_train):
    """
    Train Random Forest Regressor
    """
    rf_model = RandomForestRegressor(
        n_estimators=300, 
        random_state=42, 
        max_depth=10
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def train_random_forest_with_BPM(XX_train, yy_train):
    """
    Train Random Forest Regressor with BPM
    """
    rf_model_with_BPM = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        max_depth=10
    )
    rf_model_with_BPM.fit(XX_train, yy_train)
    return rf_model_with_BPM

def train_neural_network(X_train, y_train):
    best_params = {
        "hidden_layer_sizes": (128, 64, 32),
        "alpha": 1e-2,                 # 0.01
        "learning_rate_init": 1e-3,
    }

    mlp_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",   MLPRegressor(
                    **best_params,
                    learning_rate="adaptive",  # zostawiamy jak w gridzie
                    max_iter=5000,
                    early_stopping=True,
                    n_iter_no_change=30,
                    random_state=42,
                )
        ),
    ])

    mlp_pipe.fit(X_train, y_train)
    return mlp_pipe

def evaluate_NN_model(model, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'Mean Absolute Error (seconds)': mae,
        'Root Mean Squared Error (seconds)': rmse,
        'R-squared': r2
    }
    print('\n--------------------------------------\n')
    print('\nMetrics Neural Network:\n')
    print(metrics, y_pred)

    errors = y_test - y_pred
    print('\nErrors:\n')
    print(errors)

 # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, color="blue")
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("residual_distribution_NN.png")
    plt.close()

    # Scatter plot of predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title("Predictions vs Actual Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.tight_layout()
    plt.savefig("predictions_vs_actual_NN.png")
    plt.close()

    # Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(feature_importances)
        plt.barh(feature_names[sorted_idx], feature_importances[sorted_idx])
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.savefig("feature_importances_NN.png")
        plt.close()

    return metrics, y_pred

def evaluate_model(model, X_test, y_test, feature_names):
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

 # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, color="blue")
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("residual_distribution.png")
    plt.close()

    # Scatter plot of predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title("Predictions vs Actual Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.tight_layout()
    plt.savefig("predictions_vs_actual.png")
    plt.close()

    # Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(feature_importances)
        plt.barh(feature_names[sorted_idx], feature_importances[sorted_idx])
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.savefig("feature_importances.png")
        plt.close()


    # wandb.login(key="7cedcc8572677253cbaf3974533bf4979bb5e496")

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="praca-inzynierska"
    # )

    # wandb.log(
    #     {
    #         'Mean Absolute Error (seconds)': mae,
    #         'Root Mean Squared Error (seconds)': rmse,
    #         'R-squared': r2
    #     }
    # )

    return metrics, y_pred

def evaluate_model_with_BPM(model, XX_test, yy_test, feature_names):
    """
    Evaluate the trained model
    """
    y_pred = model.predict(XX_test)
    
    mae = mean_absolute_error(yy_test, y_pred)
    mse = mean_squared_error(yy_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(yy_test, y_pred)

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

    errors = yy_test - y_pred
    print('\nErrors:\n')
    print(errors)

 # Residual plot
    residuals = yy_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, color="blue")
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("residual_distribution2.png")
    plt.close()

    # Scatter plot of predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(yy_test, y_pred, alpha=0.5)
    plt.plot([yy_test.min(), yy_test.max()], [yy_test.min(), yy_test.max()], color='red', linestyle='--')
    plt.title("Predictions vs Actual Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.tight_layout()
    plt.savefig("predictions_vs_actual2.png")
    plt.close()

    # Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(feature_importances)
        plt.barh(feature_names[sorted_idx], feature_importances[sorted_idx])
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.savefig("feature_importances2.png")
        plt.close()

    return metrics, y_pred

def autoML(data: pd.DataFrame):
    """
    Train an AutoML model using AutoGluon TabularPredictor.
    """
    predictor = TabularPredictor(
        label="Czas treningu (seconds)", 
        eval_metric='mean_squared_error'
    ).fit(data)
    
    return predictor


def convert_seconds_to_time_str(seconds):
    """
    Convert seconds back to HH:MM:SS format
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
