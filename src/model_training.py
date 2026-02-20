import pandas as pd
import joblib
import yaml
import os
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_model():
    config = load_config()
    processed_dir = config['data']['processed_dir']
    
    # Load data
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))
    
    # MLflow tracking
    mlflow.set_experiment("NHANES_Age_Prediction")
    
    with mlflow.start_run():
        # Initialize model
        model_cfg = config['model']
        if model_cfg['type'] == 'RandomForestRegressor':
            model = RandomForestRegressor(**model_cfg['params'])
        else:
            model = RandomForestRegressor() # Default
            
        logger.info(f"Training {model_cfg['type']} model...")
        model.fit(X_train, y_train.values.ravel())
        
        # Predictions
        predictions = model.predict(X_test)
        
        # Metrics
        import numpy as np
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        logger.info(f"MAE: {mae}, RMSE: {rmse}, R2: {r2}")
        
        # Log metrics to MLflow
        mlflow.log_params(model_cfg['params'])
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Save model
        save_path = model_cfg['save_path']
        if not os.path.exists("models"):
            os.makedirs("models")
        joblib.dump(model, save_path)
        logger.info(f"Model saved to {save_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        
    return model

if __name__ == "__main__":
    train_model()
