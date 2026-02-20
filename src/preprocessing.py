import pandas as pd
import numpy as np
import os
import joblib
import yaml
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def preprocess_data(raw_data_path, config):
    df = pd.read_csv(raw_data_path)
    
    target = config['data']['target_column']
    features = config['data']['features']
    
    X = df[features]
    y = df[target]
    
    # Define preprocessing steps
    numeric_features = features # For this project, assuming most are numeric
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the preprocessor
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.joblib"))
    
    # Save processed data
    processed_dir = config['data']['processed_dir']
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    pd.DataFrame(X_train_processed, columns=features).to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test_processed, columns=features).to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
    
    logger.info("Preprocessing complete and files saved.")
    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    config = load_config()
    raw_path = os.path.join(config['data']['raw_dir'], "nhanes_combined.csv")
    if os.path.exists(raw_path):
        preprocess_data(raw_path, config)
    else:
        logger.error(f"Raw data file not found at {raw_path}")
