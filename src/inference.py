import joblib
import pandas as pd
import yaml
import os
import logging

logger = logging.getLogger(__name__)

class AgePredictor:
    def __init__(self, model_path="models/saved_model.pkl", preprocessor_path="models/preprocessor.joblib"):
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError("Model or Preprocessor file not found. Please train the model first.")
        
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
    def predict(self, input_data: pd.DataFrame):
        """
        Expects input_data to be a DataFrame with same features as training.
        """
        processed_data = self.preprocessor.transform(input_data)
        prediction = self.model.predict(processed_data)
        return prediction

if __name__ == "__main__":
    # Example usage
    # predictor = AgePredictor()
    # result = predictor.predict(sample_df)
    pass

