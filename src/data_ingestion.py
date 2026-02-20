import os
import pandas as pd
import yaml
import logging
from pydantic import BaseModel, ValidationError
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataConfig(BaseModel):
    raw_dir: str
    processed_dir: str
    merge_key: str
    files: Dict[str, str]
    target_column: str
    features: List[str]

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_local_nhanes_data(data_cfg: DataConfig):
    """
    Loads local NHANES CSV files and merges them.
    """
    data_frames = {}
    for name, filename in data_cfg.files.items():
        file_path = os.path.join(data_cfg.raw_dir, filename)
        logger.info(f"Loading {name} from {file_path}")
        try:
            # Read CSV
            df = pd.read_csv(file_path, encoding='latin1')
            data_frames[name] = df
        except Exception as e:
            logger.error(f"Error loading {name}: {e}")
            raise e
                
    # Merge datasets on merge_key
    merged_df = None
    for name, df in data_frames.items():
        if merged_df is None:
            merged_df = df
        else:
            # Merge on common key
            merged_df = merged_df.merge(df, on=data_cfg.merge_key, how="inner")
            logger.info(f"Merged {name}, current shape: {merged_df.shape}")
    
    return merged_df

def validate_data(df: pd.DataFrame, features: List[str], target: str):
    """
    Simple validation to check for required columns.
    """
    # RIAGENDR and RIDAGEYR are in demographic.csv
    # BMX... are in examination.csv
    # LBX... are in labs.csv
    required_cols = features + [target]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.warning(f"Missing columns: {missing}. Checking if they exist in raw files.")
    else:
        logger.info("Data validation successful.")

def main():
    config = load_config()
    try:
        data_cfg = DataConfig(**config['data'])
    except ValidationError as e:
        logger.error(f"Config validation error: {e}")
        return
    
    try:
        df = load_local_nhanes_data(data_cfg)
        validate_data(df, data_cfg.features, data_cfg.target_column)
        
        # Save the combined raw data for preprocessing
        combined_path = os.path.join(data_cfg.raw_dir, "nhanes_combined.csv")
        df.to_csv(combined_path, index=False)
        logger.info(f"Combined raw data saved to {combined_path}")
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")

if __name__ == "__main__":
    main()
