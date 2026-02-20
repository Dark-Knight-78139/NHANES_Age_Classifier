import pandas as pd
import yaml
from src.visualization import generate_visual_reports

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    df = pd.read_csv("data/raw/nhanes_combined.csv", encoding='latin1')
    features = config['data']['features']
    
    print("Generating visual reports...")
    generate_visual_reports(df, features)
    print("Visual reports generated successfully!")

if __name__ == "__main__":
    main()
