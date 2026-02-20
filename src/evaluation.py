import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

def evaluate_predictions(y_true, y_pred, output_dir="data/processed"):
    # Convert to series if not already
    y_true = pd.Series(y_true.values.ravel())
    y_pred = pd.Series(y_pred)
    
    # Residuals
    residuals = y_true - y_pred
    
    # Error Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title("Residuals Distribution (Errors)")
    plt.xlabel("Wait - Predicted")
    plt.savefig(os.path.join(output_dir, "residuals_dist.png"))
    plt.close()
    
    # True vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.title("True vs Predicted Age")
    plt.savefig(os.path.join(output_dir, "true_vs_pred.png"))
    plt.close()
    
    logger.info("Evaluation plots generated.")

if __name__ == "__main__":
    # Integration logic
    pass
