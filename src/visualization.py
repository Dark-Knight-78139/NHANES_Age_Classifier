import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger(__name__)

def plot_missing_values(df, output_path="data/processed/missing_heatmap.png"):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Missing values heatmap saved to {output_path}")

def plot_distributions(df, features, output_dir="data/processed"):
    for col in features:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(os.path.join(output_dir, f"dist_{col}.png"))
        plt.close()
    logger.info(f"Distribution plots saved to {output_dir}")

def plot_correlation_matrix(df, features, output_path="data/processed/correlation_matrix.png"):
    plt.figure(figsize=(10, 8))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Correlation matrix saved to {output_path}")

def plot_outliers(df, features, output_dir="data/processed"):
    for col in features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col])
        plt.title(f"Outlier Detection for {col}")
        plt.savefig(os.path.join(output_dir, f"outlier_{col}.png"))
        plt.close()
    logger.info(f"Outlier plots saved to {output_dir}")

def generate_visual_reports(df, features):
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
    plot_missing_values(df)
    plot_distributions(df, features)
    plot_correlation_matrix(df, features)
    plot_outliers(df, features)
