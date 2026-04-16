"""
TASK 1: Data Loading, Exploration, Cleaning, and Feature Engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create outputs directory if it doesn't exist
os.makedirs('../outputs', exist_ok=True)

# -------------------------------
# 1. Data Loading and Exploration
# -------------------------------

# Load the dataset
df = pd.read_csv('../input/customer_segmentation_data.csv')

# Initial exploration
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nStatistical summary:\n{df.describe()}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")

# -------------------------------
# 2. Data Cleaning and Preprocessing
# -------------------------------

# Handle missing values in 'minutes_watched' - fill with 0
if 'minutes_watched' in df.columns:
    df['minutes_watched'] = df['minutes_watched'].fillna(0)
    print("\n✓ Filled missing minutes_watched with 0")

# Save initial cleaned data
df.to_csv('../outputs/cleaned_customer_data.csv', index=False)
print("✓ Saved cleaned_customer_data.csv")

# -------------------------------
# 3. Data Visualization and Correlation Analysis
# -------------------------------

# Select numeric columns for correlation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = df[numeric_cols].corr()

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('../outputs/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved correlation_heatmap.png")

# Optional: Scatter plot of CLV vs minutes_watched
if 'CLV' in df.columns and 'minutes_watched' in df.columns:
    plt.figure(figsize=(10, 6))
    plt.scatter(df['CLV'], df['minutes_watched'], alpha=0.5, s=10)
    plt.xlabel('Customer Lifetime Value (CLV)')
    plt.ylabel('Minutes Watched')
    plt.title('CLV vs Minutes Watched')
    plt.tight_layout()
    plt.savefig('../outputs/clv_vs_minutes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved clv_vs_minutes.png")

print("\n" + "=" * 50)
print("TASK 1 COMPLETE: Data prepared and saved")
print("=" * 50)