import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load dataset
df = pd.read_csv("StressLevelDataset.csv")

# Display first 5 rows
print(df.head())
# Dataset shape
print("Dataset shape:", df.shape)

# Data types
print(df.info())

# Check missing values
print("\nMissing values:\n", df.isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())
# Separate input features and target variable
X = df.drop("stress_level", axis=1)
y = df["stress_level"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)
# Initialize scaler
scaler = StandardScaler()

# Scale features
X_scaled_array = scaler.fit_transform(X)

# Convert to DataFrame
X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)

# Create standardized dataframe
df_standardized = X_scaled.copy()
df_standardized["stress_level"] = y.reset_index(drop=True)

# Save to CSV
df_standardized.to_csv("StressLevelDataset_Standardized.csv", index=False)


# Simple statistical summary for outlier observation
print(X.describe())
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)
# Save preprocessed datasets
X_train.to_csv("X_train_preprocessed.csv", index=False)
X_test.to_csv("X_test_preprocessed.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
