import pandas as pd
import numpy as np

# Load your existing dataset
df = pd.read_csv("StressLevelDataset.csv")

# Add synthetic data for existing rows
df['age'] = np.random.randint(15, 21, size=len(df))
df['gender'] = np.random.randint(0, 2, size=len(df)) # 0: Male, 1: Female

# Reorder columns to put age and gender at the beginning, keeping stress_level at the end
cols = ['age', 'gender'] + [c for c in df.columns if c not in ['age', 'gender', 'stress_level']] + ['stress_level']
df = df[cols]

# Save it back
df.to_csv("StressLevelDataset.csv", index=False)
print("Dataset updated with age and gender.")