"""
Student Mental Stress Analysis - ML Model Training Script
Trains classification models and saves the best performer.
Stress Level: 0=Low, 1=Medium, 2=High
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
import joblib

from config import (
    BASE_DIR, DATASET_PATH, MODEL_DIR, MODEL_PATH, SCALER_PATH,
    FEATURE_COLUMNS, STRESS_LEVEL_LABELS
)

warnings.filterwarnings('ignore')


def load_dataset():
    """Load dataset from CSV. Uses root if not in dataset/ folder."""
    paths_to_try = [
        DATASET_PATH,
        os.path.join(BASE_DIR, 'StressLevelDataset.csv'),
        os.path.join(BASE_DIR, 'StressLevelDataset_Standardized.csv')
    ]
    df = None
    for path in paths_to_try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded dataset from: {path} ({len(df)} rows)")
            break
    if df is None:
        raise FileNotFoundError("StressLevelDataset.csv not found. Place it in project root or dataset/ folder.")
    return df


def prepare_data(df):
    """Split features and target, handle column names."""
    # Ensure we have the right columns
    if 'stress_level' in df.columns:
        X = df[FEATURE_COLUMNS].copy()
        y = df['stress_level'].astype(int)
    else:
        raise ValueError("Dataset must contain 'stress_level' column")
    return X, y


def train_and_evaluate_models(X_train, X_test, y_train, y_test, scaler):
    """Train multiple classifiers and return the best one with metrics."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True)
    }

    best_model = None
    best_name = None
    best_acc = 0
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': acc,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=[STRESS_LEVEL_LABELS[i] for i in range(3)],
                output_dict=True
            ),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    return best_model, best_name, results


def print_results(results):
    """Pretty print evaluation results."""
    for name, metrics in results.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print('='*60)
        print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"Precision: {metrics['precision']*100:.2f}%")
        print(f"Recall:    {metrics['recall']*100:.2f}%")
        print(f"F1-Score:  {metrics['f1']*100:.2f}%")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        for k, v in metrics['classification_report'].items():
            if isinstance(v, dict):
                print(f"  {k}: {v}")
            elif k not in ('accuracy', 'macro avg', 'weighted avg'):
                print(f"  {k}: {v}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("  Student Mental Stress Analysis - Model Training")
    print("="*60)

    # Load data
    df = load_dataset()
    X, y = prepare_data(df)

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate models
    print("\nTraining models...")
    best_model, best_name, results = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    )

    # Print results
    print_results(results)
    print(f"\n*** Best Model: {best_name} ***")

    # Save model and scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
