import os
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

import joblib

warnings.filterwarnings('ignore')


def load_dataset():
    return pd.read_csv("StressLevelDataset.csv")


def prepare_data(df):
    X = df.drop("stress_level", axis=1)
    y = df["stress_level"]
    return X, y


# def apply_pca_lda(X_train, X_test, y_train):
#     pca = PCA(n_components=0.95)
#     X_train_pca = pca.fit_transform(X_train)
#     X_test_pca = pca.transform(X_test)

#     lda = LinearDiscriminantAnalysis()
#     X_train_lda = lda.fit_transform(X_train_pca, y_train)
#     X_test_lda = lda.transform(X_test_pca)

#     # 🔥 CHANGED: Now returning the pca and lda objects so we can save them later
#     return X_train_lda, X_test_lda, pca, lda


def train_models(X_train, X_test, y_train, y_test):

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                                 use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    return results


def print_results(results):
    for name, res in results.items():
        print("\n" + "="*60)
        print(name)
        print("="*60)
        print(f"Accuracy: {res['accuracy']*100:.2f}%")
        print(f"Precision: {res['precision']*100:.2f}%")
        print(f"Recall: {res['recall']*100:.2f}%")
        print(f"F1 Score: {res['f1']*100:.2f}%")

        print("\nConfusion Matrix:")
        print(res['confusion_matrix'])


def main():
    print("\n===== Student Mental Stress Analysis =====")

    # 1. Ensure the model directory exists
    os.makedirs('model', exist_ok=True)

    df = load_dataset()
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Scale Data and Save Scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, 'model/feature_scaler.joblib')

    # # 3. Apply PCA + LDA and Save Transformers
    # X_train, X_test, pca, lda = apply_pca_lda(X_train, X_test, y_train)
    # joblib.dump(pca, 'model/pca_transformer.joblib')
    # joblib.dump(lda, 'model/lda_transformer.joblib')

    # 4. Train all models to print the comparison metrics
    results = train_models(X_train, X_test, y_train, y_test)
    print_results(results)

    # 5. Train and Save the Final Random Forest Model
    print("\n[INFO] Saving Final Random Forest Model for production...")
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(X_train, y_train)
    joblib.dump(final_model, 'model/stress_model.joblib')
    
    print("[SUCCESS] All files (Model, Scaler, PCA, LDA) saved to /model directory!")


if __name__ == "__main__":
    main()