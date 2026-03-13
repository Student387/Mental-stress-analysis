"""Incremental model retraining utilities."""
import os
import json
import threading
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import (
    FEATURE_COLUMNS, STRESS_LEVEL_LABELS, MODEL_PATH, SCALER_PATH,
    MODEL_DIR, DATASET_PATH, USER_DATA_PATH, RETRAIN_THRESHOLD
)
from utils.database import get_connection, save_model_metrics, increment_pending_count, get_pending_retrain_count, reset_pending_count

warnings.filterwarnings('ignore')
_retrain_lock = threading.Lock()


def append_user_data(form_data, stress_code, feature_dict):
    """Append new record to user_data.csv and trigger retrain if threshold met."""
    os.makedirs(os.path.dirname(USER_DATA_PATH), exist_ok=True)
    row = {**feature_dict, 'stress_level': stress_code}
    df = pd.DataFrame([row])
    if os.path.exists(USER_DATA_PATH):
        df.to_csv(USER_DATA_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(USER_DATA_PATH, mode='w', header=True, index=False)

    count = increment_pending_count()
    if count >= RETRAIN_THRESHOLD:
        threading.Thread(target=retrain_model_background, daemon=True).start()


def merge_datasets():
    """Merge master + user_data, deduplicate. Returns combined DataFrame."""
    dfs = []
    cols_needed = FEATURE_COLUMNS + ['stress_level']
    if os.path.exists(DATASET_PATH):
        df_master = pd.read_csv(DATASET_PATH)
        mc = [c for c in cols_needed if c in df_master.columns]
        if mc and 'stress_level' in mc:
            dfs.append(df_master[mc])
    if os.path.exists(USER_DATA_PATH):
        df_user = pd.read_csv(USER_DATA_PATH)
        uc = [c for c in cols_needed if c in df_user.columns]
        if uc and 'stress_level' in uc:
            dfs.append(df_user[uc])
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    common = [c for c in FEATURE_COLUMNS if c in combined.columns]
    if common:
        combined = combined.drop_duplicates(subset=common, keep='last')
    return combined


def retrain_model_background():
    """Retrain model in background, save metrics."""
    with _retrain_lock:
        reset_pending_count()
    try:
        df = merge_datasets()
        if df is None or len(df) < 20:
            return
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            return
        X = df[FEATURE_COLUMNS].astype(float)
        y = df['stress_level'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True)
        }
        best_model = None
        best_acc = 0
        best_name = None
        metrics = {}

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            metrics[name] = {
                'accuracy': float(acc),
                'precision': float(precision_score(y_test, y_pred, average='weighted')),
                'recall': float(recall_score(y_test, y_pred, average='weighted')),
                'f1': float(f1_score(y_test, y_pred, average='weighted'))
            }
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_name = name

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        save_model_metrics(best_name, metrics[best_name])
    except Exception:
        pass


def load_model_metrics():
    """Get latest model metrics from DB."""
    conn = get_connection()
    cur = conn.execute(
        "SELECT model_name, accuracy, precision, recall, f1, created_at FROM model_metrics ORDER BY created_at DESC LIMIT 1"
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            'model_name': row[0],
            'accuracy': row[1] * 100 if row[1] else 0,
            'precision': row[2] * 100 if row[2] else 0,
            'recall': row[3] * 100 if row[3] else 0,
            'f1': row[4] * 100 if row[4] else 0,
            'created_at': row[5]
        }
    return None
