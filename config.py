"""
Configuration for Student Mental Stress Analysis System
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
DATASET_PATH = os.path.join(DATASET_DIR, 'StressLevelDataset.csv')
X_TRAIN_PATH = os.path.join(DATASET_DIR, 'X_train_preprocessed.csv')
X_TEST_PATH = os.path.join(DATASET_DIR, 'X_test_preprocessed.csv')
Y_TRAIN_PATH = os.path.join(DATASET_DIR, 'y_train.csv')
Y_TEST_PATH = os.path.join(DATASET_DIR, 'y_test.csv')

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'stress_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'feature_scaler.joblib')

# Feature columns (must match dataset)
FEATURE_COLUMNS = [
    'anxiety_level', 'self_esteem', 'mental_health_history', 'depression',
    'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem',
    'noise_level', 'living_conditions', 'safety', 'basic_needs',
    'academic_performance', 'study_load', 'teacher_student_relationship',
    'future_career_concerns', 'social_support', 'peer_pressure',
    'extracurricular_activities', 'bullying'
]

# Stress level mapping: 0=Low, 1=Medium, 2=High
STRESS_LEVEL_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}

# User data and retraining
USER_DATA_PATH = os.path.join(DATASET_DIR, 'user_data.csv')
MASTER_DATASET_PATH = os.path.join(DATASET_DIR, 'StressLevelDataset.csv')
RETRAIN_THRESHOLD = 10
