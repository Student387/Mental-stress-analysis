"""
Preprocessing utilities for form inputs.
Maps user-friendly questionnaire responses to numeric features expected by the ML model.
"""

import numpy as np
import pandas as pd
from config import FEATURE_COLUMNS, STRESS_LEVEL_LABELS


# Mapping helpers: Form value -> Dataset numeric value
# Based on StressLevelDataset structure (scales typically 0-5 or 1-5)

def map_scale_1_5(val):
    """Map 1-5 scale (string or int) to int 1-5."""
    if isinstance(val, (int, float)):
        return max(1, min(5, int(val)))
    mapping = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
    return mapping.get(str(val).strip(), 3)


def map_yes_no(val):
    """Yes=1, No=0"""
    if isinstance(val, (int, float)):
        return 1 if val else 0
    return 1 if str(val).strip().lower() in ('yes', '1', 'true') else 0


def map_low_med_high(val):
    """Low=1, Medium=2/3, High=4/5"""
    v = str(val).strip().lower()
    if v in ('low', '1'): return 1
    if v in ('high', '5'): return 5
    return 3  # medium


def map_poor_avg_good(val):
    """Poor=1, Average=3, Good=5"""
    v = str(val).strip().lower()
    if v in ('poor', '1'): return 1
    if v in ('good', 'excellent', '5'): return 5
    return 3  # average


def map_never_sometimes_often(val):
    """Never=1, Sometimes=3, Often=5"""
    v = str(val).strip().lower()
    if v in ('never', '1'): return 1
    if v in ('often', 'always', '5'): return 5
    return 3  # sometimes


def map_anxiety(val):
    """Anxiety 1-5: scale up to dataset range ~4-20."""
    s = map_scale_1_5(val)
    return s * 4  # 4, 8, 12, 16, 20


def map_self_esteem(val):
    """Self-esteem: Low=5, Medium=15, High=25 (inverted - low confidence = low number in some datasets)."""
    v = str(val).strip().lower()
    if v in ('low', '1'): return 8
    if v in ('high', '5'): return 25
    return 18  # medium


def map_blood_pressure(val):
    """Yes=3, No=1 (dataset uses 1,2,3)"""
    return 3 if map_yes_no(val) else 1


def form_to_features(form_data):
    """
    Convert form dictionary to feature array matching FEATURE_COLUMNS order.
    form_data: dict with keys matching form field names
    Returns: numpy array of shape (1, 20) ready for scaler
    """
    data = form_data or {}

    features = {
        'anxiety_level': map_anxiety(data.get('anxiety_level', 3)),
        'self_esteem': map_self_esteem(data.get('self_esteem', 'medium')),
        'mental_health_history': map_yes_no(data.get('mental_health_history', 'no')),
        'depression': map_scale_1_5(data.get('depression', 3)) * 5,  # 5-25 range
        'headache': map_never_sometimes_often(data.get('headache', 'sometimes')),
        'blood_pressure': map_blood_pressure(data.get('blood_pressure', 'no')),
        'sleep_quality': map_poor_avg_good(data.get('sleep_quality', 'average')),
        'breathing_problem': map_yes_no(data.get('breathing_problem', 'no')) * 4,  # 0 or 4
        'noise_level': map_low_med_high(data.get('noise_level', 'medium')),
        'living_conditions': map_poor_avg_good(data.get('living_conditions', 'average')),
        'safety': (3 if map_yes_no(data.get('safety', 'yes')) else 1),  # 1-4
        'basic_needs': (4 if map_yes_no(data.get('basic_needs', 'yes')) else 2),
        'academic_performance': map_poor_avg_good(data.get('academic_performance', 'average')),
        'study_load': map_low_med_high(data.get('study_load', 'medium')),
        'teacher_student_relationship': map_poor_avg_good(data.get('teacher_student_relationship', 'average')),
        'future_career_concerns': map_scale_1_5(data.get('future_career_concerns', 3)),
        'social_support': map_low_med_high(data.get('social_support', 'medium')) + 1,  # 2-6 -> cap at 5
        'peer_pressure': map_low_med_high(data.get('peer_pressure', 'medium')),
        'extracurricular_activities': (4 if map_yes_no(data.get('extracurricular_activities', 'yes')) else 1),
        'bullying': map_never_sometimes_often(data.get('bullying', 'never')),
    }

    # Clamp to typical dataset ranges
    features['social_support'] = min(5, features['social_support'])
    features['breathing_problem'] = min(5, features['breathing_problem'])

    arr = np.array([[features[col] for col in FEATURE_COLUMNS]], dtype=np.float64)
    return arr


def form_to_feature_dict(form_data):
    """Return feature dict for CSV row (includes stress_level for dataset)."""
    data = form_data or {}
    features = {
        'anxiety_level': map_anxiety(data.get('anxiety_level', 3)),
        'self_esteem': map_self_esteem(data.get('self_esteem', 'medium')),
        'mental_health_history': map_yes_no(data.get('mental_health_history', 'no')),
        'depression': map_scale_1_5(data.get('depression', 3)) * 5,
        'headache': map_never_sometimes_often(data.get('headache', 'sometimes')),
        'blood_pressure': map_blood_pressure(data.get('blood_pressure', 'no')),
        'sleep_quality': map_poor_avg_good(data.get('sleep_quality', 'average')),
        'breathing_problem': min(5, map_yes_no(data.get('breathing_problem', 'no')) * 4),
        'noise_level': map_low_med_high(data.get('noise_level', 'medium')),
        'living_conditions': map_poor_avg_good(data.get('living_conditions', 'average')),
        'safety': (3 if map_yes_no(data.get('safety', 'yes')) else 1),
        'basic_needs': (4 if map_yes_no(data.get('basic_needs', 'yes')) else 2),
        'academic_performance': map_poor_avg_good(data.get('academic_performance', 'average')),
        'study_load': map_low_med_high(data.get('study_load', 'medium')),
        'teacher_student_relationship': map_poor_avg_good(data.get('teacher_student_relationship', 'average')),
        'future_career_concerns': map_scale_1_5(data.get('future_career_concerns', 3)),
        'social_support': min(5, map_low_med_high(data.get('social_support', 'medium')) + 1),
        'peer_pressure': map_low_med_high(data.get('peer_pressure', 'medium')),
        'extracurricular_activities': (4 if map_yes_no(data.get('extracurricular_activities', 'yes')) else 1),
        'bullying': map_never_sometimes_often(data.get('bullying', 'never')),
    }
    return features
