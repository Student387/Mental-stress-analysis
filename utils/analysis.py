"""
Analysis utilities for stress result explanation.
Generates human-readable factors contributing to, increasing, or reducing stress.
"""


def get_analysis_summary(stress_level, form_data):
    """
    Returns dict with:
    - contributing_factors: list of factors that contributed most to the prediction
    - increasing_stress: list of factors that may be increasing stress
    - reducing_stress: list of factors that may be reducing stress
    """
    contributing = []
    increasing = []
    reducing = []

    fd = form_data or {}

    # Anxiety
    ax = _parse_scale(fd.get('anxiety_level', 3), 5)
    if ax >= 4:
        contributing.append("High anxiety or worry levels")
        increasing.append("Frequent anxiety or worry")
    elif ax <= 2:
        reducing.append("Low anxiety levels")

    # Self-esteem
    se = str(fd.get('self_esteem', '')).lower()
    if se == 'low':
        contributing.append("Lower self-confidence")
        increasing.append("Low self-esteem")
    elif se == 'high':
        reducing.append("Good self-confidence")

    # Mental health history
    mh = str(fd.get('mental_health_history', '')).lower()
    if mh in ('yes', '1'):
        contributing.append("Previous mental health history")
        increasing.append("History of mental health concerns")

    # Depression
    dep = _parse_scale(fd.get('depression', 3), 5)
    if dep >= 4:
        contributing.append("Feelings of sadness or hopelessness")
        increasing.append("Frequent feelings of sadness or hopelessness")
    elif dep <= 2:
        reducing.append("Rare feelings of sadness")

    # Physical symptoms
    hd = str(fd.get('headache', '')).lower()
    if hd == 'often':
        contributing.append("Frequent headaches")
        increasing.append("Recurring headaches")

    bp = str(fd.get('blood_pressure', '')).lower()
    if bp in ('yes', '1'):
        contributing.append("Blood pressure concerns")
        increasing.append("Elevated or abnormal blood pressure")

    # Sleep
    sl = str(fd.get('sleep_quality', '')).lower()
    if sl in ('poor', '1'):
        contributing.append("Poor sleep quality")
        increasing.append("Inadequate or poor sleep")
    elif sl in ('good', 'excellent', '5'):
        reducing.append("Good sleep quality")

    # Breathing
    br = str(fd.get('breathing_problem', '')).lower()
    if br in ('yes', '1'):
        contributing.append("Stress-related breathing difficulties")
        increasing.append("Shortness of breath during stress")

    # Environment
    nl = str(fd.get('noise_level', '')).lower()
    if nl == 'high':
        contributing.append("High noise in study/living environment")
        increasing.append("Noisy environment")
    elif nl == 'low':
        reducing.append("Quiet study environment")

    lc = str(fd.get('living_conditions', '')).lower()
    if lc == 'poor':
        contributing.append("Uncomfortable living conditions")
        increasing.append("Poor living conditions")

    sf = str(fd.get('safety', '')).lower()
    if sf in ('no', '0'):
        contributing.append("Safety concerns")
        increasing.append("Perceived lack of safety")
    elif sf in ('yes', '1'):
        reducing.append("Feeling safe in surroundings")

    bn = str(fd.get('basic_needs', '')).lower()
    if bn in ('no', '0'):
        contributing.append("Unmet basic needs")
        increasing.append("Basic needs not fully met")
    elif bn in ('yes', '1'):
        reducing.append("Basic needs adequately met")

    # Academic
    ap = str(fd.get('academic_performance', '')).lower()
    if ap == 'poor':
        contributing.append("Academic performance concerns")
        increasing.append("Academic performance stress")
    elif ap in ('good', 'excellent'):
        reducing.append("Satisfactory academic performance")

    sload = str(fd.get('study_load', '')).lower()
    if sload in ('high', '5'):
        contributing.append("Heavy academic workload")
        increasing.append("High study load")
    elif sload in ('low', '1'):
        reducing.append("Manageable study load")

    ts = str(fd.get('teacher_student_relationship', '')).lower()
    if ts == 'poor':
        contributing.append("Limited teacher support")
        increasing.append("Lack of supportive teacher relationships")
    elif ts in ('good', 'excellent'):
        reducing.append("Supportive teacher relationships")

    # Career
    fc = _parse_scale(fd.get('future_career_concerns', 3), 5)
    if fc >= 4:
        contributing.append("Future career worries")
        increasing.append("Career anxiety")
    elif fc <= 2:
        reducing.append("Low career-related anxiety")

    # Social
    ss = str(fd.get('social_support', '')).lower()
    if ss in ('low', '1'):
        contributing.append("Limited social support")
        increasing.append("Lack of supportive friends or family")
    elif ss in ('high', '5'):
        reducing.append("Strong social support network")

    pp = str(fd.get('peer_pressure', '')).lower()
    if pp in ('high', '5'):
        contributing.append("Peer pressure")
        increasing.append("High peer pressure")
    elif pp in ('low', '1'):
        reducing.append("Low peer pressure")

    # Activities
    ext = str(fd.get('extracurricular_activities', '')).lower()
    if ext in ('yes', '1'):
        reducing.append("Participation in extracurricular activities")
    elif ext in ('no', '0'):
        increasing.append("Limited engagement in extracurricular activities")

    # Bullying
    bl = str(fd.get('bullying', '')).lower()
    if bl in ('often', 'sometimes', '5', '3'):
        contributing.append("Experience of bullying")
        increasing.append("Bullying experiences")

    # Limit to top contributors for clarity
    contributing = contributing[:5] if contributing else ["Multiple factors from your responses"]
    increasing = increasing[:6] if increasing else []
    reducing = reducing[:6] if reducing else []

    return {
        "contributing_factors": contributing,
        "increasing_stress": increasing,
        "reducing_stress": reducing
    }


def _parse_scale(val, max_val=5):
    """Parse scale value to int."""
    try:
        v = int(float(val))
        return max(1, min(max_val, v))
    except (ValueError, TypeError):
        return 3


FEATURE_LABELS = {
    'anxiety_level': 'Anxiety level',
    'self_esteem': 'Self-esteem',
    'sleep_quality': 'Sleep quality',
    'depression': 'Depression/mood',
    'headache': 'Headaches',
    'study_load': 'Study load',
    'social_support': 'Social support',
    'future_career_concerns': 'Career concerns',
    'living_conditions': 'Living conditions',
    'noise_level': 'Noise level',
    'academic_performance': 'Academic performance',
    'peer_pressure': 'Peer pressure',
    'breathing_problem': 'Breathing',
    'extracurricular_activities': 'Extracurricular activities',
    'bullying': 'Bullying',
    'mental_health_history': 'Mental health history',
    'blood_pressure': 'Blood pressure',
    'safety': 'Safety',
    'basic_needs': 'Basic needs',
    'teacher_student_relationship': 'Teacher support',
}

# Lower is better for stress: anxiety, depression, headache, study_load, peer_pressure, bullying, etc.
# Higher is better: self_esteem, sleep_quality, social_support, living_conditions, etc.
STRESS_POSITIVE = {'anxiety_level', 'depression', 'headache', 'study_load', 'peer_pressure', 'bullying',
                   'breathing_problem', 'future_career_concerns', 'noise_level', 'blood_pressure',
                   'mental_health_history'}
STRESS_NEGATIVE = {'self_esteem', 'sleep_quality', 'social_support', 'living_conditions', 'safety',
                   'basic_needs', 'academic_performance', 'teacher_student_relationship',
                   'extracurricular_activities'}


def _ordinal_val(form_data, key):
    """Get comparable ordinal value for a feature (higher = more stress typically)."""
    val = str(form_data.get(key, '')).lower()
    if key in ('anxiety_level', 'depression', 'future_career_concerns'):
        return _parse_scale(form_data.get(key, 3), 5)
    if key in ('self_esteem', 'sleep_quality', 'social_support', 'living_conditions',
               'academic_performance', 'teacher_student_relationship'):
        mapping = {'poor': 1, 'low': 1, 'average': 2, 'medium': 2, 'good': 3, 'high': 3}
        return mapping.get(val, 2)
    if key in ('study_load', 'peer_pressure', 'noise_level'):
        mapping = {'low': 1, 'medium': 2, 'high': 3}
        return mapping.get(val, 2)
    if key in ('headache', 'bullying'):
        mapping = {'never': 1, 'sometimes': 2, 'often': 3}
        return mapping.get(val, 2)
    if key in ('breathing_problem', 'blood_pressure', 'mental_health_history', 'safety', 'basic_needs',
               'extracurricular_activities'):
        mapping = {'no': 0, 'yes': 1}
        inv = {'safety': 1, 'basic_needs': 1}  # for these, yes=good so invert
        if key in inv:
            return 0 if val in ('yes', '1') else 1
        return 1 if val in ('yes', '1') else 0
    return 0


def get_stress_change_comparison(prev_responses, curr_responses, prev_stress_code, curr_stress_code):
    """
    Compare previous and current assessment. Returns:
    - trend: 'increased' | 'decreased' | 'same'
    - improved_features: list of human-readable strings
    - worsened_features: list of human-readable strings
    - what_helped: list (same as improved when trend=decreased)
    """
    trend = 'same'
    if curr_stress_code > prev_stress_code:
        trend = 'increased'
    elif curr_stress_code < prev_stress_code:
        trend = 'decreased'

    improved = []
    worsened = []
    prev = prev_responses or {}
    curr = curr_responses or {}

    for key, label in FEATURE_LABELS.items():
        pv = _ordinal_val(prev, key)
        cv = _ordinal_val(curr, key)
        if pv == cv:
            continue
        if key in STRESS_POSITIVE:
            if cv < pv:
                improved.append(label)
            else:
                worsened.append(label)
        else:
            if cv > pv:
                improved.append(label)
            else:
                worsened.append(label)

    what_helped = improved if trend == 'decreased' else []
    return {
        'trend': trend,
        'improved_features': improved,
        'worsened_features': worsened,
        'what_helped': what_helped,
        'prev_stress_code': prev_stress_code,
        'curr_stress_code': curr_stress_code
    }
