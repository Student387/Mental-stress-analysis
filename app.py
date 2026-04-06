"""
Student Mental Stress Analysis System - Flask Backend
Handles questionnaire form, ML prediction, and result display.
"""

import os
import functools
import joblib
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from io import BytesIO

from config import MODEL_PATH, SCALER_PATH, STRESS_LEVEL_LABELS
from utils.preprocessing import form_to_features, form_to_feature_dict
from utils.solutions import get_solutions
from utils.database import (
    init_db, save_response, save_user_assessment, get_user_assessments,
    get_previous_assessment, get_latest_assessment
)
from utils.analysis import get_analysis_summary, get_stress_change_comparison
from utils.user_management import (
    init_users, create_user, get_user_by_credentials, get_user_by_id
)
from utils.retrain import append_user_data, load_model_metrics

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'student-stress-analysis-secret-key-2024')

_model = None
_scaler = None


def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user_id'):
            session['next'] = request.url
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def load_model():
    """Load ML model and scaler safely."""
    global _model, _scaler

    if _model is None and os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)

    if _scaler is None and os.path.exists(SCALER_PATH):
        _scaler = joblib.load(SCALER_PATH)

    return _model, _scaler


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if not username or not email or not password:
            return render_template('register.html', error='All fields required')
        if len(password) < 6:
            return render_template('register.html', error='Password must be at least 6 characters')
        user, err = create_user(username, email, password)
        if err:
            return render_template('register.html', error=err)
        session['user_id'] = user['id']
        session['username'] = user['username']
        return redirect(url_for('dashboard'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = get_user_by_credentials(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            next_url = session.pop('next', None) or url_for('dashboard')
            return redirect(next_url)
        return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if request.method == 'POST':
        session['form_data'] = request.form.to_dict()
        return redirect(url_for('predict'))

    return render_template('questionnaire.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form_data = session.get('form_data')

    if not form_data:
        return redirect(url_for('questionnaire'))

    model, scaler = load_model()

    if model is None or scaler is None:
        return render_template('error.html', message='Model not found. Run train_model.py first.')

    # Convert form input into model features
    features = form_to_features(form_data)
    features_scaled = scaler.transform(features)

    # ✅ SAFE CONVERSION (fixes int64 error)
    pred = int(model.predict(features_scaled)[0])

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features_scaled)[0]
        proba = [float(p) for p in proba]  # ensure JSON safe
    else:
        proba = [0.33, 0.33, 0.34]

    confidence = float(proba[pred])

    stress_label = STRESS_LEVEL_LABELS.get(pred, 'Unknown')
    solutions = get_solutions(stress_label, confidence, form_data)
    analysis = get_analysis_summary(stress_label, form_data)

    proba_dict = {
        STRESS_LEVEL_LABELS[i]: round(float(proba[i]) * 100, 1)
        for i in range(len(proba))
    }

    result = {
        'stress_level': stress_label,
        'stress_code': int(pred),
        'confidence': round(confidence * 100, 1),
        'probabilities': proba_dict,
        'solutions': solutions,
        'form_data': form_data,
        'analysis': analysis,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    session['result'] = result

    user_id = session.get('user_id')
    try:
        if user_id:
            save_user_assessment(user_id, form_data, stress_label, pred, round(confidence * 100, 1), proba_dict)
            feature_dict = form_to_feature_dict(form_data)
            
            # --- DEBUGGER: Print exactly what the HTML form is sending ---
            print(f"\n[DEBUG] Raw Form Data received: {form_data}\n")
            
            current_username = session.get('username')
            if not current_username:
                user_record = get_user_by_id(user_id)
                if user_record:
                    current_username = user_record.get('username', 'Unknown Student')
                else:
                    current_username = 'Unknown Student'
            
            # ✅ ONLY CALL THIS ONCE!
            append_user_data(form_data, pred, feature_dict, current_username)

        save_response(form_data, stress_label, round(confidence * 100, 1), proba_dict)
    except Exception as e:
        print(f"[ERROR] Failed to save assessment: {e}")

    stress_change = None
    if user_id:
        prev = get_previous_assessment(user_id)
        if prev:
            stress_change = get_stress_change_comparison(
                prev.get('responses'), form_data,
                prev.get('stress_code', 0), pred
            )
    result['stress_change'] = stress_change

    return render_template('result.html', **result)


@app.route('/download-report')
@login_required
def download_report():
    result = session.get('result')
    username = session.get('username', 'Guest Student') # Mapped user

    if not result:
        return redirect(url_for('questionnaire'))

    analysis = result.get('analysis', {})
    lines = [
        'STUDENT MENTAL STRESS ANALYSIS SYSTEM',
        'OFFICIAL ASSESSMENT REPORT',
        '=' * 55,
        '',
        f"Student Name: {username.upper()}", # Show explicitly who this belongs to
        f"Report Generated: {result.get('timestamp', 'N/A')}",
        f"Assessment ID: SMAS-{datetime.now().strftime('%Y%m%d%H%M')}",
        '',
        '-' * 55,
        '1. ASSESSMENT SUMMARY',
        '-' * 55,
        '',
        f"  Predicted Stress Level: {result.get('stress_level', 'N/A')}",
        f"  Model Confidence: {result.get('confidence', 0)}%",
        '',
        '  Probability Distribution:'
    ]

    for k, v in result.get('probabilities', {}).items():
        lines.append(f"    - {k}: {v}%")

    lines.extend([
        '',
        '-' * 55,
        '2. DETAILED ANALYSIS',
        '-' * 55,
        '',
        '  Primary Contributing Factors:'
    ])
    for f in analysis.get('contributing_factors', []):
        lines.append(f"    * {f}")
    lines.extend([
        '',
        '  Factors Potentially Increasing Stress:'
    ])
    for f in analysis.get('increasing_stress', []) or ['None identified']:
        lines.append(f"    * {f}")
    lines.extend([
        '',
        '  Factors Potentially Reducing Stress:'
    ])
    for f in analysis.get('reducing_stress', []) or ['None identified']:
        lines.append(f"    * {f}")

    lines.extend([
        '',
        '-' * 55,
        '3. PERSONALIZED RECOMMENDATIONS',
        '-' * 55,
        ''
    ])
    for i, s in enumerate(result.get('solutions', []), 1):
        lines.append(f"  {i}. {s['title']}")
        lines.append(f"     {s['description']}")
        lines.append('')

    lines.extend([
        '-' * 55,
        '4. ACTION PLAN',
        '-' * 55,
        '',
        '  - Review the recommendations above and prioritize based on your needs.',
        '  - Implement changes gradually. Small consistent steps are effective.',
        '  - Track your progress and reassess after 2-4 weeks.',
        '  - If stress persists or worsens, seek professional support.',
        '',
        '-' * 55,
        '5. PRECAUTIONS & DISCLAIMER',
        '-' * 55,
        '',
        '  * This report is for informational purposes only and does NOT constitute',
        '    medical or psychological diagnosis.',
        '  * AVOID self-diagnosis. This tool is a screening aid, not a replacement',
        '    for professional evaluation.',
        '  * SEEK professional help if symptoms persist, worsen, or affect daily life.',
        '  * MAINTAIN a healthy lifestyle: adequate sleep, nutrition, and exercise.',
        '  * For crisis support, contact: National Suicide Prevention Lifeline',
        '    1-800-273-8255 (US) or your local mental health crisis line.',
        '',
        '=' * 55,
        'End of Report',
        '=' * 55
    ])

    content = '\n'.join(lines)
    buffer = BytesIO(content.encode('utf-8'))

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f'stress_report_{datetime.now().strftime("%Y%m%d_%H%M")}.txt',
        mimetype='text/plain'
    )

def get_top_contributors(stress_level, responses):
    """
    Logic to identify the most significant factors for a specific result.
    - Low Stress: Shows 'Success Factors' (Positive)
    - Med/High Stress: Shows 'Risk Factors' (Negative)
    """
    factors = []
    # Convert all responses to lowercase strings for easy comparison
    r = {k: str(v).lower() for k, v in responses.items()}

    if stress_level == 'Low':
        # --- LOOK FOR POSITIVE CONTRIBUTORS ---
        if r.get('sleep_quality') == 'good': factors.append("Healthy Sleep")
        if r.get('study_load') == 'low': factors.append("Manageable Workload")
        if r.get('anxiety_level') in ['1', '2']: factors.append("Low Anxiety")
        if r.get('social_support') == 'high': factors.append("Strong Social Support")
        if r.get('academic_performance') == 'good': factors.append("Academic Success")
        if r.get('teacher_student_relationship') == 'good': factors.append("Supportive Teachers")
        if not factors: factors = ["Balanced Lifestyle"]
        
    else:
        # --- LOOK FOR NEGATIVE CONTRIBUTORS ---
        if r.get('sleep_quality') == 'poor': factors.append("Poor Sleep Quality")
        if r.get('study_load') == 'high': factors.append("Heavy Academic Load")
        if r.get('anxiety_level') in ['4', '5']: factors.append("High Anxiety")
        if r.get('social_support') == 'low': factors.append("Lack of Social Support")
        if r.get('bullying') in ['sometimes', 'often']: factors.append("Bullying Issues")
        if r.get('depression') in ['4', '5']: factors.append("Frequent Sadness")
        if r.get('peer_pressure') == 'high': factors.append("Peer Pressure")
        if not factors: factors = ["General Academic Stress"]

    # Return the top 2 as a readable string
    return ", ".join(factors[:2])
# Inside app.py - Update your dashboard route
@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session.get('user_id')
    username = session.get('username') #
    
    # Fetch history and the latest record
    assessments = get_user_assessments(user_id) or [] # Ensure it's a list
    latest = get_latest_assessment(user_id) #
    
    # 1. Fetch result from session, or initialize as an empty dictionary
    result = session.get('result')
    
    # 2. Redirect new users who haven't taken the test yet
    if not latest and not result:
        return redirect(url_for('questionnaire')) #

    # 3. If we have a latest entry but no session 'result', rebuild it
    if not result and latest:
        result = {
            'stress_level': latest['stress_level'],
            'stress_code': latest['stress_code'],
            'confidence': latest.get('confidence', 0),
            'probabilities': latest.get('probabilities', {}),
            'timestamp': latest['created_at']
        } #
    
    # 4. SAFETY FALLBACK: If result is still None, make it an empty dict
    if result is None:
        result = {}

    # 5. Process top factors for the history table
    for a in assessments:
        # Use the logic we created earlier to identify contributors
        a['top_factors'] = get_top_contributors(a['stress_level'], a.get('responses', {}))

    return render_template(
        'dashboard.html', 
        assessments=assessments, 
        **result # result is now guaranteed to be a dictionary
    )


# Inside app.py - Update your admin route
@app.route('/admin')
@login_required
def admin():
    # 1. Get the accuracy/precision numbers
    metrics = load_model_metrics() #
    
    # 2. CALCULATE FEATURE IMPORTANCE (The missing part!)
    model, _ = load_model() #
    feature_importance = {}
    
    if model and hasattr(model, 'feature_importances_'): #
        from config import FEATURE_COLUMNS
        importances = model.feature_importances_
        # Map feature names to their importance values
        raw_importance = {name: float(imp) for name, imp in zip(FEATURE_COLUMNS, importances)}
        # Sort and take top 10
        sorted_importance = sorted(raw_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        feature_importance = {k.replace('_', ' ').title(): round(v * 100, 1) for k, v in sorted_importance}

    # 3. Send EVERYTHING to the template
    return render_template('admin.html', metrics=metrics, feature_importance=feature_importance)


@app.route('/recommendations/<topic>')
def recommendation(topic):
    """Render individual recommendation pages."""
    allowed = {'meditation', 'study-planning', 'sleep-improvement', 'personal-support', 'emergency-help'}
    if topic not in allowed:
        return redirect(url_for('index'))
    return render_template(f'recommendations/{topic}.html')


if __name__ == '__main__':
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    app.run(debug=True, port=5000)
