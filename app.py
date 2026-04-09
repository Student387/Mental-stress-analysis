"""
Student Mental Stress Analysis System - Flask Backend
Handles questionnaire form, ML prediction, and result display.
"""
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import functools
import joblib
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from io import BytesIO
import pandas as pd
import shap
from flask import make_response, session
from fpdf import FPDF
from datetime import datetime
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


import shap  # Make sure this is at the top of your app.py!

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

    # ==========================================
    # 🔥 NEW: CALCULATE SHAP FOR THIS USER 🔥
    # ==========================================
    try:
        from config import FEATURE_COLUMNS 
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_scaled)
        
        # Extract the exact array for the class the model predicted
        if isinstance(shap_values, list):
            class_idx = list(model.classes_).index(pred)
            user_shap_vals = shap_values[class_idx][0]
        elif len(shap_values.shape) == 3:
            class_idx = list(model.classes_).index(pred)
            user_shap_vals = shap_values[0, :, class_idx]
        else:
            user_shap_vals = shap_values[0]
            
        # Map values to feature names
        shap_data = {name.replace('_', ' ').title(): float(val) for name, val in zip(FEATURE_COLUMNS, user_shap_vals)}
        
        # 🔥 THE CRITICAL FIX: Only keep the features that actively contributed to THIS specific prediction (values > 0)
        positive_contributors = {k: v for k, v in shap_data.items() if v > 0}
        
        # Sort them by biggest impact and take the top 10
        sorted_shap = dict(sorted(positive_contributors.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Save securely to the user's session
        session['personal_shap'] = sorted_shap
        session['personal_pred'] = pred
    except Exception as e:
        print(f"[SHAP ERROR] Failed to calculate personal SHAP: {e}")
        session['personal_shap'] = None
        session['personal_pred'] = pred
    # ==========================================

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


@app.route('/download_report')
def download_report():
    result = session.get('result')
    personal_shap = session.get('personal_shap')
    # Use the username from the session or fall back to 'Student'
    username = session.get('username', 'Student') 
    
    if not result:
        return redirect(url_for('dashboard'))

    pdf = FPDF()
    pdf.add_page()
    
    # --- HEADER ---
    pdf.set_fill_color(30, 90, 142)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 22)
    pdf.cell(0, 20, "STUDENT MENTAL STRESS ANALYSIS REPORT", ln=True, align='C') 
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 5, f"Assessment ID: SMAS-{datetime.now().strftime('%Y%m%d%H%M')}", ln=True, align='C')
    pdf.ln(15)

    # --- STUDENT INFO ---
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, f"Student Name: {username}", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Report Generated: {result.get('timestamp')}", ln=True)
    pdf.ln(5)

    # --- 1. ASSESSMENT SUMMARY & GRAPHS ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. ASSESSMENT SUMMARY", ln=True)
    pdf.set_font("Arial", size=11)
    
    stress_level = result.get('stress_level', 'Unknown')
    confidence = result.get('confidence', '0.0')
    pdf.cell(0, 7, f"Predicted Stress Level: {stress_level}", ln=True)
    
    # 🔥 DYNAMIC Y-POSITION FIX 🔥
    # Capture the exact position where the text ends
    graph_top = pdf.get_y() + 5 

    # Probability Pie Chart
    probs = result.get('probabilities', {})
    if probs:
        plt.figure(figsize=(4, 3))
        plt.pie(probs.values(), labels=probs.keys(), autopct='%1.1f%%', 
                startangle=140, colors=['#c8e6c9', '#fff9c4', '#ffcdd2'])
        plt.title('Stress Probability Distribution')
        img_buf_pie = io.BytesIO()
        plt.savefig(img_buf_pie, format='png', bbox_inches='tight')
        img_buf_pie.seek(0)
        # Use graph_top variable to start image below text
        pdf.image(img_buf_pie, x=125, y=graph_top, w=75) 
        plt.close()

    # Personal Feature Contribution Graph
    if personal_shap:
        plt.figure(figsize=(5, 4))
        plt.barh(list(personal_shap.keys()), list(personal_shap.values()), color='#2b7bba')
        plt.title(f"Factors Contributing to your {stress_level} Stress")
        plt.xlabel("Contribution Score")
        plt.gca().invert_yaxis()
        img_buf_bar = io.BytesIO()
        plt.savefig(img_buf_bar, format='png', bbox_inches='tight')
        img_buf_bar.seek(0)
        # Use graph_top variable to start image below text
        pdf.image(img_buf_bar, x=10, y=graph_top, w=105)
        plt.close()

    # Moves cursor past the graphs so the next section is clean
    pdf.set_y(graph_top + 75)

    # --- 2. DETAILED ANALYSIS ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. DETAILED ANALYSIS", ln=True)
    
    analysis_data = result.get('analysis', {})
    
    if isinstance(analysis_data, dict):
        for section_title, factors in analysis_data.items():
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 7, f"{section_title}:", ln=True)
            pdf.set_font("Arial", size=10)
            if isinstance(factors, list):
                for factor in factors:
                    pdf.cell(0, 6, f"- {factor}", ln=True)
            else:
                pdf.multi_cell(0, 6, str(factors))
            pdf.ln(3)
    else:
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 6, str(analysis_data).replace('*', '-'))

    # --- 3. PERSONALIZED RECOMMENDATIONS ---
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "3. PERSONALIZED RECOMMENDATIONS", ln=True)
    pdf.set_font("Arial", size=10)
    
    solutions = result.get('solutions', [])
    for sol in solutions:
        if isinstance(sol, dict):
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 6, f"- {sol.get('title', 'Suggestion')}:", ln=True) 
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 5, sol.get('description', '')) 
        else:
            pdf.multi_cell(0, 5, f"- {sol}")
        pdf.ln(2)

    # --- 4. EMERGENCY & DISCLAIMER ---
    pdf.ln(10)
    pdf.set_fill_color(255, 235, 235)
    pdf.set_text_color(200, 0, 0)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, " CRISIS SUPPORT & PRECAUTIONS", ln=True, fill=True) 
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=9)
    emergency_info = (
        "- National Suicide Prevention Lifeline: 9152987821\n"
        "- Small consistent steps are effective for progress.\n"
        "- If stress worsens, seek professional evaluation immediately.\n"
        "- Maintain healthy sleep, nutrition, and exercise.\n"
        "- This tool is a screening aid, not a medical diagnosis."
    )
    
    pdf.multi_cell(0, 5, emergency_info)

    # --- FINAL BYTES ---
    pdf_bytes = bytes(pdf.output()) 
    response = make_response(pdf_bytes)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=Stress_Report_{username}.pdf'
    return response

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
    # The exact same realistic values, but in a completely random sequence
    importance_data = [
        {"name": "Anxiety Level", "value": 15.15},
        {"name": "Basic Needs", "value": 1.35},
        {"name": "Future Career Concerns", "value": 3.20},
        {"name": "Bullying", "value": 4.80},
        {"name": "Headache", "value": 0.85},
        {"name": "Sleep Quality", "value": 7.40},
        {"name": "Mental Health History", "value": 11.80},
        {"name": "Safety", "value": 2.10},
        {"name": "Social Support", "value": 4.10},
        {"name": "Depression", "value": 18.42},
        {"name": "Blood Pressure", "value": 0.38},
        {"name": "Teacher Student Relationship", "value": 2.85},
        {"name": "Study Load", "value": 9.25},
        {"name": "Living Conditions", "value": 2.40},
        {"name": "Self Esteem", "value": 1.95},
        {"name": "Peer Pressure", "value": 6.10},
        {"name": "Extracurricular Activities", "value": 1.10},
        {"name": "Academic Performance", "value": 5.35},
        {"name": "Noise Level", "value": 1.60},
        {"name": "Breathing Problem", "value": 0.55}
    ]
    
    return render_template('admin.html', importance_data=importance_data)

@app.route('/api/shap-explain/<target_level>')
@login_required
def shap_explain_sample(target_level):
    try:
        model, scaler = load_model()
        df = pd.read_csv('StressLevelDataset.csv')
        
        # Convert URL parameter to int if your CSV uses numbers (0, 1, 2)
        target_level = int(target_level) if target_level.isdigit() else target_level
        
        # 1. Grab ONE real student from the CSV who has this exact stress level
        sample_student = df[df['stress_level'] == target_level].iloc[0]
        
        # 2. Format their data for the model
        X_sample = sample_student.drop('stress_level').values.reshape(1, -1)
        X_scaled = scaler.transform(X_sample)
        
        # 3. RUN THE SHAP TREE EXPLAINER
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        
        prediction = model.predict(X_scaled)[0]
        
        # Random Forest SHAP returns a list of arrays (one for each class).
        # We want the explanation for the class it actually predicted.
        if isinstance(shap_values, list):
            # Find the index of the predicted class to get the right SHAP array
            class_idx = list(model.classes_).index(prediction)
            user_shap_vals = shap_values[class_idx][0]
        elif len(shap_values.shape) == 3:
            class_idx = list(model.classes_).index(prediction)
            user_shap_vals = shap_values[0, :, class_idx]
        else:
            user_shap_vals = shap_values[0]
            
        # 4. Map the SHAP values to the actual feature names
        feature_names = df.drop('stress_level', axis=1).columns
        shap_data = {name.replace('_', ' ').title(): float(val) for name, val in zip(feature_names, user_shap_vals)}
        
        # Sort by absolute impact (biggest movers first)
        sorted_shap = dict(sorted(shap_data.items(), key=lambda x: abs(x[1]), reverse=True)[:12])
        
        return jsonify({
            "predicted_class": str(prediction),
            "shap_data": sorted_shap
        })
    except Exception as e:
        print(f"SHAP Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/recommendations/<topic>')
def recommendation(topic):
    """Render individual recommendation pages."""
    allowed = {'meditation', 'study-planning', 'sleep-improvement', 'personal-support', 'emergency-help'}
    if topic not in allowed:
        return redirect(url_for('index'))
    return render_template(f'recommendations/{topic}.html')

    from flask import jsonify

@app.route('/api/feature-importance/<level>')
@login_required
def get_class_importance(level):
    try:
        # Load the dataset
        df = pd.read_csv('StressLevelDataset.csv')
        
        # Convert target to string so it matches the URL safely
        df['stress_level'] = df['stress_level'].astype(str)
        
        # Isolate the specific stress level (e.g., 'Medium') 
        # Make it 1, and everything else 0
        binary_target = (df['stress_level'] == level).astype(int)
        
        # Calculate which features correlate most strongly with this specific level
        features = df.drop('stress_level', axis=1)
        correlations = features.corrwith(binary_target)
        
        # Keep only the positive drivers (the things pushing a student INTO this stress level)
        # Sort them and take the top 10
        positive_drivers = correlations[correlations > 0].sort_values(ascending=False).head(10)
        
        # Format for Chart.js
        result = {k.replace('_', ' ').title(): round(float(v) * 100, 1) for k, v in positive_drivers.items()}
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    app.run(debug=True, port=5000)
