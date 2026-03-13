# Student Mental Stress Analysis System

A Final Year Project combining **Machine Learning** and **Web Application** for assessing student stress levels through a questionnaire.

## Features

- **ML Classification**: Logistic Regression, Random Forest, SVM - best model auto-selected
- **Questionnaire**: 20 stress-related questions based on dataset features
- **Personalized Solutions**: Meditation, sleep tips, study planning, counseling suggestions
- **Stress Chart**: Doughnut chart showing Low/Medium/High probabilities
- **Report Download**: Text report with results and recommendations
- **Database Storage**: Optional SQLite storage of assessment responses

## Tech Stack

- **Backend**: Python, Flask
- **ML**: scikit-learn, pandas, numpy
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5

## Project Structure

```
Final Year Project/
├── app.py                 # Flask backend
├── train_model.py         # ML training script
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── dataset/               # Place CSVs here (or keep in root)
│   ├── StressLevelDataset.csv
│   ├── X_train_preprocessed.csv  (optional - train_model uses main CSV)
│   ├── X_test_preprocessed.csv
│   ├── y_train.csv
│   └── y_test.csv
├── model/                 # Saved model (created after training)
│   ├── stress_model.joblib
│   └── feature_scaler.joblib
├── templates/
│   ├── base.html
│   ├── questionnaire.html
│   ├── result.html
│   └── error.html
├── static/
│   └── css/
│       └── style.css
└── utils/
    ├── preprocessing.py   # Form → ML features
    ├── solutions.py       # Personalized recommendations
    └── database.py        # Optional response storage
```

## Setup

### 1. Create Virtual Environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Place Dataset

Ensure `StressLevelDataset.csv` is in the project root or in `dataset/` folder.

### 4. Train the Model

```bash
python train_model.py
```

This will:
- Load the dataset
- Split 80/20 train-test
- Train Logistic Regression, Random Forest, SVM
- Print accuracy, confusion matrix, classification report
- Save the best model and scaler to `model/`

### 5. Run the Web Application

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Usage

1. **Questionnaire**: Fill out all 20 questions
2. **Submit**: Click "Get Stress Analysis"
3. **Result Page**: View stress level, confidence, chart, and personalized recommendations
4. **Download Report**: Get a text report of the assessment

## API

POST `/api/predict` with JSON body containing form fields returns:

```json
{
  "stress_level": "Medium",
  "stress_code": 1,
  "confidence": 72.5,
  "probabilities": {"Low": 15.2, "Medium": 72.5, "High": 12.3}
}
```

## License

Final Year Project - Educational Use
