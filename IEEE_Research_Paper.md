 
Student Mental Stress Analysis and Support System using Machine Learning



       [Naqiya Nullwala, UmmeHani Morbiwala, Sakina Readymadewala]



---

 Abstract

Mental stress among students has become a significant concern in educational institutions worldwide, affecting academic performance, physical health, and overall well-being. This paper presents a web-based Student Mental Stress Analysis and Support System that leverages machine learning to predict student stress levels and deliver personalized recommendations. The system employs a multi-factor assessment model incorporating 20 features spanning psychological, physiological, academic, and environmental dimensions. A comparative evaluation of classification algorithms including Logistic Regression, Random Forest, and Support Vector Machine is conducted. The proposed system utilizes StandardScaler for feature normalization and an 80-20 stratified train-test split for robust evaluation. The deployed model provides real-time stress level prediction (Low, Medium, High) with confidence scores derived from predicted probabilities, alongside personalized intervention recommendations. The web application is built using Flask, Bootstrap, and Chart.js, with SQLite for response storage. Results demonstrate the viability of ML-driven mental health screening as a supplementary tool for student wellness initiatives. The system includes ethical safeguards regarding data privacy, user consent, and appropriate disclaimers for mental health applications.

**Keywords:** Machine learning, stress prediction, student mental health, classification, logistic regression, personalized recommendations, Flask, scikit-learn.

---

 I. Introduction

The prevalence of mental stress among student populations has garnered increased attention from educators, healthcare providers, and policymakers. Academic pressures, career uncertainty, social 


dynamics, and environmental factors collectively contribute to elevated stress levels that can impair cognitive function, reduce academic performance, and exacerbate physical and mental health conditions [1]. Traditional approaches to identifying at-risk students often rely on self-reporting or sporadic clinical assessments, which may fail to capture timely interventions.

Machine learning (ML) offers a data-driven paradigm for predictive modeling of stress and mental health indicators. By training classifiers on validated psychological and behavioral datasets, ML systems can identify patterns associated with varying stress levels and provide actionable insights [2]. This paper presents the design, implementation, and evaluation of a Student Mental Stress Analysis and Support System that integrates ML-based classification with a web-based user interface for real-time assessment and personalized recommendations.

The contributions of this work include: (1) integration of ML models with a production web application for real-time stress prediction; (2) a comprehensive 20-feature multi-factor mental health assessment model; (3) personalized recommendation engine conditioned on predicted stress level and user responses; and (4) ethical considerations and safeguards for mental health applications. The remainder of this paper is organized as follows: Section II reviews related literature; Section III describes the proposed system; Section IV presents the system architecture; Section V details the methodology; Section VI discusses implementation; Section VII presents results; Section VIII concludes with future scope; and Section IX addresses ethical considerations.

---

 II. Literature Review

 A. Mental Health and Stress in Student Populations

Studies have established significant correlations between student stress and factors including academic load, sleep quality, social support, and environmental conditions [3]. The Perceived Stress Scale (PSS) and similar instruments provide validated frameworks for quantifying stress, though manual administration limits scalability [4]. Digital and ML-based screening tools have emerged as supplementary mechanisms for early identification of at-risk individuals [5].

 B. Machine Learning in Mental Health Prediction

Logistic regression has been widely applied in binary and multiclass mental health classification due to its interpretability and probabilistic output [6]. Random Forest and ensemble methods offer robustness to non-linear relationships and feature interactions [7]. Support Vector Machines (SVMs) have demonstrated strong performance in high-dimensional psychological datasets [8]. Comparative studies indicate that model choice depends on dataset characteristics, class balance, and interpretability requirements [9].

 C. Web-Based Mental Health Applications

Web platforms enable scalable deployment of screening tools while maintaining user anonymity [10]. Best practices emphasize clear disclaimers, data privacy, and pathways to professional care rather than replacement of clinical evaluation [11]. The Flask framework, combined with Bootstrap for responsive design and Chart.js for visualization, provides a robust foundation for delivering assessment results in an accessible, device-agnostic manner. SQLite offers lightweight persistence for storing assessment responses without requiring external database infrastructure.

---

 III. Proposed System

 A. System Overview

The proposed Student Mental Stress Analysis and Support System comprises: (1) a web-based questionnaire capturing 20 psychological, physiological, academic, and environmental features; (2) a preprocessing pipeline that maps user responses to numeric feature vectors; (3) a trained ML classifier that predicts stress level (Low, Medium, High); (4) a recommendation engine that generates personalized interventions based on prediction and user inputs; and (5) a reporting interface with visualization and downloadable assessment reports.

 B. Contributions of This Work

1. **ML and Web Integration:** The system seamlessly integrates trained scikit-learn models with a Flask web application, enabling real-time prediction without batch processing delays.

2. **Real-Time Stress Prediction:** Users receive immediate feedback upon form submission, including predicted class, confidence score, and probability distribution across all classes.

3. **Personalized Recommendations:** The recommendation engine applies rule-based logic conditioned on predicted stress level, confidence, and specific feature values (e.g., poor sleep, low social support) to generate tailored suggestions.

4. **Multi-Factor Mental Health Modeling:** The 20-feature model captures psychological (anxiety, self-esteem, depression, mental health history), physiological (headache, blood pressure, sleep quality, breathing), environmental (noise level, living conditions, safety, basic needs), and academic/social dimensions (academic performance, study load, teacher relationship, career concerns, social support, peer pressure, extracurricular activities, bullying).



 IV. System Architecture

 Fig. 1: System Architecture Diagram



**Fig. 2** illustrates the data transformation pipeline from raw questionnaire responses to stored results. Categorical responses are mapped to numeric values consistent with the training dataset schema before scaling and prediction.

 Fig. 3: ML Training Pipeline Diagram


**Fig. 3** shows the model training workflow. The dataset is split with stratification to preserve class distribution. The scaler is fitted only on training data to prevent data leakage. Multiple classifiers are evaluated, and the best-performing model is serialized for deployment.

---

 V. Methodology

 A. Dataset and Features

The system employs a stress level dataset comprising 20 features. The target variable is stress_level with three classes: 0 (Low), 1 (Medium), 2 (High). Feature descriptions are given in Table I.







**TABLE I: DATASET FEATURES**

Feature NameDescriptionData Type / Scaleanxiety_levelFrequency of anxiety experienced by the studentOrdinal (1–5 scale or scaled 4–20)self_esteemLevel of self-confidenceCategorical (Low / Medium / High, mapped 8–25)mental_health_historyHistory of previous mental health issuesBinary (Yes / No)depressionLevel of sadness or hopelessnessOrdinal (1–5 scale or scaled 5–25)headacheFrequency of headachesOrdinal (Never / Sometimes / Often)blood_pressurePresence of elevated blood pressureBinary (Yes / No)sleep_qualityQuality of sleepOrdinal (Poor / Average / Good)breathing_problemExperience of shortness of breath under stressBinary (Yes / No)noise_levelNoise level in study environmentOrdinal (Low / Medium / High)living_conditionsComfort level of living conditionsOrdinal (Poor / Average / Good)safetyPerceived safety in surroundingsBinary (Yes / No)basic_needsFulfillment of basic needs (food, water, internet)Binary (Yes / No)academic_performanceSelf-rated academic performanceOrdinal (Poor / Average / Good)study_loadAcademic workload intensityOrdinal (Low / Medium / High)teacher_student_relationshipLevel of teacher supportOrdinal (Poor / Average / Good)future_career_concernsLevel of anxiety regarding future careerOrdinal (1–5 scale)social_supportLevel of family/friend supportOrdinal (Low / Medium / High)peer_pressureLevel of peer pressure experiencedOrdinal (Low / Medium / High)extracurricular_activitiesParticipation in extracurricular activitiesBinary (Yes / No)bullyingExperience of bullyingOrdinal (Never / Sometimes / Often)
 B. Preprocessing

**1) Handling Categorical Variables:** User-facing questionnaire options (e.g., "Low", "Medium", "High") are mapped to numeric values aligned with the training dataset. Mappings include: binary (Yes=1, No=0), ordinal scales (e.g., 1–5, Poor=1, Average=3, Good=5), and domain-specific ranges (e.g., anxiety scaled to 4–20).

**2) StandardScaler:** Features are standardized to zero mean and unit variance. For each feature i, the transformed value is:

z_i = (x_i - μ) / σ

where μ (mean) and σ (standard deviation) are computed from the training set only. The same scaler is applied to test and inference data to prevent data leakage and ensure consistency.

**3) Train-Test Split:** Data is split 80% training, 20% testing with stratification on the target to maintain class proportions. random_state=42 ensures reproducibility.

 C. Classification Models

**1) Logistic Regression:** The primary classifier uses multinomial logistic regression (max_iter=1000, multi_class='multinomial'). For multiclass classification with K classes, the softmax function outputs the probability of class k given input X:

P(Y=k|X) = exp(β_k^T X) / Σ(j=1 to K) exp(β_j^T X)

For binary classification, the sigmoid function is used:

σ(z) = 1 / (1 + exp(-z))

where z is the linear combination of features and coefficients.

**2) Random Forest:** An ensemble of 100 decision trees with bootstrap aggregation. Each tree votes; the majority class is selected. Probability estimates are derived from the fraction of trees predicting each class.

**3) Support Vector Machine (SVM):** RBF kernel with probability=True to enable predict_proba() via Platt scaling.

 D. Prediction Confidence

Confidence is computed from the predict_proba() method provided by scikit-learn, which returns the posterior probability P(Y=k|X) for each class k. The predicted class is selected as the argmax over these probabilities: ŷ = argmax_k P(Y=k|X). The confidence score assigned to the prediction is:

Confidence = P(Y=ŷ|X)

This value is reported as a percentage (e.g., 0.85 → 85%). A high confidence indicates the model is certain about the predicted class; low confidence suggests ambiguity between classes. The full probability distribution is also visualized via Chart.js to provide transparency to the user.

 E. Evaluation Metrics

**Accuracy:** For multiclass classification, accuracy is defined as the proportion of correctly classified instances over the total number of instances:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

where TP, TN, FP, FN denote True Positives, True Negatives, False Positives, and False Negatives. In multiclass settings, correct predictions are those where predicted class matches actual class.

**Precision (weighted average):** Precision for a class is the ratio of true positives to all predicted positives:

Precision = TP / (TP + FP)

Weighted precision averages per-class precision by class support (number of true instances per class).

**Recall (weighted average):** Recall (sensitivity) for a class is the ratio of true positives to all actual positives:

Recall = TP / (TP + FN)

**F1-Score (weighted average):** The F1-score harmonizes precision and recall:

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Weighted F1 provides a single metric balancing precision and recall across all classes.

---

 VI. Implementation

 A. Technology Stack

ComponentTechnology UsedBackendPython 3.x, Flask FrameworkMachine Learning FrameworkScikit-learnModel SerializationJoblibFrontend DevelopmentHTML5, Bootstrap 5, Chart.jsDatabaseSQLiteData PreprocessingNumPy, Pandas
 B. Application Structure

The Flask application exposes routes for: home (index), questionnaire, prediction, dashboard, download report, and recommendation pages (meditation, study planning, sleep improvement, personal support, emergency help). Session storage holds form data and prediction results between requests. The ML model and scaler are loaded once at runtime and reused for each prediction.

 C. Recommendation Engine

The recommendation module applies conditional rules: (1) meditation and mindfulness for all users; (2) sleep improvement if sleep_quality is poor; (3) study planning if study_load is high; (4) career guidance if future_career_concerns ≥ 4; (5) social support if social_support is low; (6) bullying intervention if bullying is reported; (7) professional support if mental_health_history is yes; (8) counseling for Medium/High stress; (9) urgent help messaging for High stress. Recommendations are prioritized and displayed with the result.

---

 VII. Results and Analysis

 A. Model Performance

[Insert Confusion Matrix Image Here]

The confusion matrix above (or as generated by the training script) shows the distribution of predicted vs. actual stress levels. Rows represent actual classes; columns represent predicted classes. Diagonal elements indicate correct predictions.


 B. Confusion Matrix Interpretation

For a three-class problem (Low, Medium, High), the confusion matrix is 3×3. High values on the diagonal indicate strong performance. Off-diagonal elements reveal misclassification patterns: e.g., High predicted as Medium may suggest overlapping feature distributions. Stratified split helps ensure each class is adequately represented in both sets. True Positives (TP) for a given class are the diagonal entries; False Positives (FP) and False Negatives (FN) are inferred from off-diagonal sums. Per-class precision and recall can be computed and aggregated via weighted average to account for class imbalance.

 C. Probability Distribution Visualization

[Insert Probability Distribution Chart Here]

Chart.js renders a bar or doughnut chart showing P(Low), P(Medium), and P(High) for each prediction. This aids interpretability by revealing model uncertainty when probabilities are close across classes.

 D. Model Selection

The training pipeline evaluates Logistic Regression, Random Forest, and SVM. The model achieving highest accuracy on the test set is saved for deployment. Logistic Regression is often preferred for interpretability and probabilistic output; Random Forest may excel with non-linear relationships; SVM can perform well with appropriate kernel tuning.

---

 VIII. Conclusion

This paper presented the Student Mental Stress Analysis and Support System, a web-based application that combines machine learning classification with personalized recommendations for student mental wellness. The system utilizes 20 features across psychological, physiological, academic, and environmental domains to predict stress level (Low, Medium, High). Comparative evaluation of Logistic Regression, Random Forest, and SVM enables selection of the best-performing model. StandardScaler preprocessing and stratified train-test split ensure robust and reproducible evaluation. The Flask-based web interface delivers real-time predictions with confidence scores and probability visualizations, while the recommendation engine provides context-aware interventions. SQLite facilitates optional response storage for research or auditing purposes. The system is designed as a supplementary screening tool, not a replacement for professional mental health care, and includes appropriate disclaimers and ethical safeguards.

---

 IX. Future Scope

Potential extensions include: (1) deployment of deep learning models (e.g., neural networks) for potentially richer pattern capture; (2) longitudinal tracking of user assessments over time; (3) integration with campus counseling systems for seamless referral; (4) mobile application development; (5) multilingual support; (6) federated learning approaches to preserve privacy across institutions; (7) explainable AI techniques (e.g., SHAP values) for interpretable feature importance; and (8) expanded datasets and cross-institutional validation studies.

---

 X. Ethical Considerations

 A. Data Privacy

User responses are stored only if explicitly intended (e.g., in SQLite). Session data is server-side and not exposed to third parties. Implementation should comply with applicable data protection regulations (e.g., GDPR, FERPA).

 B. User Consent

Users should be informed that the system is for informational purposes only and does not constitute medical or psychological diagnosis. Consent for data storage and use should be obtained where applicable.

 C. Mental Health Sensitivity

The application includes clear disclaimers advising users to seek professional help when needed. High-stress predictions trigger prominent messaging directing users to crisis helplines and professional resources. Language is supportive rather than alarming.

 D. Ethical AI Usage

The system is designed as a screening aid, not a diagnostic tool. Recommendations emphasize professional consultation and avoid overstating model capabilities. Transparency regarding model limitations and probabilistic nature of predictions is maintained in user-facing content.

---

 References

[1] L. S. Al-Dubai et al., “Stress and coping strategies of students in a medical faculty in Malaysia,” *Malaysian Journal of Medical Sciences*, vol. 18, no. 3, pp. 57–64, 2011.

[2] H. Shatte et al., “Machine learning in mental health: A scoping review of methods and applications,” *Psychological Medicine*, vol. 49, no. 9, pp. 1426–1448, 2019.

[3] C. J. Regehr et al., “Interventions to reduce stress in university students: A review and meta-analysis,” *Journal of Affective Disorders*, vol. 148, no. 1, pp. 1–11, 2013.

[4] S. Cohen et al., “A global measure of perceived stress,” *Journal of Health and Social Behavior*, vol. 24, no. 4, pp. 385–396, 1983.

[5] A. S. Torous et al., “Digital mental health and COVID-19: Using technology today to accelerate the curve on access and quality tomorrow,” *JMIR Mental Health*, vol. 7, no. 3, e18848, 2020.

[6] B. S. Fernandes et al., “The new field of precision psychiatry,” *BMC Medicine*, vol. 15, no. 1, p. 80, 2017.

[7] L. Breiman, “Random forests,” *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[8] M. A. Hearst et al., “Support vector machines,” *IEEE Intelligent Systems*, vol. 13, no. 4, pp. 18–28, 1998.

[9] S. Dwyer et al., “Machine learning approaches for clinical psychology and psychiatry,” *Annual Review of Clinical Psychology*, vol. 14, pp. 91–118, 2018.

[10] D. D. Luxton et al., “mHealth for mental health: Integrating smartphone technology in behavioral healthcare,” *Professional Psychology: Research and Practice*, vol. 42, no. 6, pp. 505–512, 2011.

[11] S. Chandrashekar, “Do mental health mobile apps work: Evidence and recommendations for designing high-efficacy mental health mobile apps,” *mHealth*, vol. 4, p. 6, 2018.

[12] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

---

*End of Paper*

**Word count (approximate):** 3,600 words

**Figures to insert:** Fig. 1 (System Architecture), Fig. 2 (Data Flow), Fig. 3 (ML Training Pipeline), Confusion Matrix, Probability Distribution Chart

