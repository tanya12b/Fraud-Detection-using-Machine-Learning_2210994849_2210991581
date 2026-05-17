
# Fraud Detection using Machine Learning

## Project Overview

This project implements a Machine Learning based Fraud Detection System capable of identifying fraudulent financial transactions using multiple supervised and unsupervised learning algorithms. The system focuses on handling highly imbalanced transaction datasets and improving fraud detection performance using feature engineering and cost-sensitive learning techniques.

The project includes complete data preprocessing, model training, evaluation, feature importance analysis, anomaly detection, and output report generation.

---

# Features

- Fraud Detection using Machine Learning
- Supervised and Unsupervised Learning Models
- Cost-Sensitive Learning
- Feature Engineering
- Data Preprocessing and Scaling
- Model Evaluation using Multiple Metrics
- Automatic CSV and TXT Output Generation
- Model Saving using Joblib
- Real-Time Prediction Ready
- Scalable Project Structure

---

# Machine Learning Models Used

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. Isolation Forest

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Flask

---

# Dataset Information

The project uses a Credit Card Fraud Detection Dataset containing financial transaction records.

Dataset Features:
- Transaction Time
- Transaction Amount
- PCA-based anonymized features (V1 to V28)
- Class Label
    - 0 → Legitimate Transaction
    - 1 → Fraudulent Transaction

---

# Project Structure

```text
fraud-detection-ml/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   └── saved_model.pkl
│
├── outputs/
│   ├── fraud_detection_results.csv
│   ├── confusion_matrix.csv
│   ├── feature_importance.csv
│   ├── predictions_sample.csv
│   ├── classification_report.txt
│   └── model_summary.txt
│
├── train_model.py
├── utils.py
├── app.py
├── requirements.txt
└── README.md
```

---

# Data Preprocessing

The following preprocessing techniques are applied:

- Handling missing values
- Feature scaling using StandardScaler
- Feature engineering
- Behaviour score generation
- Transaction frequency analysis
- Train-test splitting
- Class imbalance handling

---

# Feature Engineering

Additional features are generated to improve fraud detection performance:

- Transaction Frequency
- Behaviour Score
- Spending Pattern Analysis
- Temporal Behaviour Analysis

Feature engineering improves model learning capability and increases overall fraud detection accuracy.

---

# Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC Score

Recall and F1-score are prioritized due to the highly imbalanced nature of fraud datasets.

---

# Output Files Generated

After successful execution, the following files are automatically generated inside the `outputs/` folder:

1. fraud_detection_results.csv
2. confusion_matrix.csv
3. feature_importance.csv
4. predictions_sample.csv
5. classification_report.txt
6. model_summary.txt

---

# How to Run the Project

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2: Place Dataset

Copy the dataset file here:

```text
data/creditcard.csv
```

---

## Step 3: Run Training Script

```bash
python train_model.py
```

---

## Step 4: Run Flask API

```bash
python app.py
```

---

# Sample Output

```text
Training Logistic Regression...
Accuracy : 0.9775
Precision: 0.064
Recall   : 0.8784
F1-score : 0.1193
AUC      : 0.969

Training Random Forest...
Accuracy : 0.9993
Precision: 0.8594
Recall   : 0.7432
F1-score : 0.7971
AUC      : 0.9571
```

---

# Key Findings

- Random Forest achieved the best overall performance.
- Cost-sensitive learning significantly improved fraud detection recall.
- Isolation Forest effectively detected anomalous transaction patterns.
- Feature engineering improved F1-score and model robustness.
- Recall is the most important metric for fraud detection systems.

---

# Future Improvements

- Real-time transaction monitoring
- Deep Learning integration
- LSTM and Transformer-based models
- Explainable AI for fraud prediction
- Cloud deployment and scalability

---

# Conclusion

This project demonstrates that Machine Learning techniques combined with feature engineering, anomaly detection, and cost-sensitive learning can significantly improve fraud detection performance in highly imbalanced financial datasets. The implemented framework provides a scalable and practical solution for modern fraud detection systems.

---

# Author

Tanya Bansal  
Chitkara University Institute of Engineering & Technology  
Chitkara University, Punjab, India
````
