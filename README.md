# 📧 Email Intelligence Dashboard & Spam Classification Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)

An enterprise-grade, machine-learning-powered application designed to detect spam emails with **>97% accuracy**. This project combines a robust Natural Language Processing (NLP) pipeline with an advanced, interactive Data Discovery Dashboard built in Streamlit. 

---

## 📑 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Key Features](#-key-features)
3. [Machine Learning Pipeline](#-machine-learning-pipeline)
4. [Detailed Analytics & Model Performance](#-detailed-analytics--model-performance)
5. [The Data Discovery Suite (Dashboard)](#-the-data-discovery-suite-dashboard)
6. [Installation & Setup](#-installation--setup)
7. [Usage Guide](#-usage-guide)
8. [Project Architecture](#-project-architecture)
9. [Tech Stack](#-tech-stack)

---

## 🚀 Executive Summary

Email spam remains one of the most persistent cybersecurity and productivity challenges. This project provides a complete, end-to-end solution: starting from raw email frequency vectors, moving through exploratory data analysis and model training, and culminating in a highly polished, interactive web application. 

The underlying engine utilizes a **Logistic Regression** model trained on a 3,000-word vocabulary feature space, strictly isolating legitimate emails (Ham) from malicious/unwanted ones (Spam).

---

## ✨ Key Features

- **Instant Email Classification:** Paste any raw email text into the dashboard to instantly classify it as Safe or Spam.
- **Dynamic NLP Preprocessing:** The application cleans inputs on-the-fly (stripping HTML, removing punctuation, converting cases) and maps it perfectly to the trained 3,000-word vocabulary matrix.
- **Interactive Analytics:** A premium dashboard featuring interactive Plotly visualizations, including donut charts, heatmaps, and violin plots to explore the dataset's topography.
- **High Accuracy Engine:** Deploys a thoroughly evaluated machine learning model achieving over 97% accuracy.

---

## 🧠 Machine Learning Pipeline

The project features a comprehensive data science pipeline documented and executed within `email_spam_analysis.ipynb`.

### 1. Data Processing
- **Dataset:** The engine processes the `emails.csv` dataset, which contains thousands of emails pre-processed into a 3,000-column word frequency matrix.
- **Cleaning:** Dropped non-numeric identifiers (`Email No.`) and validated the integrity of the dataset by checking for missing values (No null values found).
- **Splitting:** The dataset was strictly partitioned using a `75/25` Train-Test split (`random_state=42`) to prevent data leakage and ensure fair evaluation.

### 2. Model Selection & Training
Two probabilistic/linear classifiers were evaluated:
- **Multinomial Naive Bayes:** Achieved ~94.66% accuracy.
- **Logistic Regression (`max_iter=1000`):** Emerged as the superior model.

The winning Logistic Regression model was exported using `joblib` (`spam_model.pkl`), alongside its exact vocabulary feature array (`model_features.pkl`), ensuring the Streamlit application perfectly mirrors the training environment.

---

## 📊 Detailed Analytics & Model Performance

The Logistic Regression classifier demonstrated exceptional reliability in separating spam from ham. 

### Final Test Metrics:
- **Overall Accuracy:** `97.14%`
- **Precision (Spam):** `94.00%` — Extremely low false-positive rate. Legitimate emails are rarely marked as spam.
- **Recall (Spam):** `96.00%` — High detection rate. The vast majority of actual spam is successfully caught.
- **F1-Score (Spam):** `95.00%` — An excellent harmonic mean indicating a highly balanced and robust model.

*Testing Set Shape evaluated: (1293, 3000)*

---

## 📈 The Data Discovery Suite (Dashboard)

The `app.py` Streamlit dashboard is designed to look and feel like a premium analytics platform, divided into two primary tabs:

### Tab 1: Email Classification (Prediction UI)
A clean, minimalist user interface allowing users to paste text and receive instant, absolute verdicts:
- 🟢 **✅ SAFE EMAIL (HAM)**
- 🔴 **🚨 SPAM DETECTED**

### Tab 2: Model & Data Insights
A gorgeous, wide-layout analytics hub utilizing `Plotly` to render deep insights into the dataset:
1. **Class Distribution (Donut Chart):** Visualizes the exact macro-level balance between Spam and Ham in the training corpus.
2. **Top 20 Feature Dominance (Horizontal Bar Chart):** Ranks the most frequently utilized vocabulary words across the dataset using a `Plasma` color scale.
3. **Word Length Distribution (Overlaid Histogram):** Analyzes the total word counts per email, highlighting structural differences in length between spam and ham messages.
4. **Keyword Correlation Heatmap:** An interactive matrix showing the direct statistical correlation of highly predictive keywords (e.g., `money`, `click`, `deal`, `enron`) against the final spam classification.
5. **Feature Intensity Comparison (Violin & Box Plot):** Allows the user to select specific trigger words from a dropdown and visually compare their density and frequency distribution across both classes.

---

## ⚙️ Installation & Setup

Follow these steps to deploy the application on your local machine.

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Email-Spam-Detection.git
cd Email-Spam-Detection
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. Ensure Artifacts are Present
Make sure the following files exist in your root directory (these are generated by the Jupyter Notebook):
- `emails.csv` (The dataset)
- `spam_model.pkl` (The trained ML model)
- `model_features.pkl` (The vocabulary layout)

*If they are missing, run the notebook first:*
```bash
python -m nbconvert --to notebook --execute --inplace email_spam_analysis.ipynb
```

---

## 💻 Usage Guide

To launch the Streamlit dashboard, run the following command in your terminal:
```bash
streamlit run app.py
```
- A local server will start, and your default web browser will automatically open to `http://localhost:8501`.
- Navigate to the **Email Classification** tab to test custom text.
- Navigate to the **Model & Data Insights** tab to explore the interactive visual analytics.

---

## 📁 Project Architecture

```text
Email-Spam-Detection/
│
├── app.py                      # Main Streamlit application
├── emails.csv                  # Core dataset (Word frequency matrix)
├── email_spam_analysis.ipynb   # Machine learning research & training notebook
├── model_features.pkl          # Exported vocabulary array
├── spam_model.pkl              # Serialized Logistic Regression model
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🛠️ Tech Stack

- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), NumPy
- **Machine Learning Engine:** [Scikit-Learn](https://scikit-learn.org/) (Logistic Regression, Train-Test Split, Metrics)
- **Data Visualization:** [Plotly Express](https://plotly.com/python/), Plotly Graph Objects, Seaborn, Matplotlib
- **Model Serialization:** Joblib, Pickle

---

*Built utilizing Python, Machine Learning, and Advanced Data Visualization by Harshal Pednekar.*
