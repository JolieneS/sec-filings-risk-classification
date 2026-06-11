# SEC Filings Risk Classification using NLP and Machine Learning

## Overview

In this project, I built an end-to-end document intelligence and classification system using SEC 10-K financial filings. The objective was to process raw financial reports, extract meaningful information, convert text into machine-readable features, train multiple machine learning models, compare their performance, and deploy the best-performing model through a FastAPI endpoint.

This project was completed as part of the AI & Automation Task conducted by E-Cell.

---

# Problem Statement

Financial reports contain a large amount of valuable information, but manually analyzing them is time-consuming and inefficient.

The goal of this project was to:

* Load SEC 10-K filings from Hugging Face
* Preprocess and clean financial text
* Extract meaningful sections
* Convert text into numerical features
* Train multiple classification models
* Compare performance
* Deploy the best model through FastAPI

---

# Dataset

Dataset Used:

winterForestStump/10-K_sec_filings

Since the dataset is extremely large, I used Hugging Face streaming mode and loaded approximately 2000 filings instead of downloading the entire dataset.

This approach reduced memory consumption and significantly improved processing speed.

---

# Project Workflow

## Stage 1: Data Extraction & Preprocessing

### Dataset Loading

I loaded the dataset using the Hugging Face datasets library with streaming enabled.

The dataset contains several sections such as:

* Business
* Risk Factors
* Management Discussion & Analysis (MD&A)
* Financial Statements
* Legal Proceedings
* Executive Compensation

### Data Exploration

After exploring the dataset, I found that:

* Risk Factors contained very few usable entries
* Business contained rich descriptive information
* MD&A contained detailed financial discussions
* Financial Statements contained quantitative information

### Text Preparation

I combined the most useful sections:

* Business
* Management Discussion & Analysis

to create a unified document representation for each filing.

I then:

* Removed missing values
* Filtered empty records
* Standardized the text

Final dataset size after cleaning:

1540 filings

### Screenshot

<img width="863" height="372" alt="Screenshot 2026-06-11 135009" src="https://github.com/user-attachments/assets/4d3c459e-9e18-4808-b9ba-f19e5f9d2252" />


---

## Stage 2: Label Creation

The dataset did not contain direct risk labels.

Therefore, I created a custom Financial Risk Classification task.

### Risk Score Calculation

I created a list of risk-related keywords such as:

* risk
* debt
* uncertainty
* litigation
* recession
* loss
* bankruptcy

For each filing, I counted the occurrence of these terms.

This produced a numerical risk score.

### Risk Classes

Using quartile-based thresholds, I assigned:

* Low Risk
* Medium Risk
* High Risk

This transformed the problem into a supervised three-class classification task.

### Class Distribution

Low Risk: 389

Medium Risk: 767

High Risk: 384

### Screenshot

Insert class distribution screenshot here.

---

## Stage 3: Feature Engineering

Machine learning models cannot understand raw text directly.

Therefore, I converted the text into numerical vectors using TF-IDF.

### TF-IDF

TF-IDF (Term Frequency – Inverse Document Frequency) assigns importance scores to words based on:

* Frequency within a document
* Rarity across all documents

Configuration:

* Maximum Features = 5000
* English Stop Words Removed

Generated Shapes:

Training Features:

(1232, 5000)

Testing Features:

(308, 5000)

### Screenshot

<img width="128" height="113" alt="image" src="https://github.com/user-attachments/assets/a3d8a236-fef9-4a39-a338-f0420e604ba8" />


---

## Stage 4: Model Training

I trained and compared three boosting-based machine learning models.

### XGBoost

A gradient boosting algorithm that builds trees sequentially while correcting previous errors.

Accuracy:

82.14%

### AdaBoost

An adaptive boosting algorithm that focuses on difficult samples during training.

Accuracy:

85.06%

### CatBoost

A boosting algorithm optimized for categorical data.

Accuracy:

80.19%

---

# Model Evaluation

To compare models fairly, I evaluated:

* Accuracy
* F1 Score
* Classification Report
* Confusion Matrix

## Performance Comparison

| Model    | Accuracy | F1 Score |
| -------- | -------- | -------- |
| XGBoost  | 0.8214   | 0.8218   |
| AdaBoost | 0.8506   | 0.8499   |
| CatBoost | 0.8019   | 0.8024   |

### Best Model

AdaBoost achieved the highest overall Accuracy and Weighted F1 Score.

Therefore, AdaBoost was selected as the final deployment model.

### Screenshot

<img width="175" height="90" alt="image" src="https://github.com/user-attachments/assets/2f859721-903a-4ee2-a656-d101bdd643c9" />


### Screenshot

<img width="293" height="233" alt="image" src="https://github.com/user-attachments/assets/0365f7cc-9f15-43f9-b621-af040c56974e" />


---

# Model Serialization

The final deployment artifacts were saved using Joblib.

Generated files:

* best_model.pkl
* tfidf_vectorizer.pkl
* label_encoder.pkl

These files allow predictions without retraining the model.

---

# FastAPI Deployment

After selecting the best model, I created a FastAPI application.

### Endpoint

POST /predict

Input:

```json
{
  "text": "sample filing text"
}
```

Output:

```json
{
  "label": "Medium Risk",
  "confidence": 0.85
}
```

### Features

* Fast prediction endpoint
* JSON response format
* Confidence score output
* Automatic Swagger documentation

Swagger Docs:

/docs

### Screenshot

Insert FastAPI docs screenshot.

### Screenshot

Insert prediction response screenshot.

---

# Repository Structure

```text
sec-filings-risk-classification/

├── data/
├── notebooks/
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── api/
│   └── app.py
│
├── models/
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── README.md
├── report.md
└── requirements.txt
```

---

# Future Improvements

If given additional time, I would extend the project by:

* Hyperparameter tuning using Optuna
* SHAP-based explainability
* FinBERT embeddings
* BERT-based classification models
* Cloud deployment using Render or Railway

---

# Key Learnings

Through this project, I learned:

* Loading large datasets efficiently using Hugging Face streaming
* Text preprocessing for NLP applications
* Feature engineering using TF-IDF
* Training and comparing boosting algorithms
* Model evaluation using multiple metrics
* FastAPI deployment workflows
* End-to-end machine learning pipeline development

---

# Acknowledgements

I would like to sincerely thank the E-Cell team for designing this challenge.

This task provided hands-on exposure to real-world NLP pipelines, machine learning workflows, model evaluation techniques, and deployment practices. It pushed me to work with large-scale financial datasets and helped me gain practical experience in building an end-to-end AI solution.

Thank you to E-Cell for creating an engaging and valuable learning opportunity.
