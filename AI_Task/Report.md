# Model Report

## Project Overview

In this project, I built a document intelligence and financial risk classification system using SEC 10-K filings from the Hugging Face dataset. The objective was to process financial reports, extract meaningful information, transform the text into machine-readable features, train multiple classification models, and deploy the best-performing model through FastAPI.

---

# 1. Preprocessing Decisions and Text Cleaning

## Dataset Loading

The dataset was loaded from Hugging Face using streaming mode to avoid downloading the entire dataset, which is very large. Approximately 2000 filings were loaded for experimentation and model training.

## Section Selection

After exploring the dataset, I observed that the "Risk Factors" section contained very few non-empty entries. Therefore, I primarily used the following sections:

* Business
* Management's Discussion and Analysis (MD&A)
* Financial Statements and Supplementary Data

These sections contained the most informative textual content.

## Text Cleaning Steps

The following preprocessing steps were performed:

* Missing values were replaced with empty strings using `fillna("")`
* Selected sections were merged into a single text field called `combined_text`
* Documents with very short content (less than 100 characters) were removed
* Text was standardized before feature extraction

After preprocessing, the final dataset contained 1540 usable filings.

---

# 2. Features Used and Why They Were Chosen

## Risk Score Generation

Since the dataset did not provide explicit risk labels, a custom risk scoring system was created.

A list of financial risk-related keywords was defined, including:

* risk
* debt
* loss
* uncertainty
* litigation
* bankruptcy
* recession
* inflation

The frequency of these terms within each document was counted to generate a risk score.

## Label Creation

Based on the risk score distribution, each filing was assigned one of three classes:

* Low Risk
* Medium Risk
* High Risk

Class Distribution:

| Label       | Count |
| ----------- | ----- |
| Low Risk    | 389   |
| Medium Risk | 767   |
| High Risk   | 384   |

## TF-IDF Features

TF-IDF (Term Frequency – Inverse Document Frequency) was used to convert text into numerical vectors.

Configuration:

* Maximum Features: 5000
* English Stop Words Removed

TF-IDF was selected because:

* It is simple and effective for text classification
* It captures word importance across documents
* It performs well with traditional machine learning models

Generated Feature Shapes:

* Training Features: (1232, 5000)
* Testing Features: (308, 5000)

---

# 3. Model Comparison and Evaluation

Three boosting-based machine learning models were trained and evaluated:

* XGBoost
* AdaBoost
* CatBoost

Evaluation Metrics:

* Accuracy
* Weighted F1 Score
* Confusion Matrix
* Classification Report

## Results

| Model    | Accuracy | F1 Score |
| -------- | -------- | -------- |
| XGBoost  | 0.8214   | 0.8218   |
| AdaBoost | 0.8506   | 0.8499   |
| CatBoost | 0.8019   | 0.8024   |

### Interpretation

* AdaBoost achieved the highest overall accuracy.
* AdaBoost also achieved the highest weighted F1 score.
* XGBoost performed competitively but slightly underperformed compared to AdaBoost.
* CatBoost produced the lowest performance among the three models on this dataset.

The confusion matrix and classification reports indicated that the majority of misclassifications occurred between the Medium Risk and High Risk categories.

---

# 4. Best Model Selection and Justification

## Selected Model: AdaBoost

AdaBoost was selected as the final deployment model because:

* It achieved the highest Accuracy (85.06%)
* It achieved the highest Weighted F1 Score (84.99%)
* It demonstrated the best overall balance across all risk categories
* It consistently outperformed XGBoost and CatBoost on the evaluation dataset

The trained AdaBoost model was serialized and deployed using FastAPI for real-time predictions.

---

# Conclusion

This project successfully implemented an end-to-end NLP classification pipeline for SEC 10-K filings, including preprocessing, feature engineering, model training, evaluation, and deployment. Among the evaluated models, AdaBoost demonstrated the strongest performance and was selected for deployment.
