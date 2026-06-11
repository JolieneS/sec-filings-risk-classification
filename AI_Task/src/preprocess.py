# Stage 1: Data Extraction and Preprocessing


!pip install datasets

import pandas as pd
from datasets import load_dataset
dataset = load_dataset(
    "winterForestStump/10-K_sec_filings",
    streaming=True
)
'''
print(dataset)
print(dataset.keys())
data = dataset["001"]
sample = next(iter(data))
print(sample.keys())
print(sample["Risk Factors"][:1000])'''
rows = []

for i, filing in enumerate(data):
    rows.append(filing)

    if i >= 1999:
        break


df = pd.DataFrame(rows)
print(df.shape)
print(df.columns)
'''
df.head()
df.to_csv("sec_filings_2000.csv", index=False)


import os

print(os.listdir())
risk_text = df["Risk Factors"].iloc[0]

print(type(risk_text))
print(len(risk_text))
print(risk_text[:500])

df["Risk Factors"].isnull().sum()
print(df["Risk Factors"].head(10))
for i in range(10):
    text = df["Risk Factors"].iloc[i]
    print(f"Row {i}: Length = {len(str(text))}")
(df["Risk Factors"].str.len() > 0).sum()
important_cols = [
    "Business",
    "Risk Factors",
    "Management’s Discussion and Analysis of Financial Condition and Results of Operations",
    "Financial Statements and Supplementary Data"
]

for col in important_cols:
    non_empty = (df[col].fillna("").str.len() > 0).sum()
    print(col)
    print("Non-empty rows:", non_empty)
    print()
    '''
df["Business"] = df["Business"].fillna("")
df["Management’s Discussion and Analysis of Financial Condition and Results of Operations"] = df["Management’s Discussion and Analysis of Financial Condition and Results of Operations"].fillna("")
df["Financial Statements and Supplementary Data"] = df["Financial Statements and Supplementary Data"].fillna("")
df["combined_text"] = (
    df["Business"] + " " +
    df["Management’s Discussion and Analysis of Financial Condition and Results of Operations"] + " " +
    df["Financial Statements and Supplementary Data"]
)
'''print(df["combined_text"].iloc[0][:1000])'''
df = df[df["combined_text"].str.len() > 100]

print(df.shape)
df["Business"] = df["Business"].fillna("")
df["Management’s Discussion and Analysis of Financial Condition and Results of Operations"] = df["Management’s Discussion and Analysis of Financial Condition and Results of Operations"].fillna("")
df["Financial Statements and Supplementary Data"] = df["Financial Statements and Supplementary Data"].fillna("")

df["combined_text"] = (
    df["Business"] + " " +
    df["Management’s Discussion and Analysis of Financial Condition and Results of Operations"] + " " +
    df["Financial Statements and Supplementary Data"]
)

df = df[df["combined_text"].str.len() > 100]

print(df["combined_text"].iloc[0][:1000])
