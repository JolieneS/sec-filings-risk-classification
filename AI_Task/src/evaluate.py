# Stage 4: Model Evaluation

import pandas as pd

results = pd.DataFrame({
    "Model": ["XGBoost", "AdaBoost", "CatBoost"],
    "Accuracy": [xgb_acc, ada_acc, cat_acc],
    "F1 Score": [
        f1_score(y_test_encoded, xgb_preds, average="weighted"),
        ada_f1,
        cat_f1
    ]
})

results
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test_encoded, cat_preds)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("CatBoost Confusion Matrix")
plt.show()
from sklearn.metrics import classification_report

print(classification_report(
    y_test_encoded,
    cat_preds,
    target_names=le.classes_
))
