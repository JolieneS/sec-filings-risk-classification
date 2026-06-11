# Stage 3: Model Training

!pip install xgboost catboost

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

xgb_model.fit(X_train, y_train_encoded)


xgb_preds = xgb_model.predict(X_test)

from sklearn.metrics import accuracy_score

xgb_acc = accuracy_score(y_test_encoded, xgb_preds)

print("XGBoost Accuracy:", xgb_acc)
results = {
    "XGBoost": 0.8214
}
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)
ada_model.fit(X_train, y_train_encoded)
ada_preds = ada_model.predict(X_test)
ada_acc = accuracy_score(y_test_encoded, ada_preds)

print("AdaBoost Accuracy:", ada_acc)
from sklearn.metrics import f1_score

ada_f1 = f1_score(
    y_test_encoded,
    ada_preds,
    average="weighted"
)

print("AdaBoost F1:", ada_f1)

from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    verbose=0,
    random_state=42
)

cat_model.fit(X_train, y_train_encoded)

cat_preds = cat_model.predict(X_test)

cat_acc = accuracy_score(y_test_encoded, cat_preds)
cat_f1 = f1_score(y_test_encoded, cat_preds, average="weighted")

print("CatBoost Accuracy:", cat_acc)
print("CatBoost F1:", cat_f1)

