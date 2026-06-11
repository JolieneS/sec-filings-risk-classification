
import joblib

joblib.dump(xgb_model, "best_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

import os

print(os.listdir())

joblib.dump(ada_model, "best_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

import os
print(os.listdir())
