
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputText):

    features = vectorizer.transform([data.text])

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    confidence = float(np.max(probabilities))

    label = label_encoder.inverse_transform([prediction])[0]

    return {
        "label": label,
        "confidence": round(confidence,4)
    }
