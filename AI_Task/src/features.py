# Stage 2: Feature Engineering

risk_words = [
    "risk",
    "uncertainty",
    "loss",
    "litigation",
    "debt",
    "decline",
    "competition",
    "lawsuit",
    "failure",
    "bankruptcy",
    "volatility",
    "economic",
    "adverse",
    "inflation",
    "recession"
]
def calculate_risk_score(text):

    text = str(text).lower()

    score = 0

    for word in risk_words:
        score += text.count(word)

    return score
df["risk_score"] = df["combined_text"].apply(calculate_risk_score)

print(df["risk_score"].describe())
def assign_label(score):

    if score <= 9:
        return "Low Risk"

    elif score <= 92:
        return "Medium Risk"

    else:
        return "High Risk"
df["label"] = df["risk_score"].apply(assign_label)

print(df["label"].value_counts())
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)
X = vectorizer.fit_transform(df["combined_text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
