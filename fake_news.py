import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

app = Flask(__name__, template_folder="templates")

# ================= LOAD DATA =================
fake_path = "fake.csv"
true_path = "true.csv"

# Ensure files exist
if not os.path.exists(fake_path) or not os.path.exists(true_path):
    raise FileNotFoundError("CSV files 'fake.csv' and 'true.csv' must exist in the directory.")

# Load CSVs
fake = pd.read_csv(fake_path)
true = pd.read_csv(true_path)

# Ensure 'text' column exists
for df_name, df in [("fake.csv", fake), ("true.csv", true)]:
    if "text" not in df.columns:
        raise ValueError(f"{df_name} must contain a 'text' column")

# Assign labels
fake["label"] = 0
true["label"] = 1

# Combine and shuffle
data = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)
X = data["text"].astype(str)
y = data["label"]

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= PIPELINE =================
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_df=0.8,
        min_df=3,
        ngram_range=(1, 2)
    )),
    ("clf", MultinomialNB())
])

# ================= TRAIN =================
model.fit(X_train, y_train)

# ================= ACCURACY =================
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {acc:.4f}")

# ================= URL FETCH =================
def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)

def fetch_news_from_url(url):
    if not is_valid_url(url):
        raise ValueError("Invalid URL")

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, timeout=10, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text() for p in paragraphs)

    if len(text.strip()) < 50:
        raise ValueError("Not enough content from URL")

    return text.strip()

# ================= FLASK ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    confidence = ""

    if request.method == "POST":
        try:
            # ---------- INPUT ----------
            url_input = request.form.get("url", "").strip()
            text_input = request.form.get("news", "").strip()

            if url_input:
                text = fetch_news_from_url(url_input)
            elif text_input:
                text = text_input
            else:
                raise ValueError("Empty input")

            if len(text) > 10000:
                raise ValueError("Text too long")

            # ---------- ML PREDICTION ----------
            pred = model.predict([text])[0]
            prob_array = model.predict_proba([text])[0]
            prob = prob_array[int(pred)] * 100 if int(pred) < len(prob_array) else 0.0

            prediction = "REAL NEWS 🟢" if pred == 1 else "FAKE NEWS 🔴"
            confidence = f"{prob:.2f}%"

        except Exception as e:
            print("Error:", e)
            prediction = "Invalid input ❌"
            confidence = ""

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
