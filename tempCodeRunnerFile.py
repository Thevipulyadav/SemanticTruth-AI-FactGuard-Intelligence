import pandas as pd
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse

app = Flask(__name__, template_folder="templates")

# ================= FACT CHECK CONFIG =================

FACT_CHECK_API_KEY = "YOUR_API_KEY_HERE"

DISCLAIMER = [
    "❌ Does NOT fact-check (ML only)",
    "❌ Does NOT compare with trusted sources",
    "❌ Does NOT know real-time news"
]

# ================= LOAD DATA =================

fake = pd.read_csv("fake.csv")
true = pd.read_csv("true.csv")

if "text" not in fake.columns or "text" not in true.columns:
    raise ValueError("CSV files must contain a 'text' column")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true]).sample(frac=1, random_state=42)

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
    return parsed.scheme in ("http", "https")

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

# ================= FACT CHECK FUNCTION =================

def fact_check_news(text):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": text[:300],  # API limit safety
        "key": FACT_CHECK_API_KEY
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    results = []

    for claim in data.get("claims", []):
        for review in claim.get("claimReview", []):
            results.append({
                "claim": claim.get("text", "N/A"),
                "publisher": review.get("publisher", {}).get("name", "Unknown"),
                "rating": review.get("textualRating", "Unrated"),
                "url": review.get("url", "")
            })

    return results

# ================= FLASK ROUTE =================

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    confidence = ""
    fact_results = []

    if request.method == "POST":
        try:
            # ---------- INPUT ----------
            if request.form.get("url"):
                text = fetch_news_from_url(request.form["url"])
            else:
                text = request.form.get("news", "")

            if not text.strip():
                raise ValueError("Empty input")

            if len(text) > 10000:
                raise ValueError("Text too long")

            # ---------- ML PREDICTION ----------
            pred = model.predict([text])[0]
            prob = model.predict_proba([text])[0][pred] * 100

            prediction = "REAL NEWS 🟢" if pred == 1 else "FAKE NEWS 🔴"
            confidence = f"{prob:.2f}%"

            # ---------- FACT CHECK ----------
            try:
                fact_results = fact_check_news(text)
            except Exception as fc_error:
                print("Fact-check error:", fc_error)

        except Exception as e:
            print("Error:", e)
            prediction = "Invalid input ❌"
            confidence = ""

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        fact_results=fact_results,
        disclaimer=DISCLAIMER
    )

# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True)
