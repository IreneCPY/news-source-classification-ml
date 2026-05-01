import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
csv_file_path = "data/raw/url_with_headlines.csv"  # update if needed
df = pd.read_csv(csv_file_path)

# remove rows with missing headline or url
df = df.dropna(subset=["headline", "url"])

# make sure headline is string
df["headline"] = df["headline"].astype(str)

# 2. Extract label from URL
def get_source(url):
    if "foxnews" in url.lower():
        return 0
    elif "nbcnews" in url.lower():
        return 1
    else:
        return None  # ignore other sources

df["label"] = df["url"].apply(get_source)

# Drop rows that are not FoxNews/NBC
df = df.dropna(subset=["label"])

# 3. Features and labels
X = df["headline"]
y = df["label"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Train model
model = LogisticRegression(max_iter=100)
model.fit(X_train_tfidf, y_train)

# 7. Predict
y_pred = model.predict(X_test_tfidf)

# 8. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))