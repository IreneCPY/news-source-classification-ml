import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    # load data
    df = pd.read_csv("data/raw/cleaned_headlines.csv")

    # features and labels
    X = df["headline"]
    y = df["source"]

    # train/test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    max_features= [1000, 5000, 10000]
    n_grams = [1, 2, 3]
    
    results = []

    # TF-IDF vectorization
    for max_feature in max_features:    
        for n_gram in n_grams:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=max_feature,
                ngram_range=(1, n_gram)
            )
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            acc = accuracy_score(y_test, y_pred)
            results.append({
                "max_feature": max_feature,
                "n_gram": n_gram,
                "accuracy": acc
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/models/baseline_results.csv", index=False)
    print("Saved to outputs/models/baseline_results.csv")

if __name__ == "__main__":
    main()