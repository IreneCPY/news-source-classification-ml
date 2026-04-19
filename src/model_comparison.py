import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    # load data
    df = pd.read_csv("data/processed/cleaned_headlines.csv")
    
    text_col = "cleaned_headline"
    
    X = df[text_col]
    y = df["source"]
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    models = {
        "MultinomialNB": MultinomialNB(),
        "LinearSVC": LinearSVC(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
    }
    
    details = []
    results = []
    
    for model_name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        details.append({
            "model": model_name,
            "accuracy": acc,
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        })
        
        results.append({
            "model": model_name,
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        })
        
        print(f"\n==================== Model: {model_name} ====================")
        print(f"Accuracy: {acc:.4f}")
        print(f"Classification Report: {report}")
        print(f"Confusion Matrix: {cm}")
    
    os.makedirs("outputs/models", exist_ok=True)
    details_df = pd.DataFrame(details)
    details_df.sort_values(by="accuracy", ascending=False)
    details_df.to_csv("outputs/models/model_comparison_details.csv", index=False)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/models/model_comparison_results.csv", index=False)
    
    print("Saved to outputs/models/model_comparison_details.csv")
    print("Saved to outputs/models/model_comparison_results.csv")
                       
                    
if __name__ == "__main__":
    main()