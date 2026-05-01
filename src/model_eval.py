from preprocess import prepare_data
from model_svm import NewsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # Load data
    X, y = prepare_data("data/raw/url_with_headlines.csv")

    print(f"Total samples: {len(X)}")

    # Same validation split style as training
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Validation size: {len(X_val)}")

    # Load trained model from model.pt
    model = NewsClassifier(weights_path="model.pt")

    # Predict
    y_pred = model.predict(X_val)

    # Evaluate
    acc = accuracy_score(y_val, y_pred)

    print("\nValidation Accuracy:")
    print(acc)

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        y_pred,
        target_names=["Fox", "NBC"]
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))


if __name__ == "__main__":
    main()