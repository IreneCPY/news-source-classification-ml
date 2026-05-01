from preprocess import prepare_data
from model import NewsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def main():
    X, y = prepare_data("data/raw/url_with_headlines.csv")

    print(f"Total samples: {len(X)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

    best_acc = 0.0
    best_config = None
    best_model = None

    configs = [
        # Fine-tune around best C=0.25
        {"C": 0.05, "word_ngram_range": (1, 2), "char_ngram_range": (3, 5)},
        {"C": 0.10, "word_ngram_range": (1, 2), "char_ngram_range": (3, 5)},
        {"C": 0.15, "word_ngram_range": (1, 2), "char_ngram_range": (3, 5)},
        {"C": 0.20, "word_ngram_range": (1, 2), "char_ngram_range": (3, 5)},
        {"C": 0.25, "word_ngram_range": (1, 2), "char_ngram_range": (3, 5)},
        {"C": 0.30, "word_ngram_range": (1, 2), "char_ngram_range": (3, 5)},
        {"C": 0.35, "word_ngram_range": (1, 2), "char_ngram_range": (3, 5)},

        # Try no char features changed slightly
        {"C": 0.20, "word_ngram_range": (1, 2), "char_ngram_range": (2, 5)},
        {"C": 0.25, "word_ngram_range": (1, 2), "char_ngram_range": (2, 5)},
        {"C": 0.30, "word_ngram_range": (1, 2), "char_ngram_range": (2, 5)},

        # Try shorter char range
        {"C": 0.20, "word_ngram_range": (1, 2), "char_ngram_range": (3, 4)},
        {"C": 0.25, "word_ngram_range": (1, 2), "char_ngram_range": (3, 4)},
        {"C": 0.30, "word_ngram_range": (1, 2), "char_ngram_range": (3, 4)},

        # Try word unigrams only, sometimes less overfit
        {"C": 0.20, "word_ngram_range": (1, 1), "char_ngram_range": (3, 5)},
        {"C": 0.25, "word_ngram_range": (1, 1), "char_ngram_range": (3, 5)},
        {"C": 0.30, "word_ngram_range": (1, 1), "char_ngram_range": (3, 5)},
    ]

    for config in configs:
        print("\nTesting config:", config)

        model = NewsClassifier(
            C=config["C"],
            word_ngram_range=config["word_ngram_range"],
            char_ngram_range=config["char_ngram_range"],
            weights_path="__no_existing_model__.pt"
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_config = config
            best_model = model

    print("\nBest config:", best_config)
    print(f"Best validation accuracy: {best_acc:.4f}")

    y_best_pred = best_model.predict(X_val)

    print("\nClassification Report for Best Model:")
    print(classification_report(y_val, y_best_pred))

    best_model.save("model.pt")
    print("Saved best model to model.pt")


if __name__ == "__main__":
    main()