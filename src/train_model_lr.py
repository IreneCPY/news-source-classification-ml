from preprocess import prepare_data
from model_lr import NewsClassifier

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

    configs = [
        {"C": 0.05, "word_min_df": 1, "char_min_df": 2, "max_df": 0.98, "class_weight": "balanced"},
        {"C": 0.10, "word_min_df": 1, "char_min_df": 2, "max_df": 0.98, "class_weight": "balanced"},
        {"C": 0.20, "word_min_df": 1, "char_min_df": 2, "max_df": 0.98, "class_weight": "balanced"},
        {"C": 0.50, "word_min_df": 1, "char_min_df": 2, "max_df": 0.98, "class_weight": "balanced"},
        {"C": 1.00, "word_min_df": 1, "char_min_df": 2, "max_df": 0.98, "class_weight": "balanced"},
        {"C": 2.00, "word_min_df": 1, "char_min_df": 2, "max_df": 0.98, "class_weight": "balanced"},

        {"C": 0.10, "word_min_df": 2, "char_min_df": 2, "max_df": 0.98, "class_weight": "balanced"},
        {"C": 0.20, "word_min_df": 2, "char_min_df": 2, "max_df": 0.98, "class_weight": "balanced"},
        {"C": 0.50, "word_min_df": 2, "char_min_df": 2, "max_df": 0.98, "class_weight": "balanced"},
        {"C": 1.00, "word_min_df": 2, "char_min_df": 2, "max_df": 0.98, "class_weight": "balanced"},
        {"C": 2.00, "word_min_df": 2, "char_min_df": 2, "max_df": 0.98, "class_weight": "balanced"},
        {"C": 4.00, "word_min_df": 2, "char_min_df": 2, "max_df": 0.98, "class_weight": "balanced"},

        {"C": 0.50, "word_min_df": 2, "char_min_df": 3, "max_df": 0.98, "class_weight": "balanced"},
        {"C": 1.00, "word_min_df": 2, "char_min_df": 3, "max_df": 0.98, "class_weight": "balanced"},
        {"C": 2.00, "word_min_df": 2, "char_min_df": 3, "max_df": 0.98, "class_weight": "balanced"},

        {"C": 0.50, "word_min_df": 2, "char_min_df": 2, "max_df": 0.95, "class_weight": "balanced"},
        {"C": 1.00, "word_min_df": 2, "char_min_df": 2, "max_df": 0.95, "class_weight": "balanced"},
        {"C": 2.00, "word_min_df": 2, "char_min_df": 2, "max_df": 0.95, "class_weight": "balanced"},

        {"C": 0.50, "word_min_df": 2, "char_min_df": 2, "max_df": 0.98, "class_weight": None},
        {"C": 1.00, "word_min_df": 2, "char_min_df": 2, "max_df": 0.98, "class_weight": None},
        {"C": 2.00, "word_min_df": 2, "char_min_df": 2, "max_df": 0.98, "class_weight": None},
    ]

    best_acc = 0.0
    best_config = None
    best_model = None

    for config in configs:
        print("\nTesting config:", config)

        model = NewsClassifier(
            C=config["C"],
            word_ngram_range=(1, 2),
            char_ngram_range=(2, 5),
            word_min_df=config["word_min_df"],
            char_min_df=config["char_min_df"],
            max_df=config["max_df"],
            class_weight=config["class_weight"],
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

    final_model = NewsClassifier(
        C=best_config["C"],
        word_ngram_range=(1, 2),
        char_ngram_range=(2, 5),
        word_min_df=best_config["word_min_df"],
        char_min_df=best_config["char_min_df"],
        max_df=best_config["max_df"],
        class_weight=best_config["class_weight"],
        weights_path="__no_existing_model__.pt"
    )

    final_model.fit(X, y)
    final_model.save("model_lr.pt")

    print("Saved final Logistic Regression model trained on all data to model_lr.pt")


if __name__ == "__main__":
    main()