import os
import pickle
from typing import List, Optional

import torch
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


class NewsClassifier(torch.nn.Module):
    def __init__(
        self,
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
        C=0.5,
        weights_path: Optional[str] = None,
    ):
        super().__init__()

        # Dummy parameter so backend torch checkpoint loading does not fail
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        self.vectorizer = FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=word_ngram_range,
                max_features=100000,
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                lowercase=False
            )),
            ("char_tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=char_ngram_range,
                max_features=60000,
                min_df=2,
                sublinear_tf=True,
                lowercase=False
            ))
        ])

        self.classifier = LinearSVC(
            C=C,
            class_weight="balanced",
            random_state=42
        )

        self.is_trained = False

        # Auto-load model.pt if available
        self._load_saved_model(weights_path)

    def _find_model_path(self, weights_path: Optional[str]) -> Optional[str]:
        candidates = []

        if weights_path and os.path.exists(weights_path):
            candidates.append(weights_path)

        candidates.append("model.pt")
        candidates.append(os.path.join(os.path.dirname(__file__), "model.pt"))

        for path in candidates:
            if path and os.path.exists(path):
                return path

        return None

    def _load_saved_model(self, weights_path: Optional[str] = None):
        model_path = self._find_model_path(weights_path)

        if model_path is None:
            return

        checkpoint = torch.load(model_path, map_location="cpu")

        if "payload" not in checkpoint:
            return

        payload_tensor = checkpoint["payload"]
        payload_bytes = bytes(payload_tensor.cpu().numpy().tolist())
        saved = pickle.loads(payload_bytes)

        self.vectorizer = saved["vectorizer"]
        self.classifier = saved["classifier"]
        self.is_trained = True

    def fit(self, X: List[str], y: List[int]):
        X_features = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_features, y)
        self.is_trained = True
        return self

    def predict(self, X: List[str]):
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run train_model.py first to create model.pt.")

        X_features = self.vectorizer.transform(X)
        preds = self.classifier.predict(X_features)
        return preds.tolist()

    def save(self, path: str = "model.pt"):
        payload = pickle.dumps({
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
        })

        payload_tensor = torch.tensor(list(payload), dtype=torch.uint8)

        torch.save(
            {
                "dummy": self.dummy.detach().cpu(),
                "payload": payload_tensor,
            },
            path
        )


def get_model():
    return NewsClassifier()


Model = NewsClassifier