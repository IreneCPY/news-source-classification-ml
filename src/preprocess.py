import re
import pandas as pd
from typing import List, Tuple


def extract_source(url: str) -> int:
    url = str(url).lower()

    if "foxnews" in url:
        return 0
    elif "nbcnews" in url:
        return 1
    else:
        return -1


def clean_headline(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def prepare_data(path: str) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(path)

    df = df.dropna(subset=["url", "headline"])
    df["label"] = df["url"].apply(extract_source)
    df = df[df["label"] != -1]

    df["headline_clean"] = df["headline"].apply(clean_headline)
    df = df[df["headline_clean"].str.len() > 0]

    X = df["headline_clean"].tolist()
    y = df["label"].astype(int).tolist()

    return X, y