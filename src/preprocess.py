import os
import re
import pandas as pd


def infer_source(url):
    url = str(url).lower()
    if "foxnews.com" in url:
        return "Fox"
    elif "nbcnews.com" in url:
        return "NBC"
    else:
        return None


def clean_headline(headline):
    headline = str(headline).strip().lower()
    headline = re.sub(r"[^\w\s]", "", headline)
    headline = re.sub(r"\s+", " ", headline)
    return headline.strip()


def main():
    input_path = "data/raw/url_with_headlines.csv"
    output_path = "data/processed/cleaned_headlines.csv"

    df = pd.read_csv(input_path)

    # keep rows with url and headline
    df = df.dropna(subset=["url", "headline"])

    # infer source from URL
    df["source"] = df["url"].apply(infer_source)

    # keep only Fox and NBC
    df = df.dropna(subset=["source"])

    # keep original headline
    df["raw_headline"] = df["headline"]

    # clean headline
    df["cleaned_headline"] = df["headline"].apply(clean_headline)

    # remove empty cleaned headlines
    df = df[df["cleaned_headline"] != ""]

    # remove duplicate headline-source pairs
    df = df.drop_duplicates(subset=["cleaned_headline", "source"])

    print(f"Number of rows: {len(df)}")
    print(df["source"].value_counts())

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()