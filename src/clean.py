import pandas as pd

def clean_headline(headline):
    headline = str(headline).strip().lower()
    headline = pd.Series([headline]).str.replace(r"[^\w\s]", "", regex=True).iloc[0]
    headline = " ".join(headline.split())
    return headline

def main():
    df = pd.read_csv("data/raw/scraped_headlines.csv")
    df = df.dropna(subset=["source", "headline"])
    df = df.drop_duplicates(subset=["headline"])
    df["raw_headline"] = df["headline"]
    df["cleaned_headline"] = df["headline"].apply(clean_headline)
    df = df[df["cleaned_headline"] != ""]
    
    print(f"Number of rows: {len(df)}") # print the number of rows
    print(df["source"].value_counts()) # print the count of each source
    
    df.to_csv("data/processed/cleaned_headlines.csv", index=False)
    print("Saved to data/processed/cleaned_headlines.csv")
    
if __name__ == "__main__":
    main()