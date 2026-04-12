import pandas as pd

def infer_source(headline):
    if "foxnews.com" in headline.lower():
        return "Fox"
    elif "nbcnews.com" in headline.lower():
        return "NBC"
    else:
        return None

def main():
    df = pd.read_csv("data/raw/scraped_headlines.csv")
    df["source"] = df["url"].apply(infer_source)
    df["headline"] = df["headline"].astype(str).str.strip().str.lower()
    df["headline"] = df["headline"].str.replace(r"[^\w\s]", "", regex=True)
    
    df = df.dropna(subset=["source", "headline"])
    df = df.drop_duplicates(subset=["headline"])
    df = df[df["headline"] != ""]
    
    df.to_csv("data/raw/cleaned_headlines.csv", index=False)
    print("Saved to data/raw/cleaned_headlines.csv")
    print(f"Number of rows: {len(df)}") # print the number of rows
    print(df["source"].value_counts()) # print the count of each source
    
if __name__ == "__main__":
    main()