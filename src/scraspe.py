from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import pandas as pd

headers = {
    "User-Agent": "Mozilla/5.0"
}

def get_headline(url):
    try:
        res = requests.get(url, timeout=5, headers=headers)
        if res.status_code != 200:
            return None
        soup = BeautifulSoup(res.text, "html.parser")
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        
        return None
    except:
        return None


def main():
    df = pd.read_csv("data/raw/url_only_data.csv")
    results = []
    # test with first 20 rows
    df = df.head(100)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        url = row["url"]
        headline = get_headline(url)
        results.append({
            "url": url,
            "headline": headline
        })

    out_df = pd.DataFrame(results)
    # remove rows with no headline
    out_df = out_df.dropna(subset=["headline"])
    out_df.to_csv("data/raw/scraped_headlines.csv", index=False)
    print("Saved to data/raw/scraped_headlines.csv")


if __name__ == "__main__":
    main()