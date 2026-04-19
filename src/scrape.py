import errno
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import pandas as pd

headers = {
    "User-Agent": "Mozilla/5.0"
}

df = pd.read_csv("data/raw/url_only_data.csv")

fox_df = df[df["url"].str.contains("foxnews.com")]
nbc_df = df[df["url"].str.contains("nbcnews.com")]

data = pd.concat([fox_df, nbc_df]).reset_index(drop=True)

# df_sample = pd.concat([
#     fox_df.head(500),
#     nbc_df.head(500)
# ])

# df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

def infer_source(url):
    url = url.lower()
    if "foxnews.com" in url:
        return "Fox"
    elif "nbcnews.com" in url:
        return "NBC"
    else:
        return None

def get_headline(url):
    try:
        res = requests.get(url, timeout=5, headers=headers)
        if res.status_code != 200:
            return None, res.status_code
        
        soup = BeautifulSoup(res.text, "html.parser")
        
        # h1 = soup.find("h1")
        selectors = [
            ("h1", None),
            ("h1", "headline"),
            ("h1", "article-headline"),
            ("h1", "main-headline")
        ]
        for selector, attribute in selectors:
            if attribute:
                h = soup.find(selector, class_=attribute)
            else:
                h = soup.find(selector)
            if h and h.get_text(strip=True):
                return h.get_text(strip=True), res.status_code
        
        return None, res.status_code
    except Exception:
        return None, "error"

def main():
    results = []
    failed_urls = []

    for _, row in tqdm(data.iterrows(), total=len(data)):
        url = row["url"]
        headline, status_code = get_headline(url)
        record = {
            "url": url,
            "headline": headline,
            "source": infer_source(url),
            "status_code": status_code,
        }
        if headline:
            results.append(record)
        else:
            failed_urls.append(record)

    out_df = pd.DataFrame(results)
    failed_df = pd.DataFrame(failed_urls)
    
    print(f"Number of total rows: {len(data)}")
    print(f"Number of successful rows: {len(out_df)}")
    print(f"Number of failed rows: {len(failed_df)}")
    
    out_df.to_csv("data/raw/scraped_headlines.csv", index=False)
    failed_df.to_csv("data/raw/failed_urls.csv", index=False)
    
    print("Saved to data/raw/scraped_headlines.csv")
    print("Saved to data/raw/failed_urls.csv")


if __name__ == "__main__":
    main()