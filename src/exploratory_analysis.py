import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Top words for each source

df = pd.read_csv("data/processed/cleaned_headlines.csv")

fox = df[df["source"] == "Fox"]["cleaned_headline"]
nbc = df[df["source"] == "NBC"]["cleaned_headline"]

vectorizer = CountVectorizer(stop_words="english", max_features=1000)

fox_matrix = vectorizer.fit_transform(fox)
nbc_matrix = vectorizer.transform(nbc)

fox_words = fox_matrix.sum(axis=0).A1
nbc_words = nbc_matrix.sum(axis=0).A1

words = vectorizer.get_feature_names_out() # get the words

fox_words_df = pd.DataFrame({"word": words, "count": fox_words}).sort_values(by="count", ascending=False)
nbc_words_df = pd.DataFrame({"word": words, "count": nbc_words}).sort_values(by="count", ascending=False)

print(fox_words_df.head(10))
print(nbc_words_df.head(10))

# Headline length distribution

df["length"] = df["cleaned_headline"].apply(lambda x: len(x.split()))

print(df.groupby("source")["length"].describe())

#top bigrams for each source

vectorizer = CountVectorizer(stop_words="english", max_features=500, ngram_range=(2, 2))

fox_bigrams = vectorizer.fit_transform(fox)
nbc_bigrams = vectorizer.transform(nbc)

fox_bigrams_words = fox_bigrams.sum(axis=0).A1
nbc_bigrams_words = nbc_bigrams.sum(axis=0).A1

words = vectorizer.get_feature_names_out()

fox_bigrams_df = pd.DataFrame({"word": words, "count": fox_bigrams_words}).sort_values(by="count", ascending=False)
nbc_bigrams_df = pd.DataFrame({"word": words, "count": nbc_bigrams_words}).sort_values(by="count", ascending=False)

print("Fox top bigrams: ", fox_bigrams_df.head(10))
print("NBC top bigrams: ", nbc_bigrams_df.head(10))
