from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features_tfidf(data, ngram_range=(1, 1)):
    """Extract TF-IDF features from text."""
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    features = vectorizer.fit_transform(data["clean_text"])
    return features
